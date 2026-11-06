"""Opt-in trace collection for iterative AHDR rendering and epoch boundaries."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch

from .tensor_metrics import region_response, tensor_metrics


def _safe_id(value: object) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)


@torch.no_grad()
def _rgb01(value: torch.Tensor, value_range: str) -> torch.Tensor:
    image = value.detach().float()
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("PNG export requires CHW or NCHW RGB tensor")
    if value_range == "neg1_1":
        image = image * 0.5 + 0.5
    elif value_range != "zero_one":
        raise ValueError("value_range must be 'neg1_1' or 'zero_one'")
    return image.clamp(0.0, 1.0).cpu()


class DiagnosticRecorder:
    """Collect detached diagnostics without changing model state or predictions."""

    def __init__(
        self,
        output_dir: str,
        enabled: bool = False,
        save_tensors: bool = False,
        save_png: bool = True,
        run_id: str = "run",
        sample_id: str = "sample",
        call_id: str = "call_000000",
    ) -> None:
        self.enabled = bool(enabled)
        self.save_tensors = bool(save_tensors)
        self.save_png = bool(save_png)
        self.records = []
        self._timers: Dict[str, float] = {}
        self.output_dir = (
            Path(output_dir)
            / _safe_id(run_id)
            / _safe_id(sample_id)
            / _safe_id(call_id)
        )
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def start_timer(self, name: str) -> None:
        if not self.enabled:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._timers[name] = time.perf_counter()

    def stop_timer(self, name: str) -> Optional[float]:
        if not self.enabled:
            return None
        if name not in self._timers:
            raise KeyError("timer was not started: %s" % name)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - self._timers.pop(name)) * 1000.0
        self.records.append({"kind": "timing", "name": name, "elapsed_ms": elapsed_ms})
        return elapsed_ms

    def _save_png(self, name: str, value: torch.Tensor, value_range: str) -> None:
        if not self.save_png:
            return
        from PIL import Image

        image = _rgb01(value, value_range)
        array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(array, mode="RGB").save(self.output_dir / (name + ".png"))

    @torch.no_grad()
    def record_inputs(
        self,
        raw_input: torch.Tensor,
        save_im0: torch.Tensor,
    ) -> None:
        if not self.enabled:
            return
        self.records.append(
            {
                "kind": "inputs",
                "raw_input": tensor_metrics(raw_input),
                "save_im0": tensor_metrics(save_im0),
                "ranges": {"raw_input": "[-1,1]", "save_im0": "[0,1]"},
            }
        )
        self._save_png("step00_raw_input", raw_input, "neg1_1")
        self._save_png("step00_save_im0", save_im0, "zero_one")
        if self.save_tensors:
            torch.save(
                {
                    "raw_input": raw_input.detach().cpu(),
                    "save_im0": save_im0.detach().cpu(),
                },
                self.output_dir / "step00_inputs.pt",
            )

    @torch.no_grad()
    def record_ahdr_step(
        self,
        step: int,
        reference: torch.Tensor,
        guide: torch.Tensor,
        previous_output: torch.Tensor,
        raw_render: torch.Tensor,
        blended_output: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
    ) -> None:
        if not self.enabled:
            return
        if not 1 <= int(step) <= 6:
            raise ValueError("AHDR step must be in [1, 6]")
        tensors = (reference, guide, previous_output, raw_render, blended_output)
        if any(item.shape != blended_output.shape for item in tensors):
            raise ValueError("all AHDR step tensors must have identical shapes")
        step_delta = blended_output.detach() - previous_output.detach()
        reference_delta = blended_output.detach() - reference.detach()
        entry: Dict[str, Any] = {
            "kind": "ahdr_step",
            "step": int(step),
            "range": "[0,1]",
            "reference": tensor_metrics(reference),
            "guide": tensor_metrics(guide),
            "previous_output": tensor_metrics(previous_output),
            "raw_render": tensor_metrics(raw_render, reference=previous_output),
            "blended_output": tensor_metrics(blended_output, reference=previous_output),
            "step_delta": tensor_metrics(step_delta),
            "reference_delta": tensor_metrics(reference_delta),
        }
        if target_mask is not None:
            entry["reference_regions"] = region_response(reference, target_mask)
            entry["blended_regions"] = region_response(blended_output, target_mask)
            entry["step_delta_regions"] = region_response(step_delta, target_mask)
        self.records.append(entry)
        prefix = "step%02d" % step
        self._save_png(prefix + "_reference", reference, "zero_one")
        self._save_png(prefix + "_guide", guide, "zero_one")
        self._save_png(prefix + "_previous", previous_output, "zero_one")
        self._save_png(prefix + "_raw_render", raw_render, "zero_one")
        self._save_png(prefix + "_blended", blended_output, "zero_one")
        if self.save_tensors:
            torch.save(
                {
                    "reference": reference.detach().cpu(),
                    "guide": guide.detach().cpu(),
                    "previous_output": previous_output.detach().cpu(),
                    "raw_render": raw_render.detach().cpu(),
                    "blended_output": blended_output.detach().cpu(),
                    "step_delta": step_delta.cpu(),
                    "reference_delta": reference_delta.cpu(),
                },
                self.output_dir / (prefix + ".pt"),
            )

    @torch.no_grad()
    def record_state(
        self, label: str, named_tensors: Mapping[str, torch.Tensor], metadata=None
    ) -> None:
        if not self.enabled:
            return
        self.records.append(
            {
                "kind": "state",
                "label": label,
                "metadata": dict(metadata or {}),
                "tensors": {
                    name: tensor_metrics(value)
                    for name, value in named_tensors.items()
                    if torch.is_tensor(value)
                },
            }
        )

    def record_gradients(self, module: torch.nn.Module, label: str) -> None:
        if not self.enabled:
            return
        gradients = {}
        for name, parameter in module.named_parameters():
            if parameter.grad is not None:
                gradients[name] = float(parameter.grad.detach().float().norm().cpu())
        self.records.append({"kind": "gradients", "label": label, "norms": gradients})

    def record_cuda_memory(self, label: str) -> None:
        if not self.enabled or not torch.cuda.is_available():
            return
        self.records.append(
            {
                "kind": "cuda_memory",
                "label": label,
                "allocated_bytes": int(torch.cuda.memory_allocated()),
                "reserved_bytes": int(torch.cuda.memory_reserved()),
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            }
        )

    def _write_csv(self) -> Path:
        path = self.output_dir / "steps.csv"
        fields = [
            "step", "mean", "std", "l2", "high_frequency_energy",
            "cosine_to_previous", "step_change_rate",
            "step_delta_l2", "reference_delta_l2",
        ]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for record in self.records:
                if record.get("kind") != "ahdr_step":
                    continue
                blended = record["blended_output"]
                writer.writerow(
                    {
                        "step": record["step"],
                        "mean": blended["mean"],
                        "std": blended["std"],
                        "l2": blended["l2"],
                        "high_frequency_energy": blended["high_frequency_energy"],
                        "cosine_to_previous": blended["cosine_to_reference"],
                        "step_change_rate": blended["change_rate"],
                        "step_delta_l2": record["step_delta"]["l2"],
                        "reference_delta_l2": record["reference_delta"]["l2"],
                    }
                )
        return path

    def flush(self) -> Optional[Path]:
        if not self.enabled:
            return None
        path = self.output_dir / "metrics.json"
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(self.records, stream, ensure_ascii=False, indent=2)
        temporary.replace(path)
        self._write_csv()
        return path

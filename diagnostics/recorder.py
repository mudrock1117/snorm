"""Opt-in trace collection for iterative AHDR rendering and epoch boundaries."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

from .tensor_metrics import region_response, tensor_metrics


class DiagnosticRecorder:
    """Collect lightweight JSON metrics and optional detached tensors.

    The recorder is inert when disabled. Call sites must invoke record_* explicitly;
    no model hooks, mode changes, or RNG operations are performed.
    """

    def __init__(
        self,
        output_dir: str,
        enabled: bool = False,
        save_tensors: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.enabled = bool(enabled)
        self.save_tensors = bool(save_tensors)
        self.records = []
        self._timers: Dict[str, float] = {}
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

    @torch.no_grad()
    def record_ahdr_step(
        self,
        step: int,
        reference: torch.Tensor,
        guide: torch.Tensor,
        output: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
    ) -> None:
        if not self.enabled:
            return
        if not 1 <= int(step) <= 6:
            raise ValueError("AHDR step must be in [1, 6]")
        if reference.shape != output.shape:
            raise ValueError("reference and output must have identical shapes")
        residual = output.detach() - reference.detach()
        entry: Dict[str, Any] = {
            "kind": "ahdr_step",
            "step": int(step),
            "reference": tensor_metrics(reference),
            "guide": tensor_metrics(guide),
            "output": tensor_metrics(output, reference=reference),
            "residual": tensor_metrics(residual),
        }
        if target_mask is not None:
            entry["reference_regions"] = region_response(reference, target_mask)
            entry["output_regions"] = region_response(output, target_mask)
            entry["residual_regions"] = region_response(residual, target_mask)
        self.records.append(entry)
        if self.save_tensors:
            torch.save(
                {
                    "reference": reference.detach().cpu(),
                    "guide": guide.detach().cpu(),
                    "output": output.detach().cpu(),
                    "residual": residual.detach().cpu(),
                },
                self.output_dir / ("ahdr_step_%02d.pt" % step),
            )

    @torch.no_grad()
    def record_state(
        self,
        label: str,
        named_tensors: Mapping[str, torch.Tensor],
    ) -> None:
        """Snapshot SNorm parameters/memory before and after an epoch boundary."""
        if not self.enabled:
            return
        entry = {
            "kind": "state",
            "label": label,
            "tensors": {
                name: tensor_metrics(value)
                for name, value in named_tensors.items()
                if torch.is_tensor(value)
            },
        }
        self.records.append(entry)

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

    def flush(self) -> Optional[Path]:
        if not self.enabled:
            return None
        path = self.output_dir / "metrics.json"
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(self.records, stream, ensure_ascii=False, indent=2)
        temporary.replace(path)
        return path

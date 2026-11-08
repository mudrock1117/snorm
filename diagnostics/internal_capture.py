"""Temporary read-only hooks for RPN and ROI diagnostics."""
import json
from contextlib import contextmanager
from pathlib import Path

import torch


def _tensor_summary(value, module_name, field):
    tensor = value.detach()
    floating = tensor.float()
    return {
        "module": module_name,
        "field": field,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "min": float(floating.min().cpu().item()) if tensor.numel() else None,
        "max": float(floating.max().cpu().item()) if tensor.numel() else None,
        "mean": float(floating.mean().cpu().item()) if tensor.numel() else None,
        "std": (
            float(floating.std(unbiased=False).cpu().item())
            if tensor.numel() else None
        ),
    }


def _clone_cpu(value):
    return value.detach().cpu().clone()


class DetectionInternalCapture(object):
    """Capture one forward pass and remove hooks immediately afterwards."""

    def __init__(self, model, output_dir, step, save_tensors=False):
        self.model = model.module if hasattr(model, "module") else model
        self.output_dir = Path(output_dir)
        self.step = int(step)
        self.save_tensors = bool(save_tensors)
        self.handles = []
        self.raw = {}
        self.rpn_proposals_cpu = None
        self.summary = {
            "step": self.step,
            "comparison_warning": (
                "RPN proposals and ROI rows are independently generated per step; "
                "do not compare them by array index across steps"
            ),
        }

    def _rpn_hook(self, module, inputs, output):
        if not isinstance(output, (list, tuple)) or len(output) != 4:
            self.summary["rpn_error"] = "unexpected RPN output structure"
            return
        proposals, _, objectness_logits, decoded_bbox_map = output
        # Proposals are modest compared with dense RPN logits and are required
        # for class-agnostic GT coverage. Clone immediately so later steps cannot
        # overwrite or alias this forward's values.
        self.rpn_proposals_cpu = [_clone_cpu(item) for item in proposals]
        self.summary["rpn"] = {
            "module": "teacher.rpn_norm",
            "proposal_counts": [int(item.shape[0]) for item in proposals],
            "proposals": [
                _tensor_summary(item, "teacher.rpn_norm", "proposals[%d]" % index)
                for index, item in enumerate(proposals)
            ],
            "objectness_logits": _tensor_summary(
                objectness_logits, "teacher.rpn_norm", "objectness_logits"
            ),
            "decoded_bbox_map": _tensor_summary(
                decoded_bbox_map, "teacher.rpn_norm", "decoded_bbox_map"
            ),
        }
        if self.save_tensors:
            self.raw["rpn_proposals"] = [
                item.clone() for item in self.rpn_proposals_cpu
            ]
            self.raw["rpn_objectness_logits"] = _clone_cpu(objectness_logits)
            self.raw["rpn_decoded_bbox_map"] = _clone_cpu(decoded_bbox_map)

    def _roi_hook(self, module, inputs, output):
        if not isinstance(output, (list, tuple)) or len(output) != 9:
            self.summary["roi_error"] = "unexpected BoxHead output structure"
            return
        proposals = output[2]
        box_features = output[3]
        roi_features = output[4]
        class_logits = output[5]
        box_regression = output[8]
        self.summary["roi"] = {
            "module": "teacher.box_head_norm",
            "proposal_counts": [int(item.shape[0]) for item in proposals],
            "class_logits": _tensor_summary(
                class_logits, "teacher.box_head_norm", "class_logits"
            ),
            "box_regression": _tensor_summary(
                box_regression, "teacher.box_head_norm", "box_regression"
            ),
            "roi_features": _tensor_summary(
                roi_features, "teacher.box_head_norm", "roi_features"
            ),
            "box_features": _tensor_summary(
                box_features, "teacher.box_head_norm", "box_features"
            ),
        }
        if self.save_tensors:
            self.raw["roi_class_logits"] = _clone_cpu(class_logits)
            self.raw["roi_box_regression"] = _clone_cpu(box_regression)
            self.raw["roi_proposals"] = [_clone_cpu(item) for item in proposals]

    def __enter__(self):
        teacher = getattr(self.model, "teacher", None)
        if teacher is None:
            raise AttributeError("internal capture requires model.teacher")
        rpn = getattr(teacher, "rpn_norm", None)
        roi = getattr(teacher, "box_head_norm", None)
        if rpn is None or roi is None:
            raise AttributeError(
                "internal capture requires teacher.rpn_norm and teacher.box_head_norm"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.handles = [
            rpn.register_forward_hook(self._rpn_hook),
            roi.register_forward_hook(self._roi_hook),
        ]
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        return False

    def save(self):
        summary_path = self.output_dir / ("step%02d_internal.json" % self.step)
        if self.save_tensors:
            tensor_path = self.output_dir / ("step%02d_internal.pt" % self.step)
            torch.save(self.raw, tensor_path)
            self.summary["tensor_file"] = tensor_path.name
        else:
            self.summary["tensor_file"] = None
        summary_path.write_text(
            json.dumps(self.summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self.summary, summary_path

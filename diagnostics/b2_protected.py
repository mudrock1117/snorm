"""B2 keeps the B1 protected whitelist with independent optimizer labels."""
from __future__ import absolute_import, print_function

from diagnostics.a3_protected import A3_TRAINABLE_NAMES, unwrap_model
from diagnostics.b1_protected import (
    B1_GATE_LR,
    B1_OUTPUT_LR,
    audit_b1_trainable_parameters,
    b1_gate_parameter_names,
    configure_b1_trainable_parameters,
)


B2_OUTPUT_LR = B1_OUTPUT_LR
B2_GATE_LR = B1_GATE_LR


def configure_b2_trainable_parameters(model):
    return configure_b1_trainable_parameters(model)


def audit_b2_trainable_parameters(
        model, optimizer=None, output_lr=B2_OUTPUT_LR,
        gate_lr=B2_GATE_LR):
    inner = unwrap_model(model)
    base = audit_b1_trainable_parameters(
        inner, optimizer=None, output_lr=output_lr, gate_lr=gate_lr
    )
    if any(
            parameter.requires_grad
            for parameter in inner.teacher.box_head_eval.parameters()):
        raise RuntimeError("B2 box_head_eval must remain frozen")
    if optimizer is not None:
        if len(optimizer.param_groups) != 2:
            raise RuntimeError("B2 optimizer must have two groups")
        named = dict(inner.named_parameters())
        gate_names = b1_gate_parameter_names(inner)
        expected = (
            ("b2_new_32_anchor_output", A3_TRAINABLE_NAMES, output_lr),
            ("b2_object_aware_gate", gate_names, gate_lr),
        )
        seen = set()
        for group, spec in zip(optimizer.param_groups, expected):
            group_name, names, learning_rate = spec
            if group.get("name") != group_name:
                raise RuntimeError("B2 optimizer group name mismatch")
            actual_ids = set(id(parameter) for parameter in group["params"])
            expected_ids = set(id(named[name]) for name in names)
            if actual_ids != expected_ids:
                raise RuntimeError("B2 optimizer whitelist mismatch")
            if float(group["lr"]) != float(learning_rate):
                raise RuntimeError("B2 optimizer LR mismatch")
            if seen & actual_ids:
                raise RuntimeError("B2 optimizer duplicates Parameters")
            seen.update(actual_ids)
    base["experiment"] = "B2_OBJECT_AWARE_SELECTIVE_AHDR"
    return base

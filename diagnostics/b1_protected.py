"""B1 whitelist: A3 protected outputs plus the Selective-AHDR gate."""
from __future__ import absolute_import, print_function

from diagnostics.a3_protected import A3_TRAINABLE_NAMES, unwrap_model


B1_GATE_PREFIX = "teacher.selective_ahdr_gate."
B1_OUTPUT_LR = 5e-5
B1_GATE_LR = 1e-5


def b1_gate_parameter_names(model):
    inner = unwrap_model(model)
    names = tuple(
        name for name, _ in inner.named_parameters()
        if name.startswith(B1_GATE_PREFIX)
    )
    if not names:
        raise RuntimeError("B1 Selective-AHDR gate parameters are missing")
    return names


def configure_b1_trainable_parameters(model):
    inner = unwrap_model(model)
    for parameter in inner.parameters():
        parameter.requires_grad_(False)
    named = dict(inner.named_parameters())
    gate_names = b1_gate_parameter_names(inner)
    expected = tuple(A3_TRAINABLE_NAMES) + tuple(gate_names)
    missing = [name for name in expected if name not in named]
    if missing:
        raise RuntimeError("B1 trainable parameters are missing: %s" % missing)
    for name in expected:
        named[name].requires_grad_(True)
    audit_b1_trainable_parameters(inner)
    return (
        [named[name] for name in A3_TRAINABLE_NAMES],
        [named[name] for name in gate_names],
        gate_names,
    )


def audit_b1_trainable_parameters(model, optimizer=None,
                                  output_lr=B1_OUTPUT_LR,
                                  gate_lr=B1_GATE_LR):
    inner = unwrap_model(model)
    gate_names = b1_gate_parameter_names(inner)
    expected = set(A3_TRAINABLE_NAMES) | set(gate_names)
    actual = set(
        name for name, parameter in inner.named_parameters()
        if parameter.requires_grad
    )
    if actual != expected:
        raise RuntimeError(
            "B1 trainable whitelist mismatch; expected=%s actual=%s"
            % (sorted(expected), sorted(actual))
        )

    frozen_modules = (
        ("backbone_0", inner.teacher.backbone_0),
        ("backbone_1", inner.teacher.backbone_1),
        ("backbone_2", inner.teacher.backbone_2),
        ("backbone_3", inner.teacher.backbone_3),
        ("proj_norm", inner.teacher.proj_norm),
        ("rpn_norm.conv", inner.teacher.rpn_norm.conv),
        ("box_head_norm", inner.teacher.box_head_norm),
        ("rpn_eval", inner.teacher.rpn_eval),
    )
    for module_name, module in frozen_modules:
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise RuntimeError("B1 frozen module is trainable: %s" % module_name)

    if optimizer is not None:
        if len(optimizer.param_groups) != 2:
            raise RuntimeError("B1 optimizer must have exactly two groups")
        named = dict(inner.named_parameters())
        groups = (
            ("b1_new_32_anchor_output", A3_TRAINABLE_NAMES, output_lr),
            ("b1_selective_ahdr_gate", gate_names, gate_lr),
        )
        seen = set()
        for group, expected_group in zip(optimizer.param_groups, groups):
            group_name, parameter_names, learning_rate = expected_group
            if group.get("name") != group_name:
                raise RuntimeError("B1 optimizer group name mismatch")
            actual_ids = set(id(parameter) for parameter in group["params"])
            expected_ids = set(id(named[name]) for name in parameter_names)
            if actual_ids != expected_ids:
                raise RuntimeError(
                    "B1 optimizer whitelist mismatch: %s" % group_name
                )
            if float(group["lr"]) != float(learning_rate):
                raise RuntimeError(
                    "B1 optimizer LR mismatch: %s" % group_name
                )
            if seen & actual_ids:
                raise RuntimeError("B1 optimizer contains duplicate Parameters")
            seen.update(actual_ids)
        trainable_ids = set(
            id(parameter) for parameter in inner.parameters()
            if parameter.requires_grad
        )
        if seen != trainable_ids:
            raise RuntimeError("B1 optimizer does not exactly cover trainables")

    trainable_count = sum(
        parameter.numel() for parameter in inner.parameters()
        if parameter.requires_grad
    )
    frozen_count = sum(
        parameter.numel() for parameter in inner.parameters()
        if not parameter.requires_grad
    )
    gate_count = sum(
        parameter.numel() for name, parameter in inner.named_parameters()
        if name in set(gate_names)
    )
    print("[B1 PROTECTED SMALL-ANCHOR + SELECTIVE-AHDR]")
    print("trainable parameter names =", sorted(actual))
    print("gate parameter count =", gate_count)
    print("trainable parameter count =", trainable_count)
    print("frozen parameter count =", frozen_count)
    print("new-anchor output LR =", output_lr)
    print("gate LR =", gate_lr)
    return {
        "trainable_names": sorted(actual),
        "gate_names": list(gate_names),
        "gate_parameter_count": gate_count,
        "trainable_parameter_count": trainable_count,
        "frozen_parameter_count": frozen_count,
    }

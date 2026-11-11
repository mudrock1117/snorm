"""Protected A3 optimization for only the newly added 32-anchor channels."""
from __future__ import absolute_import, print_function

import torch


A3_TRAINABLE_NAMES = (
    "teacher.rpn_norm.cls_logits.weight",
    "teacher.rpn_norm.cls_logits.bias",
    "teacher.rpn_norm.bbox_pred.weight",
    "teacher.rpn_norm.bbox_pred.bias",
)
CLS_NEW_CHANNELS = (0, 5, 10)
BBOX_NEW_CHANNELS = tuple(
    list(range(0, 4)) + list(range(20, 24)) + list(range(40, 44))
)
A4_SHARED_CONV_CANDIDATES = (
    "teacher.rpn_norm.conv.weight",
    "teacher.rpn_norm.conv.bias",
)
A4_OUTPUT_LR = 5e-5
A4_SHARED_CONV_LR = 5e-6


def parse_rpn_training_output(output):
    """Parse the real RPN.forward contract used by ModuleSet mode=5."""
    if not isinstance(output, (tuple, list)) or len(output) != 4:
        raise TypeError(
            "RPN.forward must return (proposals, loss_dict, logits, bbox_pd)"
        )
    proposals, losses, logits, bbox_pd = output
    if not isinstance(losses, dict):
        raise TypeError(
            "RPN.forward second item must be a loss dict; got %s"
            % type(losses).__name__
        )
    missing = [
        key for key in ("rpn_cls_loss", "rpn_reg_loss") if key not in losses
    ]
    if missing:
        raise KeyError("RPN training loss dict is missing keys: %s" % missing)
    return proposals, losses, logits, bbox_pd


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _named_trainable(model):
    return {
        name: parameter
        for name, parameter in unwrap_model(model).named_parameters()
        if parameter.requires_grad
    }


def configure_a3_trainable_parameters(model):
    """Freeze the detector and enable only rpn_norm output-layer Parameters."""
    inner = unwrap_model(model)
    for parameter in inner.parameters():
        parameter.requires_grad_(False)
    named = dict(inner.named_parameters())
    missing = [name for name in A3_TRAINABLE_NAMES if name not in named]
    if missing:
        raise RuntimeError("A3 trainable parameters are missing: %s" % missing)
    for name in A3_TRAINABLE_NAMES:
        named[name].requires_grad_(True)
    audit_a3_trainable_parameters(inner)
    return [named[name] for name in A3_TRAINABLE_NAMES]


def a4_shared_conv_names(model):
    """Return the actual shared-conv Parameters; bias is optional by contract."""
    named = dict(unwrap_model(model).named_parameters())
    weight_name = A4_SHARED_CONV_CANDIDATES[0]
    if weight_name not in named:
        raise RuntimeError("A4 shared RPN conv weight is missing: %s" % weight_name)
    return tuple(name for name in A4_SHARED_CONV_CANDIDATES if name in named)


def configure_a4_trainable_parameters(model):
    """Freeze the detector; enable protected outputs plus the shared RPN conv."""
    inner = unwrap_model(model)
    for parameter in inner.parameters():
        parameter.requires_grad_(False)
    named = dict(inner.named_parameters())
    conv_names = a4_shared_conv_names(inner)
    expected = tuple(A3_TRAINABLE_NAMES) + tuple(conv_names)
    missing = [name for name in expected if name not in named]
    if missing:
        raise RuntimeError("A4 trainable parameters are missing: %s" % missing)
    for name in expected:
        named[name].requires_grad_(True)
    audit_a4_trainable_parameters(inner)
    return (
        [named[name] for name in A3_TRAINABLE_NAMES],
        [named[name] for name in conv_names],
        conv_names,
    )


def audit_a3_trainable_parameters(model, optimizer=None):
    inner = unwrap_model(model)
    actual = set(_named_trainable(inner).keys())
    expected = set(A3_TRAINABLE_NAMES)
    if actual != expected:
        raise RuntimeError(
            "A3 trainable whitelist mismatch; expected=%s actual=%s"
            % (sorted(expected), sorted(actual))
        )

    frozen_modules = (
        ("backbone_0", inner.teacher.backbone_0),
        ("backbone_1", inner.teacher.backbone_1),
        ("backbone_2", inner.teacher.backbone_2),
        ("backbone_3", inner.teacher.backbone_3),
        ("proj_norm", inner.teacher.proj_norm),
        ("box_head_norm", inner.teacher.box_head_norm),
        ("rpn_norm.conv", inner.teacher.rpn_norm.conv),
        ("rpn_eval", inner.teacher.rpn_eval),
    )
    for module_name, module in frozen_modules:
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise RuntimeError("A3 frozen module has trainable params: %s" % module_name)

    if optimizer is not None:
        expected_ids = set(id(parameter) for parameter in _named_trainable(inner).values())
        optimizer_parameters = []
        for group in optimizer.param_groups:
            optimizer_parameters.extend(group["params"])
        optimizer_ids = set(id(parameter) for parameter in optimizer_parameters)
        if len(optimizer_parameters) != len(optimizer_ids):
            raise RuntimeError("A3 optimizer contains duplicate Parameters")
        if optimizer_ids != expected_ids:
            raise RuntimeError(
                "A3 optimizer parameter whitelist mismatch: expected=%d actual=%d"
                % (len(expected_ids), len(optimizer_ids))
            )

    trainable_count = sum(
        parameter.numel() for parameter in inner.parameters()
        if parameter.requires_grad
    )
    frozen_count = sum(
        parameter.numel() for parameter in inner.parameters()
        if not parameter.requires_grad
    )
    print("[A3 PROTECTED SMALL-ANCHOR]")
    print("trainable modules = rpn_norm.cls_logits, rpn_norm.bbox_pred")
    print("trainable parameter names =", sorted(actual))
    print("trainable parameter count =", trainable_count)
    print("frozen parameter count =", frozen_count)
    print("cls trainable channels =", list(CLS_NEW_CHANNELS))
    print("bbox trainable blocks = [0:4,20:24,40:44]")
    return {
        "trainable_names": sorted(actual),
        "trainable_parameter_count": trainable_count,
        "frozen_parameter_count": frozen_count,
    }


def audit_a4_trainable_parameters(
        model, optimizer=None, output_lr=A4_OUTPUT_LR,
        shared_conv_lr=A4_SHARED_CONV_LR):
    inner = unwrap_model(model)
    conv_names = a4_shared_conv_names(inner)
    expected = set(A3_TRAINABLE_NAMES) | set(conv_names)
    actual = set(_named_trainable(inner).keys())
    if actual != expected:
        raise RuntimeError(
            "A4 trainable whitelist mismatch; expected=%s actual=%s"
            % (sorted(expected), sorted(actual))
        )

    frozen_modules = (
        ("backbone_0", inner.teacher.backbone_0),
        ("backbone_1", inner.teacher.backbone_1),
        ("backbone_2", inner.teacher.backbone_2),
        ("backbone_3", inner.teacher.backbone_3),
        ("proj_norm", inner.teacher.proj_norm),
        ("box_head_norm", inner.teacher.box_head_norm),
        ("rpn_eval", inner.teacher.rpn_eval),
    )
    for module_name, module in frozen_modules:
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise RuntimeError("A4 frozen module has trainable params: %s" % module_name)

    if optimizer is not None:
        if len(optimizer.param_groups) != 2:
            raise RuntimeError("A4 optimizer must contain exactly two parameter groups")
        named = dict(inner.named_parameters())
        expected_groups = (
            ("a4_new_32_anchor_output", A3_TRAINABLE_NAMES, output_lr),
            ("a4_rpn_shared_conv", conv_names, shared_conv_lr),
        )
        seen = set()
        for group, (group_name, parameter_names, learning_rate) in zip(
                optimizer.param_groups, expected_groups):
            if group.get("name") != group_name:
                raise RuntimeError(
                    "A4 optimizer group name mismatch: expected=%s actual=%s"
                    % (group_name, group.get("name"))
                )
            actual_ids = set(id(parameter) for parameter in group["params"])
            expected_ids = set(id(named[name]) for name in parameter_names)
            if actual_ids != expected_ids:
                raise RuntimeError("A4 optimizer group whitelist mismatch: %s" % group_name)
            if float(group["lr"]) != float(learning_rate):
                raise RuntimeError(
                    "A4 optimizer group LR mismatch: %s expected=%s actual=%s"
                    % (group_name, learning_rate, group["lr"])
                )
            if seen & actual_ids:
                raise RuntimeError("A4 optimizer contains duplicate Parameters")
            seen.update(actual_ids)
        if seen != set(id(parameter) for parameter in _named_trainable(inner).values()):
            raise RuntimeError("A4 optimizer does not cover the exact trainable whitelist")

    trainable_count = sum(
        parameter.numel() for parameter in inner.parameters()
        if parameter.requires_grad
    )
    print("[A4 PROTECTED-RPN]")
    print("trainable parameter names =", sorted(actual))
    print("output LR =", output_lr)
    print("shared conv LR =", shared_conv_lr)
    print("cls trainable channels =", list(CLS_NEW_CHANNELS))
    print("bbox trainable blocks = [0:4,20:24,40:44]")
    return {
        "trainable_names": sorted(actual),
        "shared_conv_names": list(conv_names),
        "trainable_parameter_count": trainable_count,
    }


def _allowed_channels(name):
    if ".cls_logits." in name:
        return CLS_NEW_CHANNELS
    if ".bbox_pred." in name:
        return BBOX_NEW_CHANNELS
    raise RuntimeError("not an A3 masked output parameter: %s" % name)


def _channel_masks(parameter, allowed_channels):
    if parameter.dim() < 1:
        raise RuntimeError("A3 output Parameter must have an output-channel axis")
    allowed = torch.zeros(
        parameter.size(0), dtype=torch.bool, device=parameter.device
    )
    allowed[list(allowed_channels)] = True
    return allowed, ~allowed


def apply_small_anchor_gradient_mask(model, extra_trainable_names=()):
    """Zero old output gradients; explicitly allow unmasked A4 shared conv."""
    trainable = _named_trainable(model)
    expected = set(A3_TRAINABLE_NAMES) | set(extra_trainable_names)
    if set(trainable.keys()) != expected:
        raise RuntimeError("refusing protected gradient mask outside audited whitelist")
    for name in A3_TRAINABLE_NAMES:
        parameter = trainable[name]
        if parameter.grad is None:
            raise RuntimeError("A3 expected a gradient for %s" % name)
        allowed, protected = _channel_masks(parameter, _allowed_channels(name))
        parameter.grad.data[protected] = 0
        if parameter.grad.data[protected].abs().max().item() != 0:
            raise RuntimeError("failed to zero protected gradient channels: %s" % name)
    return True


class ProtectedSmallAnchorStep(object):
    """Mask gradients and make old channels immune to Adam weight decay/state."""

    def __init__(self, model, extra_trainable_names=()):
        self.model = unwrap_model(model)
        self.extra_trainable_names = tuple(extra_trainable_names)
        trainable = _named_trainable(self.model)
        expected = set(A3_TRAINABLE_NAMES) | set(self.extra_trainable_names)
        if set(trainable.keys()) != expected:
            raise RuntimeError("protected controller requires configured trainable whitelist")
        self.parameters = {
            name: trainable[name] for name in A3_TRAINABLE_NAMES
        }

    def before_step(self):
        apply_small_anchor_gradient_mask(
            self.model, self.extra_trainable_names
        )
        snapshots = {}
        for name, parameter in self.parameters.items():
            _, protected = _channel_masks(parameter, _allowed_channels(name))
            snapshots[name] = parameter.detach()[protected].clone()
        return snapshots

    def after_step(self, optimizer, snapshots):
        deltas = {}
        for name, parameter in self.parameters.items():
            _, protected = _channel_masks(parameter, _allowed_channels(name))
            before = snapshots[name]
            parameter.data[protected] = before
            state = optimizer.state.get(parameter, {})
            for value in state.values():
                if torch.is_tensor(value) and tuple(value.shape) == tuple(parameter.shape):
                    value.data[protected] = 0
            delta = (
                parameter.detach()[protected] - before
            ).abs().max().item()
            deltas[name] = float(delta)
            if delta != 0:
                raise RuntimeError(
                    "A3 protected channels changed after restore: %s delta=%s"
                    % (name, delta)
                )
        return deltas

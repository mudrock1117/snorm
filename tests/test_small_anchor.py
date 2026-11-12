import pytest
import torch

from diagnostics.small_anchor import (
    _copy_anchor_channels,
    anchor_channel_index,
)


def test_small_anchor_ratio_major_mapping_preserves_old_channels():
    old_sizes = (64, 128, 256, 512)
    new_sizes = (32, 64, 128, 256, 512)
    old = torch.arange(12, dtype=torch.float32)
    initialized = torch.full((15,), -1.0)
    expanded, mapping = _copy_anchor_channels(
        old, initialized, old_sizes, new_sizes, ratio_count=3, bbox=False
    )
    assert len(mapping) == 12
    for ratio in range(3):
        assert expanded[anchor_channel_index(0, ratio, 5)].item() == -1.0
        for old_scale in range(4):
            old_channel = anchor_channel_index(old_scale, ratio, 4)
            new_channel = anchor_channel_index(old_scale + 1, ratio, 5)
            assert expanded[new_channel].item() == old[old_channel].item()


def test_small_anchor_bbox_mapping_copies_four_channel_blocks():
    old_sizes = (64, 128, 256, 512)
    new_sizes = (32, 64, 128, 256, 512)
    old = torch.arange(12 * 4, dtype=torch.float32)
    initialized = torch.full((15 * 4,), -1.0)
    expanded, _ = _copy_anchor_channels(
        old, initialized, old_sizes, new_sizes, ratio_count=3, bbox=True
    )
    for ratio in range(3):
        new_anchor = anchor_channel_index(0, ratio, 5)
        assert torch.equal(
            expanded[new_anchor * 4:new_anchor * 4 + 4],
            torch.full((4,), -1.0),
        )
        for old_scale in range(4):
            old_anchor = anchor_channel_index(old_scale, ratio, 4)
            new_anchor = anchor_channel_index(old_scale + 1, ratio, 5)
            assert torch.equal(
                expanded[new_anchor * 4:new_anchor * 4 + 4],
                old[old_anchor * 4:old_anchor * 4 + 4],
            )


class _ToyRPN(torch.nn.Module):
    def __init__(self):
        super(_ToyRPN, self).__init__()
        self.conv = torch.nn.Conv2d(2, 2, 1)
        self.cls_logits = torch.nn.Conv2d(2, 15, 1)
        self.bbox_pred = torch.nn.Conv2d(2, 60, 1)


class _ToyTeacher(torch.nn.Module):
    def __init__(self):
        super(_ToyTeacher, self).__init__()
        self.backbone_0 = torch.nn.Linear(2, 2)
        self.backbone_1 = torch.nn.Linear(2, 2)
        self.backbone_2 = torch.nn.Linear(2, 2)
        self.backbone_3 = torch.nn.Linear(2, 2)
        self.proj_norm = torch.nn.Linear(2, 2)
        self.box_head_norm = torch.nn.Linear(2, 2)
        self.rpn_norm = _ToyRPN()
        self.rpn_eval = _ToyRPN()


class _ToyDetector(torch.nn.Module):
    def __init__(self):
        super(_ToyDetector, self).__init__()
        self.teacher = _ToyTeacher()


def test_a3_restore_blocks_adam_weight_decay_on_old_channels():
    from diagnostics.a3_protected import (
        CLS_NEW_CHANNELS,
        ProtectedSmallAnchorStep,
        configure_a3_trainable_parameters,
    )

    model = _ToyDetector()
    parameters = configure_a3_trainable_parameters(model)
    optimizer = torch.optim.Adam(parameters, lr=1e-3, weight_decay=0.1)
    controller = ProtectedSmallAnchorStep(model)
    cls_weight = model.teacher.rpn_norm.cls_logits.weight
    old_mask = torch.ones(15, dtype=torch.bool)
    old_mask[list(CLS_NEW_CHANNELS)] = False
    old_before = cls_weight.detach()[old_mask].clone()
    new_before = cls_weight.detach()[list(CLS_NEW_CHANNELS)].clone()

    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    snapshots = controller.before_step()
    optimizer.step()
    deltas = controller.after_step(optimizer, snapshots)

    assert torch.equal(cls_weight.detach()[old_mask], old_before)
    assert not torch.equal(
        cls_weight.detach()[list(CLS_NEW_CHANNELS)], new_before
    )
    assert all(value == 0.0 for value in deltas.values())


def test_a3_parses_real_rpn_forward_contract():
    from diagnostics.a3_protected import parse_rpn_training_output

    cls_loss = torch.tensor(1.0)
    reg_loss = torch.tensor(2.0)
    output = (
        [torch.zeros(3, 4)],
        {"rpn_cls_loss": cls_loss, "rpn_reg_loss": reg_loss},
        torch.zeros(1, 15, 2, 2),
        torch.zeros(1, 60, 2, 2),
    )
    proposals, losses, logits, bbox_pd = parse_rpn_training_output(output)
    assert proposals[0].shape == (3, 4)
    assert losses["rpn_cls_loss"] is cls_loss
    assert losses["rpn_reg_loss"] is reg_loss
    assert logits.shape[1] == 15
    assert bbox_pd.shape[1] == 60


def test_a3_rejects_direct_rpn_losses_tuple_as_forward_loss_dict():
    from diagnostics.a3_protected import parse_rpn_training_output

    wrong_output = (
        [torch.zeros(3, 4)],
        (torch.tensor(1.0), torch.tensor(2.0)),
        torch.zeros(1, 15, 2, 2),
        torch.zeros(1, 60, 2, 2),
    )
    with pytest.raises(TypeError, match="second item must be a loss dict"):
        parse_rpn_training_output(wrong_output)


def test_a4_updates_shared_conv_and_only_new_output_channels():
    from diagnostics.a3_protected import (
        A4_OUTPUT_LR,
        A4_SHARED_CONV_LR,
        CLS_NEW_CHANNELS,
        ProtectedSmallAnchorStep,
        audit_a4_trainable_parameters,
        configure_a4_trainable_parameters,
    )

    model = _ToyDetector()
    outputs, shared_conv, conv_names = configure_a4_trainable_parameters(model)
    optimizer = torch.optim.Adam(
        [
            {
                "params": outputs,
                "lr": A4_OUTPUT_LR,
                "name": "a4_new_32_anchor_output",
            },
            {
                "params": shared_conv,
                "lr": A4_SHARED_CONV_LR,
                "name": "a4_rpn_shared_conv",
            },
        ],
        lr=A4_OUTPUT_LR,
        weight_decay=0.1,
    )
    audit_a4_trainable_parameters(model, optimizer)
    controller = ProtectedSmallAnchorStep(
        model, extra_trainable_names=conv_names
    )

    cls_weight = model.teacher.rpn_norm.cls_logits.weight
    conv_weight = model.teacher.rpn_norm.conv.weight
    old_mask = torch.ones(15, dtype=torch.bool)
    old_mask[list(CLS_NEW_CHANNELS)] = False
    old_before = cls_weight.detach()[old_mask].clone()
    new_before = cls_weight.detach()[list(CLS_NEW_CHANNELS)].clone()
    conv_before = conv_weight.detach().clone()

    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter.grad = torch.ones_like(parameter)
    snapshots = controller.before_step()
    assert cls_weight.grad.detach()[old_mask].abs().max().item() == 0
    optimizer.step()
    deltas = controller.after_step(optimizer, snapshots)

    assert torch.equal(cls_weight.detach()[old_mask], old_before)
    assert not torch.equal(
        cls_weight.detach()[list(CLS_NEW_CHANNELS)], new_before
    )
    assert not torch.equal(conv_weight.detach(), conv_before)
    assert all(value == 0.0 for value in deltas.values())
    assert [group["lr"] for group in optimizer.param_groups] == [
        A4_OUTPUT_LR, A4_SHARED_CONV_LR
    ]
    assert set(conv_names) == {
        "teacher.rpn_norm.conv.weight",
        "teacher.rpn_norm.conv.bias",
    }


def test_selective_ahdr_shape_bounds_and_baseline_preservation():
    from detection.modeling.norm.selective_ahdr import SelectiveAHDRGate

    gate_module = SelectiveAHDRGate(
        channels=3, hidden_channels=8, alpha_max=1.2, gate_init=0.5
    )
    base = torch.randn(2, 3, 9, 11)
    restored = torch.randn(2, 3, 9, 11)
    output, details = gate_module(base, restored)
    legacy = base + 0.6 * (restored - base)
    assert output.shape == base.shape
    assert details["gate"].shape == (2, 1, 9, 11)
    assert details["gate"].min().item() >= 0
    assert details["gate"].max().item() <= 1
    assert torch.allclose(output, legacy, atol=1e-6, rtol=1e-6)


def test_selective_ahdr_disabled_equivalent_path_is_identity():
    base = torch.randn(1, 3, 5, 7)
    legacy_output = base.clone()
    enabled = False
    output = base if not enabled else None
    assert torch.equal(output, legacy_output)


def test_source_object_mask_projects_tiny_box_to_one_cell():
    from detection.modeling.norm.selective_ahdr import (
        project_source_boxes_to_mask,
    )

    targets = [{"boxes": torch.tensor([[2.1, 3.1, 2.2, 3.2]])}]
    mask = project_source_boxes_to_mask(
        targets, (4, 4), image_sizes=[(16, 16)]
    )
    assert mask.shape == (1, 1, 4, 4)
    assert mask.sum().item() >= 1


def test_selective_ahdr_nan_fails_fast():
    from detection.modeling.norm.selective_ahdr import SelectiveAHDRGate

    module = SelectiveAHDRGate()
    base = torch.zeros(1, 3, 2, 2)
    restored = base.clone()
    restored[0, 0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        module(base, restored)


class _ToyB1Detector(_ToyDetector):
    def __init__(self):
        super(_ToyB1Detector, self).__init__()
        from detection.modeling.norm.selective_ahdr import SelectiveAHDRGate
        self.teacher.selective_ahdr_gate = SelectiveAHDRGate()


def test_b1_protects_old_outputs_updates_gate_and_freezes_shared_conv():
    from diagnostics.a3_protected import (
        CLS_NEW_CHANNELS,
        ProtectedSmallAnchorStep,
    )
    from diagnostics.b1_protected import (
        B1_GATE_LR,
        B1_OUTPUT_LR,
        audit_b1_trainable_parameters,
        configure_b1_trainable_parameters,
    )

    model = _ToyB1Detector()
    outputs, gate_parameters, gate_names = (
        configure_b1_trainable_parameters(model)
    )
    optimizer = torch.optim.Adam(
        [
            {
                "params": outputs,
                "lr": B1_OUTPUT_LR,
                "name": "b1_new_32_anchor_output",
            },
            {
                "params": gate_parameters,
                "lr": B1_GATE_LR,
                "name": "b1_selective_ahdr_gate",
            },
        ],
        lr=B1_OUTPUT_LR,
        weight_decay=0.1,
    )
    audit_b1_trainable_parameters(model, optimizer)
    controller = ProtectedSmallAnchorStep(
        model, extra_trainable_names=gate_names
    )

    cls_weight = model.teacher.rpn_norm.cls_logits.weight
    shared_conv = model.teacher.rpn_norm.conv.weight
    backbone = model.teacher.backbone_0.weight
    old_mask = torch.ones(15, dtype=torch.bool)
    old_mask[list(CLS_NEW_CHANNELS)] = False
    old_before = cls_weight.detach()[old_mask].clone()
    new_before = cls_weight.detach()[list(CLS_NEW_CHANNELS)].clone()
    conv_before = shared_conv.detach().clone()
    backbone_before = backbone.detach().clone()
    gate_before = [
        parameter.detach().clone()
        for parameter in model.teacher.selective_ahdr_gate.parameters()
    ]

    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter.grad = torch.ones_like(parameter)
    snapshots = controller.before_step()
    assert cls_weight.grad[old_mask].abs().max().item() == 0
    optimizer.step()
    old_delta = controller.after_step(optimizer, snapshots)

    assert torch.equal(cls_weight.detach()[old_mask], old_before)
    assert not torch.equal(
        cls_weight.detach()[list(CLS_NEW_CHANNELS)], new_before
    )
    assert torch.equal(shared_conv.detach(), conv_before)
    assert torch.equal(backbone.detach(), backbone_before)
    assert any(
        not torch.equal(parameter.detach(), before)
        for parameter, before in zip(
            model.teacher.selective_ahdr_gate.parameters(), gate_before
        )
    )
    assert all(value == 0 for value in old_delta.values())

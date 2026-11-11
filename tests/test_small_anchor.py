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

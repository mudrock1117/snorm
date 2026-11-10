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

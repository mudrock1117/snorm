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

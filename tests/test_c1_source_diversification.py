from __future__ import absolute_import, print_function

import random
import torch

from detection.config import cfg
from detection.data.source_diversification import SourceAdaptiveDiversification


def _config():
    local = cfg.clone()
    local.defrost()
    local.C1_SOURCE_DIVERSIFICATION.ENABLED = True
    local.C1_SOURCE_DIVERSIFICATION.APPLY_PROB = 1.0
    local.freeze()
    return local.C1_SOURCE_DIVERSIFICATION


def _batch(with_boxes=True):
    image = torch.linspace(-1, 1, 3 * 48 * 80).view(1, 3, 48, 80)
    boxes = torch.tensor([[10., 7., 11., 8.], [30., 15., 60., 35.]])
    if not with_boxes:
        boxes = boxes[:0]
    return image, [{"img_shape": (73, 45)}], [{"boxes": boxes, "labels": torch.ones(len(boxes), dtype=torch.long)}]


def test_c1_off_is_identity():
    local = cfg.clone()
    local.defrost()
    local.C1_SOURCE_DIVERSIFICATION.ENABLED = False
    local.freeze()
    image, metas, targets = _batch()
    output, stats = SourceAdaptiveDiversification(local.C1_SOURCE_DIVERSIFICATION)(image, metas, targets)
    assert output.data_ptr() == image.data_ptr()
    assert not stats["applied"]


def test_c1_shape_range_boxes_and_finite():
    image, metas, targets = _batch()
    old_boxes = targets[0]["boxes"].clone()
    random.seed(7)
    torch.manual_seed(7)
    output, stats = SourceAdaptiveDiversification(_config())(image, metas, targets)
    assert output.shape == image.shape
    assert torch.equal(targets[0]["boxes"], old_boxes)
    assert torch.isfinite(output).all()
    assert float(output.min()) >= -1.00001
    assert float(output.max()) <= 1.00001
    assert stats["object_mask"].shape == (1, 1, 48, 80)
    assert stats["small_mask"].sum() > 0


def test_c1_seed_reproducible_and_no_object_safe():
    image, metas, targets = _batch(False)
    random.seed(11); torch.manual_seed(11)
    first, first_stats = SourceAdaptiveDiversification(_config())(image, metas, targets)
    random.seed(11); torch.manual_seed(11)
    second, second_stats = SourceAdaptiveDiversification(_config())(image, metas, targets)
    assert torch.equal(first, second)
    assert first_stats["object_coverage"] == 0
    assert second_stats["small_coverage"] == 0

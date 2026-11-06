import json

import pytest
import torch

from diagnostics import DiagnosticRecorder, region_response, tensor_metrics


def test_tensor_metrics_do_not_mutate_input():
    value = torch.randn(2, 3, 8, 8, requires_grad=True)
    before = value.detach().clone()
    metrics = tensor_metrics(value, reference=value)
    assert torch.equal(value.detach(), before)
    assert value.requires_grad
    assert metrics["change_rate"] == pytest.approx(0.0)
    assert metrics["cosine_to_reference"] == pytest.approx(1.0)


def test_disabled_recorder_has_no_side_effects(tmp_path):
    recorder = DiagnosticRecorder(str(tmp_path / "missing"), enabled=False, save_tensors=True)
    value = torch.randn(1, 3, 4, 4)
    recorder.record_ahdr_step(1, value, value, value)
    assert recorder.flush() is None
    assert not (tmp_path / "missing").exists()


def test_records_six_steps_and_regions(tmp_path):
    recorder = DiagnosticRecorder(str(tmp_path), enabled=True)
    reference = torch.ones(1, 3, 4, 4)
    guide = torch.ones_like(reference) * 0.5
    mask = torch.zeros(4, 4)
    mask[:2, :2] = 1
    for step in range(1, 7):
        output = reference + step * 0.01
        recorder.record_ahdr_step(
            step, reference, guide, reference, output, output, mask
        )
    path = recorder.flush()
    records = json.loads(path.read_text(encoding="utf-8"))
    assert [item["step"] for item in records] == list(range(1, 7))
    assert all("blended_regions" in item for item in records)
    assert all("step_delta" in item and "reference_delta" in item for item in records)


def test_region_response_validates_shape():
    with pytest.raises(ValueError):
        region_response(torch.ones(3, 4, 4), torch.ones(4, 4))


def test_unique_run_sample_call_directories(tmp_path):
    value = torch.zeros(1, 3, 4, 4)
    first = DiagnosticRecorder(
        str(tmp_path), enabled=True, run_id="run", sample_id="fog/001", call_id="call_1"
    )
    second = DiagnosticRecorder(
        str(tmp_path), enabled=True, run_id="run", sample_id="fog/001", call_id="call_2"
    )
    first.record_inputs(value, value)
    second.record_inputs(value, value)
    assert first.flush() != second.flush()

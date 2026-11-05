"""Optional diagnostics; importing this package does not enable collection."""

from .recorder import DiagnosticRecorder
from .tensor_metrics import (
    cosine_similarity,
    high_frequency_energy,
    region_response,
    tensor_metrics,
)

__all__ = [
    "DiagnosticRecorder",
    "cosine_similarity",
    "high_frequency_energy",
    "region_response",
    "tensor_metrics",
]

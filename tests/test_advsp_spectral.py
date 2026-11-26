"""Lightweight CPU coverage for AdvSP full and one-sided FFT layouts."""
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detection.modeling.advsp import (
    SpectralPerturbationModule,
    build_frequency_radius,
)


class AdvSPSpectralTest(unittest.TestCase):
    def test_full_and_onesided_frequency_shapes(self):
        height, width = 7, 10
        full = build_frequency_radius(
            height, width, width, device=torch.device('cpu'),
            dtype=torch.float32,
        )
        onesided = build_frequency_radius(
            height, width // 2 + 1, width, device=torch.device('cpu'),
            dtype=torch.float32,
        )
        self.assertEqual(tuple(full.shape), (height, width))
        self.assertEqual(tuple(onesided.shape), (height, width // 2 + 1))
        self.assertTrue(torch.allclose(full[:, :width // 2 + 1], onesided))
        self.assertAlmostEqual(float(onesided[0, 0]), 0.0)

    def test_delta_sampling_per_image_and_reuse(self):
        images = torch.randn(3, 3, 6, 8)
        module = SpectralPerturbationModule(
            csg=False, tsg=False, scg=False,
            delta_min=0.0, delta_max=0.01,
        )
        first, deltas = module(images, return_deltas=True)
        second, reused = module(images, deltas=deltas, return_deltas=True)
        self.assertEqual(tuple(deltas.shape), (3,))
        self.assertGreaterEqual(float(deltas.min()), 0.0)
        self.assertLessEqual(float(deltas.max()), 0.01)
        self.assertTrue(torch.equal(deltas, reused))
        self.assertTrue(torch.allclose(first, second, atol=1e-5, rtol=1e-5))

    def test_legacy_onesided_reconstruction_shape(self):
        height, width = 7, 10
        image = torch.randn(3, height, width)
        module = SpectralPerturbationModule(
            csg=True, tsg=True, scg=True, delta_min=0.0, delta_max=0.01,
        )

        def legacy_forward(value):
            spectrum = torch.fft.rfft2(value, dim=(-2, -1), norm='backward')
            return torch.view_as_real(spectrum), False

        def legacy_inverse(spectrum, modern, out_h, out_w):
            self.assertFalse(modern)
            self.assertEqual(tuple(spectrum.shape[-3:-1]),
                             (height, width // 2 + 1))
            return torch.fft.irfft2(
                torch.view_as_complex(spectrum), s=(out_h, out_w),
                dim=(-2, -1), norm='backward',
            )

        with mock.patch('detection.modeling.advsp._fft2_real',
                        side_effect=legacy_forward), mock.patch(
                'detection.modeling.advsp._ifft2_real',
                side_effect=legacy_inverse):
            output = module._one(
                image, torch.zeros(3), torch.zeros(1), torch.zeros(3),
                torch.tensor(0.005),
            )
        self.assertEqual(tuple(output.shape), tuple(image.shape))
        self.assertTrue(torch.allclose(output, image, atol=1e-5, rtol=1e-5))


if __name__ == '__main__':
    unittest.main()

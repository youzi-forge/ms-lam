from __future__ import annotations

import unittest

import numpy as np

from mslam.preprocessing.shift_suite import (
    apply_gamma_brightness,
    apply_gaussian_noise,
    apply_resolution_shift_down_up,
    brain_percentiles,
    resize_trilinear,
    stable_seed,
)


class ShiftSuiteTest(unittest.TestCase):
    def test_stable_seed_is_deterministic(self) -> None:
        self.assertEqual(stable_seed("patient01", "gamma", "1"), stable_seed("patient01", "gamma", "1"))
        self.assertNotEqual(stable_seed("patient01", "gamma", "1"), stable_seed("patient01", "gamma", "2"))

    def test_brain_percentiles_handle_empty_and_flat_inputs(self) -> None:
        empty = np.zeros((2, 2, 2), dtype=np.float32)
        lo, hi = brain_percentiles(empty, np.zeros_like(empty, dtype=bool))
        self.assertEqual((lo, hi), (0.0, 1.0))

        flat = np.ones((2, 2, 2), dtype=np.float32) * 5.0
        lo, hi = brain_percentiles(flat, np.ones_like(flat, dtype=bool))
        self.assertLess(lo, hi)

    def test_apply_gamma_brightness_identity_preserves_values_in_range(self) -> None:
        x = np.linspace(0.0, 1.0, 8, dtype=np.float32).reshape(2, 2, 2)
        out = apply_gamma_brightness(x, brainmask=np.ones_like(x, dtype=bool), gamma=1.0, brightness_shift=0.0)
        expected = np.clip(x, 0.01, 0.99)
        np.testing.assert_allclose(out, expected)

    def test_apply_gaussian_noise_zero_sigma_reduces_to_clipping(self) -> None:
        x = np.linspace(0.0, 1.0, 8, dtype=np.float32).reshape(2, 2, 2)
        out = apply_gaussian_noise(x, brainmask=np.ones_like(x, dtype=bool), sigma_rel=0.0, seed=123)
        expected = np.clip(x, 0.01, 0.99)
        np.testing.assert_allclose(out, expected)

    def test_resize_trilinear_requires_3d_input(self) -> None:
        with self.assertRaises(ValueError):
            resize_trilinear(np.ones((3, 3), dtype=np.float32), (3, 3, 3))

    def test_resolution_shift_preserves_shape(self) -> None:
        x = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
        out = apply_resolution_shift_down_up(x, factor_xy=2.0, factor_z=2.0)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()

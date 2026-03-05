from __future__ import annotations

import unittest

import numpy as np

from mslam.metrics.longitudinal_metrics import (
    dice_coefficient,
    dilate_binary_mm,
    intensity_change_stats,
    percentile_normalize_in_mask,
    remove_small_components,
)


class LongitudinalMetricsTest(unittest.TestCase):
    def test_dice_coefficient_handles_empty_masks(self) -> None:
        a = np.zeros((2, 2, 2), dtype=np.uint8)
        b = np.zeros((2, 2, 2), dtype=np.uint8)
        self.assertEqual(dice_coefficient(a, b), 1.0)

        b[0, 0, 0] = 1
        self.assertEqual(dice_coefficient(a, b), 0.0)

    def test_percentile_normalize_in_mask_empty_mask_returns_zero_image(self) -> None:
        image = np.ones((3, 3, 3), dtype=np.float32)
        mask = np.zeros_like(image, dtype=bool)
        norm, stats = percentile_normalize_in_mask(image, mask)
        self.assertTrue(np.all(norm == 0.0))
        self.assertEqual(stats.v_lo, 0.0)
        self.assertEqual(stats.v_hi, 1.0)

    def test_intensity_change_stats_are_zero_for_identical_inputs(self) -> None:
        flair = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
        brainmask = np.ones_like(flair, dtype=bool)
        t0, t1, diff = intensity_change_stats(flair, flair.copy(), brainmask)
        self.assertAlmostEqual(t0["median"], t1["median"])
        self.assertEqual(diff["mean"], 0.0)
        self.assertEqual(diff["p95"], 0.0)
        self.assertEqual(diff["frac_gt"], 0.0)

    def test_dilate_binary_mm_expands_single_voxel(self) -> None:
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[2, 2, 2] = True
        dilated = dilate_binary_mm(mask, spacing_xyz=(1.0, 1.0, 1.0), radius_mm=1.0)
        self.assertGreater(int(dilated.sum()), 1)
        self.assertTrue(dilated[2, 2, 2])

    def test_remove_small_components_keeps_large_component_only(self) -> None:
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[1, 1, 1] = True
        mask[3, 3, 3] = True
        mask[3, 3, 4] = True
        cleaned = remove_small_components(mask, min_voxels=2, connectivity=26)
        self.assertFalse(cleaned[1, 1, 1])
        self.assertTrue(cleaned[3, 3, 3])
        self.assertTrue(cleaned[3, 3, 4])


if __name__ == "__main__":
    unittest.main()


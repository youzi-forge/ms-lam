from __future__ import annotations

import math
import unittest

import numpy as np

from mslam.metrics.uncertainty_metrics import (
    finite_quantile,
    make_boundary_band,
    prob_entropy,
    prob_p1mp,
    summarize_in_mask,
    variance_across_probs,
)


class UncertaintyMetricsTest(unittest.TestCase):
    def test_prob_p1mp_and_entropy_clip_probabilities(self) -> None:
        probs = np.array([-1.0, 0.5, 2.0], dtype=np.float32)
        p1mp = prob_p1mp(probs)
        entropy = prob_entropy(probs)
        self.assertTrue(np.all(p1mp >= 0.0))
        self.assertGreater(float(entropy[1]), float(entropy[0]))

    def test_variance_requires_multiple_maps(self) -> None:
        with self.assertRaises(ValueError):
            variance_across_probs([np.zeros((2, 2, 2), dtype=np.float32)])

    def test_summarize_in_mask_handles_empty_masks(self) -> None:
        summary = summarize_in_mask(np.ones((2, 2, 2), dtype=np.float32), np.zeros((2, 2, 2), dtype=bool))
        self.assertEqual(summary.n_voxels, 0)
        self.assertTrue(math.isnan(summary.mean))
        self.assertTrue(math.isnan(summary.p95))

    def test_make_boundary_band_wraps_lesion_without_including_it(self) -> None:
        lesion = np.zeros((5, 5, 5), dtype=bool)
        lesion[2, 2, 2] = True
        brainmask = np.ones_like(lesion, dtype=bool)
        band = make_boundary_band(lesion, brainmask=brainmask, spacing_xyz=(1.0, 1.0, 1.0), radius_mm=1.0)
        self.assertFalse(band[2, 2, 2])
        self.assertGreater(int(band.sum()), 0)

    def test_finite_quantile_ignores_non_finite_values(self) -> None:
        value = finite_quantile([1.0, float("nan"), 3.0, float("inf")], 0.5)
        self.assertEqual(value, 2.0)
        with self.assertRaises(ValueError):
            finite_quantile([1.0, 2.0], 2.0)


if __name__ == "__main__":
    unittest.main()

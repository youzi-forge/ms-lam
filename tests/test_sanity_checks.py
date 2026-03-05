from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from mslam.preprocessing.sanity_checks import check_patient_from_paths


class PatientSanityChecksTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="mslam-sanity-test-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir)

    def _write_nifti(
        self,
        name: str,
        data: np.ndarray,
        *,
        affine: np.ndarray | None = None,
        zooms: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Path:
        path = self.tmpdir / name
        image = nib.Nifti1Image(data.astype(np.float32), np.eye(4) if affine is None else affine)
        image.header.set_zooms(zooms)
        nib.save(image, str(path))
        return path

    def _paths(self) -> dict[str, Path]:
        return {
            "t0_flair": self.tmpdir / "t0_flair.nii.gz",
            "t1_flair": self.tmpdir / "t1_flair.nii.gz",
            "brainmask": self.tmpdir / "brainmask.nii.gz",
            "gt_change": self.tmpdir / "gt_change.nii.gz",
            "t0_t1w": self.tmpdir / "t0_t1w.nii.gz",
            "t1_t1w": self.tmpdir / "t1_t1w.nii.gz",
        }

    def test_missing_files_are_reported(self) -> None:
        paths = self._paths()
        self._write_nifti("t0_flair.nii.gz", np.zeros((2, 2, 2), dtype=np.uint8))
        result = check_patient_from_paths("patient01", paths)
        self.assertFalse(result.ok)
        self.assertIn("t1_flair", result.missing_files)
        self.assertIn("gt_change", result.missing_files)

    def test_ok_case_computes_gt_volume(self) -> None:
        affine = np.diag([1.0, 2.0, 3.0, 1.0]).astype(np.float64)
        gt = np.zeros((2, 2, 2), dtype=np.uint8)
        gt[0, 0, 0] = 1
        gt[1, 1, 1] = 1
        for name in self._paths():
            data = gt if name == "gt_change" else np.zeros((2, 2, 2), dtype=np.uint8)
            self._write_nifti(f"{name}.nii.gz", data, affine=affine, zooms=(1.0, 2.0, 3.0))

        result = check_patient_from_paths("patient01", self._paths())
        self.assertTrue(result.ok)
        self.assertEqual(result.gt_voxels, 2)
        self.assertAlmostEqual(result.gt_volume_mm3 or 0.0, 12.0)
        self.assertTrue(result.shape_ok)
        self.assertTrue(result.affine_ok)
        self.assertTrue(result.zooms_ok)

    def test_affine_mismatch_is_flagged(self) -> None:
        affine_ok = np.eye(4)
        affine_bad = np.array(
            [
                [1.0, 0.0, 0.0, 5.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        for name in self._paths():
            affine = affine_bad if name == "gt_change" else affine_ok
            self._write_nifti(f"{name}.nii.gz", np.zeros((2, 2, 2), dtype=np.uint8), affine=affine)

        result = check_patient_from_paths("patient01", self._paths())
        self.assertFalse(result.ok)
        self.assertFalse(result.affine_ok)
        self.assertIn("affine mismatch", result.notes)


if __name__ == "__main__":
    unittest.main()


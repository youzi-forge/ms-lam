from __future__ import annotations

import gzip
import shutil
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from mslam.io.nifti import allclose_affine, allclose_zooms, read_array, read_header


class NiftiIoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="mslam-nifti-test-"))

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

    def test_read_header_and_array_round_trip(self) -> None:
        affine = np.array(
            [
                [1.0, 0.0, 0.0, 2.0],
                [0.0, 2.0, 0.0, 3.0],
                [0.0, 0.0, 3.0, 4.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        path = self._write_nifti("sample.nii.gz", data, affine=affine, zooms=(1.0, 2.0, 3.0))

        header = read_header(path)
        self.assertEqual(header.shape, (2, 3, 4))
        self.assertEqual(header.zooms, (1.0, 2.0, 3.0))
        self.assertTrue(allclose_affine(header.affine, affine))
        self.assertAlmostEqual(header.voxel_volume_mm3, 6.0)

        header2, array = read_array(path)
        self.assertEqual(header2.shape, header.shape)
        np.testing.assert_allclose(array, data)

    def test_short_file_raises(self) -> None:
        path = self.tmpdir / "broken.nii.gz"
        with gzip.open(path, "wb") as f:
            f.write(b"short")
        with self.assertRaises(ValueError):
            read_header(path)

    def test_allclose_zooms(self) -> None:
        self.assertTrue(allclose_zooms((1.0, 1.0, 1.0), (1.0, 1.0, 1.0 + 1e-7)))
        self.assertFalse(allclose_zooms((1.0, 1.0, 1.0), (1.0, 1.0, 1.01)))


if __name__ == "__main__":
    unittest.main()


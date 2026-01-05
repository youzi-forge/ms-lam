from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mslam.io.nifti import allclose_affine, allclose_zooms, read_array, read_header


@dataclass(frozen=True)
class PatientSanityResult:
    patient_id: str
    ok: bool
    missing_files: list[str]
    shape_ok: bool
    affine_ok: bool
    zooms_ok: bool
    spacing_x: float | None
    spacing_y: float | None
    spacing_z: float | None
    gt_voxels: int | None
    gt_volume_mm3: float | None
    notes: str

    def to_row(self) -> dict[str, str]:
        return {
            "patient_id": self.patient_id,
            "ok": str(self.ok),
            "missing_files": ",".join(self.missing_files),
            "shape_ok": str(self.shape_ok),
            "affine_ok": str(self.affine_ok),
            "zooms_ok": str(self.zooms_ok),
            "spacing_x": "" if self.spacing_x is None else f"{self.spacing_x:.9f}",
            "spacing_y": "" if self.spacing_y is None else f"{self.spacing_y:.9f}",
            "spacing_z": "" if self.spacing_z is None else f"{self.spacing_z:.9f}",
            "gt_voxels": "" if self.gt_voxels is None else str(self.gt_voxels),
            "gt_volume_mm3": "" if self.gt_volume_mm3 is None else f"{self.gt_volume_mm3:.3f}",
            "notes": self.notes,
        }


def check_patient_from_paths(patient_id: str, paths: dict[str, str | Path]) -> PatientSanityResult:
    required = ["t0_flair", "t1_flair", "brainmask", "gt_change", "t0_t1w", "t1_t1w"]
    missing = [k for k in required if not Path(paths[k]).exists()]
    if missing:
        return PatientSanityResult(
            patient_id=patient_id,
            ok=False,
            missing_files=missing,
            shape_ok=False,
            affine_ok=False,
            zooms_ok=False,
            spacing_x=None,
            spacing_y=None,
            spacing_z=None,
            gt_voxels=None,
            gt_volume_mm3=None,
            notes="missing required files",
        )

    # Header-based checks
    hdr_t0 = read_header(paths["t0_flair"])
    hdr_t1 = read_header(paths["t1_flair"])
    hdr_bm = read_header(paths["brainmask"])
    hdr_gt = read_header(paths["gt_change"])

    shape_ok = (hdr_t0.shape == hdr_t1.shape == hdr_bm.shape == hdr_gt.shape)
    affine_ok = allclose_affine(hdr_t0.affine, hdr_t1.affine) and allclose_affine(hdr_t0.affine, hdr_bm.affine) and allclose_affine(hdr_t0.affine, hdr_gt.affine)
    zooms_ok = allclose_zooms(hdr_t0.zooms, hdr_t1.zooms) and allclose_zooms(hdr_t0.zooms, hdr_bm.zooms) and allclose_zooms(hdr_t0.zooms, hdr_gt.zooms)

    spacing_x, spacing_y, spacing_z = hdr_t0.zooms

    # Data-based GT volume
    _, gt = read_array(paths["gt_change"])
    gt_voxels = int(np.sum(gt > 0))
    gt_volume_mm3 = float(gt_voxels * hdr_t0.voxel_volume_mm3)

    ok = bool(shape_ok and affine_ok)
    notes = ""
    if not shape_ok:
        notes += "shape mismatch; "
    if not affine_ok:
        notes += "affine mismatch; "
    if not zooms_ok:
        notes += "zooms differ (tiny float diffs are possible); "
    notes = notes.strip()

    return PatientSanityResult(
        patient_id=patient_id,
        ok=ok,
        missing_files=[],
        shape_ok=bool(shape_ok),
        affine_ok=bool(affine_ok),
        zooms_ok=bool(zooms_ok),
        spacing_x=float(spacing_x),
        spacing_y=float(spacing_y),
        spacing_z=float(spacing_z),
        gt_voxels=gt_voxels,
        gt_volume_mm3=gt_volume_mm3,
        notes=notes,
    )


from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def parse_csv_list(values: str) -> list[str]:
    return [value.strip() for value in values.split(",") if value.strip()]


def parse_patients_arg(patients: str) -> set[str]:
    return set(parse_csv_list(patients))


def parse_int_list(values: str) -> list[int]:
    return [int(value) for value in parse_csv_list(values)]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: object) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(obj, indent=2))


def safe_float(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else float("nan")


def save_nifti_like(ref_path: Path, out_path: Path, data: object) -> None:
    import nibabel as nib
    import numpy as np

    ref_img = nib.load(str(ref_path))
    arr = np.asarray(data)
    if tuple(arr.shape) != tuple(ref_img.shape[:3]):
        raise ValueError(f"Shape mismatch for write: ref={ref_img.shape} vs data={arr.shape}")
    out_img = nib.Nifti1Image(arr.astype(np.float32, copy=False), ref_img.affine, ref_img.header)
    out_img.header.set_data_dtype(np.float32)
    try:
        out_img.header.set_zooms(ref_img.header.get_zooms()[:3])
    except Exception:
        pass
    ensure_parent(out_path)
    nib.save(out_img, str(out_path))


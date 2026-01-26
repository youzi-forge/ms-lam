from __future__ import annotations

import gzip
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class NiftiHeader:
    shape: tuple[int, int, int]
    zooms: tuple[float, float, float]
    datatype: int
    bitpix: int
    vox_offset: float
    scl_slope: float
    scl_inter: float
    qform_code: int
    sform_code: int
    srow_x: tuple[float, float, float, float]
    srow_y: tuple[float, float, float, float]
    srow_z: tuple[float, float, float, float]
    endian: Literal["<", ">"]

    @property
    def affine(self) -> np.ndarray:
        return np.array([self.srow_x, self.srow_y, self.srow_z, (0.0, 0.0, 0.0, 1.0)], dtype=np.float64)

    @property
    def voxel_volume_mm3(self) -> float:
        sx, sy, sz = self.zooms
        return float(sx * sy * sz)


_DTYPE_MAP: dict[int, np.dtype] = {
    2: np.dtype(np.uint8),
    4: np.dtype(np.int16),
    8: np.dtype(np.int32),
    16: np.dtype(np.float32),
    64: np.dtype(np.float64),
    256: np.dtype(np.int8),
    512: np.dtype(np.uint16),
    768: np.dtype(np.uint32),
    1024: np.dtype(np.int64),
    1280: np.dtype(np.uint64),
}


def _detect_endian(hdr: bytes) -> Literal["<", ">"]:
    if len(hdr) < 4:
        raise ValueError("Header too short")
    sizeof_hdr_le = struct.unpack_from("<i", hdr, 0)[0]
    if sizeof_hdr_le == 348:
        return "<"
    sizeof_hdr_be = struct.unpack_from(">i", hdr, 0)[0]
    if sizeof_hdr_be == 348:
        return ">"
    raise ValueError(f"Invalid NIfTI header (sizeof_hdr={sizeof_hdr_le}/{sizeof_hdr_be})")


def read_header(path: str | Path) -> NiftiHeader:
    path = Path(path)
    with gzip.open(path, "rb") as f:
        hdr = f.read(348)
    if len(hdr) != 348:
        raise ValueError(f"Short NIfTI header: {path}")

    endian = _detect_endian(hdr)
    dim = struct.unpack_from(endian + "8h", hdr, 40)
    pixdim = struct.unpack_from(endian + "8f", hdr, 76)
    datatype = int(struct.unpack_from(endian + "h", hdr, 70)[0])
    bitpix = int(struct.unpack_from(endian + "h", hdr, 72)[0])
    vox_offset = float(struct.unpack_from(endian + "f", hdr, 108)[0])
    scl_slope = float(struct.unpack_from(endian + "f", hdr, 112)[0])
    scl_inter = float(struct.unpack_from(endian + "f", hdr, 116)[0])
    qform_code = int(struct.unpack_from(endian + "h", hdr, 252)[0])
    sform_code = int(struct.unpack_from(endian + "h", hdr, 254)[0])
    srow_x = struct.unpack_from(endian + "4f", hdr, 280)
    srow_y = struct.unpack_from(endian + "4f", hdr, 296)
    srow_z = struct.unpack_from(endian + "4f", hdr, 312)

    shape = (int(dim[1]), int(dim[2]), int(dim[3]))
    zooms = (float(pixdim[1]), float(pixdim[2]), float(pixdim[3]))

    return NiftiHeader(
        shape=shape,
        zooms=zooms,
        datatype=datatype,
        bitpix=bitpix,
        vox_offset=vox_offset,
        scl_slope=scl_slope,
        scl_inter=scl_inter,
        qform_code=qform_code,
        sform_code=sform_code,
        srow_x=tuple(float(x) for x in srow_x),
        srow_y=tuple(float(x) for x in srow_y),
        srow_z=tuple(float(x) for x in srow_z),
        endian=endian,
    )


def allclose_affine(a: np.ndarray, b: np.ndarray, atol: float = 1e-3) -> bool:
    if a.shape != (4, 4) or b.shape != (4, 4):
        return False
    diff = np.abs(a - b)
    return bool(np.all(diff <= atol))


def allclose_zooms(a: tuple[float, float, float], b: tuple[float, float, float], atol: float = 1e-6) -> bool:
    return all(abs(x - y) <= atol for x, y in zip(a, b, strict=True))


def read_array(path: str | Path) -> tuple[NiftiHeader, np.ndarray]:
    path = Path(path)
    with gzip.open(path, "rb") as f:
        raw = f.read()
    if len(raw) < 348:
        raise ValueError(f"Short NIfTI file: {path}")

    endian = _detect_endian(raw[:348])
    hdr = read_header(path)

    dtype = _DTYPE_MAP.get(hdr.datatype)
    if dtype is None:
        raise ValueError(f"Unsupported NIfTI datatype {hdr.datatype} in {path}")

    offset = int(round(hdr.vox_offset))
    nvox = int(np.prod(hdr.shape))
    arr = np.frombuffer(raw[offset:], dtype=dtype, count=nvox)
    if arr.size != nvox:
        raise ValueError(f"Short NIfTI data: {path} ({arr.size} < {nvox})")

    arr = arr.reshape(hdr.shape, order="F")

    if not (math.isfinite(hdr.scl_slope) and math.isfinite(hdr.scl_inter)):
        return hdr, arr

    if hdr.scl_slope not in (0.0, 1.0) or hdr.scl_inter != 0.0:
        if hdr.scl_slope != 0.0:
            arr = arr.astype(np.float32, copy=False) * float(hdr.scl_slope) + float(hdr.scl_inter)
    return hdr, arr


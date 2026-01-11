from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from mslam.io.nifti import read_header


@dataclass(frozen=True)
class LstAiDockerConfig:
    image: str = "jqmcginnis/lst-ai:v1.2.0"
    platform: str | None = "linux/amd64"
    device: str = "cpu"  # "cpu" or GPU id like "0"
    threads: int | None = None
    segment_only: bool = True
    stripped: bool = False
    threshold: float = 0.5
    lesion_threshold: int = 0
    clipping: tuple[float, float] = (0.5, 99.5)
    fast_mode: bool = False
    probability_map: bool = True


@dataclass(frozen=True)
class LstAiRunResult:
    cmd: list[str]
    returncode: int
    runtime_sec: float
    run_dir: Path
    lst_output_dir: Path
    lst_temp_dir: Path | None
    lesion_mask: Path
    lesion_prob: Path | None
    lesion_prob_model1: Path | None
    lesion_prob_model2: Path | None
    lesion_prob_model3: Path | None
    stdout_log: Path
    stderr_log: Path
    metadata_json: Path


def _as_abs(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _build_docker_cmd(
    *,
    t1_path: Path,
    flair_path: Path,
    output_dir: Path,
    temp_dir: Path | None,
    cfg: LstAiDockerConfig,
) -> list[str]:
    cmd: list[str] = ["docker", "run", "--rm"]
    if cfg.platform:
        cmd += ["--platform", cfg.platform]

    # Inputs: mount as read-only files to avoid path issues/spaces.
    cmd += [
        "-v",
        f"{t1_path}:/input/t1.nii.gz:ro",
        "-v",
        f"{flair_path}:/input/flair.nii.gz:ro",
        "-v",
        f"{output_dir}:/output",
    ]
    if temp_dir is not None:
        cmd += ["-v", f"{temp_dir}:/temp"]

    cmd += [
        cfg.image,
        "--t1",
        "/input/t1.nii.gz",
        "--flair",
        "/input/flair.nii.gz",
        "--output",
        "/output",
        "--device",
        str(cfg.device),
        "--threshold",
        str(cfg.threshold),
        "--lesion_threshold",
        str(cfg.lesion_threshold),
        "--clipping",
        str(cfg.clipping[0]),
        str(cfg.clipping[1]),
    ]
    if cfg.threads is not None:
        cmd += ["--threads", str(cfg.threads)]
    if cfg.segment_only:
        cmd.append("--segment_only")
    if cfg.stripped:
        cmd.append("--stripped")
    if cfg.fast_mode:
        cmd.append("--fast-mode")
    if cfg.probability_map:
        if temp_dir is None:
            raise ValueError("probability_map=True requires a temp_dir (LST-AI stores probmaps under --temp).")
        cmd.append("--probability_map")
        cmd += ["--temp", "/temp"]
    elif temp_dir is not None:
        # If user provided temp_dir, keep intermediates.
        cmd += ["--temp", "/temp"]
    return cmd


def run_lstai_docker(
    *,
    t1_path: str | Path,
    flair_path: str | Path,
    run_dir: str | Path,
    cfg: LstAiDockerConfig | None = None,
    keep_temp: bool = False,
    overwrite: bool = False,
    timeout_sec: float | None = None,
) -> LstAiRunResult:
    """
    Run LST-AI (Docker) once and export canonical outputs into `run_dir`:
      - lesion_mask.nii.gz
      - lesion_prob.nii.gz (ensemble, if probability_map enabled)
      - lesion_prob_model{1,2,3}.nii.gz (if probability_map enabled)

    Notes:
    - LST-AI `--output` is a DIRECTORY (despite some help text). It will write
      `space-flair_seg-lst.nii.gz` into that directory.
    - With `--probability_map`, LST-AI writes probmaps into `--temp` and does NOT
      copy them into `--output`; this wrapper copies them out.
    """
    cfg = cfg or LstAiDockerConfig()

    t1_path = _as_abs(t1_path)
    flair_path = _as_abs(flair_path)
    run_dir = _as_abs(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    lst_output_dir = run_dir / "_lstai_output"
    lst_temp_dir = (run_dir / "_lstai_temp") if (cfg.probability_map or keep_temp) else None

    if overwrite:
        if lst_output_dir.exists():
            shutil.rmtree(lst_output_dir)
        if lst_temp_dir is not None and lst_temp_dir.exists():
            shutil.rmtree(lst_temp_dir)
    lst_output_dir.mkdir(parents=True, exist_ok=True)
    if lst_temp_dir is not None:
        lst_temp_dir.mkdir(parents=True, exist_ok=True)

    stdout_log = run_dir / "lstai_stdout.txt"
    stderr_log = run_dir / "lstai_stderr.txt"
    metadata_json = run_dir / "lstai_run.json"

    cmd = _build_docker_cmd(
        t1_path=t1_path,
        flair_path=flair_path,
        output_dir=lst_output_dir,
        temp_dir=lst_temp_dir,
        cfg=cfg,
    )

    started = time.time()
    with stdout_log.open("w") as out_f, stderr_log.open("w") as err_f:
        proc = subprocess.run(
            cmd,
            stdout=out_f,
            stderr=err_f,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    runtime_sec = time.time() - started

    def write_metadata(extra: dict[str, object] | None = None) -> None:
        meta = {
            "t1_path": str(t1_path),
            "flair_path": str(flair_path),
            "run_dir": str(run_dir),
            "lst_output_dir": str(lst_output_dir),
            "lst_temp_dir": "" if lst_temp_dir is None else str(lst_temp_dir),
            "docker_image": cfg.image,
            "docker_platform": cfg.platform,
            "device": cfg.device,
            "threads": cfg.threads,
            "segment_only": cfg.segment_only,
            "stripped": cfg.stripped,
            "threshold": cfg.threshold,
            "lesion_threshold": cfg.lesion_threshold,
            "clipping": list(cfg.clipping),
            "fast_mode": cfg.fast_mode,
            "probability_map": cfg.probability_map,
            "cmd": cmd,
            "returncode": proc.returncode,
            "runtime_sec": runtime_sec,
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
        }
        if extra:
            meta.update(extra)
        metadata_json.write_text(json.dumps(meta, indent=2))

    # Always write base metadata immediately after the run.
    write_metadata()

    # Canonical output paths
    lesion_mask = run_dir / "lesion_mask.nii.gz"
    lesion_prob = run_dir / "lesion_prob.nii.gz"
    lesion_prob_model1 = run_dir / "lesion_prob_model1.nii.gz"
    lesion_prob_model2 = run_dir / "lesion_prob_model2.nii.gz"
    lesion_prob_model3 = run_dir / "lesion_prob_model3.nii.gz"

    # Export segmentation mask to canonical name
    mask_src = lst_output_dir / "space-flair_seg-lst.nii.gz"
    try:
        if proc.returncode == 0:
            if not mask_src.exists():
                raise FileNotFoundError(
                    "LST-AI finished with returncode 0 but expected output was not found: "
                    f"{mask_src}. Check {stdout_log} and {stderr_log}."
                )
            shutil.copyfile(mask_src, lesion_mask)

            # Export probability maps (if enabled)
            if cfg.probability_map:
                if lst_temp_dir is None:
                    raise RuntimeError("Internal error: probability_map=True but lst_temp_dir is None.")
                prob_src = lst_temp_dir / "sub-X_ses-Y_space-FLAIR_seg-lst_prob.nii.gz"
                p1_src = lst_temp_dir / "sub-X_ses-Y_space-FLAIR_seg-lst_prob_1.nii.gz"
                p2_src = lst_temp_dir / "sub-X_ses-Y_space-FLAIR_seg-lst_prob_2.nii.gz"
                p3_src = lst_temp_dir / "sub-X_ses-Y_space-FLAIR_seg-lst_prob_3.nii.gz"

                missing = [p for p in [prob_src, p1_src, p2_src, p3_src] if not p.exists()]
                if missing:
                    raise FileNotFoundError(
                        "LST-AI ran with --probability_map but expected probmaps were not found under --temp. "
                        f"Missing: {', '.join(str(p) for p in missing)}. Check {stdout_log} and {stderr_log}."
                    )
                shutil.copyfile(prob_src, lesion_prob)
                shutil.copyfile(p1_src, lesion_prob_model1)
                shutil.copyfile(p2_src, lesion_prob_model2)
                shutil.copyfile(p3_src, lesion_prob_model3)

                if not keep_temp and lst_temp_dir.exists():
                    shutil.rmtree(lst_temp_dir)
                    lst_temp_dir = None
    except Exception as e:
        # Persist error context to metadata for easier debugging, then re-raise.
        write_metadata({"export_error": f"{type(e).__name__}: {e}"})
        raise

    # Update metadata after exporting (e.g. temp dir cleaned up).
    write_metadata(
        {
            "lesion_mask": str(lesion_mask) if lesion_mask.exists() else "",
            "lesion_prob": str(lesion_prob) if lesion_prob.exists() else "",
            "lesion_prob_model1": str(lesion_prob_model1) if lesion_prob_model1.exists() else "",
            "lesion_prob_model2": str(lesion_prob_model2) if lesion_prob_model2.exists() else "",
            "lesion_prob_model3": str(lesion_prob_model3) if lesion_prob_model3.exists() else "",
            "lst_temp_dir": "" if lst_temp_dir is None else str(lst_temp_dir),
        }
    )

    # Optional sanity: output shape matches input FLAIR shape (when successful)
    if proc.returncode == 0 and lesion_mask.exists():
        flair_hdr = read_header(flair_path)
        mask_hdr = read_header(lesion_mask)
        if flair_hdr.shape != mask_hdr.shape:
            raise ValueError(f"Shape mismatch: FLAIR {flair_hdr.shape} vs lesion_mask {mask_hdr.shape} ({lesion_mask})")
        if cfg.probability_map and lesion_prob.exists():
            prob_hdr = read_header(lesion_prob)
            if flair_hdr.shape != prob_hdr.shape:
                raise ValueError(f"Shape mismatch: FLAIR {flair_hdr.shape} vs lesion_prob {prob_hdr.shape} ({lesion_prob})")

    return LstAiRunResult(
        cmd=cmd,
        returncode=proc.returncode,
        runtime_sec=runtime_sec,
        run_dir=run_dir,
        lst_output_dir=lst_output_dir,
        lst_temp_dir=lst_temp_dir,
        lesion_mask=lesion_mask,
        lesion_prob=(lesion_prob if (cfg.probability_map and lesion_prob.exists()) else None),
        lesion_prob_model1=(lesion_prob_model1 if (cfg.probability_map and lesion_prob_model1.exists()) else None),
        lesion_prob_model2=(lesion_prob_model2 if (cfg.probability_map and lesion_prob_model2.exists()) else None),
        lesion_prob_model3=(lesion_prob_model3 if (cfg.probability_map and lesion_prob_model3.exists()) else None),
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        metadata_json=metadata_json,
    )

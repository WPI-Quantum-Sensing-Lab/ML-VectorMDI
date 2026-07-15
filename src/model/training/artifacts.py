"""Helpers for writing unified run artifacts."""
from __future__ import annotations

import csv
import datetime as _dt
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


RUN_ID_TIME_FORMAT = "%Y%m%d_%H%M"


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _run_git_command(args: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _to_serialisable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, _dt.datetime):
        return value.isoformat()
    if isinstance(value, _dt.date):
        return value.isoformat()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Mapping):
        return {k: _to_serialisable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serialisable(v) for v in value]
    return str(value)


def generate_run_id(timestamp: _dt.datetime | None = None) -> str:
    stamp = (timestamp or _utcnow()).strftime(RUN_ID_TIME_FORMAT)
    short_hash = _run_git_command(["rev-parse", "--short=6", "HEAD"]) or "nogit"
    return f"{stamp}_{short_hash}"


def init_run(
    *,
    run_root: str | Path = "runs",
    run_dir: str | Path | None = None,
    run_id: str | None = None,
    model_name: str | None = None,
    dataset_path: Any = None,
    seed: int | None = None,
    cli_args: Any = None,
) -> tuple[Path, str]:
    timestamp = _utcnow()
    if run_dir is None:
        run_id = run_id or generate_run_id(timestamp)
        run_dir_path = Path(run_root) / run_id
    else:
        run_dir_path = Path(run_dir)
        run_id = run_id or run_dir_path.name or generate_run_id(timestamp)

    run_dir_path.mkdir(parents=True, exist_ok=True)
    for subdir in ("metrics", "plots", "diagnostics", "arrays"):
        (run_dir_path / subdir).mkdir(parents=True, exist_ok=True)

    run_json_path = run_dir_path / "run.json"
    if not run_json_path.exists():
        run_payload = {
            "run_id": run_id,
            "timestamp": timestamp.isoformat(),
            "git_commit": _run_git_command(["rev-parse", "HEAD"]),
            "git_dirty": bool(_run_git_command(["status", "--porcelain"])),
            "model_name": model_name,
            "dataset_path": _to_serialisable(dataset_path),
            "seed": seed,
            "cli_args": _to_serialisable(vars(cli_args) if hasattr(cli_args, "__dict__") else cli_args),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
        }
        with run_json_path.open("w", encoding="utf-8") as handle:
            json.dump(_to_serialisable(run_payload), handle, indent=2)

    artifacts_path = run_dir_path / "artifacts.json"
    if not artifacts_path.exists():
        with artifacts_path.open("w", encoding="utf-8") as handle:
            json.dump([], handle, indent=2)

    return run_dir_path, run_id


def _relative_to_run(run_dir: Path, path: str | Path) -> str:
    if path is None:
        return ""
    path_obj = Path(path)
    if not path_obj.is_absolute():
        return str(path_obj)
    try:
        return str(path_obj.relative_to(run_dir))
    except ValueError:
        return str(path_obj)


def register_artifact(run_dir: str | Path, artifact_type: str, path: str | Path) -> None:
    run_dir_path = Path(run_dir)
    artifacts_path = run_dir_path / "artifacts.json"
    created_at = _utcnow().isoformat()
    entry = {
        "type": artifact_type,
        "relative_path": _relative_to_run(run_dir_path, path),
        "created_at": created_at,
    }
    artifacts: list[dict[str, Any]] = []
    if artifacts_path.exists():
        try:
            with artifacts_path.open("r", encoding="utf-8") as handle:
                artifacts = json.load(handle) or []
        except json.JSONDecodeError:
            artifacts = []
    artifacts.append(entry)
    with artifacts_path.open("w", encoding="utf-8") as handle:
        json.dump(artifacts, handle, indent=2)


def write_epoch_metrics(
    run_dir: str | Path,
    run_id: str,
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    output_path = Path(run_dir) / "metrics" / "epoch_metrics.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_id", "epoch", "split", "loss", "mag_error", "incl_error", "decl_error"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_dict = dict(row)
            row_dict["run_id"] = row_dict.get("run_id") or run_id
            writer.writerow({key: row_dict.get(key) for key in fieldnames})
    register_artifact(run_dir, "metrics", output_path)
    return output_path


def save_plot_data(
    run_dir: str | Path,
    name: str,
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    output_path = Path(run_dir) / "plots" / f"{name}_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["series", "x", "y", "split"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_dict = dict(row)
            writer.writerow({key: row_dict.get(key) for key in fieldnames})
    register_artifact(run_dir, "plot_data", output_path)
    return output_path


def build_epoch_rows_from_metrics(
    metrics: Mapping[str, Sequence[Any]],
    *,
    run_id: str,
    epoch_offset: int = 0,
    angles_in_degrees: bool = False,
) -> list[dict[str, Any]]:
    def _safe_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _angle(value: Any) -> float | None:
        numeric = _safe_float(value)
        if numeric is None:
            return None
        if angles_in_degrees:
            return numeric
        return math.degrees(numeric)

    rows: list[dict[str, Any]] = []
    train_loss = list(metrics.get("train_loss") or [])
    train_mag = list(metrics.get("train_mag_p") or [])
    train_incl = list(metrics.get("train_incl") or [])
    train_decl = list(metrics.get("train_decl") or [])
    val_loss = list(metrics.get("val_loss") or [])
    val_mag = list(metrics.get("val_mag_p") or [])
    val_incl = list(metrics.get("val_incl") or [])
    val_decl = list(metrics.get("val_decl") or [])

    for idx in range(len(train_loss)):
        rows.append(
            {
                "run_id": run_id,
                "epoch": epoch_offset + idx,
                "split": "train",
                "loss": _safe_float(train_loss[idx]),
                "mag_error": _safe_float(train_mag[idx]) if idx < len(train_mag) else None,
                "incl_error": _angle(train_incl[idx]) if idx < len(train_incl) else None,
                "decl_error": _angle(train_decl[idx]) if idx < len(train_decl) else None,
            }
        )

    for idx in range(len(val_loss)):
        rows.append(
            {
                "run_id": run_id,
                "epoch": epoch_offset + idx,
                "split": "val",
                "loss": _safe_float(val_loss[idx]),
                "mag_error": _safe_float(val_mag[idx]) if idx < len(val_mag) else None,
                "incl_error": _angle(val_incl[idx]) if idx < len(val_incl) else None,
                "decl_error": _angle(val_decl[idx]) if idx < len(val_decl) else None,
            }
        )

    return rows


def build_training_plot_rows(
    metrics: Mapping[str, Sequence[Any]],
    *,
    angles_in_degrees: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def _iter_series(values: Sequence[Any] | None, series: str, split: str, *, to_degrees: bool) -> None:
        if not values:
            return
        for idx, value in enumerate(values):
            if value is None:
                continue
            numeric = float(value)
            if to_degrees and not angles_in_degrees:
                numeric = math.degrees(numeric)
            rows.append({"series": series, "x": idx, "y": numeric, "split": split})

    _iter_series(metrics.get("train_loss"), "loss", "train", to_degrees=False)
    _iter_series(metrics.get("val_loss"), "loss", "val", to_degrees=False)
    _iter_series(metrics.get("train_mag_p"), "mag_error", "train", to_degrees=False)
    _iter_series(metrics.get("val_mag_p"), "mag_error", "val", to_degrees=False)
    _iter_series(metrics.get("train_incl"), "incl_error", "train", to_degrees=True)
    _iter_series(metrics.get("val_incl"), "incl_error", "val", to_degrees=True)
    _iter_series(metrics.get("train_decl"), "decl_error", "train", to_degrees=True)
    _iter_series(metrics.get("val_decl"), "decl_error", "val", to_degrees=True)

    return rows

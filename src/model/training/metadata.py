"""Utilities for building and persisting structured training metadata."""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import torch

SCHEMA_VERSION = "1.0"
PROJECT_NAME = "VecMagNet"


def _ensure_timezone(dt: _dt.datetime) -> _dt.datetime:
    """Return *dt* with an explicit timezone (defaults to UTC)."""

    if dt.tzinfo is None:
        return dt.replace(tzinfo=_dt.timezone.utc)
    return dt.astimezone(_dt.timezone.utc)


def _to_serializable(obj: Any) -> Any:
    """Convert *obj* into a JSON-serializable structure."""

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, _dt.datetime):
        return _ensure_timezone(obj).isoformat()
    if isinstance(obj, _dt.date):
        return obj.isoformat()
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, Mapping):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_serializable(v) for v in obj]
    return str(obj)


def _run_git_command(args: Sequence[str]) -> str | None:
    """Execute a git command and return stripped stdout or ``None`` on failure."""

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


def _collect_code_state() -> Dict[str, Any]:
    """Gather git repository metadata when available."""

    commit = _run_git_command(["rev-parse", "HEAD"])
    status = _run_git_command(["status", "--porcelain"])
    remote = _run_git_command(["remote", "get-url", "origin"])

    return {
        "git_commit": commit,
        "git_dirty": bool(status),
        "repo": remote,
    }


def _collect_hardware_info(device: torch.device) -> Dict[str, Any]:
    """Return a dictionary describing the runtime hardware."""

    gpu_names: list[str] = []
    if torch.cuda.is_available():
        gpu_names = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]

    cpu_name = platform.processor() or platform.machine()
    ram_gb: float | None = None
    try:
        import psutil

        ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        # Fallback for environments without psutil (e.g. some CI systems).
        try:
            pagesize = os.sysconf("SC_PAGE_SIZE")
            pages = os.sysconf("SC_PHYS_PAGES")
            ram_gb = round(pagesize * pages / (1024**3), 2)
        except (AttributeError, OSError, ValueError):
            ram_gb = None

    return {
        "device": str(device),
        "gpus": gpu_names,
        "num_gpus": len(gpu_names),
        "cpu": cpu_name,
        "ram_gb": ram_gb,
    }


def _collect_env_info() -> Dict[str, Any]:
    """Return software environment metadata."""

    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "os": platform.platform(),
    }


def _infer_snr_from_path(path: str) -> Dict[str, Any] | None:
    """Attempt to infer SNR information from *path*.

    The function searches for ``snr`` tokens and parses common range/value
    encodings such as ``snr20`` or ``snr_0.05-20``.  When parsing fails the raw
    fragment is returned instead so that downstream tools may still leverage the
    hint provided by the folder name.
    """

    lower = path.lower()
    if "snr" not in lower:
        return None

    pattern = re.compile(r"snr[_-]?([0-9.]+\s*(?:-|to)\s*[0-9.]+|[0-9.]+)")
    match = pattern.search(lower)
    if not match:
        return {"type": "unknown", "raw": lower}

    raw_value = match.group(1).replace("to", "-").replace(" ", "")
    if "-" in raw_value:
        lo, hi = raw_value.split("-", 1)
        try:
            return {
                "type": "range",
                "min": float(lo),
                "max": float(hi),
                "raw": match.group(1),
            }
        except ValueError:
            return {"type": "unknown", "raw": match.group(1)}
    try:
        return {"type": "value", "value": float(raw_value), "raw": match.group(1)}
    except ValueError:
        return {"type": "unknown", "raw": match.group(1)}


def dataset_stage_metadata(
    *,
    stage_index: int,
    dataset_folder: str,
    num_samples: int,
    train_samples: int,
    val_samples: int,
    test_samples: int | None = None,
    epochs: int | None = None,
    notes: str | None = None,
) -> Dict[str, Any]:
    """Build a metadata dictionary describing one training stage."""

    folder_path = Path(dataset_folder)
    dataset_id = folder_path.name or folder_path.stem or "dataset"

    return {
        "stage": stage_index,
        "order": stage_index,
        "dataset_id": dataset_id,
        "dataset_path": str(folder_path),
        "total_samples": num_samples,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
        "snr": _infer_snr_from_path(str(folder_path)),
        "epochs": epochs,
        "notes": notes,
    }


def _serialize_optimizer(
    optimizer: torch.optim.Optimizer | None,
) -> Dict[str, Any] | None:
    if optimizer is None:
        return None
    return {
        "name": optimizer.__class__.__name__,
        "defaults": _to_serializable(dict(optimizer.defaults)),
    }


def _serialize_scheduler(
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
) -> Dict[str, Any] | None:
    if scheduler is None:
        return None
    config: Dict[str, Any] = {"name": scheduler.__class__.__name__}
    # Extract a minimal subset of configuration parameters when present.
    for attr in ("step_size", "gamma", "T_max", "eta_min"):
        if hasattr(scheduler, attr):
            config[attr] = getattr(scheduler, attr)
    return _to_serializable(config)


def _serialize_criterion(criterion: Any) -> Dict[str, Any]:
    name = getattr(criterion, "__name__", criterion.__class__.__name__)
    config: Dict[str, Any] = {"name": name}
    for attr in ("reduction", "delta"):
        if hasattr(criterion, attr):
            config[attr] = getattr(criterion, attr)
    return _to_serializable(config)


def _namespace_to_dict(args: Any) -> Dict[str, Any] | None:
    if args is None:
        return None
    if isinstance(args, Mapping):
        return _to_serializable(dict(args))
    if hasattr(args, "__dict__"):
        return _to_serializable(vars(args))
    return _to_serializable(args)


def build_training_metadata(
    *,
    args: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    criterion: Any,
    dataset_stages: Sequence[MutableMapping[str, Any]],
    metrics: Mapping[str, Any],
    start_time: _dt.datetime,
    end_time: _dt.datetime,
    checkpoint_prefix: str | None = None,
    best_checkpoint: str | None = None,
    final_model_path: str | None = None,
    order_type: str = "single_dataset",
    run_id: str | None = None,
    run_notes: str | None = None,
    artifacts_extra: Mapping[str, Any] | None = None,
    trainer_script: str | None = None,
) -> Dict[str, Any]:
    """Assemble the metadata dictionary for a training run."""

    start = _ensure_timezone(start_time)
    end = _ensure_timezone(end_time)
    duration = (end - start).total_seconds()

    device = torch.device(getattr(model, "device", torch.device("cpu")))
    model_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    stages_serialized = [_to_serializable(dict(stage)) for stage in dataset_stages]
    total_train = sum(int(stage.get("train_samples") or 0) for stage in dataset_stages)
    total_val = sum(int(stage.get("val_samples") or 0) for stage in dataset_stages)
    total_test = sum(
        int(stage.get("test_samples") or 0)
        for stage in dataset_stages
        if stage.get("test_samples")
    )
    total_samples = sum(
        int(stage.get("total_samples") or 0) for stage in dataset_stages
    )

    run_identifier = run_id or start.strftime("%Y%m%d_%H%M%S_UTC")

    per_epoch_metrics = {
        "train_loss": metrics.get("train_loss"),
        "train_mag_p": metrics.get("train_mag_p"),
        "train_incl": metrics.get("train_incl"),
        "train_decl": metrics.get("train_decl"),
    }
    evaluation_summary = {
        "primary_metric": "val_loss",
        "metrics": {
            "best_val_loss": metrics.get("best_val_loss"),
            "train_loss_final": (metrics.get("train_loss") or [None])[-1],
        },
        "per_epoch": _to_serializable(
            {k: v for k, v in per_epoch_metrics.items() if v is not None}
        ),
    }
    if "stage_metrics" in metrics:
        evaluation_summary["stage_metrics"] = _to_serializable(metrics["stage_metrics"])
    known_keys = {
        "train_loss",
        "train_mag_p",
        "train_incl",
        "train_decl",
        "best_val_loss",
        "stage_metrics",
    }
    additional_metrics = {
        key: value for key, value in metrics.items() if key not in known_keys
    }
    if additional_metrics:
        evaluation_summary["additional"] = _to_serializable(additional_metrics)

    artifacts = {
        "checkpoint_prefix": checkpoint_prefix,
        "best_checkpoint_path": best_checkpoint,
        "final_model_path": final_model_path,
    }
    if artifacts_extra:
        artifacts.update(_to_serializable(dict(artifacts_extra)))

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "project": PROJECT_NAME,
        "run_id": run_identifier,
        "model": {
            "name": model.__class__.__name__,
            "params": model_params,
            "trainable_params": trainable_params,
            "device": str(device),
        },
        "code_state": _collect_code_state(),
        "hardware": _collect_hardware_info(device),
        "env": _collect_env_info(),
        "data": {
            "stages": stages_serialized,
            "total_samples": total_samples,
            "total_train_samples": total_train,
            "total_val_samples": total_val,
            "total_test_samples": total_test,
        },
        "training": {
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "duration_sec": duration,
            "total_epochs": sum(
                int(stage.get("epochs") or 0) for stage in dataset_stages
            )
            or None,
            "batch_size": getattr(args, "batch_size", None),
            "optimizer": _serialize_optimizer(optimizer),
            "scheduler": _serialize_scheduler(scheduler),
            "loss_fn": _serialize_criterion(criterion),
        },
        "curriculum": {
            "order_type": order_type,
            "train_sequence": stages_serialized,
        },
        "evaluation": evaluation_summary,
        "artifacts": _to_serializable(artifacts),
        "provenance": {
            "trainer_script": trainer_script,
            "cli_args": _namespace_to_dict(args),
            "notes": run_notes,
        },
    }
    return metadata


def save_training_metadata(
    metadata: Mapping[str, Any], output_path: str | Path
) -> Path:
    """Persist *metadata* to *output_path* in JSON format."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(_to_serializable(metadata), fp, indent=2)
    return path

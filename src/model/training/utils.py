"""Shared training utilities for the baby-steps curriculum pipeline.

Consolidates the helper functions that the curriculum trainer previously
imported from the sequential- and ordered-SNR training scripts, which are not
part of this release.  Only the DipoleNet (--dipolenet) architecture and the
smooth-l1 / full-vector loss modes are supported here.
"""
from __future__ import annotations

import argparse
import datetime
import math
import random
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from model.training.stage_runner import full_vector_loss

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_ROOT = _REPO_ROOT / "data" / "datasets"

_ONE_DIPOLE_RE = re.compile(r"^training_1dipoles_snr_?(?P<snr>\d+(?:[p.]\d+)?)(?:_|$)")
_MULTI_DIPOLE_RE = re.compile(
    r"^training_(?P<dipoles>\d+)dipoles(?:_Distance)?_snr_?(?P<snr>\d+(?:[p.]\d+)?)(?:_|$)"
)
_DATE_SUFFIX_RE = re.compile(r"(?P<stamp>\d{4}-\d{2}-\d{2})$")


def _resolve_device(requested: str | None) -> torch.device:
    """Return the torch device respecting an explicit request."""

    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available in this environment.")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _identifier_from_path(path: str) -> str:
    """Return a filesystem-friendly identifier derived from *path*."""

    name = Path(path).name or "dataset"
    return "_".join(filter(None, name.split())) or name


_SNR_TOKEN_RE = re.compile(r"snr_?(\d+(?:[p.]\d+)?)", re.IGNORECASE)


def _infer_snr_from_folder(folder: str) -> float | None:
    name = Path(folder).name
    match = _SNR_TOKEN_RE.search(name)
    if not match:
        return None
    token = match.group(1).lower().replace("p", ".")
    try:
        return float(token)
    except ValueError:
        return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _bn_strength_from_snr(
    snr: float | None,
    *,
    bn_min: float,
    bn_max: float,
    snr_min: float,
    snr_max: float,
    curve: str,
) -> float:
    if snr is None:
        return _clamp(bn_min, 0.0, 1.0)
    if snr_max <= snr_min:
        return _clamp(bn_max, 0.0, 1.0)
    if curve == "log":
        snr_safe = max(snr, 1e-8)
        snr_min_safe = max(snr_min, 1e-8)
        snr_max_safe = max(snr_max, snr_min_safe)
        denom = math.log10(snr_max_safe) - math.log10(snr_min_safe)
        t = (math.log10(snr_safe) - math.log10(snr_min_safe)) / denom if denom else 1.0
    else:
        t = (snr - snr_min) / (snr_max - snr_min)
    t = _clamp(t, 0.0, 1.0)
    strength = bn_max - t * (bn_max - bn_min)
    return _clamp(strength, 0.0, 1.0)


def _make_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    *,
    optimizer_name: str = "adamw",
    extra_parameters: list[torch.nn.Parameter] | None = None,
) -> torch.optim.Optimizer:
    """Construct a fresh optimizer for *model* and any auxiliary parameters."""

    parameter_groups = list(model.parameters())
    if extra_parameters:
        parameter_groups.extend(list(extra_parameters))
    if optimizer_name == "lbfgs":
        if weight_decay:
            warnings.warn(
                "LBFGS ignores weight_decay because torch.optim.LBFGS does not expose that option.",
                RuntimeWarning,
                stacklevel=2,
            )
        return torch.optim.LBFGS(parameter_groups, lr=lr)
    return torch.optim.AdamW(parameter_groups, lr=lr, weight_decay=weight_decay)


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    step_every: int,
    gamma: float,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    """Construct the StepLR scheduler if enabled."""

    if step_every and step_every > 0:
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_every, gamma=gamma)
    return None


def _warm_restart_scheduler(
    optimizer: torch.optim.Optimizer,
    step_every: int,
    gamma: float,
    base_lr: float,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    """Restore the optimizer learning rate and recreate the StepLR scheduler."""

    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = base_lr
        parameter_group["initial_lr"] = base_lr
    return _make_scheduler(optimizer, step_every=step_every, gamma=gamma)


def _seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    try:
        torch.use_deterministic_algorithms(False)
    except TypeError:
        pass


def _ensure_unique_output_dir(output_dir: Path, *, reuse: bool) -> Path:
    if output_dir.exists():
        if output_dir.is_file():
            raise ValueError(f"--output_dir must be a directory, got file: {output_dir}")
        if not reuse and any(output_dir.iterdir()):
            timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
            output_dir = output_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _resolve_step_stats_csv_dir(
    step_stats_csv_dir: str | None,
    output_dir: Path,
    original_output_dir: Path,
) -> str | None:
    if step_stats_csv_dir is None:
        return None
    step_stats_path = Path(step_stats_csv_dir)
    if not step_stats_path.is_absolute():
        return str(output_dir / step_stats_path)
    try:
        relative = step_stats_path.relative_to(original_output_dir)
    except ValueError:
        return str(step_stats_path)
    return str(output_dir / relative)



def _select_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    raise NotImplementedError(
        "The legacy CNN architectures are not part of this release; "
        "run with --dipolenet to use the published AttentionDipoleModel."
    )


def _describe_model(args: argparse.Namespace) -> str:
    return "AttentionDipoleModel" if getattr(args, "dipolenet", False) else "legacy (not included)"


def _select_loss(args: argparse.Namespace, device: torch.device):
    """Return (criterion_callable, loss_module, extra_optimizer_parameters)."""
    if args.loss_mode == "full-vector":
        return full_vector_loss, None, None
    if args.loss_mode == "smooth-l1":
        return torch.nn.SmoothL1Loss(), None, None
    raise NotImplementedError(
        f"loss_mode={args.loss_mode!r} is not part of this release; "
        "use smooth-l1 (default) or full-vector."
    )


def _parse_snr_value(text: str) -> float:
    cleaned = text.strip().lower().replace("p", ".")
    return float(cleaned)


def _parse_snr_sequence(text: str) -> List[float]:
    if not text:
        raise ValueError("Expected a comma-separated list of SNR values.")
    values: List[float] = []
    for item in text.split(","):
        cleaned = item.strip()
        if not cleaned:
            raise ValueError("Empty entry found while parsing SNR sequence.")
        values.append(_parse_snr_value(cleaned))
    return values


def _resolve_dataset_root(dataset_root: str) -> Path:
    """Resolve ``--dataset_root``, anchoring relative values to the repo root.

    Anchoring to ``_REPO_ROOT`` rather than the working directory keeps the
    configs valid from ``src/`` and from the repo root alike -- but it also
    means a config value of ``../data/datasets`` resolves one level *above*
    the repository, so relative values must be written as ``data/datasets``.
    """

    root = Path(dataset_root).expanduser()
    if not root.is_absolute():
        root = (_REPO_ROOT / root).resolve()
    return root


def _dataset_preference_key(path: Path) -> Tuple[int, int, str, str]:
    """Rank dataset folders so newer non-legacy variants win duplicate matches."""

    name = path.name
    is_distance_variant = "_Distance_" in name
    has_standoff = "standoff" in name.lower()
    date_match = _DATE_SUFFIX_RE.search(name)
    date_stamp = date_match.group("stamp") if date_match else ""
    return (
        0 if is_distance_variant else 1,
        1 if has_standoff else 0,
        date_stamp,
        name,
    )


def _discover_datasets(
    dataset_root: Path,
    min_dipoles: int,
    max_dipoles: int,
) -> Dict[float, Dict[int, str]]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset root does not exist: {dataset_root}\n"
            "Relative --dataset_root values are resolved against the repository "
            f"root ({_REPO_ROOT}), so use `data/datasets`, not `../data/datasets`."
        )

    mapping: Dict[float, Dict[int, str]] = {}
    duplicates: List[Tuple[float, int, str, str]] = []
    unmatched: List[str] = []

    for entry in dataset_root.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        dipoles: int | None = None
        snr_text: str | None = None

        match = _ONE_DIPOLE_RE.match(name)
        if match:
            dipoles = 1
            snr_text = match.group("snr")
        else:
            match = _MULTI_DIPOLE_RE.match(name)
            if match:
                dipoles = int(match.group("dipoles"))
                snr_text = match.group("snr")

        if dipoles is None or snr_text is None:
            unmatched.append(name)
            continue
        if dipoles < min_dipoles or dipoles > max_dipoles:
            unmatched.append(f"{name} (dipoles={dipoles} outside {min_dipoles}-{max_dipoles})")
            continue

        snr_value = _parse_snr_value(snr_text)
        bucket = mapping.setdefault(snr_value, {})
        if dipoles in bucket:
            existing = Path(bucket[dipoles])
            chosen = max((existing, entry), key=_dataset_preference_key)
            other = entry if chosen == existing else existing
            if _dataset_preference_key(chosen) == _dataset_preference_key(other):
                duplicates.append((snr_value, dipoles, str(existing), str(entry)))
                continue
            bucket[dipoles] = str(chosen)
        else:
            bucket[dipoles] = str(entry)

    if duplicates:
        details = ", ".join(
            f"snr={snr} dipoles={dipoles} ({first} vs {second})"
            for snr, dipoles, first, second in duplicates
        )
        raise ValueError(f"Duplicate dataset matches found: {details}")

    if not mapping:
        listing = "\n".join(f"  - {name}" for name in sorted(unmatched)[:20]) or "  (no subdirectories)"
        raise ValueError(
            f"No matching training datasets found under {dataset_root}\n"
            "Dataset folders must be named `training_<N>dipoles[_Distance]_snr_<S>` "
            "(e.g. training_1dipoles_snr_0.6). Set [snr_sort] name in "
            "configs/blend_noise.ini to a matching prefix.\n"
            f"Folders present that did not match:\n{listing}"
        )

    return mapping



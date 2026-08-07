"""Train with a 2D Baby-Steps curriculum: diagonal unlock order + competence-gated pacing.

Implements the Baby-Steps curriculum over two difficulty axes:
  - Complexity axis (c): number of magnetic dipoles (1 → max_dipoles)
  - Noise axis (η): signal-to-noise ratio (high SNR → low SNR)

Each (dipoles, SNR) cell is scored as:

    d(c, η) = α · c̃ + (1 − α) · η̃

where c̃ = (c − c_min) / (c_max − c_min) and η̃ = 1 − log-normalized SNR (so low SNR
is hard). The hyperparameter α ∈ (0, 1) controls which axis dominates early
training. For ill-posed inverse problems (like dipole localization from magnetic
field images), α > 0.5 prioritizes structural complexity over noise.

Training loop (competence-gated Baby-Steps):
  1. Sort all cells from easiest to hardest.
  2. Unlock the next cell.
  3. Build the training set as the union of ALL previously unlocked cells
     (Baby-Steps mixing) — preventing catastrophic forgetting of easy examples.
  4. Use the newly-unlocked cell's held-out validation set as the advancement
     criterion: train until val_loss < --advancement_threshold, or until
     --max_epochs_per_stage is reached.
  5. Repeat from step 2.

Reference: Soviany et al. (2022), "Curriculum Learning: A Survey";
           Platanios et al. (2019), "Competence-based Curriculum Learning";
           Hacohen & Weinshall (2019), "On the Power of Curriculum Learning in
           Training Deep Networks".
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import time

import h5py
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from alive_progress import alive_bar

from model.architecture import (
    AttentionDipoleModel,
    DipoleNetLoss,
    declination_error,
    inclination_error,
    mag_p_error,
    make_measurement_grid,
)

from model.training.artifacts import (
    build_epoch_rows_from_metrics,
    build_training_plot_rows,
    init_run,
    register_artifact,
    save_plot_data,
    write_epoch_metrics,
)
from model.training.metadata import (
    build_training_metadata,
    dataset_stage_metadata,
    save_training_metadata,
)
from model.training.utils import (
    DEFAULT_DATASET_ROOT,
    _bn_strength_from_snr,
    _describe_model,
    _discover_datasets,
    _ensure_unique_output_dir,
    _make_optimizer,
    _make_scheduler,
    _parse_snr_sequence,
    _resolve_dataset_root,
    _resolve_device,
    _resolve_step_stats_csv_dir,
    _seed_everything,
    _select_loss,
    _select_model,
    _warm_restart_scheduler,
)
from model.training.stage_runner import (
    _build_best_checkpoint_path,
    load_state_dict_checkpoint,
    load_xy_from_h5,
    make_loaders,
    plot_training,
    save_model,
    train_one_stage,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellInfo:
    """One (dipoles, SNR) grid cell in the 2D curriculum."""

    dipoles: int
    snr: float
    folder: str
    difficulty: float  # pre-computed diagonal score ∈ [0, 1]

    @property
    def identifier(self) -> str:
        snr_token = str(int(self.snr)) if float(self.snr).is_integer() else f"{self.snr:g}".replace(".", "p")
        return f"c{self.dipoles:02d}_snr{snr_token}"


# ---------------------------------------------------------------------------
# Difficulty scoring
# ---------------------------------------------------------------------------

def _compute_cell_difficulty(
    dipoles: int,
    snr: float,
    *,
    c_min: int,
    c_max: int,
    snr_min: float,
    snr_max: float,
    alpha: float,
) -> float:
    """Diagonal difficulty score: 0 = easiest, 1 = hardest.

    The complexity rank c̃ increases with dipole count.
    The noise rank η̃ increases as SNR decreases (harder noise regime).

    Args:
        dipoles: Number of dipoles for this cell.
        snr: Signal-to-noise ratio for this cell.
        c_min: Minimum dipole count across all cells.
        c_max: Maximum dipole count across all cells.
        snr_min: Minimum (hardest) SNR across all cells.
        snr_max: Maximum (easiest) SNR across all cells.
        alpha: Weight given to the complexity axis (0 < alpha < 1).
            alpha > 0.5 makes complexity the dominant axis.

    Returns:
        Difficulty score in [0, 1].
    """
    if c_max > c_min:
        c_norm = (dipoles - c_min) / (c_max - c_min)
    else:
        c_norm = 0.0

    # Log-normalize SNR so perceptually equal intervals → equal rank increments.
    if snr_max > snr_min and snr > 0 and snr_min > 0:
        log_snr = math.log(snr)
        log_snr_min = math.log(snr_min)
        log_snr_max = math.log(snr_max)
        snr_norm = (log_snr - log_snr_min) / (log_snr_max - log_snr_min)
    else:
        snr_norm = 1.0  # unknown SNR → treat as easy (high SNR)

    # High SNR → easy → low η̃; low SNR → hard → high η̃
    eta_difficulty = 1.0 - max(0.0, min(1.0, snr_norm))
    c_norm = max(0.0, min(1.0, c_norm))

    return alpha * c_norm + (1.0 - alpha) * eta_difficulty


def _build_unlock_order(
    mapping: Dict[float, Dict[int, str]],
    *,
    alpha: float,
) -> List[CellInfo]:
    """Return all cells sorted from easiest to hardest (diagonal sweep).

    Args:
        mapping: {snr: {dipoles: folder}} as returned by _discover_datasets.
        alpha: Complexity-axis weight for the diagonal scoring function.

    Returns:
        Sorted list of CellInfo objects, easiest first.
    """
    all_cells: List[Tuple[int, float, str]] = [
        (dipoles, snr, folder)
        for snr, dipole_map in mapping.items()
        for dipoles, folder in dipole_map.items()
    ]
    if not all_cells:
        return []

    all_dipoles = [c[0] for c in all_cells]
    all_snrs = [c[1] for c in all_cells]
    c_min, c_max = min(all_dipoles), max(all_dipoles)
    snr_min, snr_max = min(all_snrs), max(all_snrs)

    cell_infos: List[CellInfo] = []
    for dipoles, snr, folder in all_cells:
        difficulty = _compute_cell_difficulty(
            dipoles,
            snr,
            c_min=c_min,
            c_max=c_max,
            snr_min=snr_min,
            snr_max=snr_max,
            alpha=alpha,
        )
        cell_infos.append(CellInfo(dipoles=dipoles, snr=snr, folder=folder, difficulty=difficulty))

    # Tie-break: lower SNR first within same difficulty (harder noise second axis),
    # then higher dipoles for further ties. This keeps the order deterministic.
    cell_infos.sort(key=lambda ci: (ci.difficulty, -ci.snr, ci.dipoles))
    return cell_infos


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def _build_stage_loaders(
    unlocked: Sequence[CellInfo],
    primary: CellInfo,
    *,
    stage_samples: int,
    w_n_data: int,
    batch_size: int,
    train_val_split: float,
    seed: int | None,
    device: torch.device | None = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int, List[Dict[str, Any]]]:
    """Build the mixed training loader and a primary-cell validation loader.

    The training set is the union (Baby-Steps) of all unlocked cells with an
    equal per-cell sample budget. The validation loader contains only the
    primary (newly unlocked) cell's held-out data and is used both for
    checkpoint selection and the competence-gated advancement check.

    Args:
        unlocked: All cells unlocked so far, including the primary cell.
        primary: The newly unlocked cell (used for competence check).
        stage_samples: Target total training samples across all cells.
        w_n_data: Per-cell cap on loaded samples.
        batch_size: Batch size for both loaders.
        train_val_split: Fraction of data used for training.
        seed: Base random seed.
        device: Torch device passed to load_xy_from_h5.

    Returns:
        (train_loader, primary_val_loader, total_train_samples, component_records)
    """
    n_cells = len(unlocked)
    n_per_cell = max(1, min(w_n_data, stage_samples // n_cells))

    all_x: List[torch.Tensor] = []
    all_y: List[torch.Tensor] = []
    component_records: List[Dict[str, Any]] = []
    primary_x: torch.Tensor | None = None
    primary_y: torch.Tensor | None = None
    primary_seed: int | None = None

    for cell_idx, cell in enumerate(unlocked):
        cell_seed = None if seed is None else (seed + cell_idx * 1009 + int(cell.snr * 100)) % (2**31)
        x, y, n_loaded = load_xy_from_h5(
            cell.folder,
            w_n_data=n_per_cell,
            device=device,
            seed=cell_seed,
            shuffle=True,
        )
        all_x.append(x)
        all_y.append(y)
        component_records.append(
            {
                "dipoles": cell.dipoles,
                "snr": cell.snr,
                "difficulty": cell.difficulty,
                "folder": cell.folder,
                "n_per_cell": n_per_cell,
                "loaded": n_loaded,
            }
        )
        if cell.dipoles == primary.dipoles and cell.snr == primary.snr:
            primary_x = x
            primary_y = y
            primary_seed = cell_seed

    mixed_x = torch.cat(all_x, dim=0)
    mixed_y = torch.cat(all_y, dim=0)

    # Mixed training loader (entire Baby-Steps union, 80% for training).
    mixed_shuffle_seed = None if seed is None else seed + 7
    train_loader, _ = make_loaders(
        mixed_x,
        mixed_y,
        batch_size=batch_size,
        train_val_split=train_val_split,
        seed=mixed_shuffle_seed,
    )

    # Primary-cell validation loader for competence-gated advancement.
    # Re-use the same data that was loaded for the primary cell above, applying
    # the same train/val split so the held-out 20% is consistent.
    assert primary_x is not None, "Primary cell not found in unlocked list."
    _, primary_val_loader = make_loaders(
        primary_x,
        primary_y,
        batch_size=batch_size,
        train_val_split=train_val_split,
        seed=primary_seed,
    )

    total_train_samples = len(train_loader.dataset)
    return train_loader, primary_val_loader, total_train_samples, component_records


# ---------------------------------------------------------------------------
# Per-cell HDF5 → .npy conversion
# ---------------------------------------------------------------------------

def _convert_cells_to_npy(
    unlock_order: List["CellInfo"],
    precomputed_dir: Path,
    *,
    n_samples: int,
) -> Dict[str, Tuple[Path, Path]]:
    """Convert each unique cell's HDF5 to uncompressed float32 .npy files.

    Reads the first ``n_samples`` rows **sequentially** from each HDF5 (optimal
    for gzip-chunked datasets — no random-seek overhead), converts to
    (N, C, H, W) float32, and writes two files per cell:
      ``precomputed_dir/cells/<cell_name>_I.npy``  — field images
      ``precomputed_dir/cells/<cell_name>_m.npy``  — dipole moments

    Conversion is skipped for cells whose .npy files already exist, so it is
    safe to interrupt and resume, and subsequent runs reuse the cached files.

    Args:
        unlock_order: Full curriculum cell list (only unique folders converted).
        precomputed_dir: Root dir for precomputed data.
        n_samples: Maximum samples to cache per cell (cap on each HDF5 read).
            Typically set to ``w_n_data`` so memory usage is bounded.

    Returns:
        Mapping from ``cell.folder`` → ``(I_npy_path, m_npy_path)``.
    """
    cells_dir = precomputed_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    # De-duplicate by folder path
    seen: Dict[str, Tuple[Path, Path]] = {}
    to_convert: List[Tuple[str, Path, Path]] = []

    for cell in unlock_order:
        if cell.folder in seen:
            continue
        cell_name = Path(cell.folder).name
        x_npy = cells_dir / f"{cell_name}_I.npy"
        m_npy = cells_dir / f"{cell_name}_m.npy"
        seen[cell.folder] = (x_npy, m_npy)
        if not x_npy.exists() or not m_npy.exists():
            to_convert.append((cell.folder, x_npy, m_npy))

    if not to_convert:
        print(f"  All {len(seen)} cell .npy files already exist — skipping conversion.")
        return seen

    approx_gb = len(seen) * n_samples * (100 * 100 * 3 + 3) * 4 / (1024 ** 3)
    print(
        f"\nConverting {len(to_convert)} cell HDF5 files to uncompressed .npy"
        f" ({len(seen) - len(to_convert)} already cached)."
        f"\nTarget: {cells_dir}"
        f"\nEstimated total disk: ~{approx_gb:.1f} GB"
    )

    with alive_bar(len(to_convert), title="Converting cells", unit="cell") as bar:
        for folder, x_npy, m_npy in to_convert:
            h5_path = Path(folder) / "dataset.h5"
            with h5py.File(str(h5_path), "r") as f:
                total = min(len(f["I"]), len(f["m"]))
                n = min(n_samples, total)
                # Sequential slice: optimal read order for gzip chunks
                x_data = f["I"][:n].astype(np.float32).transpose(0, 3, 1, 2)
                m_data = f["m"][:n].astype(np.float32)
            np.save(str(x_npy), x_data)
            np.save(str(m_npy), m_data)
            bar()

    return seen


# ---------------------------------------------------------------------------
# Stage precomputation
# ---------------------------------------------------------------------------

def _precompute_stage(
    stage_index: int,
    primary_cell: "CellInfo",
    unlocked: Sequence["CellInfo"],
    *,
    stage_samples: int,
    w_n_data: int,
    train_val_split: float,
    seed: int | None,
    precomputed_dir: Path,
    index_h5: "h5py.File",
    cell_npy_map: Dict[str, Tuple[Path, Path]] | None = None,
) -> Path:
    """Precompute one curriculum stage: sample indices, write to index H5, save .npy.

    When ``cell_npy_map`` is provided, reads each cell from the pre-converted
    uncompressed .npy files (fast mmap + sorted-index read) instead of the
    original gzip HDF5.  Falls back to HDF5 for any cell not in the map.

    The index H5 group records every sampled index (relative to the original
    HDF5 file) so the exact dataset can be reconstructed without the .npy files.

    Args:
        stage_index: 1-based stage number.
        primary_cell: The newly unlocked cell for this stage.
        unlocked: All cells unlocked so far (including primary_cell).
        stage_samples: Target total training samples across all cells.
        w_n_data: Per-cell sample cap.
        train_val_split: Fraction used for training (stored in meta).
        seed: Global random seed (None = unseeded).
        precomputed_dir: Directory where stage subdirs and .npy files are written.
        index_h5: Open h5py.File for writing index groups.
        cell_npy_map: Optional mapping from cell.folder -> (I_npy_path, m_npy_path)
            produced by _convert_cells_to_npy.  When supplied each cell is read
            from its .npy file for dramatically faster I/O.

    Returns:
        Path to the stage subdirectory containing X.npy, y.npy, meta.json.
    """
    n_cells = len(unlocked)
    n_per_cell = max(1, min(w_n_data, stage_samples // n_cells))
    stage_seed = None if seed is None else (seed + stage_index)
    mixed_shuffle_seed = None if stage_seed is None else stage_seed + 7

    stage_name = f"stage_{stage_index:03d}_{primary_cell.identifier}"
    stage_dir = precomputed_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)

    # H5 index group for this stage
    grp = index_h5.create_group(f"stage_{stage_index:03d}")
    grp.attrs["stage_index"] = stage_index
    grp.attrs["primary_dipoles"] = primary_cell.dipoles
    grp.attrs["primary_snr"] = float(primary_cell.snr)
    grp.attrs["primary_identifier"] = primary_cell.identifier
    grp.attrs["n_cells"] = n_cells
    grp.attrs["n_per_cell"] = n_per_cell
    grp.attrs["stage_seed"] = stage_seed if stage_seed is not None else -1
    grp.attrs["mixed_shuffle_seed"] = mixed_shuffle_seed if mixed_shuffle_seed is not None else -1

    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    n_per_cell_list: List[int] = []
    component_records: List[Dict[str, Any]] = []
    primary_cell_idx_in_unlocked = -1
    primary_seed: int | None = None

    for cell_idx, cell in enumerate(unlocked):
        cell_seed = (
            None if stage_seed is None
            else (stage_seed + cell_idx * 1009 + int(cell.snr * 100)) % (2 ** 31)
        )

        h5_path = Path(cell.folder) / "dataset.h5"

        # Determine pool size and open the data source.
        # When a cell .npy exists, sample from it (indices 0..n_cache-1).
        # Because _convert_cells_to_npy reads the FIRST n_cache HDF5 rows in
        # order, the sampled indices are also valid HDF5 row numbers, so
        # the stage_indices.h5 records remain reproducible from HDF5.
        if cell_npy_map and cell.folder in cell_npy_map:
            x_npy, m_npy = cell_npy_map[cell.folder]
            X_mmap = np.load(str(x_npy), mmap_mode="r")
            M_mmap = np.load(str(m_npy), mmap_mode="r")
            total = len(X_mmap)
            use_npy = True
        else:
            with h5py.File(str(h5_path), "r") as f:
                total = min(len(f["I"]), len(f["m"]))
            use_npy = False

        n = min(n_per_cell, total)
        rng = np.random.default_rng(cell_seed)
        if n < total:
            indices = rng.choice(total, size=n, replace=False)
        else:
            indices = np.arange(total)
            rng.shuffle(indices)

        sorted_idx = np.sort(indices)

        # Record indices in H5
        cell_grp = grp.create_group(f"cell_{cell_idx:03d}")
        cell_grp.attrs["folder"] = str(cell.folder)
        cell_grp.attrs["dipoles"] = cell.dipoles
        cell_grp.attrs["snr"] = float(cell.snr)
        cell_grp.attrs["difficulty"] = float(cell.difficulty)
        cell_grp.attrs["n_loaded"] = n
        cell_grp.attrs["total_samples"] = total
        cell_grp.attrs["seed"] = cell_seed if cell_seed is not None else -1
        cell_grp.create_dataset("indices", data=indices, dtype="int64")

        if use_npy:
            # Fast path: sorted-index read on flat uncompressed binary = near-sequential I/O
            x_chunk = X_mmap[sorted_idx].copy()  # (n, C, H, W) float32 — already transposed
            y_chunk = M_mmap[sorted_idx].copy()  # (n, 3) float32
        else:
            # Slow fallback: gzip HDF5 read
            with h5py.File(str(h5_path), "r") as f:
                x_chunk = f["I"][sorted_idx]  # (n, H, W, C)
                y_chunk = f["m"][sorted_idx]  # (n, 3)
            x_chunk = x_chunk.astype(np.float32).transpose(0, 3, 1, 2)
            y_chunk = y_chunk.astype(np.float32)

        all_X.append(x_chunk)
        all_y.append(y_chunk)
        n_per_cell_list.append(n)

        component_records.append({
            "dipoles": cell.dipoles,
            "snr": cell.snr,
            "difficulty": cell.difficulty,
            "folder": cell.folder,
            "n_per_cell": n_per_cell,
            "loaded": n,
        })

        if cell.dipoles == primary_cell.dipoles and cell.snr == primary_cell.snr:
            primary_cell_idx_in_unlocked = cell_idx
            primary_seed = cell_seed

    # Concatenate: data is ordered by cell (cell_0 first, cell_1 next, ...)
    mixed_X = np.concatenate(all_X, axis=0)
    mixed_y = np.concatenate(all_y, axis=0)

    # Save meta for loader reconstruction
    meta: Dict[str, Any] = {
        "stage_index": stage_index,
        "n_cells": n_cells,
        "n_per_cell_list": n_per_cell_list,
        "primary_cell_idx": primary_cell_idx_in_unlocked,
        "primary_seed": primary_seed,
        "mixed_shuffle_seed": mixed_shuffle_seed,
        "train_val_split": train_val_split,
        "primary_dipoles": primary_cell.dipoles,
        "primary_snr": float(primary_cell.snr),
        "component_records": component_records,
    }
    with (stage_dir / "meta.json").open("w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)

    np.save(str(stage_dir / "X.npy"), mixed_X)
    np.save(str(stage_dir / "y.npy"), mixed_y)

    return stage_dir


def _load_precomputed_stage(
    stage_dir: Path,
    *,
    batch_size: int,
) -> Tuple[
    "torch.utils.data.DataLoader",
    "torch.utils.data.DataLoader",
    int,
    List[Dict[str, Any]],
]:
    """Load a precomputed stage from .npy files and return loaders + metadata.

    Exactly mirrors the return signature of _build_stage_loaders so the
    training loop needs no branching beyond the initial loader-selection step.

    Args:
        stage_dir: Directory produced by _precompute_stage (contains X.npy,
            y.npy, meta.json).
        batch_size: Batch size for both loaders.

    Returns:
        (train_loader, primary_val_loader, n_train_samples, component_records)
    """
    with (stage_dir / "meta.json").open(encoding="utf-8") as fp:
        meta = json.load(fp)

    # Load uncompressed .npy — sequential read, OS-cached after first access
    X = np.load(str(stage_dir / "X.npy"))
    y = np.load(str(stage_dir / "y.npy"))

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    # Identify primary cell slice (cells are concatenated in order)
    primary_idx = meta["primary_cell_idx"]
    n_per_cell_list: List[int] = meta["n_per_cell_list"]
    primary_start = sum(n_per_cell_list[:primary_idx])
    primary_end = primary_start + n_per_cell_list[primary_idx]

    primary_x = X_t[primary_start:primary_end]
    primary_y = y_t[primary_start:primary_end]

    train_val_split: float = meta["train_val_split"]
    mixed_shuffle_seed: int | None = meta["mixed_shuffle_seed"]
    primary_seed: int | None = meta["primary_seed"]

    train_loader, _ = make_loaders(
        X_t, y_t,
        batch_size=batch_size,
        train_val_split=train_val_split,
        seed=mixed_shuffle_seed,
    )
    _, primary_val_loader = make_loaders(
        primary_x, primary_y,
        batch_size=batch_size,
        train_val_split=train_val_split,
        seed=primary_seed,
    )

    n_train = len(train_loader.dataset)
    component_records: List[Dict[str, Any]] = meta["component_records"]
    return train_loader, primary_val_loader, n_train, component_records


# ---------------------------------------------------------------------------
# Simple evaluation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_val_loss(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: Any,
    device: torch.device,
) -> float:
    """Compute mean batch loss on val_loader without training.

    Used to measure full-mixed-set validation loss after each stage for
    logging, independently of the primary-cell val loss used for gating.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for bx, by in val_loader:
        bx = bx.to(device)
        by = by.to(device)
        out = model(bx)
        loss = criterion(out, by)
        total_loss += float(loss.item())
        n_batches += 1
    model.train()
    return total_loss / n_batches if n_batches > 0 else float("inf")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the vector model with 2D Baby-Steps curriculum: diagonal unlock order "
            "and competence-gated stage advancement."
        ),
    )

    # ---- dataset discovery ----
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help="Root directory containing training_* dataset folders.",
    )
    parser.add_argument("--min_dipoles", type=int, default=1)
    parser.add_argument("--max_dipoles", type=int, default=10)
    parser.add_argument(
        "--snr_levels",
        type=str,
        default=None,
        help="Optional comma-separated SNR list. When omitted, SNRs are auto-discovered.",
    )
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Skip missing (dipoles, SNR) combinations instead of raising an error.",
    )

    # ---- curriculum hyperparameters ----
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help=(
            "Complexity-axis weight for the diagonal difficulty score (0 < alpha < 1). "
            "alpha > 0.5 prioritises structural complexity (dipole count) over noise (SNR). "
            "Recommended ≥ 0.5 for ill-posed inverse problems."
        ),
    )
    parser.add_argument(
        "--advancement_threshold",
        type=float,
        default=None,
        help=(
            "Competence gate: advance to the next cell only when the primary cell's "
            "validation loss falls below this value. Supports negative thresholds. "
            "Omit (default) to disable gating and always advance after max_epochs_per_stage epochs."
        ),
    )
    parser.add_argument(
        "--max_epochs_per_stage",
        type=int,
        default=300,
        help="Hard epoch ceiling per curriculum stage (safety net for competence gate).",
    )
    parser.add_argument(
        "--stage_samples",
        type=int,
        default=50_000,
        help=(
            "Target total training samples for each stage, divided equally across all "
            "unlocked cells. Per-cell sample count = stage_samples // num_unlocked_cells, "
            "capped by --w_n_data."
        ),
    )
    parser.add_argument(
        "--w_n_data",
        type=int,
        default=10_000,
        help="Per-cell sample cap. Bounds memory: max memory ∝ w_n_data × num_unlocked.",
    )

    # ---- standard training args ----
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--train_val_split", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--optimizer",
        choices=("adamw", "lbfgs"),
        default="adamw",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--clip_grad_norm", type=float, default=None)
    parser.add_argument("--step_every", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=20,
        help="Early stopping patience within a stage; set 0 to disable.",
    )
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)

    # ---- BN scheduling ----
    parser.add_argument("--bn_min", type=float, default=0.0)
    parser.add_argument("--bn_max", type=float, default=0.0)
    parser.add_argument("--bn_snr_min", type=float, default=0.05)
    parser.add_argument("--bn_snr_max", type=float, default=20.0)
    parser.add_argument("--bn_curve", choices=("linear", "log"), default="log")

    # ---- model selection ----
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--deep_model", action="store_true")
    parser.add_argument("--minimal_model", action="store_true")
    parser.add_argument("--standard_bn_model", action="store_true")
    parser.add_argument("--no_bn_model", action="store_true")
    parser.add_argument("--regularized_model", action="store_true")
    parser.add_argument("--deeper_regularized_model", action="store_true")
    parser.add_argument(
        "--loss_mode",
        choices=("directional-magnitude", "full-vector", "smooth-l1"),
        default="smooth-l1",
    )

    # ---- DipoleNet model (overrides all legacy model/loss flags) ----
    parser.add_argument(
        "--dipolenet",
        action="store_true",
        help=(
            "Use AttentionDipoleModel + DipoleNetLoss instead of the legacy CNN "
            "selection flags. Enables attention pooling, separate magnitude/direction "
            "heads, Kendall weighting, superposition augmentation, and optional "
            "physics-consistency loss."
        ),
    )
    parser.add_argument("--base_channels", type=int, default=32, help="AttentionDipoleModel base channel count.")
    parser.add_argument(
        "--physics_weight",
        type=float,
        default=0.0,
        help="Forward-consistency physics loss weight γ (0 disables).",
    )
    parser.add_argument("--grid_size", type=int, default=100, help="Physics loss measurement grid size (pixels/side).")
    parser.add_argument("--grid_extent", type=float, default=1.0, help="Physics loss grid extent in generation units (1.0 = full [0,1] image span).")
    parser.add_argument("--grid_z_offset", type=float, default=0.1, help="Physics loss measurement plane height in generation units (match dataset mapheight).")
    parser.add_argument(
        "--aug_prob",
        type=float,
        default=0.5,
        help="Per-batch probability of superposition augmentation (DipoleNet only).",
    )

    # ---- optimizer / checkpoint management ----
    parser.add_argument(
        "--keep_optimizer",
        action="store_true",
        help=(
            "Keep optimizer and scheduler state between stages. "
            "By default the optimizer is reset at each new cell."
        ),
    )
    parser.add_argument(
        "--reset_scheduler_each_stage",
        action="store_true",
        help="Warm-restart scheduler between stages while keeping optimizer momentum.",
    )
    parser.add_argument(
        "--load_best_per_stage",
        action="store_true",
        help="Reload the best checkpoint before advancing to the next cell.",
    )

    # ---- output / run management ----
    parser.add_argument("--output_dir", type=str, default="sequential_runs/baby_steps_curriculum")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--step_stats_csv", action="store_true")
    parser.add_argument("--step_stats_csv_dir", type=str, default="step_logs")
    parser.add_argument("--final_model", type=str, default="vector_model_baby_steps_final.pth")
    parser.add_argument(
        "--metrics_json",
        type=str,
        default="baby_steps_curriculum_metrics.json",
        help="JSON filename for per-stage metrics.",
    )
    parser.add_argument("--reuse_output_dir", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot_curves", action="store_true")
    parser.add_argument(
        "--start_stage",
        type=int,
        default=1,
        help=(
            "Resume training from this stage index (1-based). Stages before it are "
            "skipped (unlocked-cell history is still reconstructed correctly). "
            "Use with --pretrained pointing at the last completed stage's checkpoint."
        ),
    )
    parser.add_argument("--ckpt_prefix", type=str, default="checkpoint_minval")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--show_progress", action="store_true")

    # ---- precomputation ----
    parser.add_argument(
        "--no_precompute",
        action="store_true",
        help=(
            "Disable stage precomputation and use the original HDF5 loading path. "
            "By default each stage is precomputed as an uncompressed .npy file before "
            "training and deleted afterward, which eliminates gzip-decompression overhead."
        ),
    )
    parser.add_argument(
        "--precompute_all",
        action="store_true",
        help=(
            "Precompute ALL stages to disk before training begins. "
            "Requires ~stage_samples * n_stages * 120 KB of free disk space. "
            "By default only the current stage is precomputed (streaming mode)."
        ),
    )
    parser.add_argument(
        "--keep_precomputed",
        action="store_true",
        help="Keep .npy stage files on disk after training (default: delete after each stage).",
    )
    parser.add_argument(
        "--precomputed_dir",
        type=str,
        default=None,
        help=(
            "Directory for precomputed stage .npy files and stage_indices.h5. "
            "Defaults to <output_dir>/precomputed."
        ),
    )
    parser.add_argument(
        "--cell_cache_samples",
        type=int,
        default=None,
        help=(
            "Number of samples to cache per cell when converting HDF5 -> .npy. "
            "Defaults to --w_n_data. Higher values give more sampling variety "
            "across stages at the cost of more disk space per cell "
            "(each sample ~120 KB for 100x100x3 images)."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_for_json(value: object) -> object:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    return value


def _print_stage_summary(
    stage_index: int,
    primary: CellInfo,
    unlocked: Sequence[CellInfo],
    component_records: Sequence[Dict[str, Any]],
    advancement_threshold: float,
) -> None:
    gate_str = f"(gate: val_loss < {advancement_threshold:.4f})" if advancement_threshold is not None else "(no gate)"
    print(f"\n=== Baby-Steps stage {stage_index} :: {primary.identifier} {gate_str} ===")
    print(
        f"Primary cell: {primary.dipoles} dipoles, SNR {primary.snr:g}  "
        f"(difficulty={primary.difficulty:.3f})"
    )
    print(f"Unlocked cells: {len(unlocked)}  |  Baby-Steps mixture:")
    for rec in component_records:
        print(
            f"  d={rec['difficulty']:.3f}  {rec['dipoles']:2d} dipoles | "
            f"SNR {rec['snr']:6g} | n={rec['loaded']} | {Path(rec['folder']).name}"
        )


# ---------------------------------------------------------------------------
# DipoleNet training helpers
# ---------------------------------------------------------------------------

def _superposition_augment(
    x: torch.Tensor,
    y: torch.Tensor,
    prob: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix random pairs within the batch via exact magnetic superposition."""
    if torch.rand(1).item() > prob:
        return x, y
    B = x.shape[0]
    idx = torch.randperm(B, device=x.device)
    log_scale_a = torch.empty(B, device=x.device).uniform_(math.log(0.5), math.log(2.0))
    log_scale_b = torch.empty(B, device=x.device).uniform_(math.log(0.5), math.log(2.0))
    scale_a = log_scale_a.exp().view(B, 1, 1, 1)
    scale_b = log_scale_b.exp().view(B, 1, 1, 1)
    x_aug = scale_a * x + scale_b * x[idx]
    y_aug = scale_a.squeeze(-1).squeeze(-1) * y + scale_b.squeeze(-1).squeeze(-1) * y[idx]
    return x_aug, y_aug


@torch.no_grad()
def _eval_dipolenet_metrics(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: DipoleNetLoss,
    device: torch.device,
    r_grid: torch.Tensor | None,
) -> dict[str, float]:
    """Return averaged loss, mag%, inclination, and declination over val_loader."""
    model.eval()
    total_loss = total_mag = total_incl = total_decl = 0.0
    n_batches = 0
    for bx, by in val_loader:
        bx, by = bx.to(device), by.to(device)
        moment, direction, log_mag = model(bx)
        loss, _ = criterion(
            moment, direction, log_mag, by,
            bx if r_grid is not None else None,
            r_grid,
        )
        total_loss += float(loss.item())
        total_mag += mag_p_error(moment, by).item()
        total_incl += inclination_error(moment, by).item()
        total_decl += declination_error(moment, by).item()
        n_batches += 1
    model.train()
    n = max(n_batches, 1)
    return {
        "loss": total_loss / n,
        "mag_p": total_mag / n,
        "incl": total_incl / n,
        "decl": total_decl / n,
    }


def _train_dipolenet_stage(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: DipoleNetLoss,
    optimizer: torch.optim.Optimizer,
    scheduler,
    *,
    epochs: int,
    device: torch.device,
    r_grid: torch.Tensor | None,
    aug_prob: float,
    clip_grad_norm: float | None,
    checkpoint_prefix: str,
    log_every: int = 50,
    show_progress: bool = False,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
) -> dict:
    """Train one curriculum stage with DipoleNet and return a metrics dict.

    Compatible with the existing baby-steps stage-record and artifact helpers.
    """
    best_ckpt_path = _build_best_checkpoint_path(checkpoint_prefix)
    best_val = float("inf")
    best_ckpt = None
    epochs_since_improvement = 0
    early_stop_epoch = None

    train_loss_hist, train_mag_hist, train_incl_hist, train_decl_hist = [], [], [], []
    val_loss_hist, val_mag_hist, val_incl_hist, val_decl_hist = [], [], [], []


    clip_params = None
    if clip_grad_norm is not None and clip_grad_norm > 0:
        clip_params = [p for group in optimizer.param_groups for p in group["params"]]

    bar_ctx = alive_bar(epochs, title=f"DipoleNet {Path(checkpoint_prefix).stem}") if show_progress else None
    bar_fn = bar_ctx.__enter__() if bar_ctx else None

    try:
        for epoch in range(epochs):
            model.train()
            ep_loss = ep_mag = ep_incl = ep_decl = 0.0
            n_batches = 0

            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                xb, yb = _superposition_augment(xb, yb, aug_prob)

                optimizer.zero_grad(set_to_none=True)
                moment, direction, log_mag = model(xb)
                loss, _ = criterion(
                    moment, direction, log_mag, yb,
                    xb if r_grid is not None else None,
                    r_grid,
                )
                loss.backward()
                if clip_params is not None:
                    torch.nn.utils.clip_grad_norm_(clip_params, max_norm=clip_grad_norm)
                optimizer.step()

                ep_loss += loss.item()
                with torch.no_grad():
                    ep_mag += mag_p_error(moment, yb).item()
                    ep_incl += inclination_error(moment, yb).item()
                    ep_decl += declination_error(moment, yb).item()
                n_batches += 1

            if scheduler is not None:
                scheduler.step()

            n = max(n_batches, 1)
            train_loss_hist.append(ep_loss / n)
            train_mag_hist.append(ep_mag / n)
            train_incl_hist.append(ep_incl / n)
            train_decl_hist.append(ep_decl / n)

            val_metrics = _eval_dipolenet_metrics(model, val_loader, criterion, device, r_grid)
            val_loss = val_metrics["loss"]
            val_loss_hist.append(val_loss)
            val_mag_hist.append(val_metrics["mag_p"])
            val_incl_hist.append(val_metrics["incl"])
            val_decl_hist.append(val_metrics["decl"])

            if (epoch + 1) % log_every == 0 or epoch == 0:
                print(
                    f"  epoch {epoch+1:>4}/{epochs} | "
                    f"train={train_loss_hist[-1]:.4f}  val={val_loss:.4f} | "
                    f"mag%={train_mag_hist[-1]*100:.1f}  "
                    f"incl={math.degrees(train_incl_hist[-1]):.1f}°  "
                    f"decl={math.degrees(train_decl_hist[-1]):.1f}°"
                )

            if val_loss < best_val - early_stopping_min_delta:
                best_val = val_loss
                torch.save(model.state_dict(), str(best_ckpt_path))
                criterion_ckpt_path = best_ckpt_path.with_name(
                    best_ckpt_path.stem + "_criterion" + best_ckpt_path.suffix
                )
                torch.save(criterion.state_dict(), str(criterion_ckpt_path))
                best_ckpt = str(best_ckpt_path)
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if early_stopping_patience and epochs_since_improvement >= early_stopping_patience:
                early_stop_epoch = epoch + 1
                print(f"  Early stopping at epoch {early_stop_epoch} (patience={early_stopping_patience})")
                break

            if bar_fn is not None:
                bar_fn()
    finally:
        if bar_ctx is not None:
            bar_ctx.__exit__(None, None, None)

    return {
        "train_loss": train_loss_hist,
        "train_mag_p": train_mag_hist,
        "train_incl": train_incl_hist,
        "train_decl": train_decl_hist,
        "val_loss": val_loss_hist,
        "best_val_loss": best_val,
        "best_ckpt": best_ckpt,
        "early_stop_epoch": early_stop_epoch,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    if not (0.0 < args.train_val_split < 1.0):
        raise ValueError("--train_val_split must be within (0, 1).")
    if not (0.0 < args.alpha < 1.0):
        raise ValueError("--alpha must be within (0, 1).")

    device = _resolve_device(args.device)
    print(f"Using device: {device}")

    seed = args.seed if args.seed is not None and args.seed >= 0 else None
    _seed_everything(seed)

    dataset_root = _resolve_dataset_root(args.dataset_root)
    snr_levels_filter = _parse_snr_sequence(args.snr_levels) if args.snr_levels else None
    mapping = _discover_datasets(dataset_root, min_dipoles=args.min_dipoles, max_dipoles=args.max_dipoles)

    if not mapping:
        raise ValueError(f"No matching training datasets found under {dataset_root}")

    if snr_levels_filter is not None:
        mapping = {snr: mapping[snr] for snr in snr_levels_filter if snr in mapping}
        if not mapping:
            raise ValueError("None of the requested --snr_levels were found in the dataset root.")

    unlock_order = _build_unlock_order(mapping, alpha=args.alpha)
    if not unlock_order:
        raise ValueError("No cells found after filtering. Check --min_dipoles / --max_dipoles / --snr_levels.")

    print(f"\nCurriculum: {len(unlock_order)} cells, alpha={args.alpha:.2f}")
    print("Unlock order (easiest -> hardest):")
    for idx, cell in enumerate(unlock_order, 1):
        print(f"  {idx:3d}. d={cell.difficulty:.3f}  {cell.dipoles:2d} dipoles  SNR {cell.snr:6g}")

    # ---- run / output directory setup ----
    run_dir = None
    run_id = None
    if args.run_dir:
        run_dir, run_id = init_run(
            run_dir=args.run_dir,
            model_name="AttentionDipoleModel" if args.dipolenet else _describe_model(args),
            dataset_path=[cell.folder for cell in unlock_order],
            seed=seed,
            cli_args=args,
        )
        args.reuse_output_dir = True
        args.output_dir = str(run_dir)
        if args.metrics_json:
            metrics_path = Path(args.metrics_json)
            if not metrics_path.is_absolute() and metrics_path.parent == Path("."):
                args.metrics_json = str(Path("metrics") / metrics_path.name)

    original_output_dir = Path(args.output_dir)
    output_dir = _ensure_unique_output_dir(original_output_dir, reuse=args.reuse_output_dir)
    args.step_stats_csv_dir = _resolve_step_stats_csv_dir(
        args.step_stats_csv_dir if args.step_stats_csv else None,
        output_dir=output_dir,
        original_output_dir=original_output_dir,
    )
    if output_dir != original_output_dir:
        print(f"Output directory not empty; writing to {output_dir}")
        args.output_dir = str(output_dir)

    selected_model_flags = [
        getattr(args, f, False)
        for f in ("deep_model", "minimal_model", "standard_bn_model", "no_bn_model",
                  "regularized_model", "deeper_regularized_model")
    ]
    if sum(bool(f) for f in selected_model_flags) > 1:
        raise ValueError("Select only one model architecture flag.")

    if args.dipolenet:
        model = AttentionDipoleModel(base_channels=args.base_channels).to(device)
        model.device = device
        if args.pretrained:
            state = load_state_dict_checkpoint(args.pretrained, map_location=device)
            model.load_state_dict(state)
            print(f"Loaded pretrained weights from {args.pretrained}")
        dipolenet_criterion = DipoleNetLoss(physics_weight=args.physics_weight).to(device)
        r_grid = (
            make_measurement_grid(
                grid_size=args.grid_size,
                extent=args.grid_extent,
                z_offset=args.grid_z_offset,
                device=device,
            )
            if args.physics_weight > 0.0
            else None
        )
        criterion_callable = dipolenet_criterion
        extra_optimizer_parameters = list(dipolenet_criterion.parameters())
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"AttentionDipoleModel | params={n_params:,} | physics_weight={args.physics_weight}")
    else:
        model = _select_model(args, device=device)
        dipolenet_criterion = None
        r_grid = None
        criterion_callable, _, extra_optimizer_parameters = _select_loss(args, device=device)

    bn_enabled = not (math.isclose(args.bn_min, 0.0) and math.isclose(args.bn_max, 0.0))
    if bn_enabled and not hasattr(model, "set_bn_strength"):
        print("Warning: BN schedule requested but model does not support set_bn_strength; ignoring.")

    optimizer = _make_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer,
        extra_parameters=extra_optimizer_parameters,
    )
    scheduler = _make_scheduler(optimizer, step_every=args.step_every, gamma=args.gamma)

    advancement_threshold = args.advancement_threshold

    # ---- Precomputation setup ----
    use_precompute = not args.no_precompute
    precomputed_dir = (
        Path(args.precomputed_dir) if args.precomputed_dir
        else output_dir / "precomputed"
    )
    index_h5_path = precomputed_dir / "stage_indices.h5"

    cell_npy_map: Dict[str, Tuple[Path, Path]] | None = None

    if use_precompute:
        precomputed_dir.mkdir(parents=True, exist_ok=True)
        n_stages = len(unlock_order)

        # Convert each unique cell HDF5 → uncompressed .npy (skips existing)
        cell_cache_samples = args.cell_cache_samples or args.w_n_data
        cell_npy_map = _convert_cells_to_npy(
            unlock_order,
            precomputed_dir,
            n_samples=cell_cache_samples,
        )
        cells_gb = len(cell_npy_map) * cell_cache_samples * (100 * 100 * 3 + 3) * 4 / (1024 ** 3)

        # Estimate stage .npy disk usage: each stage ≈ stage_samples × sample_bytes
        bytes_per_stage = args.stage_samples * (100 * 100 * 3 + 3) * 4
        if args.precompute_all:
            total_stage_gb = bytes_per_stage * n_stages / (1024 ** 3)
            print(
                f"\nPrecomputing all {n_stages} stages upfront -> {precomputed_dir}"
                f"\nCell cache: ~{cells_gb:.1f} GB | Stage files: ~{total_stage_gb:.1f} GB"
            )
        else:
            print(
                f"\nStreaming precompute: each stage built from cell .npy before training, "
                f"then {'kept' if args.keep_precomputed else 'deleted'}."
                f"\nCell cache: ~{cells_gb:.1f} GB | Max per-stage overhead: "
                f"~{bytes_per_stage / (1024**3):.2f} GB"
            )

    # ---- Optional: precompute ALL stages upfront before training begins ----
    if use_precompute and args.precompute_all:
        precomputed_stage_dirs: Dict[int, Path] = {}
        unlock_scratch: List[CellInfo] = []
        with h5py.File(str(index_h5_path), "w") as idx_f:
            idx_f.attrs["n_stages"] = len(unlock_order)
            idx_f.attrs["stage_samples"] = args.stage_samples
            idx_f.attrs["w_n_data"] = args.w_n_data
            idx_f.attrs["seed"] = seed if seed is not None else -1
            with alive_bar(len(unlock_order), title="Precomputing stages", unit="stage") as bar:
                for si, pc in enumerate(unlock_order, start=1):
                    unlock_scratch.append(pc)
                    stage_dir = _precompute_stage(
                        si, pc, unlock_scratch,
                        stage_samples=args.stage_samples,
                        w_n_data=args.w_n_data,
                        train_val_split=args.train_val_split,
                        seed=seed,
                        precomputed_dir=precomputed_dir,
                        index_h5=idx_f,
                        cell_npy_map=cell_npy_map,
                    )
                    precomputed_stage_dirs[si] = stage_dir
                    bar()
        print(f"All stages precomputed. Index saved to {index_h5_path}")
    elif use_precompute:
        # Streaming: open index H5 once; stages will be appended inside the loop
        _index_h5_handle = h5py.File(str(index_h5_path), "w")
        _index_h5_handle.attrs["n_stages"] = len(unlock_order)
        _index_h5_handle.attrs["stage_samples"] = args.stage_samples
        _index_h5_handle.attrs["w_n_data"] = args.w_n_data
        _index_h5_handle.attrs["seed"] = seed if seed is not None else -1
        precomputed_stage_dirs = {}

    run_start = datetime.datetime.now(datetime.timezone.utc)
    stage_records: List[Dict[str, Any]] = []
    epoch_rows: List[Dict[str, Any]] = []
    epoch_offset = 0
    unlocked: List[CellInfo] = []

    try:
      for stage_index, primary_cell in enumerate(unlock_order, start=1):
        unlocked.append(primary_cell)

        # Skip already-completed stages when resuming (--start_stage).
        # unlocked is still built above so later stages see the full history.
        if stage_index < args.start_stage:
            continue

        # ---- BN strength for the primary cell's SNR ----
        if bn_enabled and hasattr(model, "set_bn_strength"):
            strength = _bn_strength_from_snr(
                primary_cell.snr,
                bn_min=args.bn_min,
                bn_max=args.bn_max,
                snr_min=args.bn_snr_min,
                snr_max=args.bn_snr_max,
                curve=args.bn_curve,
            )
            model.set_bn_strength(strength)

        # ---- Build / load Baby-Steps dataset ----
        _t_load_start = time.perf_counter()
        if use_precompute:
            if stage_index not in precomputed_stage_dirs:
                # Streaming mode: precompute this stage now
                print(f"\n  [precompute] stage {stage_index}/{len(unlock_order)} "
                      f"({len(unlocked)} cells) ...", end=" ", flush=True)
                stage_dir = _precompute_stage(
                    stage_index, primary_cell, unlocked,
                    stage_samples=args.stage_samples,
                    w_n_data=args.w_n_data,
                    train_val_split=args.train_val_split,
                    seed=seed,
                    precomputed_dir=precomputed_dir,
                    index_h5=_index_h5_handle,
                    cell_npy_map=cell_npy_map,
                )
                precomputed_stage_dirs[stage_index] = stage_dir
            train_loader, primary_val_loader, n_train, component_records = _load_precomputed_stage(
                precomputed_stage_dirs[stage_index],
                batch_size=args.batch_size,
            )
        else:
            stage_seed = None if seed is None else (seed + stage_index)
            train_loader, primary_val_loader, n_train, component_records = _build_stage_loaders(
                unlocked,
                primary_cell,
                stage_samples=args.stage_samples,
                w_n_data=args.w_n_data,
                batch_size=args.batch_size,
                train_val_split=args.train_val_split,
                seed=stage_seed,
                device=None,
            )
        _t_load_end = time.perf_counter()

        _print_stage_summary(stage_index, primary_cell, unlocked, component_records, advancement_threshold)
        print(f"Training samples (mixed): {n_train}")
        print(f"  [timing] data loading: {_t_load_end - _t_load_start:.1f}s")

        # ---- Optimizer / scheduler reset ----
        if not args.keep_optimizer:
            optimizer = _make_optimizer(
                model,
                lr=args.lr,
                weight_decay=args.weight_decay,
                optimizer_name=args.optimizer,
                extra_parameters=extra_optimizer_parameters,
            )
            scheduler = _make_scheduler(optimizer, step_every=args.step_every, gamma=args.gamma)
        elif scheduler is not None and args.reset_scheduler_each_stage:
            scheduler = _warm_restart_scheduler(
                optimizer=optimizer,
                step_every=args.step_every,
                gamma=args.gamma,
                base_lr=args.lr,
            )

        # ---- Checkpoint path for this stage ----
        ckpt_prefix = output_dir / f"{args.ckpt_prefix}_stage{stage_index:03d}_{primary_cell.identifier}"

        # ---- Step stats CSV ----
        step_stats_csv_path = None
        if args.step_stats_csv_dir:
            step_stats_dir = Path(args.step_stats_csv_dir)
            step_stats_dir.mkdir(parents=True, exist_ok=True)
            step_stats_csv_path = step_stats_dir / f"stage{stage_index:03d}_{primary_cell.identifier}_steps.csv"

        # ---- Train: primary-cell val loader drives checkpoint selection + competence gate ----
        if args.dipolenet:
            metrics = _train_dipolenet_stage(
                model=model,
                train_loader=train_loader,
                val_loader=primary_val_loader,
                criterion=dipolenet_criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=args.max_epochs_per_stage,
                device=device,
                r_grid=r_grid,
                aug_prob=args.aug_prob,
                clip_grad_norm=args.clip_grad_norm,
                checkpoint_prefix=str(ckpt_prefix),
                log_every=args.log_every,
                show_progress=args.show_progress,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
            )
        else:
            metrics = train_one_stage(
                model=model,
                train_loader=train_loader,
                val_loader=primary_val_loader,
                epochs=args.max_epochs_per_stage,
                optimizer=optimizer,
                scheduler=scheduler,
                clip_grad_norm=args.clip_grad_norm,
                criterion=criterion_callable,
                checkpoint_prefix=str(ckpt_prefix),
                log_every=args.log_every,
                show_progress=args.show_progress,
                step_stats_csv_path=str(step_stats_csv_path) if step_stats_csv_path else None,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
            )

        _t_train_end = time.perf_counter()
        print(f"  [timing] training:      {_t_train_end - _t_load_end:.1f}s")

        best_val_loss = metrics.get("best_val_loss", float("inf"))

        # ---- Competence-gated advancement check ----
        if advancement_threshold is not None:
            competence_reached = best_val_loss < advancement_threshold
            status = "PASSED" if competence_reached else "NOT MET (advancing anyway)"
            print(
                f"Competence gate: best_val_loss={best_val_loss:.4e} "
                f"vs threshold={advancement_threshold:.4e} → {status}"
            )
        else:
            competence_reached = True  # gating disabled

        print(f"Best checkpoint: {metrics.get('best_ckpt')}")

        # ---- Reload best checkpoint before next stage ----
        if args.load_best_per_stage and metrics.get("best_ckpt"):
            best_ckpt_path = Path(metrics["best_ckpt"])
            if not best_ckpt_path.is_absolute():
                best_ckpt_path = ckpt_prefix.parent / best_ckpt_path.name
            print(f"Reloading best checkpoint from {best_ckpt_path}")
            state_dict = load_state_dict_checkpoint(best_ckpt_path, map_location=device)
            model.load_state_dict(state_dict)

        if run_dir and metrics.get("best_ckpt"):
            register_artifact(run_dir, "checkpoint", metrics.get("best_ckpt"))

        # ---- Save training curves ----
        if args.plot_curves:
            plot_path = output_dir / f"stage{stage_index:03d}_{primary_cell.identifier}_training.png"
            if run_dir:
                plot_path = Path(run_dir) / "plots" / plot_path.name
            plot_training(metrics, out_png=str(plot_path))
            print(f"Saved training curves to {plot_path}")
            if run_dir:
                plot_rows = build_training_plot_rows(metrics, angles_in_degrees=False)
                save_plot_data(run_dir, plot_path.stem, plot_rows)
                register_artifact(run_dir, "plot", plot_path)

        # ---- Epoch-level metric rows ----
        if run_dir:
            stage_epoch_rows = build_epoch_rows_from_metrics(
                metrics,
                run_id=run_id or "",
                epoch_offset=epoch_offset,
                angles_in_degrees=False,
            )
            epoch_rows.extend(stage_epoch_rows)
            epoch_offset += len(metrics.get("train_loss") or [])

        # ---- Stage record ----
        stage_record: Dict[str, Any] = {
            "stage": stage_index,
            "primary_dipoles": primary_cell.dipoles,
            "primary_snr": primary_cell.snr,
            "difficulty": primary_cell.difficulty,
            "identifier": primary_cell.identifier,
            "unlocked_cells": len(unlocked),
            "total_train_samples": n_train,
            "components": component_records,
            "best_val_loss": best_val_loss,
            "best_ckpt": metrics.get("best_ckpt"),
            "early_stop_epoch": metrics.get("early_stop_epoch"),
            "competence_reached": competence_reached,
            "train_loss": metrics.get("train_loss"),
            "train_mag_p": metrics.get("train_mag_p"),
            "train_incl": metrics.get("train_incl"),
            "train_decl": metrics.get("train_decl"),
        }
        stage_records.append(_sanitize_for_json(stage_record))

        # ---- Clean up precomputed .npy for this stage (streaming mode) ----
        if use_precompute and not args.keep_precomputed and not args.precompute_all:
            stage_dir = precomputed_stage_dirs.get(stage_index)
            if stage_dir:
                for _f in (stage_dir / "X.npy", stage_dir / "y.npy"):
                    if _f.exists():
                        _f.unlink()

    finally:
        # Close the streaming H5 index file if it was opened
        if use_precompute and not args.precompute_all and "_index_h5_handle" in dir():
            try:
                _index_h5_handle.close()
            except Exception:
                pass

    # ---- Delete cell .npy cache now that training is complete ----
    if use_precompute and not args.keep_precomputed:
        cells_dir = precomputed_dir / "cells"
        if cells_dir.exists():
            deleted = 0
            for _f in cells_dir.glob("*.npy"):
                _f.unlink()
                deleted += 1
            if deleted:
                print(f"Deleted {deleted} cell cache .npy files from {cells_dir}")

    # ---- Save final model ----
    final_model_path = output_dir / args.final_model
    save_model(model, out_path=str(final_model_path))
    print(f"\nSaved final model to {final_model_path}")
    if args.dipolenet:
        final_criterion_path = final_model_path.with_name(
            final_model_path.stem + "_criterion" + final_model_path.suffix
        )
        torch.save(dipolenet_criterion.state_dict(), str(final_criterion_path))
        print(f"Saved final criterion to {final_criterion_path}")
    # ---- Post-save logging (non-fatal: model is already on disk) ----
    try:
        if run_dir:
            register_artifact(run_dir, "model", final_model_path)

        run_end = datetime.datetime.now(datetime.timezone.utc)

        # ---- Save metrics JSON ----
        artifacts_extra: Dict[str, Any] = {
            "output_dir": str(output_dir),
            "unlock_order": [
                {"dipoles": ci.dipoles, "snr": ci.snr, "difficulty": ci.difficulty}
                for ci in unlock_order
            ],
            "alpha": args.alpha,
            "advancement_threshold": advancement_threshold,
        }

        if args.metrics_json:
            metrics_path = output_dir / args.metrics_json
            with metrics_path.open("w", encoding="utf-8") as fp:
                json.dump(stage_records, fp, indent=2)
            print(f"Persisted stage metrics to {metrics_path}")
            artifacts_extra["stage_metrics_path"] = str(metrics_path)
            if run_dir:
                register_artifact(run_dir, "metrics", metrics_path)

        if run_dir and epoch_rows:
            write_epoch_metrics(run_dir, run_id or "", epoch_rows)

        # ---- Training metadata ----
        summary_metrics: Dict[str, Any] = {
            "stage_metrics": stage_records,
            "best_val_loss": stage_records[-1].get("best_val_loss") if stage_records else None,
            "total_stages": len(stage_records),
            "cells_reaching_competence": sum(
                1 for r in stage_records if r.get("competence_reached", False)
            ),
        }

        metadata = build_training_metadata(
            args=args,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion_callable,
            dataset_stages=stage_records,
            metrics=summary_metrics,
            start_time=run_start,
            end_time=run_end,
            checkpoint_prefix=str(output_dir / args.ckpt_prefix),
            best_checkpoint=None,
            final_model_path=str(final_model_path),
            order_type="baby_steps_2d",
            run_id=run_id,
            artifacts_extra=artifacts_extra,
            trainer_script=str(Path(__file__).resolve()),
        )
        metadata_filename = f"{metadata['run_id']}_metadata.json"
        metadata_path = save_training_metadata(metadata, output_dir / metadata_filename)
        print(f"Recorded training metadata to {metadata_path}")
        if run_dir:
            register_artifact(run_dir, "metadata", metadata_path)
    except Exception as _post_save_exc:
        print(f"\nWARNING: post-training logging failed: {_post_save_exc}")
        print("The model checkpoint was saved successfully; this is non-fatal.")


if __name__ == "__main__":
    main()

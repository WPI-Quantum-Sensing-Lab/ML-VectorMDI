"""Ensemble uncertainty training for a curriculum-trained AttentionDipoleModel.

Runs the full baby-steps curriculum training N times, each with a different
random seed, to produce a proper deep ensemble (Lakshminarayanan et al. 2017).
Diversity comes from independent weight initialisation and stochastic training
dynamics rather than fine-tuning a shared checkpoint.

Each ensemble member is an independent subprocess call to
``model.training.curriculum`` with:
  - ``--seed base_seed + k``
  - ``--output_dir <ensemble_output_dir>/member_kk``
  - all other flags forwarded unchanged

Disk space:
  By default ``--no_precompute`` is passed to each member run so no cell
  .npy caches are written (training reads HDF5 directly).  If you have
  153 GB free and want faster I/O, pass ``--shared_cell_cache <dir>``; the
  cells are generated once into that directory and reused by every member
  (each member still gets its own stage_indices.h5 inside its output dir).

After all members finish, ``ensemble_manifest.json`` is written to
``<ensemble_output_dir>`` listing every member checkpoint and best val loss.

Example
-------

    # No cell cache (safe, slower per-stage I/O), run from src/:
    python -m model.training.ensemble \\
        --dataset_root data/datasets \\
        --min_dipoles 1 --max_dipoles 10 \\
        --alpha 0.6 --advancement_threshold 0.05 \\
        --max_epochs_per_stage 300 --stage_samples 20000 --w_n_data 8000 \\
        --batch_size 64 --dipolenet --aug_prob 0.5 --physics_weight 0.1 \\
        --grid_extent 1.0 --grid_z_offset 0.1 \\
        --load_best_per_stage --plot_curves \\
        --metrics_json baby_steps_trial_metrics.json \\
        --ensemble_size 5 --base_seed 42 \\
        --ensemble_output_dir runs/baby_steps_ensemble

    # Shared cell cache (generate .npy files once, reuse across members):
    python -m model.training.ensemble \\
        ... (same flags) ...
        --shared_cell_cache runs/baby_steps_ensemble/shared_cells

Inference
---------

    from model.training.ensemble import EnsemblePredictor
    import json, torch
    from pathlib import Path

    with open("runs/baby_steps_ensemble/ensemble_manifest.json") as f:
        manifest = json.load(f)

    predictor = EnsemblePredictor.from_checkpoints(
        [Path(p) for p in manifest["member_final_checkpoints"]],
        device=torch.device("cuda"),
    )
    mean_moment, std_moment = predictor(x)   # x: [B, 3, 100, 100]
    scalar_uncertainty = std_moment.norm(dim=-1)   # [B]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from model.architecture import AttentionDipoleModel

# src/ directory (two levels above this file); child processes need it on
# PYTHONPATH so `-m model.training.curriculum` resolves regardless of cwd.
_SRC_DIR = Path(__file__).resolve().parents[2]


def _child_env() -> Dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{_SRC_DIR}{os.pathsep}{existing}" if existing else str(_SRC_DIR)
    )
    return env


# ---------------------------------------------------------------------------
# Ensemble inference (used post-training)
# ---------------------------------------------------------------------------

class EnsemblePredictor(nn.Module):
    """Wraps N AttentionDipoleModel checkpoints for uncertainty-aware inference.

    Usage::

        predictor = EnsemblePredictor.from_checkpoints(paths, device=device)
        mean_moment, std_moment = predictor(x)  # x: [B, 3, 100, 100]
        scalar_uncertainty = std_moment.norm(dim=-1)  # [B]
    """

    def __init__(self, members: List[AttentionDipoleModel]) -> None:
        super().__init__()
        self.members = nn.ModuleList(members)

    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_paths: List[Path],
        *,
        base_channels: int = 32,
        device: torch.device,
    ) -> "EnsemblePredictor":
        members = []
        for p in checkpoint_paths:
            m = AttentionDipoleModel(base_channels=base_channels).to(device)
            state = torch.load(str(p), map_location=device, weights_only=True)
            m.load_state_dict(state, strict=True)
            m.eval()
            members.append(m)
        print(f"Loaded {len(members)} ensemble members from disk.")
        return cls(members)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """Run all members and return (mean_moment, std_moment).

        Args:
            x: Input field images [B, 3, H, W].

        Returns:
            mean_moment: Mean predicted moment [B, 3].
            std_moment:  Per-component std across members [B, 3].
                         ``std_moment.norm(dim=-1)`` gives scalar uncertainty [B].
        """
        all_moments = []
        for member in self.members:
            moment, _, _ = member(x)
            all_moments.append(moment)
        stacked = torch.stack(all_moments, dim=0)  # [N, B, 3]
        return stacked.mean(dim=0), stacked.std(dim=0)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run the full baby-steps curriculum training N times with different "
            "seeds to build a proper deep ensemble."
        ),
        # Let unknown args through so the user can add any baby-steps flag
        # without this script needing to know about it.
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- ensemble-specific ----
    p.add_argument(
        "--ensemble_size",
        type=int,
        default=5,
        help="Number of ensemble members (independent curriculum training runs).",
    )
    p.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Seed for member 0; member k uses base_seed + k.",
    )
    p.add_argument(
        "--ensemble_output_dir",
        type=str,
        default="runs/baby_steps_ensemble",
        help="Root directory; each member writes to <ensemble_output_dir>/member_kk.",
    )
    p.add_argument(
        "--shared_cell_cache",
        type=str,
        default=None,
        help=(
            "Optional: shared precomputed_dir for the cell .npy cache. "
            "Cell files are generated once and reused across all members. "
            "When omitted, --no_precompute is passed to every member run "
            "so no cell caches are written."
        ),
    )
    p.add_argument(
        "--curriculum_script",
        type=str,
        default=None,
        help=(
            "Optional path to an alternative curriculum-trainer script. "
            "Defaults to running the model.training.curriculum module."
        ),
    )

    # ---- forward everything else to the curriculum script ----
    # These are the baby-steps flags that are the same for every member.
    # --seed and --output_dir are injected per-member; do not pass them here.
    p.add_argument("--dataset_root", type=str, default="data/datasets")
    p.add_argument("--min_dipoles", type=int, default=1)
    p.add_argument("--max_dipoles", type=int, default=10)
    p.add_argument("--snr_levels", type=str, default=None)
    p.add_argument("--allow_missing", action="store_true")
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--advancement_threshold", type=float, default=0.05)
    p.add_argument("--max_epochs_per_stage", type=int, default=300)
    p.add_argument("--stage_samples", type=int, default=20000)
    p.add_argument("--w_n_data", type=int, default=8000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--train_val_split", type=float, default=0.8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--optimizer", choices=("adamw", "lbfgs"), default="adamw")
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--clip_grad_norm", type=float, default=None)
    p.add_argument("--step_every", type=int, default=0)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--early_stopping_patience", type=int, default=20)
    p.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    p.add_argument("--bn_min", type=float, default=0.0)
    p.add_argument("--bn_max", type=float, default=0.0)
    p.add_argument("--bn_snr_min", type=float, default=0.05)
    p.add_argument("--bn_snr_max", type=float, default=20.0)
    p.add_argument("--bn_curve", choices=("linear", "log"), default="log")
    p.add_argument("--pretrained", type=str, default=None)
    p.add_argument("--deep_model", action="store_true")
    p.add_argument("--minimal_model", action="store_true")
    p.add_argument("--standard_bn_model", action="store_true")
    p.add_argument("--no_bn_model", action="store_true")
    p.add_argument("--regularized_model", action="store_true")
    p.add_argument("--deeper_regularized_model", action="store_true")
    p.add_argument(
        "--loss_mode",
        choices=("directional-magnitude", "full-vector", "smooth-l1"),
        default="smooth-l1",
    )
    p.add_argument("--dipolenet", action="store_true")
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--physics_weight", type=float, default=0.0)
    p.add_argument("--grid_size", type=int, default=100)
    p.add_argument("--grid_extent", type=float, default=1.0)
    p.add_argument("--grid_z_offset", type=float, default=0.1)
    p.add_argument("--aug_prob", type=float, default=0.5)
    p.add_argument("--keep_optimizer", action="store_true")
    p.add_argument("--reset_scheduler_each_stage", action="store_true")
    p.add_argument("--load_best_per_stage", action="store_true")
    p.add_argument("--step_stats_csv", action="store_true")
    p.add_argument("--step_stats_csv_dir", type=str, default="step_logs")
    p.add_argument(
        "--final_model",
        type=str,
        default="vector_model_baby_steps_final.pth",
        help="Filename for each member's final model (saved inside its output dir).",
    )
    p.add_argument(
        "--metrics_json",
        type=str,
        default="baby_steps_curriculum_metrics.json",
    )
    p.add_argument("--plot_curves", action="store_true")
    p.add_argument("--ckpt_prefix", type=str, default="checkpoint_minval")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--show_progress", action="store_true")
    p.add_argument("--cell_cache_samples", type=int, default=None)

    # ---- resume ----
    p.add_argument(
        "--start_member",
        type=int,
        default=0,
        help=(
            "Resume ensemble training from this member index (0-based). "
            "Members before it are assumed already complete; their results are "
            "reconstructed from their output dirs. For the first active member "
            "(start_member itself), the highest completed stage is auto-detected "
            "from existing checkpoints and --start_stage / --pretrained are injected "
            "automatically."
        ),
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _detect_resume_stage(member_output_dir: Path, ckpt_prefix: str) -> tuple[int, Optional[Path]]:
    """Return (next_stage, pretrained_path) for resuming a partially-completed member.

    Scans *member_output_dir* for checkpoint files matching the naming pattern
    ``<ckpt_prefix>_stage<NNN>_*_min_loss.pth`` and returns the stage after the
    highest completed one, plus the path to that checkpoint.

    Returns (1, None) when no checkpoints are found (fresh start).
    """
    import re
    pattern = re.compile(rf"{re.escape(ckpt_prefix)}_stage(\d+)_.*_min_loss\.pth$")
    best_stage = 0
    best_path: Optional[Path] = None
    for p in member_output_dir.glob(f"{ckpt_prefix}_stage*_min_loss.pth"):
        m = pattern.match(p.name)
        if m:
            s = int(m.group(1))
            if s > best_stage:
                best_stage = s
                best_path = p
    return (best_stage + 1, best_path)


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------

def _build_member_cmd(
    args: argparse.Namespace,
    member_idx: int,
    member_output_dir: Path,
    curriculum_cmd: List[str],
    *,
    start_stage: int = 1,
    pretrained_ckpt: Optional[Path] = None,
) -> List[str]:
    """Construct the subprocess command for one ensemble member."""
    seed = args.base_seed + member_idx

    cmd = [
        *curriculum_cmd,
        "--dataset_root", args.dataset_root,
        "--min_dipoles", str(args.min_dipoles),
        "--max_dipoles", str(args.max_dipoles),
        "--alpha", str(args.alpha),
        "--advancement_threshold", str(args.advancement_threshold),
        "--max_epochs_per_stage", str(args.max_epochs_per_stage),
        "--stage_samples", str(args.stage_samples),
        "--w_n_data", str(args.w_n_data),
        "--batch_size", str(args.batch_size),
        "--train_val_split", str(args.train_val_split),
        "--lr", str(args.lr),
        "--optimizer", args.optimizer,
        "--weight_decay", str(args.weight_decay),
        "--step_every", str(args.step_every),
        "--gamma", str(args.gamma),
        "--log_every", str(args.log_every),
        "--early_stopping_patience", str(args.early_stopping_patience),
        "--early_stopping_min_delta", str(args.early_stopping_min_delta),
        "--bn_min", str(args.bn_min),
        "--bn_max", str(args.bn_max),
        "--bn_snr_min", str(args.bn_snr_min),
        "--bn_snr_max", str(args.bn_snr_max),
        "--bn_curve", args.bn_curve,
        "--loss_mode", args.loss_mode,
        "--base_channels", str(args.base_channels),
        "--physics_weight", str(args.physics_weight),
        "--grid_size", str(args.grid_size),
        "--grid_extent", str(args.grid_extent),
        "--grid_z_offset", str(args.grid_z_offset),
        "--aug_prob", str(args.aug_prob),
        "--step_stats_csv_dir", args.step_stats_csv_dir,
        "--final_model", args.final_model,
        "--metrics_json", args.metrics_json,
        "--ckpt_prefix", args.ckpt_prefix,
        "--seed", str(seed),
        "--output_dir", str(member_output_dir),
        "--reuse_output_dir",  # member_output_dir is freshly created by us
    ]

    # Boolean flags
    if args.allow_missing:
        cmd.append("--allow_missing")
    if args.deep_model:
        cmd.append("--deep_model")
    if args.minimal_model:
        cmd.append("--minimal_model")
    if args.standard_bn_model:
        cmd.append("--standard_bn_model")
    if args.no_bn_model:
        cmd.append("--no_bn_model")
    if args.regularized_model:
        cmd.append("--regularized_model")
    if args.deeper_regularized_model:
        cmd.append("--deeper_regularized_model")
    if args.dipolenet:
        cmd.append("--dipolenet")
    if args.keep_optimizer:
        cmd.append("--keep_optimizer")
    if args.reset_scheduler_each_stage:
        cmd.append("--reset_scheduler_each_stage")
    if args.load_best_per_stage:
        cmd.append("--load_best_per_stage")
    if args.step_stats_csv:
        cmd.append("--step_stats_csv")
    if args.plot_curves:
        cmd.append("--plot_curves")
    if args.show_progress:
        cmd.append("--show_progress")

    # Optional values
    if args.snr_levels is not None:
        cmd += ["--snr_levels", args.snr_levels]
    # pretrained_ckpt (resume) takes priority over the global --pretrained flag
    effective_pretrained = str(pretrained_ckpt) if pretrained_ckpt is not None else args.pretrained
    if effective_pretrained is not None:
        cmd += ["--pretrained", effective_pretrained]
    if start_stage > 1:
        cmd += ["--start_stage", str(start_stage)]
    if args.clip_grad_norm is not None:
        cmd += ["--clip_grad_norm", str(args.clip_grad_norm)]
    if args.device is not None:
        cmd += ["--device", args.device]
    if args.cell_cache_samples is not None:
        cmd += ["--cell_cache_samples", str(args.cell_cache_samples)]

    # Precomputation: shared cell cache OR no precompute
    if args.shared_cell_cache is not None:
        # Point all members at the same cells dir.
        # Each member writes its own stage_indices.h5 inside its output dir,
        # but the cells/ subdir is shared so .npy files are only generated once.
        shared_dir = Path(args.shared_cell_cache)
        member_precomputed_dir = shared_dir / f"member_{member_idx:02d}"
        member_precomputed_dir.mkdir(parents=True, exist_ok=True)
        # Symlink cells/ → shared_dir/cells so all members share the same files.
        shared_cells = shared_dir / "cells"
        shared_cells.mkdir(parents=True, exist_ok=True)
        member_cells_link = member_precomputed_dir / "cells"
        if not member_cells_link.exists():
            try:
                member_cells_link.symlink_to(shared_cells.resolve())
            except OSError:
                # On Windows without developer mode, fall back to a junction
                import subprocess as _sp
                _sp.run(
                    ["cmd", "/c", "mklink", "/J",
                     str(member_cells_link), str(shared_cells.resolve())],
                    check=True,
                )
        cmd += ["--precomputed_dir", str(member_precomputed_dir)]
    else:
        # No cell cache: read HDF5 directly.  No disk overhead beyond checkpoints.
        cmd.append("--no_precompute")

    return cmd


# ---------------------------------------------------------------------------
# ETA tracking
# ---------------------------------------------------------------------------

def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as Xh Ym Zs."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _print_eta(
    completed: int,
    total: int,
    elapsed_times: List[float],
    window: int = 3,
) -> None:
    """Print a moving-average ETA after each member completes.

    Args:
        completed: Number of members finished so far.
        total: Total ensemble size.
        elapsed_times: Wall-clock seconds for each completed member.
        window: Moving-average window size.
    """
    remaining = total - completed
    if remaining == 0:
        return

    recent = elapsed_times[-window:]
    avg_time = sum(recent) / len(recent)
    eta_seconds = avg_time * remaining
    finish_at = datetime.datetime.now() + datetime.timedelta(seconds=eta_seconds)

    total_elapsed = sum(elapsed_times)
    print(
        f"\n  ETA  |  members done: {completed}/{total}"
        f"  |  avg/member (last {len(recent)}): {_fmt_duration(avg_time)}"
        f"  |  remaining: ~{_fmt_duration(eta_seconds)}"
        f"  |  est. finish: {finish_at.strftime('%Y-%m-%d %H:%M:%S')}"
        f"  |  total elapsed: {_fmt_duration(total_elapsed)}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    ensemble_output_dir = Path(args.ensemble_output_dir)
    ensemble_output_dir.mkdir(parents=True, exist_ok=True)

    if args.curriculum_script:
        curriculum_script = Path(args.curriculum_script)
        if not curriculum_script.exists():
            raise FileNotFoundError(f"Curriculum script not found: {curriculum_script}")
        curriculum_cmd = [sys.executable, str(curriculum_script)]
    else:
        curriculum_cmd = [sys.executable, "-m", "model.training.curriculum"]

    if args.shared_cell_cache is not None and args.shared_cell_cache.lower() in ("true", "false", "1", "0", "yes", "no"):
        raise ValueError(
            f"--shared_cell_cache expects a directory path, not '{args.shared_cell_cache}'. "
            "Example: --shared_cell_cache runs/baby_steps_ensemble/shared_cells"
        )

    print(f"\n{'='*60}")
    print(f"Deep ensemble: {args.ensemble_size} members")
    print(f"Seeds: {args.base_seed} .. {args.base_seed + args.ensemble_size - 1}")
    print(f"Output: {ensemble_output_dir}")
    if args.start_member > 0:
        print(f"Resuming from member {args.start_member:02d} (skipping 0..{args.start_member - 1:02d})")
    if args.shared_cell_cache:
        print(f"Shared cell cache: {args.shared_cell_cache}")
    else:
        print("Precompute: disabled (reading HDF5 directly)")
    print(f"{'='*60}\n")

    member_results: List[Dict] = []
    elapsed_times: List[float] = []
    ensemble_start = time.monotonic()

    for k in range(args.ensemble_size):
        member_output_dir = ensemble_output_dir / f"member_{k:02d}"
        member_output_dir.mkdir(parents=True, exist_ok=True)

        # ---- Skip already-completed members ----
        if k < args.start_member:
            final_ckpt = member_output_dir / args.final_model
            member_results.append(
                {
                    "member_idx": k,
                    "seed": args.base_seed + k,
                    "output_dir": str(member_output_dir),
                    "final_checkpoint": str(final_ckpt) if final_ckpt.exists() else None,
                    "returncode": 0,
                    "success": final_ckpt.exists(),
                    "elapsed_seconds": 0.0,
                    "skipped": True,
                }
            )
            print(f"Member {k:02d}: skipped (assumed complete, final_ckpt={'found' if final_ckpt.exists() else 'MISSING'})")
            continue

        # ---- Auto-detect resume point for every active member ----
        resume_start_stage, resume_pretrained = _detect_resume_stage(
            member_output_dir, args.ckpt_prefix
        )
        if resume_pretrained is not None:
            print(
                f"Member {k:02d}: resuming from stage {resume_start_stage} "
                f"(pretrained: {resume_pretrained.name})"
            )
        else:
            print(f"Member {k:02d}: no checkpoints found, starting from stage 1")

        cmd = _build_member_cmd(
            args, k, member_output_dir, curriculum_cmd,
            start_stage=resume_start_stage,
            pretrained_ckpt=resume_pretrained,
        )

        print(f"\n{'─'*60}")
        print(f"Member {k:02d}  (seed={args.base_seed + k})  →  {member_output_dir}")
        print(f"{'─'*60}")
        print("Command:\n  " + " \\\n    ".join(cmd))
        print()

        member_start = time.monotonic()
        result = subprocess.run(cmd, check=False, env=_child_env())
        member_elapsed = time.monotonic() - member_start
        elapsed_times.append(member_elapsed)

        final_ckpt = member_output_dir / args.final_model
        success = final_ckpt.exists()

        member_results.append(
            {
                "member_idx": k,
                "seed": args.base_seed + k,
                "output_dir": str(member_output_dir),
                "final_checkpoint": str(final_ckpt) if final_ckpt.exists() else None,
                "returncode": result.returncode,
                "success": success,
                "elapsed_seconds": member_elapsed,
            }
        )

        if not success:
            print(
                f"WARNING: member {k:02d} exited with code {result.returncode} "
                f"and final checkpoint not found at {final_ckpt}. Continuing."
            )
        elif result.returncode != 0:
            print(
                f"NOTE: member {k:02d} exited with code {result.returncode} "
                f"but final checkpoint exists — counting as success."
            )

        _print_eta(k + 1, args.ensemble_size, elapsed_times)

    # ---- Clean up shared cell cache .npy files (keep stage_indices.h5) ----
    if args.shared_cell_cache is not None:
        shared_cells = Path(args.shared_cell_cache) / "cells"
        npy_files = list(shared_cells.glob("*.npy"))
        if npy_files:
            print(f"\nCleaning up {len(npy_files)} cell .npy files from {shared_cells} ...")
            for f in npy_files:
                f.unlink()
            print("  Done.")

    # ---- Ensemble manifest ----
    successful = [r for r in member_results if r["success"]]
    manifest = {
        "ensemble_size": args.ensemble_size,
        "base_seed": args.base_seed,
        "ensemble_output_dir": str(ensemble_output_dir),
        "member_final_checkpoints": [r["final_checkpoint"] for r in successful],
        "member_results": member_results,
        "hyperparams": {
            "dataset_root": args.dataset_root,
            "min_dipoles": args.min_dipoles,
            "max_dipoles": args.max_dipoles,
            "alpha": args.alpha,
            "advancement_threshold": args.advancement_threshold,
            "max_epochs_per_stage": args.max_epochs_per_stage,
            "stage_samples": args.stage_samples,
            "w_n_data": args.w_n_data,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "aug_prob": args.aug_prob,
            "physics_weight": args.physics_weight,
            "dipolenet": args.dipolenet,
        },
    }
    manifest_path = ensemble_output_dir / "ensemble_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    total_wall = time.monotonic() - ensemble_start
    print(f"\n{'='*60}")
    print(f"Ensemble complete: {len(successful)}/{args.ensemble_size} members succeeded.")
    print(f"Total wall time:   {_fmt_duration(total_wall)}")
    print(f"Manifest: {manifest_path}")
    print(f"{'='*60}")

    if len(successful) < args.ensemble_size:
        sys.exit(1)


if __name__ == "__main__":
    main()

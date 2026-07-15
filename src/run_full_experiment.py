"""
run_full_experiment.py

Trains the published deep ensemble with aug_prob (mixup rate) 0.1.

This release contains the no-physics variant of the experiment: a 6-member
deep ensemble of curriculum-trained AttentionDipoleModel networks with
physics_weight = 0.

Execution order
---------------
  1. Train ensemble — no physics loss
  2. Collect models → <results_dir>/<variant>/

Crash safety
------------
A JSON state file tracks which steps are complete.  A human-readable
safety_log.txt records every transition with timestamps and resume hints.
Re-running this script resumes from the last completed step.

Usage (from src/):
    python run_full_experiment.py
    python run_full_experiment.py --ensemble-size 6 --results-dir big_results
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parent
_DATA = (_SRC.parent / "data" / "datasets").resolve()

FINAL_MODEL_NAME = "vector_model_baby_steps_final.pth"

COMMON_TRAIN_FLAGS = [
    "--dataset_root", str(_DATA),
    "--min_dipoles", "1",
    "--max_dipoles", "10",
    "--alpha", "0.6",
    "--advancement_threshold", "0.05",
    "--max_epochs_per_stage", "300",
    "--stage_samples", "20000",
    "--w_n_data", "8000",
    "--batch_size", "64",
    "--dipolenet",
    "--aug_prob", "0.1",
    "--load_best_per_stage",
    "--plot_curves",
    "--metrics_json", "baby_steps_trial_metrics.json",
    "--base_seed", "42",
]

VARIANTS = [
    {
        "key": "no_physics",
        "label": "No Physics Loss",
        "extra_train_flags": ["--physics_weight", "0.0"],
    },
]


# ---------------------------------------------------------------------------
# Safety log
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


class SafetyLog:
    """Persistent state tracker.  JSON for machine reads, txt for humans."""

    STEPS = ["training", "collect_results"]

    def __init__(self, results_dir: Path) -> None:
        self._dir = results_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._state_path = results_dir / "state.json"
        self._log_path = results_dir / "safety_log.txt"
        self._state = self._load()

    def _load(self) -> dict:
        if self._state_path.exists():
            raw = json.loads(self._state_path.read_text())
            for vkey in raw:
                for step in self.STEPS:
                    val = raw[vkey].get(step, "pending")
                    if isinstance(val, str):
                        raw[vkey][step] = self._blank_step(val)
            return raw
        return {v["key"]: {s: self._blank_step() for s in self.STEPS} for v in VARIANTS}

    @staticmethod
    def _blank_step(status: str = "pending") -> dict:
        return {"status": status, "started_at": None, "completed_at": None,
                "wall_seconds": None, "total_epochs": None}

    def _save(self) -> None:
        tmp = self._state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, indent=2))
        tmp.replace(self._state_path)  # atomic on same filesystem

    def _append_log(self, msg: str) -> None:
        line = f"{_now()}  {msg}\n"
        with self._log_path.open("a") as f:
            f.write(line)
        print(line, end="")

    def is_complete(self, variant_key: str, step: str) -> bool:
        return self._state.get(variant_key, {}).get(step, {}).get("status") == "complete"

    def mark_started(self, variant_key: str, step: str, resume_hint: str = "") -> None:
        self._state[variant_key][step]["status"] = "started"
        self._state[variant_key][step]["started_at"] = datetime.now(timezone.utc).isoformat()
        self._save()
        hint = f"  resume: {resume_hint}" if resume_hint else ""
        self._append_log(f"[{variant_key}]  {step:<20} STARTED{hint}")

    def mark_complete(
        self,
        variant_key: str,
        step: str,
        *,
        wall_seconds: float | None = None,
        total_epochs: int | None = None,
    ) -> None:
        s = self._state[variant_key][step]
        s["status"] = "complete"
        s["completed_at"] = datetime.now(timezone.utc).isoformat()
        if wall_seconds is not None:
            s["wall_seconds"] = round(wall_seconds, 1)
        if total_epochs is not None:
            s["total_epochs"] = total_epochs
        self._save()
        timing = ""
        if wall_seconds is not None:
            timing = f"  {wall_seconds:.0f}s ({wall_seconds/3600:.2f}h)"
        if total_epochs is not None:
            timing += f"  {total_epochs} epochs"
        self._append_log(f"[{variant_key}]  {step:<20} COMPLETE{timing}")

    def mark_failed(self, variant_key: str, step: str, reason: str) -> None:
        self._state[variant_key][step]["status"] = "failed"
        self._save()
        self._append_log(f"[{variant_key}]  {step:<20} FAILED  {reason}")

    def print_status(self) -> None:
        print("\n" + "=" * 60)
        print("Experiment status")
        print("=" * 60)
        for v in VARIANTS:
            k = v["key"]
            print(f"\n  [{k}]  {v['label']}")
            for step in self.STEPS:
                s = self._state.get(k, {}).get(step, {})
                if isinstance(s, str):
                    status, wall, epochs = s, None, None
                else:
                    status = s.get("status", "pending")
                    wall   = s.get("wall_seconds")
                    epochs = s.get("total_epochs")
                icon = {"complete": "✓", "started": "→", "failed": "✗", "pending": "·"}.get(status, "?")
                extra = ""
                if wall is not None:
                    extra += f"  {wall:.0f}s ({wall/3600:.2f}h)"
                if epochs is not None:
                    extra += f"  {epochs} epochs"
                print(f"    {icon} {step:<22} {status}{extra}")
        print()

    def write_timing_summary(self) -> None:
        summary: dict = {}
        for v in VARIANTS:
            k = v["key"]
            summary[k] = {}
            for step in self.STEPS:
                s = self._state.get(k, {}).get(step, {})
                if not isinstance(s, dict):
                    continue
                if s.get("wall_seconds") is not None or s.get("total_epochs") is not None:
                    summary[k][step] = {
                        "wall_seconds": s.get("wall_seconds"),
                        "total_epochs": s.get("total_epochs"),
                    }
        out = self._dir / "timing_summary.json"
        out.write_text(json.dumps(summary, indent=2))
        print(f"Timing summary written to {out}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], label: str) -> None:
    """Run a subprocess, streaming output.  Raises on non-zero exit."""
    print(f"\n{'─'*60}\n{label}\n{'─'*60}")
    print("CMD:", " ".join(cmd), "\n")
    result = subprocess.run(cmd, cwd=str(_SRC))
    if result.returncode != 0:
        raise RuntimeError(f"{label} exited with code {result.returncode}")


def _read_total_epochs(ensemble_dir: Path) -> int | None:
    """Sum epochs across all curriculum stages from member_00's metrics JSON."""
    metrics_path = ensemble_dir / "member_00" / "baby_steps_trial_metrics.json"
    if not metrics_path.exists():
        return None
    try:
        data = json.loads(metrics_path.read_text())
        return sum(len(stage.get("train_loss") or []) for stage in data)
    except Exception:
        return None


def _count_completed_members(ensemble_dir: Path) -> int:
    """Count member dirs that have a final model checkpoint."""
    count = 0
    for member_dir in sorted(ensemble_dir.glob("member_*")):
        if (member_dir / FINAL_MODEL_NAME).exists():
            count += 1
    return count


def _member0_path(ensemble_dir: Path) -> Path:
    return ensemble_dir / "member_00" / FINAL_MODEL_NAME


def _manifest_path(ensemble_dir: Path) -> Path:
    return ensemble_dir / "ensemble_manifest.json"


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_train(variant: dict, args: argparse.Namespace, log: SafetyLog) -> None:
    key = variant["key"]
    ensemble_dir = _SRC / "runs" / f"ensemble_{key}"
    manifest = _manifest_path(ensemble_dir)

    if log.is_complete(key, "training"):
        print(f"[{key}] training already complete — skipping.")
        return

    # Detect partial completion for crash resume
    completed = _count_completed_members(ensemble_dir)
    start_member = completed  # 0-indexed: next member to run

    resume_hint = (
        f"--start_member {start_member} (members 0–{completed-1} already done)"
        if completed > 0 else ""
    )
    log.mark_started(key, "training", resume_hint)

    shared_cache = str(ensemble_dir / "shared_cells")
    train_flags = _apply_dry_overrides(COMMON_TRAIN_FLAGS) if getattr(args, "dry_run", False) else COMMON_TRAIN_FLAGS
    cmd = [
        sys.executable,
        "-m", "model.training.ensemble",
        *train_flags,
        *variant["extra_train_flags"],
        "--ensemble_size", str(args.ensemble_size),
        "--ensemble_output_dir", str(ensemble_dir),
        "--shared_cell_cache", shared_cache,
    ]
    if completed > 0:
        cmd += ["--start_member", str(start_member)]

    t0 = time.time()
    _run(cmd, f"TRAINING [{key}]")
    wall_seconds = time.time() - t0

    if not manifest.exists():
        log.mark_failed(key, "training", "ensemble_manifest.json not found after run")
        raise RuntimeError(f"Training appeared to finish but no manifest at {manifest}")

    log.mark_complete(key, "training",
                      wall_seconds=wall_seconds,
                      total_epochs=_read_total_epochs(ensemble_dir))


def step_collect_results(variant: dict, args: argparse.Namespace, log: SafetyLog) -> None:
    key = variant["key"]
    ensemble_dir = _SRC / "runs" / f"ensemble_{key}"
    dest = _SRC / args.results_dir / key

    if log.is_complete(key, "collect_results"):
        print(f"[{key}] collect_results already complete — skipping.")
        return

    log.mark_started(key, "collect_results")
    dest.mkdir(parents=True, exist_ok=True)

    # Copy manifest
    manifest = _manifest_path(ensemble_dir)
    if manifest.exists():
        shutil.copy2(manifest, dest / "ensemble_manifest.json")

    # Copy member 0 model
    model_path = _member0_path(ensemble_dir)
    if model_path.exists():
        models_dir = dest / "models"
        models_dir.mkdir(exist_ok=True)
        shutil.copy2(model_path, models_dir / FINAL_MODEL_NAME)

    # Copy all figures from the run dir
    figs_dir = dest / "figures"
    figs_dir.mkdir(exist_ok=True)
    for png in ensemble_dir.rglob("*.png"):
        shutil.copy2(png, figs_dir / png.name)

    print(f"[{key}] Results collected → {dest}")
    log.mark_complete(key, "collect_results")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ensemble-size", type=int, default=6)
    p.add_argument("--results-dir", default="big_results_new",
                   help="Root directory for collected results (relative to src/).")
    p.add_argument("--skip-training", action="store_true",
                   help="Skip training steps (useful if models already exist).")
    p.add_argument("--dry-run", action="store_true",
                   help="Smoke-test the full pipeline: 1 member, 2 epochs/stage, "
                        "2 dipole stages, 100 eval samples. Results in dry_run_results/.")
    return p.parse_args()


# Flags that --dry-run overrides in COMMON_TRAIN_FLAGS
_DRY_TRAIN_OVERRIDES: dict[str, str] = {
    "--max_dipoles": "2",
    "--max_epochs_per_stage": "2",
    "--stage_samples": "500",
    "--w_n_data": "500",
}


def _apply_dry_overrides(flags: list[str]) -> list[str]:
    """Return a copy of *flags* with dry-run values substituted in-place."""
    out = list(flags)
    for key, val in _DRY_TRAIN_OVERRIDES.items():
        try:
            idx = out.index(key)
            out[idx + 1] = val
        except ValueError:
            out += [key, val]
    return out


def main() -> None:
    args = parse_args()
    if args.dry_run:
        args.ensemble_size = 1
        if not args.results_dir or args.results_dir == "big_results":
            args.results_dir = "dry_run_results"
    results_root = _SRC / args.results_dir
    log = SafetyLog(results_root)
    log.print_status()

    for variant in VARIANTS:
        key = variant["key"]
        print(f"\n{'='*60}\nVariant: {variant['label']}\n{'='*60}")

        if not args.skip_training:
            step_train(variant, args, log)
        else:
            print(f"[{key}] --skip-training set, skipping training.")

        step_collect_results(variant, args, log)

    log.write_timing_summary()
    log.print_status()
    print(f"\nAll steps complete.  Results in: {results_root}")


if __name__ == "__main__":
    main()

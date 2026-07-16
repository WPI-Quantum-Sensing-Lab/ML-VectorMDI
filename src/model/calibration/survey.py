"""Magnitude calibration survey across all tested datasets.

For each dataset directory in a run manifest, subsample N examples, run the model,
collect (true_mag, pred_mag) pairs, and fit a least-squares slope through the origin
(pred = slope * true).  Calibration factor = 1 / slope.

The pooled slope (canonical single number) is computed only over rows that match the
--pool-dipoles / --pool-snr-min / --pool-snr-max filters, so you can derive a factor
from a specific subset (e.g. hard 10-dipole low-SNR data) while still seeing per-dataset
numbers for every condition.

Use --from-csv to skip inference entirely and recompute pooled factors from an existing
per-dataset CSV — instant, no GPU required.

Usage
-----
    cd src
    # Full survey (runs model on all datasets; settings in configs/calibrate.ini):
    python calibrate.py

    # Derive factor from 10-dipole, SNR <= 0.5 only (no re-inference):
    python calibrate.py --from-csv figures/calibration_survey/calibration_survey.csv \
        --pool-dipoles 10 --pool-snr-max 0.5
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model.calibration import predict
from model.calibration.predict import load_model

# ---------------------------------------------------------------------------
# Dataset-name parsing helpers (inlined from the evaluation tooling, which is
# not part of this release)
# ---------------------------------------------------------------------------

from typing import Any

DATASET_FILENAME = "dataset.h5"

SNR_PATTERN = re.compile(r"snr(?:lv)?[_-]?([0-9]+(?:[\.p][0-9]+)?)", re.IGNORECASE)
_DIPOLE_PATTERNS = (
    re.compile(r"(?P<value>\d+)[_-]?dipoles?", re.IGNORECASE),
    re.compile(r"dipoles?[_-]?(?P<value>\d+)", re.IGNORECASE),
    re.compile(r"dipole(?:count|num|number)?[_-]?(?P<value>\d+)", re.IGNORECASE),
    re.compile(r"multidipole[_-]?(?P<value>\d+)", re.IGNORECASE),
    re.compile(r"multipole[_-]?(?P<value>\d+)", re.IGNORECASE),
)



def _normalize_numeric_attr(value: Any) -> float | None:
    """Convert HDF5 attribute values into floats when possible."""

    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            pass
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        cleaned = value.strip().lower().replace("p", ".")
        try:
            return float(cleaned)
        except ValueError:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None



def _dataset_h5_path(dataset_path: Path) -> Path:
    """Return the concrete ``dataset.h5`` path for *dataset_path*."""

    if dataset_path.is_dir():
        return dataset_path / DATASET_FILENAME
    if dataset_path.is_file():
        return dataset_path
    raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")



def infer_dipole_count(dataset_directory: Path) -> int:
    """Infer the dipole count from the dataset path or HDF5 metadata."""

    directory_text = str(dataset_directory)
    for pattern in _DIPOLE_PATTERNS:
        match = pattern.search(directory_text)
        if match:
            return int(match.group("value"))

    h5_path = _dataset_h5_path(dataset_directory)
    if h5py is None:
        raise ValueError(
            f"Could not infer dipole count from {dataset_directory} and h5py is unavailable for metadata fallback."
        )
    with h5py.File(h5_path, "r") as handle:
        for attribute_name in ("dipoles", "dipoleLimit", "dipole_num", "dipole_number"):
            value = _normalize_numeric_attr(handle.attrs.get(attribute_name))
            if value is not None:
                return int(value)

    raise ValueError(f"Could not infer dipole count for dataset: {dataset_directory}")


def infer_snr(dataset_directory: Path) -> float:
    """Infer the SNR from the dataset path or HDF5 metadata."""

    match = SNR_PATTERN.search(str(dataset_directory))
    if match:
        return float(match.group(1).lower().replace("p", "."))

    parsed_by_predict = predict.extract_snr(dataset_directory)
    if parsed_by_predict is not None:
        return parsed_by_predict

    h5_path = _dataset_h5_path(dataset_directory)
    if h5py is None:
        raise ValueError(
            f"Could not infer SNR from {dataset_directory} and h5py is unavailable for metadata fallback."
        )
    with h5py.File(h5_path, "r") as handle:
        for attribute_name in ("snr", "snr_level"):
            value = _normalize_numeric_attr(handle.attrs.get(attribute_name))
            if value is not None:
                return value

    raise ValueError(f"Could not infer SNR for dataset: {dataset_directory}")



_DEFAULT_MANIFEST = _SRC / "big_results_new" / "no_physics" / "eval_error" / "run_manifest.json"
_DEFAULT_OUTPUT   = _SRC / "figures" / "calibration_survey" / "calibration_survey.csv"
CALIBRATION_SUMMARY_JSON     = _SRC / "figures" / "calibration_survey" / "calibration_summary.json"
CALIBRATION_TABLE_PER_SNR    = _SRC / "figures" / "calibration_survey" / "calibration_table_per_snr.json"
CALIBRATION_TABLE_PER_DIPOLE = _SRC / "figures" / "calibration_survey" / "calibration_table_per_dipole.json"
CALIBRATION_TABLE_ISOTONIC        = _SRC / "figures" / "calibration_survey" / "calibration_table_isotonic.json"
CALIBRATION_TABLE_ISOTONIC_GLOBAL = _SRC / "figures" / "calibration_survey" / "calibration_table_isotonic_global.json"



def _fit_slope_with_stats(
    true_mag: np.ndarray, pred_mag: np.ndarray
) -> tuple[float, float, float, float]:
    """Slope through origin with SE, R², and 95% CI half-width.

    Returns (slope, se, r2, ci95) where:
      se   = standard error of slope estimate
      r2   = coefficient of determination (standard, not forced-through-origin)
      ci95 = 1.96 * se  (half-width of 95% confidence interval)
    """
    n = len(true_mag)
    ss_xx = float(np.dot(true_mag, true_mag))
    if ss_xx <= 0 or n < 2:
        nan = float("nan")
        return nan, nan, nan, nan

    slope = float(np.dot(true_mag, pred_mag) / ss_xx)
    residuals = pred_mag - slope * true_mag
    ss_res = float(np.dot(residuals, residuals))
    sigma2 = ss_res / (n - 1)                          # unbiased variance of residuals
    se = float(np.sqrt(sigma2 / ss_xx))

    ss_tot = float(np.var(pred_mag) * n)               # sum of squared deviations from mean
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return slope, se, r2, 1.96 * se


@torch.no_grad()
def _run_model(model: torch.nn.Module, x_np: np.ndarray,
               batch_size: int, device: torch.device) -> np.ndarray:
    preds: list[np.ndarray] = []
    for start in range(0, x_np.shape[0], batch_size):
        batch = torch.from_numpy(x_np[start : start + batch_size]).to(device)
        if batch.ndim == 4 and batch.shape[-1] == 3:
            batch = batch.permute(0, 3, 1, 2)   # NHWC → NCHW
        out = model(batch)
        if isinstance(out, tuple):
            out = out[0]
        preds.append(out.cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def _subsample(total: int, n: int, rng: np.random.Generator) -> np.ndarray | None:
    if n >= total:
        return None
    return np.sort(rng.choice(total, size=n, replace=False))


def _magnitudes(cart: np.ndarray) -> np.ndarray:
    return np.linalg.norm(cart, axis=-1)


def survey(
    manifest_path: Path,
    *,
    samples_per_dataset: int,
    seed: int,
    batch_size: int,
    device: torch.device,
    scale_samples_by_dipoles: bool = False,
    r2_min: float = 0.85,
) -> tuple:
    with manifest_path.open() as f:
        manifest = json.load(f)

    model_path = Path(manifest["model_path"])
    model_type = manifest["model_type"]
    directories = [Path(d) for d in manifest["dataset_directories"]]

    print(f"Model : {model_path}")
    print(f"Type  : {model_type}")
    print(f"Dirs  : {len(directories)}")
    print(f"Device: {device}\n")

    model = load_model(model_path, device, model_type)
    rng   = np.random.default_rng(seed)

    # Global accumulators for pooled slope
    global_sum_tp = 0.0   # sum(true * pred)
    global_sum_tt = 0.0   # sum(true^2)

    _WARN_SE_THRESHOLD    = 0.02   # slope SE above this → unreliable estimate
    _WARN_R2_THRESHOLD    = 0.80   # R² below this → poor linear fit
    _WARN_SLOPE_LO        = 0.50   # slope below this → implausible (model broken?)
    _WARN_SLOPE_HI        = 1.50   # slope above this → implausible

    rows: list[dict] = []
    warnings: list[str] = []
    # Each bucket: (pred_mags, true_mags, r2_weights) — weight = max(0, r2) per sample
    iso_buckets_all: dict[int, tuple[list, list, list]] = {}
    iso_buckets_rel: dict[int, tuple[list, list, list]] = {}

    for i, directory in enumerate(directories):
        h5_path = directory / "dataset.h5"
        if not h5_path.exists():
            print(f"  [{i+1}/{len(directories)}] SKIP (no dataset.h5): {directory.name}")
            continue

        try:
            snr        = infer_snr(directory)
            dipoles    = infer_dipole_count(directory)
        except ValueError as exc:
            print(f"  [{i+1}/{len(directories)}] SKIP (metadata): {exc}")
            continue

        with h5py.File(h5_path, "r") as fh:
            total = fh["m"].shape[0]
            effective_n = samples_per_dataset * dipoles if scale_samples_by_dipoles else samples_per_dataset
            idx   = _subsample(total, effective_n, rng)
            if idx is None:
                m_true = fh["m"][:].astype(np.float32)
                x_np   = fh["I"][:].astype(np.float32)
            else:
                m_true = fh["m"][idx.tolist()].astype(np.float32)
                x_np   = fh["I"][idx.tolist()].astype(np.float32)

        n_used = len(m_true)
        capped = n_used < effective_n   # dataset smaller than requested
        pred_cart  = _run_model(model, x_np, batch_size, device)
        true_mag   = _magnitudes(m_true)
        pred_mag   = _magnitudes(pred_cart)

        slope, se, r2, ci95 = _fit_slope_with_stats(true_mag, pred_mag)
        calib  = 1.0 / slope if slope > 0 else float("nan")

        global_sum_tp += float(np.dot(true_mag, pred_mag))
        global_sum_tt += float(np.dot(true_mag, true_mag))

        w = max(0.0, r2) if not np.isnan(r2) else 0.0
        p_all, t_all, w_all = iso_buckets_all.setdefault(dipoles, ([], [], []))
        p_all.extend(pred_mag.tolist())
        t_all.extend(true_mag.tolist())
        w_all.extend([w] * len(pred_mag))
        if not np.isnan(r2) and r2 >= r2_min:
            p_rel, t_rel, w_rel = iso_buckets_rel.setdefault(dipoles, ([], [], []))
            p_rel.extend(pred_mag.tolist())
            t_rel.extend(true_mag.tolist())
            w_rel.extend([w] * len(pred_mag))

        # ── QA flags ──────────────────────────────────────────────────────
        flags: list[str] = []
        if capped:
            flags.append(f"CAPPED(only {n_used}/{effective_n} avail)")
        if se > _WARN_SE_THRESHOLD:
            flags.append(f"HIGH-SE({se:.4f})")
        if r2 < _WARN_R2_THRESHOLD:
            flags.append(f"LOW-R2({r2:.3f})")
        if slope < _WARN_SLOPE_LO or slope > _WARN_SLOPE_HI:
            flags.append(f"IMPLAUSIBLE-SLOPE({slope:.4f})")

        flag_str = "  ⚠ " + ", ".join(flags) if flags else ""
        if flags:
            warnings.append(f"  dipoles={dipoles:>2}  snr={snr:>5g}  {', '.join(flags)}  [{directory.name}]")

        rows.append({
            "directory"             : str(directory),
            "dipoles"               : dipoles,
            "snr"                   : snr,
            "n_samples"             : n_used,
            "mag_slope"             : slope,
            "slope_se"              : se,
            "slope_ci95"            : ci95,
            "r2"                    : r2,
            "capped"                : capped,
            "mag_calibration_factor": calib,
        })
        print(f"  [{i+1:>3}/{len(directories)}] dipoles={dipoles:>2}  snr={snr:>5g}"
              f"  n={n_used:>5}  slope={slope:.4f}±{ci95:.4f}  R²={r2:.3f}"
              f"  calib={calib:.4f}{flag_str}")

    if warnings:
        print(f"\n{'!'*70}")
        print(f"QA WARNINGS  ({len(warnings)} datasets flagged)")
        for w in warnings:
            print(w)
        print("!" * 70)

    # Merge: prefer R²-passing conditions; fall back to all samples when none pass
    iso_buckets: dict[int, tuple[list, list, list]] = {}
    for d in iso_buckets_all:
        iso_buckets[d] = iso_buckets_rel[d] if d in iso_buckets_rel else iso_buckets_all[d]

    df = pd.DataFrame(rows)
    return df, global_sum_tp, global_sum_tt, iso_buckets


def pooled_slope_from_df(
    df: pd.DataFrame,
    *,
    dipoles: list[int] | None = None,
    snr_min: float | None = None,
    snr_max: float | None = None,
    r2_min: float | None = None,
) -> tuple[float, float, int]:
    """Recompute pooled slope from per-dataset rows using per-dataset slopes as proxies.

    Since per-dataset slopes approximate pred = slope * true, the pooled slope across
    a subset is the sample-count-weighted mean slope (close enough for calibration).
    Returns (pooled_slope, calibration_factor, n_datasets_used).
    """
    mask = pd.Series([True] * len(df), index=df.index)
    if dipoles is not None:
        mask &= df["dipoles"].isin(dipoles)
    if snr_min is not None:
        mask &= df["snr"] >= snr_min
    if snr_max is not None:
        mask &= df["snr"] <= snr_max
    if r2_min is not None and "r2" in df.columns:
        mask &= df["r2"] >= r2_min
    subset = df[mask]
    if subset.empty:
        return float("nan"), float("nan"), 0
    # Weighted mean slope (weight by number of samples used per dataset)
    weights = subset["n_samples"].astype(float)
    pooled_slope = float((subset["mag_slope"] * weights).sum() / weights.sum())
    calib = 1.0 / pooled_slope if pooled_slope > 0 else float("nan")
    return pooled_slope, calib, int(len(subset))


def max_strictly_improving_scalar(df: pd.DataFrame) -> tuple[float, str]:
    """Find the largest c that strictly reduces |slope*c - 1| vs |slope - 1| for every dataset.

    For slope s < 1: strict improvement requires 1 < c < 2/s - 1.
    The binding constraint is the dataset with the highest slope (closest to 1.0),
    because 2/s - 1 is smallest there.

    Returns (c_max, binding_dataset_name) where c_max is the supremum — any c strictly
    less than c_max strictly improves every condition. At exactly c_max the binding
    dataset ties its raw residual (not worse, not better).

    If any slope >= 1.0, c > 1 worsens that dataset; returns (nan, reason).
    """
    slopes = df["mag_slope"].astype(float).values
    dirs   = df["directory"].astype(str).values if "directory" in df.columns else [str(i) for i in range(len(slopes))]

    over_one = [(s, d) for s, d in zip(slopes, dirs) if s >= 1.0]
    if over_one:
        return float("nan"), f"slope >= 1.0 in {len(over_one)} dataset(s) — c>1 worsens them"

    upper_bounds = [(2.0 / s - 1.0, d) for s, d in zip(slopes, dirs) if s > 0]
    if not upper_bounds:
        return float("nan"), "no valid slopes"

    c_max, binding = min(upper_bounds, key=lambda x: x[0])
    return c_max, Path(binding).name


def optimal_global_scalar(df: pd.DataFrame, *, r2_min: float | None = None) -> float:
    """Find c* minimizing sum_i n_i * (slope_i * c - 1)^2.

    Closed form: c* = sum(n_i * slope_i) / sum(n_i * slope_i^2)
    This is the single scalar that best corrects all datasets simultaneously,
    weighted by sample count.
    """
    subset = df
    if r2_min is not None and "r2" in df.columns:
        subset = df[df["r2"] >= r2_min]
    if subset.empty:
        return float("nan")
    n = subset["n_samples"].astype(float).values
    s = subset["mag_slope"].astype(float).values
    return float((n * s).sum() / (n * s ** 2).sum())


def build_per_snr_table(df: pd.DataFrame, fallback: float) -> dict:
    """Compute a per-SNR calibration table from per-dataset survey rows.

    Uses sample-count-weighted mean slope per SNR bin.  Returns a dict ready
    to be serialized as JSON and used for nearest-SNR lookup at inference time.
    """
    rows = []
    for snr, group in df.groupby("snr"):
        w = group["n_samples"].astype(float)
        slope = float((group["mag_slope"] * w).sum() / w.sum())
        calib = 1.0 / slope if slope > 0 else fallback
        rows.append({"snr": float(snr), "slope": slope, "calibration_factor": calib,
                     "n_datasets": int(len(group)), "n_samples": int(w.sum())})
    rows.sort(key=lambda r: r["snr"])
    return {"per_snr": rows, "fallback_global": fallback}


def build_per_dipole_table(df: pd.DataFrame, fallback: float, *, r2_min: float = 0.85) -> dict:
    """Per-dipole-count optimal calibration scalar (minimizes weighted squared residuals within each count).

    Only datasets with R² >= r2_min are used to fit the scalar; low-SNR datasets where
    the linear relationship is noise-dominated are excluded.  Defaults to r2_min=0.85.
    """
    rows = []
    for dipoles, group in df.groupby("dipoles"):
        reliable = group[group["r2"] >= r2_min] if "r2" in group.columns else group
        if reliable.empty:
            reliable = group   # fall back to all if nothing passes threshold
        n = reliable["n_samples"].astype(float)
        s = reliable["mag_slope"].astype(float)
        c_opt = float((n * s).sum() / (n * s ** 2).sum())
        rows.append({
            "dipoles": int(dipoles),
            "mean_slope": float((s * n).sum() / n.sum()),
            "calibration_factor": c_opt,
            "n_datasets_used": int(len(reliable)),
            "n_datasets_total": int(len(group)),
            "n_samples": int(n.sum()),
            "r2_min_used": r2_min,
        })
    rows.sort(key=lambda r: r["dipoles"])
    return {"per_dipole": rows, "fallback_global": fallback, "r2_min": r2_min}


def build_isotonic_table(
    iso_buckets: dict[int, tuple[list, list, list]],
    *,
    r2_min: float = 0.85,
) -> dict:
    """Fit one R²-weighted IsotonicRegression per dipole count.

    Each sample carries weight = max(0, r2_condition) so high-SNR conditions
    (where the slope estimate is reliable) dominate the fitted curve.
    Falls back to uniform weights when all condition R²s are ≤ 0.
    """
    from sklearn.isotonic import IsotonicRegression

    per_dipole: dict[str, dict] = {}
    for d in sorted(iso_buckets):
        preds, trues, weights = iso_buckets[d]
        if not preds:
            continue
        w = np.array(weights, dtype=np.float64)
        sample_weight = w if w.sum() > 0 else None  # uniform fallback
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        iso.fit(preds, trues, sample_weight=sample_weight)
        per_dipole[str(d)] = {
            "X_thresholds": iso.X_thresholds_.tolist(),
            "y_thresholds": iso.y_thresholds_.tolist(),
            "n_samples": len(preds),
        }
        print(f"  dipoles={d:2d}  isotonic n={len(preds):>6}  "
              f"X=[{min(preds):.3f}, {max(preds):.3f}]  w_sum={w.sum():.2f}")
    return {"per_dipole": per_dipole, "r2_min": r2_min}


def build_global_isotonic_table(
    iso_buckets: dict[int, tuple[list, list, list]],
) -> dict:
    """Fit a single R²-weighted isotonic regressor across ALL dipole counts.

    Intended for deployment when the dipole count is unknown. Pools all
    (pred_mag, true_mag) pairs from every condition; each pair is weighted by
    max(0, r2) of its source condition so high-SNR fits dominate the curve.
    Falls back to uniform weights when all condition R²s are ≤ 0.
    """
    from sklearn.isotonic import IsotonicRegression

    all_preds: list[float] = []
    all_trues: list[float] = []
    all_weights: list[float] = []
    for preds, trues, weights in iso_buckets.values():
        all_preds.extend(preds)
        all_trues.extend(trues)
        all_weights.extend(weights)

    if not all_preds:
        return {"global": None}

    w = np.array(all_weights, dtype=np.float64)
    sample_weight = w if w.sum() > 0 else None
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(all_preds, all_trues, sample_weight=sample_weight)
    print(f"  global   isotonic n={len(all_preds):>7}  "
          f"X=[{min(all_preds):.3f}, {max(all_preds):.3f}]  w_sum={w.sum():.2f}")
    return {
        "global": {
            "X_thresholds": iso.X_thresholds_.tolist(),
            "y_thresholds": iso.y_thresholds_.tolist(),
            "n_samples": len(all_preds),
        }
    }


def lookup_dipole_calibration(table: dict, dipoles: int) -> float:
    """Return the calibration factor for the nearest dipole count in *table*."""
    entries = table.get("per_dipole", [])
    if not entries:
        return float(table.get("fallback_global", 1.0))
    exact = [e for e in entries if e["dipoles"] == dipoles]
    if exact:
        return float(exact[0]["calibration_factor"])
    nearest = min(entries, key=lambda r: abs(r["dipoles"] - dipoles))
    return float(nearest["calibration_factor"])


def lookup_snr_calibration(table: dict, snr: float) -> float:
    """Return the calibration factor for the nearest SNR in *table*."""
    entries = table.get("per_snr", [])
    if not entries:
        return float(table.get("fallback_global", 1.0))
    nearest = min(entries, key=lambda r: abs(r["snr"] - snr))
    return float(nearest["calibration_factor"])


def print_summary(df: pd.DataFrame, global_sum_tp: float, global_sum_tt: float) -> None:
    global_slope = global_sum_tp / global_sum_tt if global_sum_tt > 0 else float("nan")
    global_calib = 1.0 / global_slope if global_slope > 0 else float("nan")

    print("\n" + "=" * 70)
    print("GLOBAL POOLED  (all datasets, all SNRs, all dipole counts)")
    print(f"  slope = {global_slope:.6f}   calibration factor = {global_calib:.6f}")

    print("\n" + "=" * 70)
    print("MEDIAN CALIBRATION FACTOR BY DIPOLE COUNT")
    by_dip = (
        df.groupby("dipoles")["mag_calibration_factor"]
        .agg(["median", "std", "count"])
        .reset_index()
        .sort_values("dipoles")
    )
    print(by_dip.to_string(index=False))

    print("\n" + "=" * 70)
    print("MEDIAN CALIBRATION FACTOR BY SNR")
    by_snr = (
        df.groupby("snr")["mag_calibration_factor"]
        .agg(["median", "std", "count"])
        .reset_index()
        .sort_values("snr")
    )
    print(by_snr.to_string(index=False))

    print("\n" + "=" * 70)
    overall_median = df["mag_calibration_factor"].median()
    overall_std    = df["mag_calibration_factor"].std()
    overall_min    = df["mag_calibration_factor"].min()
    overall_max    = df["mag_calibration_factor"].max()
    print("OVERALL ACROSS ALL CONDITIONS")
    print(f"  median = {overall_median:.6f}   std = {overall_std:.6f}"
          f"   min = {overall_min:.6f}   max = {overall_max:.6f}")
    print(f"\n  Recommendation: use global pooled factor = {global_calib:.6f}")
    print("  (pools all pairs before fitting — gives more weight to large-magnitude samples)")
    print("=" * 70)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST,
                   help="Path to run_manifest.json from a prior eval run.")
    p.add_argument("--samples", type=int, default=200,
                   help="Max samples per dataset (default: 200).")
    p.add_argument("--scale-samples-by-dipoles", action="store_true",
                   help="Multiply --samples by the dipole count for each dataset. "
                        "Compensates for higher variance in multi-dipole magnitude predictions "
                        "(e.g. --samples 200 gives 200 for 1-dipole, 2000 for 10-dipole).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT,
                   help="Where to write the per-dataset CSV.")
    p.add_argument("--device", default=None)
    p.add_argument("--from-csv", type=Path, default=None, metavar="CSV",
                   help="Skip inference; recompute pooled slope from an existing per-dataset CSV.")
    p.add_argument("--pool-dipoles", type=int, nargs="+", default=None, metavar="N",
                   help="Only include these dipole counts in the pooled slope (e.g. --pool-dipoles 10).")
    p.add_argument("--pool-snr-min", type=float, default=None,
                   help="Minimum SNR (inclusive) for pooled slope.")
    p.add_argument("--pool-snr-max", type=float, default=None,
                   help="Maximum SNR (inclusive) for pooled slope.")
    p.add_argument("--pool-r2-min", type=float, default=0.85,
                   help="Minimum R² for a dataset to be included in calibration fits (default: 0.85). "
                        "Excludes noisy low-SNR conditions where the slope estimate is unreliable.")
    args = p.parse_args()

    # ── Load or run survey ────────────────────────────────────────────────
    if args.from_csv is not None:
        print(f"Loading existing per-dataset CSV: {args.from_csv}")
        df = pd.read_csv(args.from_csv)
        global_sum_tp = global_sum_tt = None  # not available from CSV
    else:
        device = torch.device(
            args.device if args.device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        df, global_sum_tp, global_sum_tt, iso_buckets = survey(
            args.manifest,
            samples_per_dataset=args.samples,
            seed=args.seed,
            batch_size=args.batch_size,
            device=device,
            scale_samples_by_dipoles=args.scale_samples_by_dipoles,
            r2_min=args.pool_r2_min,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\nPer-dataset results saved to: {args.output}")

    # ── Pooled slope (optionally filtered) ───────────────────────────────
    has_filter = args.pool_dipoles or args.pool_snr_min or args.pool_snr_max
    pooled_slope, pooled_calib, n_pooled = pooled_slope_from_df(
        df,
        dipoles=args.pool_dipoles,
        snr_min=args.pool_snr_min,
        snr_max=args.pool_snr_max,
        r2_min=args.pool_r2_min,
    )

    filter_desc = "all datasets"
    if has_filter:
        parts = []
        if args.pool_dipoles:
            parts.append(f"dipoles={args.pool_dipoles}")
        if args.pool_snr_min is not None:
            parts.append(f"snr>={args.pool_snr_min}")
        if args.pool_snr_max is not None:
            parts.append(f"snr<={args.pool_snr_max}")
        filter_desc = ", ".join(parts)

    optimal_c = optimal_global_scalar(df, r2_min=args.pool_r2_min)
    c_max, binding = max_strictly_improving_scalar(df)

    print(f"\n{'='*70}")
    print(f"OPTIMAL GLOBAL SCALAR  (minimizes weighted sum of squared residuals, all {len(df)} datasets)")
    print(f"  c* = {optimal_c:.6f}  (c* = sum(n*slope) / sum(n*slope^2))")

    print(f"\n{'='*70}")
    print(f"MAX STRICTLY-IMPROVING SCALAR  (any c strictly below this improves EVERY condition)")
    if not (c_max != c_max):   # not NaN
        print(f"  c_max = {c_max:.6f}  (binding dataset: {binding})")
        print(f"  Safe range for guaranteed improvement in all conditions: 1.0 < c < {c_max:.6f}")
    else:
        print(f"  c_max = N/A  ({binding})")

    print(f"\n{'='*70}")
    print(f"POOLED SLOPE  ({filter_desc}, n={n_pooled} datasets)")
    print(f"  slope = {pooled_slope:.6f}   calibration factor = {pooled_calib:.6f}")

    # ── Cross-validation: per-dataset (dipole, SNR) breakdown ────────────
    _GRN = "\033[32m"
    _RED = "\033[31m"
    _RST = "\033[0m"

    def _color_res(res: float, raw_res: float, width: int = 10) -> str:
        s = f"{res:{width}.4f}"
        if res < raw_res - 1e-6:
            return f"{_GRN}{s}{_RST}"
        if res > raw_res + 1e-6:
            return f"{_RED}{s}{_RST}"
        return s

    per_dipole_table = build_per_dipole_table(df, fallback=optimal_c, r2_min=args.pool_r2_min)
    dipole_to_calib = {r["dipoles"]: r["calibration_factor"] for r in per_dipole_table["per_dipole"]}

    display = (
        df[["dipoles", "snr", "mag_slope", "n_samples"]]
        .copy()
        .sort_values(["dipoles", "snr"])
        .reset_index(drop=True)
    )
    display["raw_res"]        = (display["mag_slope"] - 1).abs()
    display["raw_pct_err"]    = (display["mag_slope"] - 1) * 100
    display["res_optimal"]    = (display["mag_slope"] * optimal_c - 1).abs()
    display["delta_pct_opt"]  = (display["raw_res"] - display["res_optimal"]) * 100
    display["calib_dip"]      = display["dipoles"].map(dipole_to_calib)
    display["res_dip"]        = (display["mag_slope"] * display["calib_dip"] - 1).abs()
    display["delta_pct_dip"]  = (display["raw_res"] - display["res_dip"]) * 100

    def _color_delta(delta: float, width: int = 8) -> str:
        s = f"{delta:+{width}.2f}%"
        if delta > 1e-4:
            return f"{_GRN}{s}{_RST}"
        if delta < -1e-4:
            return f"{_RED}{s}{_RST}"
        return s

    print(f"\n{'='*100}")
    print(f"PER-CONDITION PERFORMANCE  (optimal c*={optimal_c:.4f}  vs  per-dipole-count scalar)")
    print(f"  {_GRN}green = better than uncalibrated{_RST}  {_RED}red = worse than uncalibrated{_RST}")
    hdr = (f"  {'dip':>4}  {'SNR':>6}  {'slope':>7}  {'raw%err':>8}"
           f"  {'res(c*)':>8}  {'Δ%(c*)':>9}"
           f"  {'c(dip)':>7}  {'res(dip)':>9}  {'Δ%(dip)':>9}")
    print(hdr)
    print(f"  {'-'*4}  {'-'*6}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*9}  {'-'*9}")
    prev_dip = None
    for _, row in display.iterrows():
        dip = int(row["dipoles"])
        if prev_dip is not None and dip != prev_dip:
            print()
        raw_res = float(row["raw_res"])
        print(f"  {dip:>4}  {row['snr']:>6g}  {row['mag_slope']:>7.4f}  {row['raw_pct_err']:>+7.2f}%"
              f"  {_color_res(row['res_optimal'], raw_res, width=8)}  {_color_delta(row['delta_pct_opt'])}"
              f"  {row['calib_dip']:>7.4f}  {_color_res(row['res_dip'], raw_res, width=9)}  {_color_delta(row['delta_pct_dip'])}")
        prev_dip = dip

    # ── Save summary JSON ─────────────────────────────────────────────────
    summary = {
        "global_calibration_factor"       : optimal_c,
        "optimal_scalar_c"                : optimal_c,
        "max_strictly_improving_scalar"   : c_max if c_max == c_max else None,
        "max_strictly_improving_binding"  : binding,
        "pooled_calib_filtered"           : pooled_calib,
        "pool_filter"                     : filter_desc,
        "n_datasets_pooled"               : n_pooled,
        "median_calibration_factor"       : float(df["mag_calibration_factor"].median()),
        "std_calibration_factor"          : float(df["mag_calibration_factor"].std()),
        "n_datasets_total"                : len(df),
    }
    CALIBRATION_SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\nCalibration summary saved to: {CALIBRATION_SUMMARY_JSON}")

    per_dipole_table_out = build_per_dipole_table(df, fallback=optimal_c)
    CALIBRATION_TABLE_PER_DIPOLE.write_text(json.dumps(per_dipole_table_out, indent=2))
    print(f"Per-dipole table saved to:    {CALIBRATION_TABLE_PER_DIPOLE}")
    print(f"\nPer-dipole calibration factors  (R²≥{args.pool_r2_min} filter applied):")
    print(f"  {'dipoles':>7}  {'slope':>8}  {'calib':>8}  {'used/total':>10}  {'samples':>8}")
    for row in per_dipole_table_out["per_dipole"]:
        used, total = row["n_datasets_used"], row["n_datasets_total"]
        flag = "  ⚠ all excluded, used all" if used == total and row.get("r2_min_used", 0) > 0 and used < total else ""
        print(f"  {row['dipoles']:>7}  {row['mean_slope']:>8.4f}  {row['calibration_factor']:>8.4f}"
              f"  {used:>4}/{total:<5}  {row['n_samples']:>8}{flag}")

    per_snr_table = build_per_snr_table(df, fallback=pooled_calib)
    CALIBRATION_TABLE_PER_SNR.write_text(json.dumps(per_snr_table, indent=2))
    print(f"Per-SNR table saved to:       {CALIBRATION_TABLE_PER_SNR}")

    if args.from_csv is None:
        print(f"\nFitting per-dipole isotonic regressors (R²-weighted, R²≥{args.pool_r2_min} filter):")
        iso_table = build_isotonic_table(iso_buckets, r2_min=args.pool_r2_min)
        CALIBRATION_TABLE_ISOTONIC.write_text(json.dumps(iso_table, indent=2))
        print(f"Isotonic table saved to:      {CALIBRATION_TABLE_ISOTONIC}")

        print(f"\nFitting global isotonic regressor (R²-weighted, all dipole counts pooled):")
        iso_global_table = build_global_isotonic_table(iso_buckets)
        CALIBRATION_TABLE_ISOTONIC_GLOBAL.write_text(json.dumps(iso_global_table, indent=2))
        print(f"Global isotonic table saved:  {CALIBRATION_TABLE_ISOTONIC_GLOBAL}")
    else:
        print("\nSkipping isotonic fit (--from-csv mode, no raw samples available).")

    if global_sum_tp is not None:
        print_summary(df, global_sum_tp, global_sum_tt)


if __name__ == "__main__":
    main()

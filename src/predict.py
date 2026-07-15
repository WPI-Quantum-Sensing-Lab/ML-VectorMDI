"""Task 5 — Simple single prediction with a trained model or ensemble.

Loads one field map (index ``index`` of a ``dataset.h5``), runs the model,
and prints the predicted dipole moment — plus the ensemble uncertainty when a
manifest is used, and the calibrated magnitude when a calibration table is
configured.

Settings come from ``configs/predict.ini``; pass ``--config PATH`` to use
another file.  Any CLI flag overrides its config value.  Run from ``src/``:

    python predict.py
    python predict.py --input path/to/dataset.h5 --index 3
"""
from __future__ import annotations

import argparse
import configparser
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from model.architecture import AttentionDipoleModel
from model.training.ensemble import EnsemblePredictor

_SRC = Path(__file__).resolve().parent
DEFAULT_CONFIG = _SRC.parent / "configs" / "predict.ini"


def _read_config(path: Path) -> dict:
    parser = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    if not path.exists():
        raise SystemExit(f"Config file not found: {path}")
    parser.read(path)
    if "predict" not in parser:
        raise SystemExit(f"Config file {path} has no [predict] section")
    return {k: v.strip() for k, v in parser["predict"].items() if v.strip()}


def _parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    pre_args, remaining = pre.parse_known_args()
    cfg = _read_config(pre_args.config)

    p = argparse.ArgumentParser(description=__doc__, parents=[pre])
    p.add_argument("--manifest", type=str, default=cfg.get("manifest"),
                   help="Ensemble manifest JSON (predicts with all members + uncertainty).")
    p.add_argument("--checkpoint", type=str, default=cfg.get("checkpoint"),
                   help="Single model state-dict .pth (used when no manifest is given).")
    p.add_argument("--input", type=str, default=cfg.get("input"),
                   help="dataset.h5 file (or its folder) holding field maps under 'I'.")
    p.add_argument("--index", type=int, default=int(cfg.get("index", 0)),
                   help="Sample index inside the dataset (default 0).")
    p.add_argument("--device", type=str, default=cfg.get("device", "cpu"))
    p.add_argument("--base_channels", type=int, default=int(cfg.get("base_channels", 32)))
    p.add_argument("--calibration_table", type=str, default=cfg.get("calibration_table"),
                   help="Optional isotonic calibration table JSON (global table).")
    return p.parse_args(remaining)


def _resolve_member_path(raw: str, manifest_path: Path) -> Path:
    """Manifest checkpoint paths are relative to the repository root."""
    p = Path(raw)
    if p.is_absolute():
        return p
    for base in (manifest_path.parent.parent, manifest_path.parent, Path.cwd()):
        candidate = base / p
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Checkpoint from manifest not found: {raw}")


def _load_predictor(args: argparse.Namespace, device: torch.device):
    if args.manifest:
        manifest_path = Path(args.manifest).resolve()
        manifest = json.loads(manifest_path.read_text())
        paths = [
            _resolve_member_path(p, manifest_path)
            for p in manifest["member_final_checkpoints"]
        ]
        return EnsemblePredictor.from_checkpoints(
            paths, base_channels=args.base_channels, device=device
        ), True
    if args.checkpoint:
        m = AttentionDipoleModel(base_channels=args.base_channels).to(device)
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        m.load_state_dict(state)
        m.eval()
        return m, False
    raise SystemExit("Set `manifest` or `checkpoint` in the config (or pass the flag).")


def _load_field(input_path: str, index: int) -> tuple[torch.Tensor, np.ndarray | None]:
    p = Path(input_path)
    if p.is_dir():
        p = p / "dataset.h5"
    if not p.exists():
        raise SystemExit(f"Input dataset not found: {p}")
    with h5py.File(p, "r") as f:
        x = np.asarray(f["I"][index], dtype=np.float32)  # (H, W, 3)
        true_m = np.asarray(f["m"][index]) if "m" in f else None
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0), true_m


def _calibrated_magnitude(magnitude: float, table_path: str) -> float:
    table = json.loads(Path(table_path).read_text())
    entry = table.get("global", table)
    return float(np.interp(magnitude, entry["X_thresholds"], entry["y_thresholds"]))


def main() -> None:
    args = _parse_args()
    if not args.input:
        raise SystemExit("Set `input` in the config (or pass --input) to a dataset.h5.")
    device = torch.device(args.device)

    predictor, is_ensemble = _load_predictor(args, device)
    x, true_m = _load_field(args.input, args.index)
    x = x.to(device)

    with torch.no_grad():
        if is_ensemble:
            mean, std = predictor(x)
            moment = mean[0].cpu().numpy().astype(np.float64)
            uncertainty = std[0].cpu().numpy().astype(np.float64)
        else:
            # AttentionDipoleModel.forward returns (moment, attention, features)
            moment = predictor(x)[0][0].cpu().numpy().astype(np.float64)
            uncertainty = None

    magnitude = float(np.linalg.norm(moment))
    print(f"input      : {args.input} [index {args.index}]")
    print(f"moment     : [{moment[0]: .6e}, {moment[1]: .6e}, {moment[2]: .6e}]")
    print(f"magnitude  : {magnitude:.6e}")
    if uncertainty is not None:
        print(f"std        : [{uncertainty[0]: .6e}, {uncertainty[1]: .6e}, {uncertainty[2]: .6e}]")
        print(f"|std|      : {float(np.linalg.norm(uncertainty)):.6e}")
    if args.calibration_table:
        print(f"calibrated : {_calibrated_magnitude(magnitude, args.calibration_table):.6e}"
              f"  (magnitude via {Path(args.calibration_table).name})")
    if true_m is not None:
        true_mag = float(np.linalg.norm(true_m))
        print(f"true moment: [{true_m[0]: .6e}, {true_m[1]: .6e}, {true_m[2]: .6e}]"
              f"  (|m| = {true_mag:.6e})")


if __name__ == "__main__":
    main()

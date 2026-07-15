"""Task 3 — Train a single curriculum model.

Settings come from ``configs/train_model.ini``; pass ``--config PATH`` to use
another file.  Any extra CLI flags override the config.  Run from ``src/``:

    python train_model.py
    python train_model.py --max_epochs_per_stage 5   # quick smoke test
"""
import runpy
import sys

from _config import build_argv

if __name__ == "__main__":
    sys.argv = build_argv("train_model")
    runpy.run_module("model.training.curriculum", run_name="__main__")

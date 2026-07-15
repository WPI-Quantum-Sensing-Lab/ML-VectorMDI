"""Task 4 — Train a deep ensemble of curriculum models (one per seed).

Settings come from ``configs/train_ensemble.ini``; pass ``--config PATH`` to
use another file.  Any extra CLI flags override the config.  Run from ``src/``:

    python train_ensemble.py

To reproduce the published experiment end-to-end (with crash-safe resume and
result collection), use ``python run_full_experiment.py`` instead.
"""
import runpy
import sys

from _config import build_argv

if __name__ == "__main__":
    sys.argv = build_argv("train_ensemble")
    runpy.run_module("model.training.ensemble", run_name="__main__")

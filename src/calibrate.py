"""Task 3.5 — Fit the isotonic magnitude calibration from model predictions.

Runs the trained model over the datasets listed in a run manifest, pools
(true, predicted) magnitudes, and writes the isotonic calibration tables.

Settings come from ``configs/calibrate.ini``; pass ``--config PATH`` to use
another file.  Any extra CLI flags override the config.  Run from ``src/``:

    python calibrate.py
"""
import runpy
import sys

from _config import build_argv

if __name__ == "__main__":
    sys.argv = build_argv("calibrate")
    runpy.run_module("model.calibration.survey", run_name="__main__")

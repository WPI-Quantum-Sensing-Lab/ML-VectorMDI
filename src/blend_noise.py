"""Task 2 — Blend noiseless source maps with experimental blank noise at set SNRs.

Settings come from ``configs/blend_noise.ini`` ([snr_sort] section);
pass ``--config PATH`` to use another file.  Run from ``src/``:

    python blend_noise.py
"""
import runpy

if __name__ == "__main__":
    runpy.run_module("data_creation.snr.snr_blend_and_sort", run_name="__main__")

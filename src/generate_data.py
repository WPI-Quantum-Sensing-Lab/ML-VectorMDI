"""Task 1 — Generate synthetic multi-dipole source maps.

Settings come from ``configs/generate_data.ini`` ([map_generation] section);
pass ``--config PATH`` to use another file.  Run from ``src/``:

    python generate_data.py

For the exact multi-dataset sweep used in the publication (one dataset per
dipole count with distance constraints), see
``data_creation/generation/multi_source_maker.py``.
"""
import runpy

if __name__ == "__main__":
    runpy.run_module("data_creation.generation.magnetic_moment_gen_mp", run_name="__main__")

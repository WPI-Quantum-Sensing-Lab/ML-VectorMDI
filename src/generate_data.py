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
    # alter_sys=True points sys.modules["__main__"] at the module being run for the
    # duration of execution. Without it (the runpy default), functions defined in the
    # target module get __module__ == "__main__" while sys.modules["__main__"] still
    # refers to this script, so multiprocessing's spawn workers fail to look up the
    # pool initializer ("attribute lookup _pool_worker_seed_initializer on __main__ failed").
    runpy.run_module(
        "data_creation.generation.magnetic_moment_gen_mp",
        run_name="__main__",
        alter_sys=True,
    )

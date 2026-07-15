"""Compatibility helpers for working with VecMagNet HDF5 datasets.

The original :mod:`dataset_conversion` module shipped a small ``H5Dataset``
wrapper that many scripts in this repository import.  The file was removed
during a recent cleanup which left those imports failing at runtime.  This
module reintroduces a lightweight implementation that covers the behaviour
used throughout the codebase so downstream scripts keep functioning without
modification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import h5py

PathLike = Union[str, Path]


class H5Dataset:
    """Thin convenience wrapper around :class:`h5py.File`.

    Parameters
    ----------
    path:
        Location of the ``.h5`` file on disk.
    mode:
        File open mode compatible with :func:`h5py.File`.  Defaults to ``"r"``.

    Notes
    -----
    Only the subset of the original API that is still in active use is
    implemented here: context-manager support, ``len()`` and ``get_data``.  The
    wrapper purposely stays minimal and simply exposes the underlying
    :class:`h5py.File` handle through the ``hf`` attribute for legacy callers.
    """

    def __init__(self, path: PathLike, mode: str = "r") -> None:
        self.path = Path(path)
        self.mode = mode
        self.hf = h5py.File(self.path, mode)

    def __enter__(self) -> "H5Dataset":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> Optional[bool]:
        self.close()
        return None

    def __len__(self) -> int:
        """Return the number of entries contained in the first dataset.

        Many of our HDF5 files store images under the ``"I"`` key, but the exact
        layout varies between scripts.  To stay flexible the function looks for
        a dataset with a ``len()`` first and falls back to zero when the file is
        empty.
        """

        for key in ("I", "images", "data"):
            if key in self.hf:
                return len(self.hf[key])
        for dataset in self.hf.values():
            # ``len`` raises ``TypeError`` for scalar datasets.
            try:
                return len(dataset)
            except TypeError:
                continue
        return 0

    def get_data(self, key: str, index: Optional[int] = None):
        """Retrieve data stored under *key*.

        When *index* is provided the call returns a single element from the
        dataset; otherwise the underlying :class:`h5py.Dataset` object is
        returned so callers can slice without copying.
        """

        if key not in self.hf:
            raise KeyError(f"Dataset '{key}' not found in {self.path}")

        dataset = self.hf[key]
        if index is None:
            return dataset
        return dataset[index]

    def close(self) -> None:
        """Close the underlying HDF5 handle if it is still open."""

        if getattr(self.hf, "id", None) is not None and self.hf.id:
            self.hf.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

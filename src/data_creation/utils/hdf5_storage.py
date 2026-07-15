"""Shared HDF5 storage defaults for VecMagNet datasets."""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np


DEFAULT_HDF5_COMPRESSION = "gzip"
DEFAULT_HDF5_COMPRESSION_LEVEL = 4
DEFAULT_HDF5_SHUFFLE = True
DEFAULT_IMAGE_DTYPE = np.float32
DEFAULT_VECTOR_DTYPE = np.float32
DEFAULT_IMAGE_CHUNKS = (16, 100, 100, 3)
DEFAULT_VECTOR_CHUNKS = (2048, 3)
DEFAULT_HASH_CHUNKS = (4096,)


def normalize_compression(compression: str | None) -> str | None:
    """Return an HDF5-ready compression value."""

    if compression is None:
        return None
    normalized = str(compression).strip().lower()
    if normalized in {"", "none", "false", "off"}:
        return None
    return normalized


def clamp_chunk_shape(
    shape: tuple[int, ...],
    preferred_chunks: tuple[int, ...] | None,
) -> tuple[int, ...] | None:
    """Clamp chunk dimensions so they fit the dataset shape when needed."""

    if preferred_chunks is None:
        return None

    resolved: list[int] = []
    for dim, preferred in zip(shape, preferred_chunks):
        resolved.append(preferred if dim == 0 else min(int(dim), preferred))
    return tuple(resolved)


def build_hdf5_dataset_kwargs(
    *,
    shape: tuple[int, ...] | None = None,
    maxshape: tuple[int | None, ...] | None = None,
    chunks: tuple[int, ...] | None = None,
    compression: str | None = DEFAULT_HDF5_COMPRESSION,
    compression_opts: int | None = DEFAULT_HDF5_COMPRESSION_LEVEL,
    shuffle: bool = DEFAULT_HDF5_SHUFFLE,
) -> dict[str, Any]:
    """Build consistent create_dataset keyword arguments."""

    normalized_compression = normalize_compression(compression)
    kwargs: dict[str, Any] = {}

    if maxshape is not None:
        kwargs["maxshape"] = maxshape

    if chunks is not None:
        kwargs["chunks"] = clamp_chunk_shape(shape, chunks) if shape is not None else chunks
    elif normalized_compression is not None:
        kwargs["chunks"] = True

    if normalized_compression is not None:
        kwargs["compression"] = normalized_compression
        if normalized_compression == "gzip" and compression_opts is not None:
            kwargs["compression_opts"] = compression_opts
        if shuffle:
            kwargs["shuffle"] = True

    return kwargs


def image_dataset_kwargs(
    *,
    shape: tuple[int, ...] | None = None,
    maxshape: tuple[int | None, ...] | None = None,
    compression: str | None = DEFAULT_HDF5_COMPRESSION,
    compression_opts: int | None = DEFAULT_HDF5_COMPRESSION_LEVEL,
) -> dict[str, Any]:
    """Return dataset kwargs for VecMagNet image tensors."""

    chunks = DEFAULT_IMAGE_CHUNKS if shape is None else (DEFAULT_IMAGE_CHUNKS[0],) + tuple(shape[1:])
    return build_hdf5_dataset_kwargs(
        shape=shape,
        maxshape=maxshape,
        chunks=chunks,
        compression=compression,
        compression_opts=compression_opts,
    )


def vector_dataset_kwargs(
    *,
    shape: tuple[int, ...] | None = None,
    maxshape: tuple[int | None, ...] | None = None,
    compression: str | None = DEFAULT_HDF5_COMPRESSION,
    compression_opts: int | None = DEFAULT_HDF5_COMPRESSION_LEVEL,
) -> dict[str, Any]:
    """Return dataset kwargs for ``m`` and ``r`` vectors."""

    return build_hdf5_dataset_kwargs(
        shape=shape,
        maxshape=maxshape,
        chunks=DEFAULT_VECTOR_CHUNKS,
        compression=compression,
        compression_opts=compression_opts,
    )


def hash_dataset_kwargs(
    *,
    shape: tuple[int, ...] | None = None,
    maxshape: tuple[int | None, ...] | None = None,
    compression: str | None = DEFAULT_HDF5_COMPRESSION,
    compression_opts: int | None = DEFAULT_HDF5_COMPRESSION_LEVEL,
) -> dict[str, Any]:
    """Return dataset kwargs for string hash datasets."""

    return build_hdf5_dataset_kwargs(
        shape=shape,
        maxshape=maxshape,
        chunks=DEFAULT_HASH_CHUNKS,
        compression=compression,
        compression_opts=compression_opts,
    )


def create_vecmagnet_datasets(
    h5_file: h5py.File,
    *,
    image_shape: tuple[int, ...],
    vector_shape: tuple[int, ...] = (0, 3),
    image_dtype: Any = DEFAULT_IMAGE_DTYPE,
    vector_dtype: Any = DEFAULT_VECTOR_DTYPE,
    include_hash: bool = True,
    compression: str | None = DEFAULT_HDF5_COMPRESSION,
    compression_opts: int | None = DEFAULT_HDF5_COMPRESSION_LEVEL,
) -> None:
    """Create the standard ``I``, ``m``, ``r`` and optional ``hash`` datasets."""

    h5_file.create_dataset(
        "I",
        image_shape,
        dtype=image_dtype,
        **image_dataset_kwargs(
            shape=image_shape,
            maxshape=(None,) + tuple(image_shape[1:]),
            compression=compression,
            compression_opts=compression_opts,
        ),
    )
    h5_file.create_dataset(
        "m",
        vector_shape,
        dtype=vector_dtype,
        **vector_dataset_kwargs(
            shape=vector_shape,
            maxshape=(None, vector_shape[1]),
            compression=compression,
            compression_opts=compression_opts,
        ),
    )
    h5_file.create_dataset(
        "r",
        vector_shape,
        dtype=vector_dtype,
        **vector_dataset_kwargs(
            shape=vector_shape,
            maxshape=(None, vector_shape[1]),
            compression=compression,
            compression_opts=compression_opts,
        ),
    )
    if include_hash:
        hash_shape = (image_shape[0],)
        h5_file.create_dataset(
            "hash",
            hash_shape,
            dtype=h5py.string_dtype(encoding="utf-8"),
            **hash_dataset_kwargs(
                shape=hash_shape,
                maxshape=(None,),
                compression=compression,
                compression_opts=compression_opts,
            ),
        )

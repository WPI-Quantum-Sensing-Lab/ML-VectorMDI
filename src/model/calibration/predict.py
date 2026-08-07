"""Utility script for running inference over evaluation datasets.

The original version of this script was tightly coupled to one developer's
Windows directory layout.  This refactor focuses on making it portable,
configurable, and a little bit easier to maintain without changing the
underlying workflow: load a trained vector model, run predictions for each
`dataset.h5` inside the supplied glob, compute summary statistics, and write
both the per-sample errors and a CSV recap. Per-dataset predictions default to
``dataset_answers.h5`` but can be suffixed via ``--prediction-suffix`` to avoid
overwriting prior runs.
"""

from __future__ import annotations

import argparse
import glob
import logging
import math
import pickle
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

try:
    import alive_progress
except ModuleNotFoundError:
    alive_progress = None
try:
    import h5py
except ModuleNotFoundError:
    h5py = None

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None
try:
    from scipy.ndimage import gaussian_filter
except ModuleNotFoundError:
    gaussian_filter = None

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from model.architecture import (
        AttentionDipoleModel,
        declination_error,
        inclination_error,
        mag_p_error,
        make_measurement_grid,
        physics_refine,
    )
except (ModuleNotFoundError, ImportError):
    AttentionDipoleModel = None
    declination_error = None
    inclination_error = None
    mag_p_error = None
    make_measurement_grid = None
    physics_refine = None

# The legacy architectures (SimpleCNN family, equivariant ResNet) are not part
# of this release; the published model is AttentionDipoleModel ("dipolenet").
EquivariantHeadedResNet = None
SimpleCNN = None
SimpleCNNNoBN = None
SimpleCNNStandardBN = None
try:
    from augment_dataset_rotations import (
        DEFAULT_GRID_MAX,
        DEFAULT_GRID_MIN,
        DEFAULT_MU0_OVER_4PI,
        build_grid_points,
        recompute_field_from_m,
    )
except (ModuleNotFoundError, ImportError):
    DEFAULT_GRID_MAX = None
    DEFAULT_GRID_MIN = None
    DEFAULT_MU0_OVER_4PI = None
    build_grid_points = None
    recompute_field_from_m = None

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = REPO_ROOT / "src" / "pth" / "primary_model.pth"
DEFAULT_DATASET_PATTERN = str(REPO_ROOT / "data" / "datasets" / "testing_on_1_dipole_SNRlv*")
DATASET_FILENAME = "dataset.h5"
PREDICTION_FILENAME = "dataset_answers.h5"
HDF5_DATASET_NAMES = {"magnitude_error": "m_p_err", "inclination_error": "inc_err", "declination_error": "dec_err"}
LOGGED_PREDICTION_LIMIT = 10
DEFAULT_FIELD_SCALING_MODE = "none"
FIELD_SCALING_MODES = (
    DEFAULT_FIELD_SCALING_MODE,
    "gaussian_vector_percentile",
    "gradient_iqr",
    "rms",
)


@dataclass(frozen=True)
class FieldScalingConfig:
    """Inference-time field-scaling configuration.

    Modes
    -----
    gaussian_vector_percentile
        Original mode.  Gaussian-smooth the field, then use the
        (upper_percentile - lower_percentile) range of the vector-magnitude
        map as the amplitude estimate.
    gradient_iqr
        Option A.  Gaussian-smooth the field, compute the spatial gradient
        of the vector-magnitude map, and use the IQR (Q75 - Q25) of that
        gradient map.  Less sensitive to the number of dipoles because
        gradients are driven by the sharpest local feature rather than
        total field power.
    rms
        Option B.  Gaussian-smooth the field, then use the RMS of the
        vector-magnitude map.  Principled noise-floor correction via
        quadrature subtraction when *noise_floor_subtraction* is True.

    noise_floor_subtraction
        When True, estimate the noise floor from the outermost two rows/
        columns of the field and subtract it from the measured amplitude
        before computing the scale ratio.  Applied linearly for
        ``gaussian_vector_percentile`` and ``gradient_iqr``, in quadrature
        for ``rms``.
    """

    mode: str = DEFAULT_FIELD_SCALING_MODE
    sigma: float = 2.0
    lower_percentile: float = 1.0
    upper_percentile: float = 99.0
    mapheight: float | None = None
    noise_floor_subtraction: bool = False

    @property
    def enabled(self) -> bool:
        return self.mode != DEFAULT_FIELD_SCALING_MODE


@dataclass(frozen=True)
class PhysicsRefineConfig:
    """Configuration for test-time physics refinement.

    After the neural network produces an initial moment prediction, gradient
    descent is run on ||B_dipole(m; r_grid) - B_obs||² to correct errors in
    magnitude and direction.  Most effective for N=1 dipole samples where the
    single-dipole forward model is exact.  For N>1 the forward model is wrong
    (superposition ≠ single dipole), so refinement is skipped for datasets
    whose dipole count exceeds ``max_dipoles``.
    """

    n_steps: int = 50
    lr: float = 1e-3
    grid_extent: float = 1.0
    grid_z_offset: float = 0.1
    grid_size: int = 100
    residual_threshold: float = 0.15


def _require_dependency(module, name: str):
    """Raise a helpful error if an optional dependency is missing."""

    if module is None:
        raise ModuleNotFoundError(
            f"Required dependency '{name}' is not installed. "
            "Install it via `pip install -r src/requirements.txt` or an equivalent command."
        )
    return module


def _missing_dependencies(model_type: str) -> List[str]:
    """Return a list of human-friendly dependency names that are unavailable."""

    missing = []
    model_dependencies = {
        "simple_cnn": (SimpleCNN, "model.architecture.SimpleCNN"),
        "simple_cnn_no_bn": (
            SimpleCNNNoBN,
            "model.architecture.SimpleCNNNoBN",
        ),
        "simple_cnn_standard_bn": (
            SimpleCNNStandardBN,
            "model.architecture.SimpleCNNStandardBN",
        ),
        "equivariant_resnet": (EquivariantHeadedResNet, "model.architecture.EquivariantHeadedResNet"),
        "dipolenet": (AttentionDipoleModel, "model.architecture.AttentionDipoleModel"),
    }

    for module, name in (
        (torch, "torch"),
        (np, "numpy"),
        (pd, "pandas"),
        (h5py, "h5py"),
        model_dependencies[model_type],
        (mag_p_error, "model.architecture.mag_p_error"),
        (inclination_error, "model.architecture.inclination_error"),
        (declination_error, "model.architecture.declination_error"),
    ):
        if module is None:
            missing.append(name)
    return missing


def build_field_scaling_config(
    *,
    mode: str = DEFAULT_FIELD_SCALING_MODE,
    sigma: float = 2.0,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    mapheight: float | None = None,
    noise_floor_subtraction: bool = False,
) -> FieldScalingConfig:
    """Validate and return an immutable field-scaling configuration."""

    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in FIELD_SCALING_MODES:
        raise ValueError(
            f"Unsupported field scaling mode: {mode!r}. Expected one of {FIELD_SCALING_MODES!r}."
        )

    sigma_value = float(sigma)
    lower_value = float(lower_percentile)
    upper_value = float(upper_percentile)
    if sigma_value < 0.0:
        raise ValueError("field scaling sigma must be non-negative.")
    if not (0.0 <= lower_value < upper_value <= 100.0):
        raise ValueError(
            "field scaling percentiles must satisfy 0 <= lower_percentile < upper_percentile <= 100."
        )

    return FieldScalingConfig(
        mode=normalized_mode,
        sigma=sigma_value,
        lower_percentile=lower_value,
        upper_percentile=upper_value,
        mapheight=None if mapheight is None else float(mapheight),
        noise_floor_subtraction=bool(noise_floor_subtraction),
    )


def field_scaling_signature_payload(config: FieldScalingConfig) -> Dict[str, float | str | bool | None]:
    """Return a stable JSON-serializable summary for caching and manifests."""

    return {
        "mode": config.mode,
        "sigma": config.sigma,
        "lower_percentile": config.lower_percentile,
        "upper_percentile": config.upper_percentile,
        "mapheight": config.mapheight,
        "noise_floor_subtraction": config.noise_floor_subtraction,
    }


def missing_field_scaling_dependencies(config: FieldScalingConfig) -> List[str]:
    """Return missing optional dependencies required by the requested scaling mode."""

    if not config.enabled:
        return []

    missing = []
    # scipy gaussian_filter is optional — only required when sigma > 0
    if config.sigma > 0.0 and gaussian_filter is None:
        missing.append("scipy.ndimage.gaussian_filter")
    for module, name in (
        (build_grid_points, "augment_dataset_rotations.build_grid_points"),
        (recompute_field_from_m, "augment_dataset_rotations.recompute_field_from_m"),
    ):
        if module is None:
            missing.append(name)
    return missing


def _detect_simple_cnn_variant_from_state_dict(state_dict: Mapping[str, object]) -> str | None:
    """Infer which SimpleCNN-family variant produced *state_dict*."""

    keys = set(state_dict.keys())
    if any(key.startswith("bn1.bn.") or key.startswith("residual.bn1.bn.") for key in keys):
        return "simple_cnn"
    if any(
        key.startswith("bn1.") or key.startswith("residual.bn1.")
        for key in keys
    ):
        return "simple_cnn_standard_bn"
    if "conv1.weight" in keys and "fc.weight" in keys and not any(".bn" in key for key in keys):
        return "simple_cnn_no_bn"
    # Legacy sequential-style checkpoint (net.0 / net.2 / net.4) with no BN
    if "net.0.weight" in keys and "net.4.weight" in keys and not any(".bn" in key for key in keys):
        return "simple_cnn_no_bn"
    return None


def _remap_legacy_sequential_keys(state_dict: Mapping[str, object]) -> dict:
    """Remap old nn.Sequential-style keys to current named-attribute keys.

    Old checkpoints stored SimpleCNNNoBN as an ``nn.Sequential`` so keys look
    like ``net.0.weight``, ``net.2.conv1.weight``, ``net.4.weight``.  The
    current architecture uses named attributes (``conv1``, ``residual``, ``fc``).
    """
    remap = {
        "net.0.": "conv1.",
        "net.2.": "residual.",
        "net.4.": "fc.",
    }
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in remap.items():
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                break
        remapped[new_key] = value
    return remapped


@contextmanager
def progress_bar(total: int, title: str, enabled: bool) -> Iterator[Callable[[], None]]:
    """Context manager that optionally yields an ``alive_progress`` bar."""

    if enabled and total > 0 and alive_progress is not None:
        with alive_progress.alive_bar(total, title=title) as bar:
            yield bar
    else:
        yield lambda *_, **__: None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for inference configuration."""

    parser = argparse.ArgumentParser(description="Run inference and summarize vector model errors.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained model checkpoint (default: %(default)s).",
    )
    parser.add_argument(
        "--model-type",
        choices=["simple_cnn", "simple_cnn_no_bn", "simple_cnn_standard_bn", "equivariant_resnet"],
        default="simple_cnn",
        help=(
            "Architecture used by the checkpoint. Select 'equivariant_resnet' for models trained "
            "with equivariant_resnet_training.py."
        ),
    )
    parser.add_argument(
        "--dataset-pattern",
        type=str,
        default=DEFAULT_DATASET_PATTERN,
        help="Glob pattern pointing to directories that contain dataset.h5 files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional explicit path for the aggregated CSV output.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the aggregated CSV when --output-csv is omitted.",
    )
    parser.add_argument(
        "--prediction-suffix",
        type=str,
        default="",
        help=(
            "Optional suffix appended to dataset_answers.h5 when writing per-dataset predictions. "
            "Defaults to an empty string, preserving the historic dataset_answers.h5 output."
        ),
    )
    parser.add_argument(
        "--mag-calibration-factor",
        type=float,
        default=1.0,
        help="Multiply predicted magnitude by this scalar after inference (default: 1.0, i.e. no correction).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="Images per inference batch (default: 64). Increase for higher GPU throughput; reduce if you run out of memory.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars (useful when running in logging-only environments).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Python logging level (default: %(default)s).",
    )
    parser.add_argument(
        "--print-predictions",
        action="store_true",
        help="Log each predicted vector to the console while processing datasets.",
    )
    parser.add_argument(
        "--field-scaling-mode",
        choices=FIELD_SCALING_MODES,
        default=DEFAULT_FIELD_SCALING_MODE,
        help=(
            "Optional inference-time rescaling for predicted magnetic moments. "
            "'gaussian_vector_percentile' smooths the measured and forward-modeled fields "
            "with a Gaussian filter, then matches their robust 1st/99th-percentile amplitudes."
        ),
    )
    parser.add_argument(
        "--field-scaling-sigma",
        type=float,
        default=2.0,
        help="Gaussian sigma used by inference-time field scaling.",
    )
    parser.add_argument(
        "--field-scaling-lower-percentile",
        type=float,
        default=1.0,
        help="Lower percentile used to estimate the measured/predicted field scale.",
    )
    parser.add_argument(
        "--field-scaling-upper-percentile",
        type=float,
        default=99.0,
        help="Upper percentile used to estimate the measured/predicted field scale.",
    )
    parser.add_argument(
        "--field-scaling-mapheight",
        type=float,
        default=None,
        help="Optional override for the dataset mapheight/standoff used by field scaling.",
    )
    parser.add_argument(
        "--field-scaling-noise-floor-subtraction",
        action="store_true",
        default=False,
        help=(
            "Estimate the noise floor from the outer border pixels and subtract it from "
            "the measured field amplitude before computing the scale ratio."
        ),
    )
    return parser.parse_args()


def load_input_norm_stats(
    model_path: Path,
) -> "tuple[torch.Tensor, torch.Tensor] | tuple[None, None]":
    """Load per-channel z-score stats saved alongside a checkpoint, if present.

    The pipeline saves ``input_norm_stats.json`` next to its checkpoints with
    ``{"mean": [c0, c1, c2], "std": [c0, c1, c2]}``.  Returns (mean, std) as
    (1, C, 1, 1) CPU float tensors, or (None, None) if no file is found.
    """
    import json

    torch_module = _require_dependency(torch, "torch")
    stats_path = model_path.expanduser().parent / "input_norm_stats.json"
    if not stats_path.exists():
        return None, None
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        mean = torch_module.tensor(stats["mean"], dtype=torch_module.float32).view(1, -1, 1, 1)
        std = torch_module.tensor(stats["std"], dtype=torch_module.float32).view(1, -1, 1, 1).clamp(min=1e-8)
        logging.info("Loaded input normalization stats from %s", stats_path)
        return mean, std
    except Exception as exc:
        logging.warning("Could not load input_norm_stats.json (%s); inputs will not be normalized.", exc)
        return None, None


def load_model(model_path: Path, device: "torch.device", model_type: str) -> "torch.nn.Module":
    """Load the checkpoint onto the requested device."""

    torch_module = _require_dependency(torch, "torch")
    expanded_path = model_path.expanduser()
    try:
        state_dict = torch_module.load(expanded_path, map_location=device, weights_only=True)
    except pickle.UnpicklingError as safe_load_error:
        # weights_only=True understands only a restricted pickle dialect; a
        # checkpoint saved by a newer torch or with a higher pickle protocol
        # trips it even when the payload is a plain state dict. These files are
        # produced by this project's own training runs, so retry loudly.
        logging.warning(
            "Safe (weights_only=True) load failed for %s: %s. Retrying with "
            "weights_only=False -- only appropriate for checkpoints you trust.",
            expanded_path,
            safe_load_error,
        )
        state_dict = torch_module.load(expanded_path, map_location=device, weights_only=False)
    if isinstance(state_dict, Mapping) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Unsupported checkpoint payload in {expanded_path}.")

    resolved_model_type = model_type
    detected_variant = _detect_simple_cnn_variant_from_state_dict(state_dict)
    if model_type in {"simple_cnn", "simple_cnn_no_bn", "simple_cnn_standard_bn"} and detected_variant is not None:
        if detected_variant != model_type:
            logging.warning(
                "Checkpoint %s appears to use %s; overriding requested model_type=%s.",
                expanded_path,
                detected_variant,
                model_type,
            )
        resolved_model_type = detected_variant

    model_builders = {
        "simple_cnn": lambda: _require_dependency(SimpleCNN, "model.architecture.SimpleCNN")(),
        "simple_cnn_no_bn": lambda: _require_dependency(
            SimpleCNNNoBN, "model.architecture.SimpleCNNNoBN"
        )(),
        "simple_cnn_standard_bn": lambda: _require_dependency(
            SimpleCNNStandardBN, "model.architecture.SimpleCNNStandardBN"
        )(),
        "equivariant_resnet": lambda: _require_dependency(
            EquivariantHeadedResNet, "model.architecture.EquivariantHeadedResNet"
        )(),
        "dipolenet": lambda: _require_dependency(
            AttentionDipoleModel, "model.architecture.AttentionDipoleModel"
        )(),
    }

    if resolved_model_type not in model_builders:
        raise ValueError(f"Unsupported model type: {resolved_model_type}")

    if "net.0.weight" in state_dict:
        state_dict = _remap_legacy_sequential_keys(state_dict)

    model = model_builders[resolved_model_type]()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Attach pre-computed input normalization stats if saved alongside the checkpoint.
    # These are produced by legacy_cnn_pipeline.py for models that were trained with
    # per-channel z-score normalization (legacySimpleCNN / simple_cnn_no_bn).
    norm_mean, norm_std = load_input_norm_stats(model_path)
    if norm_mean is not None:
        model._input_norm_mean = norm_mean
        model._input_norm_std = norm_std

    return model


def extract_snr(path: Path) -> Optional[float]:
    """Extract the numeric SNR encoded in a directory name if present."""

    match = re.search(r"SNR(?:lv)?[_-]?([0-9]+(?:\.[0-9]+)?)", str(path))
    if match:
        return float(match.group(1))
    return None


def _infer_mapheight(h5_file: "h5py.File", override: float | None) -> float:
    """Return the generator-space mapheight recorded for a dataset."""

    if override is not None:
        return float(override)
    for key in ("mapheight", "standoff"):
        if key in h5_file.attrs:
            return float(h5_file.attrs[key])
    raise ValueError("mapheight not provided and not found in HDF5 attrs.")


def _zero_center_field_components(field: "np.ndarray") -> "np.ndarray":
    """Remove the per-channel mean from a 2D or 3D field image."""

    np_module = _require_dependency(np, "numpy")
    field_array = np_module.asarray(field, dtype=np.float64)
    if field_array.ndim == 2:
        return field_array - float(field_array.mean())
    if field_array.ndim == 3:
        return field_array - field_array.mean(axis=(0, 1), keepdims=True)
    raise ValueError(f"Expected a 2D or 3D field image, received shape {field_array.shape!r}.")


def _smooth_field(field_array: "np.ndarray", config: FieldScalingConfig) -> "np.ndarray":
    """Return a Gaussian-smoothed copy of *field_array* when sigma > 0."""

    if config.sigma <= 0.0:
        return field_array
    gaussian = _require_dependency(gaussian_filter, "scipy.ndimage.gaussian_filter")
    if field_array.ndim == 2:
        return gaussian(field_array, sigma=config.sigma)
    return gaussian(field_array, sigma=(config.sigma, config.sigma, 0.0))


def _vector_magnitude_map(field: "np.ndarray") -> "np.ndarray":
    """Return a 2D map of per-pixel vector magnitudes for a 2D or 3D field array."""

    np_module = _require_dependency(np, "numpy")
    if field.ndim == 2:
        return np_module.abs(field)
    if field.ndim == 3 and field.shape[-1] == 1:
        return np_module.abs(field[..., 0])
    if field.ndim == 3:
        return np_module.linalg.norm(field, axis=-1)
    raise ValueError(f"Expected a 2D or 3D field image, received shape {field.shape!r}.")


def _estimate_noise_floor_rms(field_array: "np.ndarray", border_width: int = 2) -> float:
    """Estimate noise RMS from the outermost border pixels of the field map.

    The dipole signal is negligible near the map edges, so the border pixels
    give a reliable estimate of the instrument/digitisation noise floor.
    """

    np_module = _require_dependency(np, "numpy")
    mag = _vector_magnitude_map(field_array)
    h, w = mag.shape
    if h <= 2 * border_width or w <= 2 * border_width:
        return float(np_module.std(mag))
    top = mag[:border_width, :].ravel()
    bottom = mag[-border_width:, :].ravel()
    left = mag[border_width:-border_width, :border_width].ravel()
    right = mag[border_width:-border_width, -border_width:].ravel()
    border = np_module.concatenate([top, bottom, left, right])
    return float(np_module.std(border))


def _gaussian_percentile_amplitude(smoothed: "np.ndarray", config: FieldScalingConfig) -> float:
    """Original mode: (upper_pct - lower_pct) range of the vector-magnitude map."""

    np_module = _require_dependency(np, "numpy")
    mag = _vector_magnitude_map(smoothed)
    lower = float(np_module.percentile(mag, config.lower_percentile))
    upper = float(np_module.percentile(mag, config.upper_percentile))
    return max(upper - lower, 0.0)


def _gradient_iqr_amplitude(smoothed: "np.ndarray") -> float:
    """Option A: IQR of the spatial gradient magnitude of the vector-magnitude map.

    Gradient-based statistics are driven by the sharpest local feature rather
    than total field power, making this measure less sensitive to the number of
    dipole sources present.
    """

    np_module = _require_dependency(np, "numpy")
    mag = _vector_magnitude_map(smoothed)
    gy, gx = np_module.gradient(mag)
    grad_mag = np_module.sqrt(gx ** 2 + gy ** 2)
    q75 = float(np_module.percentile(grad_mag, 75))
    q25 = float(np_module.percentile(grad_mag, 25))
    return max(q75 - q25, 0.0)


def _rms_amplitude(smoothed: "np.ndarray") -> float:
    """Option B: RMS of the vector-magnitude map.

    Provides a principled basis for noise-floor subtraction in quadrature.
    """

    np_module = _require_dependency(np, "numpy")
    mag = _vector_magnitude_map(smoothed)
    return float(np_module.sqrt(np_module.mean(mag ** 2)))



def _robust_field_scale(field: "np.ndarray", config: FieldScalingConfig) -> float:
    """Return the scalar robust amplitude used for inference-time field rescaling.

    Dispatches to the appropriate estimator based on *config.mode* and
    optionally subtracts the estimated noise floor.
    """

    import math as _math

    np_module = _require_dependency(np, "numpy")
    field_array = _zero_center_field_components(np_module.asarray(field))

    if config.mode == DEFAULT_FIELD_SCALING_MODE:
        raise ValueError("Field scaling amplitude is undefined when field scaling is disabled.")

    smoothed = _smooth_field(field_array, config)

    if config.mode == "gaussian_vector_percentile":
        scale = _gaussian_percentile_amplitude(smoothed, config)
    elif config.mode == "gradient_iqr":
        scale = _gradient_iqr_amplitude(smoothed)
    elif config.mode == "rms":
        scale = _rms_amplitude(smoothed)
    else:
        raise ValueError(f"Unsupported field scaling mode: {config.mode!r}.")

    if config.noise_floor_subtraction and scale > 0.0:
        noise_floor = _estimate_noise_floor_rms(field_array)
        if config.mode == "rms":
            scale = _math.sqrt(max(scale ** 2 - noise_floor ** 2, 0.0))
        else:
            scale = max(scale - noise_floor, 0.0)

    return max(scale, 0.0)


def _forward_model_field_from_prediction(
    prediction_vector: "np.ndarray",
    *,
    image_shape: tuple[int, ...],
    mapheight: float,
) -> "np.ndarray":
    """Generate the predicted field corresponding to one magnetic-moment vector."""

    np_module = _require_dependency(np, "numpy")
    grid_builder = _require_dependency(build_grid_points, "augment_dataset_rotations.build_grid_points")
    field_recompute = _require_dependency(
        recompute_field_from_m,
        "augment_dataset_rotations.recompute_field_from_m",
    )
    grid_min = float(_require_dependency(DEFAULT_GRID_MIN, "augment_dataset_rotations.DEFAULT_GRID_MIN"))
    grid_max = float(_require_dependency(DEFAULT_GRID_MAX, "augment_dataset_rotations.DEFAULT_GRID_MAX"))
    mu0_over_4pi = float(
        _require_dependency(DEFAULT_MU0_OVER_4PI, "augment_dataset_rotations.DEFAULT_MU0_OVER_4PI")
    )

    if len(image_shape) == 2:
        size_x, size_y = image_shape
        channel_count = 1
    elif len(image_shape) == 3:
        size_x, size_y, channel_count = image_shape
    else:
        raise ValueError(f"Expected image shape with 2 or 3 dimensions, received {image_shape!r}.")
    if size_x != size_y:
        raise ValueError(
            f"Field scaling expects square field maps for forward modeling, received {image_shape!r}."
        )

    grid_points = grid_builder(
        int(size_x),
        int(size_y),
        float(mapheight),
        grid_min,
        grid_max,
        np.float64,
    )
    predicted_field = field_recompute(
        np_module.asarray(prediction_vector, dtype=np.float64),
        np_module.asarray([0.5, 0.5, 0.0], dtype=np.float64),
        grid_points,
        mu0_over_4pi=mu0_over_4pi,
    )
    if channel_count == 1:
        n111 = np_module.asarray([1.0, 1.0, 1.0], dtype=np.float64)
        n111 /= np_module.linalg.norm(n111)
        projected_field = np_module.einsum("ijk,k->ij", predicted_field, n111)
        return _zero_center_field_components(projected_field)
    return _zero_center_field_components(np_module.asarray(predicted_field, dtype=np.float64))


def _apply_field_scaling_to_prediction(
    prediction_vector: "np.ndarray",
    measured_field: "np.ndarray",
    *,
    mapheight: float,
    config: FieldScalingConfig,
) -> tuple["np.ndarray", float]:
    """Rescale *prediction_vector* using robust amplitudes from measured/predicted fields."""

    np_module = _require_dependency(np, "numpy")
    prediction_array = np_module.asarray(prediction_vector, dtype=np.float64).reshape(-1)
    if not config.enabled:
        return prediction_array, 1.0

    measured_scale = _robust_field_scale(measured_field, config)
    predicted_field = _forward_model_field_from_prediction(
        prediction_array,
        image_shape=np_module.asarray(measured_field).shape,
        mapheight=mapheight,
    )
    predicted_scale = _robust_field_scale(predicted_field, config)
    if not np_module.isfinite(measured_scale) or not np_module.isfinite(predicted_scale):
        return prediction_array, 1.0
    if measured_scale <= 0.0 or predicted_scale <= 0.0:
        return prediction_array, 1.0

    scale_factor = float(measured_scale / predicted_scale)
    if not np_module.isfinite(scale_factor) or scale_factor <= 0.0:
        return prediction_array, 1.0
    return prediction_array * scale_factor, scale_factor


def _prepare_images(image_array: "np.ndarray") -> "torch.Tensor":
    """Convert raw HDF5 image data into a Torch tensor ready for inference."""

    np_module = _require_dependency(np, "numpy")
    torch_module = _require_dependency(torch, "torch")
    if image_array.ndim != 4:
        raise ValueError(f"Expected image tensor with 4 dimensions, received shape {image_array.shape!r}.")
    # HDF5 stores images as NHWC; Torch expects NCHW.
    tensor = torch_module.from_numpy(np_module.asarray(image_array)).float().permute(0, 3, 1, 2)
    return tensor


def _normalize_equivariant_images(images: "torch.Tensor") -> "torch.Tensor":
    """Return channel-standardized copies of images for the equivariant model."""

    torch_module = _require_dependency(torch, "torch")
    if images.ndim != 4:
        raise ValueError(
            "Equivariant ResNet expects a 4D image tensor shaped (batch, channels, height, width)."
        )

    channel_mean = images.mean(dim=(0, 2, 3), keepdim=True)
    channel_std = images.std(dim=(0, 2, 3), unbiased=False, keepdim=True)
    safe_channel_std = channel_std.clamp_min(torch_module.tensor(1e-6, dtype=images.dtype))
    return (images - channel_mean) / safe_channel_std


def _circular_mean_degrees(
    angles_radians: "torch.Tensor",
    *,
    valid_mask: Optional["torch.Tensor"] = None,
) -> float:
    """Return the circular mean of *angles_radians* in degrees."""

    torch_module = _require_dependency(torch, "torch")
    if valid_mask is not None:
        angles_radians = angles_radians[valid_mask]
    if angles_radians.numel() == 0:
        return 0.0

    mean_sin = torch_module.sin(angles_radians).mean()
    mean_cos = torch_module.cos(angles_radians).mean()
    return float(torch_module.rad2deg(torch_module.atan2(mean_sin, mean_cos)).item())


def _compute_diagnostic_statistics(
    predictions: "torch.Tensor",
    ground_truth: "torch.Tensor",
    *,
    predicted_magnitude: "torch.Tensor",
    true_magnitude: "torch.Tensor",
    predicted_inclination: "torch.Tensor",
    true_inclination: "torch.Tensor",
    predicted_declination: "torch.Tensor",
    true_declination: "torch.Tensor",
    valid_declination_mask: "torch.Tensor",
) -> Dict[str, float]:
    """Compute summary diagnostics that help detect prediction collapse."""

    torch_module = _require_dependency(torch, "torch")
    mean_prediction_vector = predictions.mean(dim=0, keepdim=True)
    mean_target_vector = ground_truth.mean(dim=0, keepdim=True)

    mean_prediction_xy_norm = torch_module.norm(mean_prediction_vector[:, :2], dim=1)
    mean_target_xy_norm = torch_module.norm(mean_target_vector[:, :2], dim=1)
    mean_prediction_inclination = torch_module.atan2(
        mean_prediction_vector[:, 2], mean_prediction_xy_norm
    )
    mean_target_inclination = torch_module.atan2(mean_target_vector[:, 2], mean_target_xy_norm)
    mean_prediction_declination = torch_module.atan2(
        mean_prediction_vector[:, 1], mean_prediction_vector[:, 0]
    )
    mean_target_declination = torch_module.atan2(mean_target_vector[:, 1], mean_target_vector[:, 0])

    constant_prediction = mean_prediction_vector.expand_as(predictions)

    return {
        "mean_predicted_vector_norm": float(predicted_magnitude.mean().item()),
        "mean_target_vector_norm": float(true_magnitude.mean().item()),
        "mean_prediction_vector_norm": float(torch_module.norm(mean_prediction_vector, dim=1).item()),
        "mean_target_vector_collapsed_norm": float(torch_module.norm(mean_target_vector, dim=1).item()),
        "mean_predicted_inclination_deg": float(torch_module.rad2deg(predicted_inclination.mean()).item()),
        "mean_target_inclination_deg": float(torch_module.rad2deg(true_inclination.mean()).item()),
        "mean_predicted_declination_deg": _circular_mean_degrees(predicted_declination),
        "mean_target_declination_deg": _circular_mean_degrees(
            true_declination,
            valid_mask=valid_declination_mask,
        ),
        "mean_prediction_vector_inclination_deg": float(
            torch_module.rad2deg(mean_prediction_inclination.squeeze(0)).item()
        ),
        "mean_target_vector_inclination_deg": float(
            torch_module.rad2deg(mean_target_inclination.squeeze(0)).item()
        ),
        "mean_prediction_vector_declination_deg": float(
            torch_module.rad2deg(mean_prediction_declination.squeeze(0)).item()
        ),
        "mean_target_vector_declination_deg": float(
            torch_module.rad2deg(mean_target_declination.squeeze(0)).item()
        ),
        "declination_valid_fraction": float(valid_declination_mask.float().mean().item()),
        "constant_mean_prediction_magnitude_error": float(
            mag_p_error(constant_prediction, ground_truth).item() * 100.0
        ),
        "constant_mean_prediction_inclination_error": float(
            math.degrees(inclination_error(constant_prediction, ground_truth).item())
        ),
        "constant_mean_prediction_declination_error": float(
            math.degrees(declination_error(constant_prediction, ground_truth).item())
        ),
    }


def _compute_error_statistics(
    predictions: "torch.Tensor", ground_truth: "torch.Tensor"
) -> Tuple[Dict[str, float], Dict[str, "np.ndarray"]]:
    """Compute error statistics given predictions and ground-truth vectors."""

    torch_module = _require_dependency(torch, "torch")
    if predictions.shape != ground_truth.shape:
        raise ValueError(
            "Predictions and ground truth must have matching shapes; "
            f"received {predictions.shape} vs {ground_truth.shape}."
        )
    if mag_p_error is None or inclination_error is None or declination_error is None:
        raise ModuleNotFoundError(
            "Required model.architecture metrics are unavailable. Install dependencies via ``pip install -r src/requirements.txt``."
        )

    eps = torch_module.tensor(1e-8, dtype=predictions.dtype, device=predictions.device)

    predicted_magnitude = torch_module.norm(predictions, dim=1)
    true_magnitude = torch_module.norm(ground_truth, dim=1)
    magnitude_percent_error = torch_module.abs(
        (predicted_magnitude - true_magnitude) / (true_magnitude + eps)
    ) * 100.0

    predicted_xy_norm = torch_module.norm(predictions[:, :2], dim=1)
    true_xy_norm = torch_module.norm(ground_truth[:, :2], dim=1)

    predicted_inclination = torch_module.atan2(predictions[:, 2], predicted_xy_norm)
    true_inclination = torch_module.atan2(ground_truth[:, 2], true_xy_norm)
    inclination_difference = (
        (predicted_inclination - true_inclination + torch_module.pi) % (2 * torch_module.pi)
    ) - torch_module.pi
    inclination_error_deg = torch_module.abs(torch_module.rad2deg(inclination_difference))

    predicted_declination = torch_module.atan2(predictions[:, 1], predictions[:, 0])
    true_declination = torch_module.atan2(ground_truth[:, 1], ground_truth[:, 0])
    declination_difference = (
        (predicted_declination - true_declination + torch_module.pi) % (2 * torch_module.pi)
    ) - torch_module.pi
    declination_error_deg_all = torch_module.abs(torch_module.rad2deg(declination_difference))

    horizontal_threshold = 1e-3  # magnetic moment horizontal magnitude threshold (A·m^2)
    valid_declination_mask = true_xy_norm > horizontal_threshold
    declination_error_deg = torch_module.where(
        valid_declination_mask,
        declination_error_deg_all,
        torch_module.full_like(declination_error_deg_all, float("nan")),
    )

    stats = {
        "average_magnitude_error": mag_p_error(predictions, ground_truth).item() * 100.0,
        "average_inclination_error": math.degrees(inclination_error(predictions, ground_truth).item()),
        "average_declination_error": math.degrees(declination_error(predictions, ground_truth).item()),
    }
    stats.update(
        _compute_diagnostic_statistics(
            predictions,
            ground_truth,
            predicted_magnitude=predicted_magnitude,
            true_magnitude=true_magnitude,
            predicted_inclination=predicted_inclination,
            true_inclination=true_inclination,
            predicted_declination=predicted_declination,
            true_declination=true_declination,
            valid_declination_mask=valid_declination_mask,
        )
    )

    per_example = {
        "magnitude_error": magnitude_percent_error.cpu().numpy(),
        "inclination_error": inclination_error_deg.cpu().numpy(),
        "declination_error": declination_error_deg.cpu().numpy(),
    }
    return stats, per_example


def _normalize_string_attribute(value: Any) -> Optional[str]:
    """Return a lowercase string for byte or string attributes when available."""

    if isinstance(value, bytes):
        return value.decode("utf-8").strip().lower()
    if isinstance(value, str):
        return value.strip().lower()
    return None


def _describe_target_representation(m_dataset: "h5py.Dataset") -> str:
    """Describe whether targets are stored as Cartesian or angular components."""

    attribute_candidates = (
        "representation",
        "coordinate_system",
        "target_representation",
        "vector_format",
    )

    for attribute in attribute_candidates:
        normalized_value = _normalize_string_attribute(m_dataset.attrs.get(attribute))
        if normalized_value is None:
            continue
        if normalized_value in {"mx_my_mz", "cartesian", "xyz"}:
            return "Mx/My/Mz"
        if normalized_value in {"mag_inc_dec", "magnitude_inclination_declination", "mid"}:
            return "Magnitude/Inclination/Declination"

    component_labels = m_dataset.attrs.get("component_labels")
    if isinstance(component_labels, (list, tuple)):
        normalized_labels: List[str] = []
        for label in component_labels:
            normalized_label = _normalize_string_attribute(label)
            if normalized_label is not None:
                normalized_labels.append(normalized_label)

        if {"mx", "my", "mz"}.issubset(set(normalized_labels)):
            return "Mx/My/Mz"
        if {"mag", "inc", "dec"}.issubset(set(normalized_labels)):
            return "Magnitude/Inclination/Declination"

    return "Mx/My/Mz"


def _should_log_prediction(log_each_prediction: bool, image_index: int, max_logged_predictions: int) -> bool:
    """Return True when the current prediction should be logged."""

    return log_each_prediction and image_index < max_logged_predictions


def _save_errors(
    output_path: Path,
    errors: Dict[str, "np.ndarray"],
    stats: Dict[str, float],
    *,
    field_scaling_factors: "np.ndarray | None" = None,
    field_scaling_config: FieldScalingConfig | None = None,
    sample_indices: "Optional[Any]" = None,
    predictions_raw: "np.ndarray | None" = None,
) -> None:
    """Persist per-sample errors and aggregate statistics to an HDF5 file."""

    h5_module = _require_dependency(h5py, "h5py")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5_module.File(output_path, "w") as h5_file:
        for key, dataset_name in HDF5_DATASET_NAMES.items():
            h5_file.create_dataset(dataset_name, data=errors[key])
        if field_scaling_factors is not None:
            h5_file.create_dataset("field_scaling_factor", data=field_scaling_factors)
        if sample_indices is not None:
            h5_file.create_dataset("subsample_indices", data=sample_indices)
        if predictions_raw is not None:
            h5_file.create_dataset("predictions_raw", data=predictions_raw)
        for key, value in stats.items():
            h5_file.attrs[key] = value
        if field_scaling_config is not None:
            h5_file.attrs["field_scaling_mode"] = field_scaling_config.mode
            h5_file.attrs["field_scaling_sigma"] = field_scaling_config.sigma
            h5_file.attrs["field_scaling_lower_percentile"] = field_scaling_config.lower_percentile
            h5_file.attrs["field_scaling_upper_percentile"] = field_scaling_config.upper_percentile
            if field_scaling_config.mapheight is not None:
                h5_file.attrs["field_scaling_mapheight"] = field_scaling_config.mapheight


def _apply_isotonic_mag(vector: "torch.Tensor", iso: dict) -> "torch.Tensor":
    """Rescale vector magnitude via isotonic lookup without changing direction."""
    np_module = _require_dependency(np, "numpy")
    pred_mag = float(vector.norm())
    if pred_mag <= 0.0:
        return vector
    corrected = float(np_module.interp(pred_mag, iso["X_thresholds"], iso["y_thresholds"]))
    return vector * (corrected / pred_mag)


def _apply_isotonic_mag_batch(vectors: "torch.Tensor", iso: dict) -> "torch.Tensor":
    """Vectorized isotonic rescaling over a batch of prediction vectors (B, 3)."""
    np_module = _require_dependency(np, "numpy")
    pred_mags = vectors.norm(dim=1)                                      # (B,)
    corrected = torch.from_numpy(
        np_module.interp(pred_mags.numpy(), iso["X_thresholds"], iso["y_thresholds"])
    ).to(dtype=vectors.dtype)
    safe_mags = pred_mags.clamp(min=1e-12)
    return vectors * (corrected / safe_mags).unsqueeze(1)


def predict_and_save(
    model: "torch.nn.Module",
    h5_file_path: Path,
    output_path: Path,
    device: "torch.device",
    show_progress: bool,
    log_each_prediction: bool,
    field_scaling_config: FieldScalingConfig | None = None,
    physics_refine_config: PhysicsRefineConfig | None = None,
    sample_indices: "Optional[Any]" = None,
    mag_calibration_factor: float = 1.0,
    mag_calibration_isotonic: "dict | None" = None,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Run inference for a single dataset and persist the results."""

    h5_module = _require_dependency(h5py, "h5py")
    np_module = _require_dependency(np, "numpy")
    torch_module = _require_dependency(torch, "torch")
    scaling_config = field_scaling_config or build_field_scaling_config()
    with h5_module.File(h5_file_path, "r") as h5_file:
        m_dataset = h5_file["m"]
        target_representation = _describe_target_representation(m_dataset)
        if sample_indices is not None:
            idx = sample_indices.tolist()
            m_values = torch_module.from_numpy(np_module.asarray(m_dataset[idx])).float()
            raw_images = np_module.asarray(h5_file["I"][idx], dtype=np.float32)
            logging.info("Subsampled to %d examples using provided indices.", len(sample_indices))
        else:
            m_values = torch_module.from_numpy(m_dataset[:]).float()
            raw_images = np_module.asarray(h5_file["I"][:], dtype=np.float32)
        mapheight = None
        if scaling_config.enabled:
            mapheight = _infer_mapheight(h5_file, scaling_config.mapheight)
    images = _prepare_images(raw_images)
    if EquivariantHeadedResNet is not None and isinstance(model, EquivariantHeadedResNet):
        images = _normalize_equivariant_images(images)

    # Apply per-channel z-score normalization when the checkpoint was trained with it.
    if hasattr(model, "_input_norm_mean") and model._input_norm_mean is not None:
        images = (images - model._input_norm_mean) / model._input_norm_std

    # Build physics refinement grid once if requested (only supported for dipolenet).
    refine_fn = physics_refine if physics_refine_config is not None else None
    r_grid = None
    if physics_refine_config is not None and refine_fn is not None and make_measurement_grid is not None:
        if not isinstance(model, AttentionDipoleModel):
            logging.warning("Physics refinement is only supported for dipolenet; skipping.")
            refine_fn = None
        else:
            r_grid = make_measurement_grid(
                grid_size=physics_refine_config.grid_size,
                extent=physics_refine_config.grid_extent,
                z_offset=physics_refine_config.grid_z_offset,
                device=device,
            )

    batch_predictions: List["torch.Tensor"] = []
    batch_predictions_raw: List["torch.Tensor"] = []   # pre-calibration, for --recalibrate
    field_scaling_factors: list[float] = []
    total_images = images.shape[0]

    # Fast batched path: field scaling and physics refinement both require per-sample
    # access to raw images / the forward model, so they stay in the per-sample loop.
    use_fast_path = not scaling_config.enabled and refine_fn is None

    with torch_module.no_grad(), progress_bar(total_images, "Processing Images", show_progress) as bar:
        if use_fast_path:
            for start in range(0, total_images, batch_size):
                chunk = images[start:start + batch_size].to(device)
                out = model(chunk)
                if isinstance(out, tuple):
                    out = out[0]
                raw = out.cpu()  # (B, 3)

                for local_i, pred in enumerate(raw.unbind(0)):
                    global_i = start + local_i
                    field_scaling_factors.append(1.0)
                    if _should_log_prediction(log_each_prediction, global_i, LOGGED_PREDICTION_LIMIT):
                        logging.info(
                            "Prediction %d for %s (%s): prediction=%s | target=%s",
                            global_i, h5_file_path.name, target_representation,
                            pred.tolist(), m_values[global_i].tolist(),
                        )
                    elif log_each_prediction and global_i == LOGGED_PREDICTION_LIMIT:
                        remaining_predictions = max(total_images - LOGGED_PREDICTION_LIMIT, 0)
                        if remaining_predictions:
                            logging.info(
                                "Logging limited to the first %d predictions for %s; skipping %d additional items.",
                                LOGGED_PREDICTION_LIMIT, h5_file_path.name, remaining_predictions,
                            )
                    bar()

                batch_predictions_raw.extend(raw.unbind(0))
                if mag_calibration_isotonic is not None:
                    cal = _apply_isotonic_mag_batch(raw, mag_calibration_isotonic)
                elif mag_calibration_factor != 1.0:
                    cal = raw * mag_calibration_factor
                else:
                    cal = raw
                batch_predictions.extend(cal.unbind(0))

        else:
            # Per-sample fallback: required when field scaling or physics refinement is active.
            for image_index, image in enumerate(images):
                outputs = model(image.unsqueeze(0).to(device))
                # AttentionDipoleModel returns (moment, direction, log_mag); take the moment vector.
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                prediction_vector = outputs.cpu().squeeze(0)
                scale_factor = 1.0
                if scaling_config.enabled:
                    scaled_prediction, scale_factor = _apply_field_scaling_to_prediction(
                        prediction_vector.numpy(),
                        raw_images[image_index],
                        mapheight=float(mapheight),
                        config=scaling_config,
                    )
                    prediction_vector = torch_module.as_tensor(
                        scaled_prediction,
                        dtype=prediction_vector.dtype,
                    )
                field_scaling_factors.append(float(scale_factor))
                # Physics refinement: minimize ||B_dipole(m) - B_obs||² w.r.t. m.
                if refine_fn is not None and r_grid is not None:
                    B_obs = torch_module.from_numpy(raw_images[image_index]).float().permute(2, 0, 1)
                    with torch_module.enable_grad():
                        refined, rel_residual = refine_fn(
                            prediction_vector.to(device),
                            B_obs,
                            r_grid,
                            n_steps=physics_refine_config.n_steps,
                            lr=physics_refine_config.lr,
                        )
                    if rel_residual <= physics_refine_config.residual_threshold:
                        prediction_vector = refined.cpu()
                    else:
                        logging.debug(
                            "Physics refinement rejected for sample %d (rel_residual=%.3f > %.3f); "
                            "keeping NN prediction.",
                            image_index, rel_residual, physics_refine_config.residual_threshold,
                        )
                if _should_log_prediction(log_each_prediction, image_index, LOGGED_PREDICTION_LIMIT):
                    logging.info(
                        "Prediction %d for %s (%s): prediction=%s | target=%s%s",
                        image_index, h5_file_path.name, target_representation,
                        prediction_vector.tolist(), m_values[image_index].tolist(),
                        f" | field_scale={scale_factor:.4f}" if scaling_config.enabled else "",
                    )
                elif log_each_prediction and image_index == LOGGED_PREDICTION_LIMIT:
                    remaining_predictions = max(total_images - LOGGED_PREDICTION_LIMIT, 0)
                    if remaining_predictions:
                        logging.info(
                            "Logging limited to the first %d predictions for %s; skipping %d additional items.",
                            LOGGED_PREDICTION_LIMIT, h5_file_path.name, remaining_predictions,
                        )
                batch_predictions_raw.append(prediction_vector.clone())
                if mag_calibration_isotonic is not None:
                    prediction_vector = _apply_isotonic_mag(prediction_vector, mag_calibration_isotonic)
                elif mag_calibration_factor != 1.0:
                    prediction_vector = prediction_vector * mag_calibration_factor
                batch_predictions.append(prediction_vector)
                bar()

    if not batch_predictions:
        raise ValueError(f"No images found in dataset {h5_file_path}.")

    predictions = torch_module.stack(batch_predictions, dim=0)
    predictions_raw = torch_module.stack(batch_predictions_raw, dim=0)
    stats, per_example_errors = _compute_error_statistics(predictions, m_values)
    field_scaling_array = None
    if scaling_config.enabled:
        np_module = _require_dependency(np, "numpy")
        field_scaling_array = np_module.asarray(field_scaling_factors, dtype=np.float64)
        stats.update(
            {
                "mean_field_scaling_factor": float(field_scaling_array.mean()),
                "median_field_scaling_factor": float(np_module.median(field_scaling_array)),
                "std_field_scaling_factor": float(field_scaling_array.std()),
            }
        )
    _save_errors(
        output_path,
        per_example_errors,
        stats,
        field_scaling_factors=field_scaling_array,
        field_scaling_config=scaling_config if scaling_config.enabled else None,
        sample_indices=sample_indices,
        predictions_raw=predictions_raw.numpy(),
    )
    return stats


def process_directory(
    model: "torch.nn.Module",
    directory: Path,
    device: "torch.device",
    show_progress: bool,
    prediction_suffix: str,
    log_each_prediction: bool,
    field_scaling_config: FieldScalingConfig | None = None,
    physics_refine_config: PhysicsRefineConfig | None = None,
    sample_indices: "Optional[Any]" = None,
    mag_calibration_factor: float = 1.0,
    mag_calibration_isotonic: "dict | None" = None,
    batch_size: int = 64,
) -> Optional[Tuple[Path, Dict[str, float]]]:
    """Process a dataset directory, returning the output path and stats on success."""

    logging.info("Processing directory: %s", directory)
    h5_file_path = directory / DATASET_FILENAME
    if not h5_file_path.exists():
        logging.warning("Dataset file not found: %s", h5_file_path)
        return None

    base_prediction = Path(PREDICTION_FILENAME)
    if prediction_suffix:
        output_name = f"{base_prediction.stem}{prediction_suffix}{base_prediction.suffix}"
    else:
        output_name = base_prediction.name
    output_path = directory / output_name
    logging.debug("Predictions will be written to %s", output_path)
    stats = predict_and_save(
        model,
        h5_file_path,
        output_path,
        device=device,
        show_progress=show_progress,
        log_each_prediction=log_each_prediction,
        field_scaling_config=field_scaling_config,
        physics_refine_config=physics_refine_config,
        sample_indices=sample_indices,
        mag_calibration_factor=mag_calibration_factor,
        mag_calibration_isotonic=mag_calibration_isotonic,
        batch_size=batch_size,
    )
    logging.info("Predictions and errors saved to %s", output_path)
    return output_path, stats


def _build_csv_path(model_path: Path, dataset_pattern: str, output_dir: Optional[Path]) -> Path:
    """Determine where to write the aggregated CSV file."""

    model_name = model_path.stem
    dataset_name = Path(dataset_pattern).name.replace("*", "X") or "datasets"
    if output_dir is not None:
        base_dir = output_dir.expanduser()
    else:
        base_dir = Path(dataset_pattern).expanduser().parent
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"sorted_errors_{model_name}_{dataset_name}.csv"


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    try:
        field_scaling_config = build_field_scaling_config(
            mode=args.field_scaling_mode,
            sigma=args.field_scaling_sigma,
            lower_percentile=args.field_scaling_lower_percentile,
            upper_percentile=args.field_scaling_upper_percentile,
            mapheight=args.field_scaling_mapheight,
            noise_floor_subtraction=args.field_scaling_noise_floor_subtraction,
        )
    except ValueError as exc:
        logging.error("Invalid field scaling configuration: %s", exc)
        return 1

    missing = _missing_dependencies(args.model_type) + missing_field_scaling_dependencies(field_scaling_config)
    if missing:
        logging.error("Missing required dependencies: %s", ", ".join(missing))
        logging.error("Install them via `pip install -r src/requirements.txt` or an equivalent command.")
        return 1

    torch_module = _require_dependency(torch, "torch")
    pd_module = _require_dependency(pd, "pandas")

    device = torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)
    try:
        model = load_model(args.model_path, device, args.model_type)
    except FileNotFoundError:
        logging.error("Model checkpoint not found at %s", args.model_path)
        return 1

    dataset_pattern = str(Path(args.dataset_pattern).expanduser())
    directories = []
    for match in glob.glob(dataset_pattern):
        path = Path(match)
        if path.is_dir():
            directories.append(path)
        else:
            logging.debug("Skipping non-directory glob match: %s", path)
    directories.sort()
    if not directories:
        logging.warning("No directories found matching %s", dataset_pattern)
        return 1

    summary_rows = []
    for directory in directories:
        result = process_directory(
            model,
            directory,
            device=device,
            show_progress=not args.no_progress,
            prediction_suffix=args.prediction_suffix,
            log_each_prediction=args.print_predictions,
            field_scaling_config=field_scaling_config,
            mag_calibration_factor=args.mag_calibration_factor,
        )
        if result is None:
            continue
        _, stats = result
        summary_rows.append({
            "directory": str(directory),
            "snr": extract_snr(directory),
            **stats,
        })

    if not summary_rows:
        logging.error("All dataset directories failed to process; no summary generated.")
        return 1

    summary_df = pd_module.DataFrame(summary_rows)
    summary_df["snr"] = pd_module.to_numeric(summary_df["snr"], errors="coerce")
    summary_df = summary_df.sort_values("snr", na_position="last").reset_index(drop=True)

    output_dir = args.output_dir.expanduser() if args.output_dir is not None else None

    if args.output_csv is not None:
        csv_path = args.output_csv.expanduser()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        csv_path = _build_csv_path(args.model_path.expanduser(), dataset_pattern, output_dir)

    summary_df.to_csv(csv_path, index=False)
    logging.info("Wrote aggregated summary to %s", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

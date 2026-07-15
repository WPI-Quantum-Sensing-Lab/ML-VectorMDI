import argparse
import configparser
import datetime
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
from PyQt5 import QtWidgets
from tqdm import tqdm


_SRC_DIR = Path(__file__).resolve().parents[2]
if __package__ in (None, "") and str(_SRC_DIR) not in sys.path:
    # Allow running this file directly as a script (outside `python -m`).
    sys.path.insert(0, str(_SRC_DIR))

from data_creation.utils.dataset_conversion import H5Dataset
import pyqtgraph as pg

from data_creation.utils.hdf5_storage import create_vecmagnet_datasets


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

#Argparsing
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default=None,
    help="Config file with a [snr_sort] section (default: configs/blend_noise.ini)",
)
parser.add_argument("--source_path", type=str)
parser.add_argument("--new_name", type=str)
parser.add_argument("--testing_bool", action="store_true")
parser.add_argument("--seed", type=int, help="Optional random seed for reproducible SNR generation")
parser.add_argument(
    "--size",
    type=int,
    default=None,
    help="Target number of samples to generate for each SNR dataset.",
)
parser.add_argument(
    "--disable_extra_channel_noise",
    action="store_true",
    help=(
        "Disable the 25%% chance of injecting an additional noisy Bx/By/Bz channel "
        "after the baseline SNR scaling."
    ),
)
parser.add_argument(
    "--blank-hdf5-path",
    type=Path,
    default=None,
    help=(
        "Path to an HDF5 archive containing blank sections stored under the Bx, "
        "By, and Bz datasets. When provided, the archive supplies all noise "
        "templates instead of loading individual .npy files."
    ),
)
parser.add_argument(
    "--blank-directory",
    type=Path,
    default=None,
    help=(
        "Directory containing individual blank .npy cubes. Used when an HDF5 "
        "archive is not supplied."
    ),
)
parser.add_argument(
    "--rotation-k-choices",
    type=str,
    default="1,2,3",
    help=(
        "Comma-separated np.rot90 k values sampled for the genuine z-axis blank "
        "rotation applied to --testing_bool datasets (default '1,2,3' = "
        "90/180/270 deg). Use '2' for a 180-deg-only test set, which maps each "
        "field component to +/-itself and so preserves channel-bound noise "
        "statistics."
    ),
)
args, _ = parser.parse_known_args()

# Lazily created global random generator reused across the module. It avoids the
# repeated global state manipulation in ``numpy.random`` which can become a
# bottleneck when sampling millions of indices.
_GLOBAL_RANDOM_GENERATOR = (
    np.random.default_rng(int(args.seed)) if args.seed is not None else np.random.default_rng()
)


def build_snr_dataset_directory_name(name_prefix: str, snr_value: float) -> str:
    """Return a dataset directory name for a given SNR value."""

    if name_prefix.endswith(("_", "-", os.sep, "/", "\\")):
        return f"{name_prefix}{snr_value}"
    return f"{name_prefix}_{snr_value}"


# Define constants
config = configparser.ConfigParser()
config.read(args.config if args.config else _SRC_DIR.parent / "configs" / "blend_noise.ini")
map_source_path = args.source_path if args.source_path is not None else config["snr_sort"]["path"]
dataset_name_prefix = args.new_name if args.new_name is not None else config["snr_sort"]["name"]
test_dataset = args.testing_bool
rotation_k_choices: tuple[int, ...] = tuple(
    int(k) for k in args.rotation_k_choices.split(",") if k.strip()
)
if not rotation_k_choices or any(k % 4 == 0 for k in rotation_k_choices):
    raise ValueError(
        f"--rotation-k-choices must be non-empty values in {{1,2,3}} (mod 4); "
        f"got {args.rotation_k_choices!r}"
    )
use_gaussian_noise = False
RELATIVE_SNR_TOLERANCE = 0.005  # 0.5% relative tolerance on per-component SNR
DEFAULT_EXTRA_CHANNEL_NOISE_PROBABILITY = 0.25
DEFAULT_EXTRA_CHANNEL_NOISE_MULTIPLIERS: tuple[int, ...] = tuple(range(5, 51, 5))
extra_channel_noise_probability = (
    0.0 if args.disable_extra_channel_noise else DEFAULT_EXTRA_CHANNEL_NOISE_PROBABILITY
)

config_blank_directory_value = config["snr_sort"].get("blank_directory", "").strip()
config_blank_hdf5_value = config["snr_sort"].get("blank_hdf5_path", "").strip()

repository_root = Path(__file__).resolve().parents[3]
user_home = Path.home()

_DEFAULT_BLANK_DIRECTORY_CANDIDATES = [
    Path(config_blank_directory_value) if config_blank_directory_value else None,
    repository_root / "data" / "blanks" / "250408" / "sections",
    user_home / "GitHub" / "VecMagNet" / "data" / "blanks" / "250408" / "sections",
]
_DEFAULT_BLANK_DIRECTORY = next(
    (
        candidate
        for candidate in _DEFAULT_BLANK_DIRECTORY_CANDIDATES
        if candidate is not None and candidate.exists()
    ),
    _DEFAULT_BLANK_DIRECTORY_CANDIDATES[0]
    or _DEFAULT_BLANK_DIRECTORY_CANDIDATES[1],
)

_DEFAULT_BLANK_HDF5_CANDIDATES = [
    Path(config_blank_hdf5_value) if config_blank_hdf5_value else None,
    repository_root / "data" / "blanks" / "250408" / "combined_hdf5" / "blanks.h5",
    repository_root / "data" / "blanks" / "blank_sections.h5",
    user_home / "GitHub" / "VecMagNet" / "data" / "blanks" / "250408" / "combined_hdf5" / "blanks.h5",
]
_DEFAULT_BLANK_HDF5_PATH = next(
    (
        candidate
        for candidate in _DEFAULT_BLANK_HDF5_CANDIDATES
        if candidate is not None and candidate.exists()
    ),
    None,
)

selected_blank_hdf5_path = args.blank_hdf5_path or _DEFAULT_BLANK_HDF5_PATH
selected_blank_directory = args.blank_directory or _DEFAULT_BLANK_DIRECTORY


logging.info("Config loaded successfully.")

data_size = (
    int(args.size)
    if args.size is not None
    else config.getint("snr_sort", "size", fallback=30000)
)
# if there are desired snrs in config, use them
if "desired_snrs" in config["snr_sort"]:
    desired_snr_values = config["snr_sort"]["desired_snrs"]
    desired_snr_values = desired_snr_values.split(",")
    desired_snr_values = [int(s) for s in desired_snr_values]
else:
    desired_snr_values = []
    x = 0.6
    for i in range(6):
        desired_snr_values.append(x)
        if i < 2:
            x -= 0.1  # Subtracting 0.1 for the first two values
        else:
            x /= 2  # Halving the value for subsequent numbers
    for i in range(0, 20, 2):
        desired_snr_values.append(i)
    desired_snr_values.append(1)
    desired_snr_values.append(20)
    desired_snr_values = set(desired_snr_values)
    if 0 in desired_snr_values:
        desired_snr_values.remove(0)
    # Effective if untouched, [0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 1, 2, 4, 6, 8 10, 12, 14, 16, 18, 20]
desired_snr_values = sorted(desired_snr_values)
logging.info(f"Desired SNRs: {desired_snr_values}")

_MIN_NOISE_STD = 1e-12  # Prevent division by zero when noise is almost zero (Tesla)
_IMAGE_SHAPE = (100, 100, 3)
_BYTES_PER_FLOAT32 = np.dtype(np.float32).itemsize
_BYTES_PER_IMAGE_SAMPLE = int(np.prod(_IMAGE_SHAPE)) * _BYTES_PER_FLOAT32
_BYTES_PER_VECTOR_SAMPLE = 3 * _BYTES_PER_FLOAT32
_BYTES_PER_DATASET_SAMPLE = _BYTES_PER_IMAGE_SAMPLE + (2 * _BYTES_PER_VECTOR_SAMPLE)
_DISK_SPACE_SAFETY_MARGIN_BYTES = 256 * 1024 * 1024
_DISK_SPACE_SAFETY_MULTIPLIER = 1.05


def load_blank_noise_cubes_from_hdf5(
    blank_sections_hdf5_path: Path,
) -> tuple[list[str], list[np.ndarray]]:
    """Load blank noise cubes from an HDF5 archive exposing Bx, By, and Bz datasets."""

    with h5py.File(blank_sections_hdf5_path, "r") as blank_file:
        try:
            bx_stack = blank_file["Bx"][:]
            by_stack = blank_file["By"][:]
            bz_stack = blank_file["Bz"][:]
        except KeyError as error:
            raise KeyError(
                "Blank HDF5 archives must contain 'Bx', 'By', and 'Bz' datasets."
            ) from error

    if not (bx_stack.shape == by_stack.shape == bz_stack.shape):
        raise ValueError(
            "Blank component stacks must share identical shapes inside the HDF5 archive."
        )

    blank_stack = np.stack((bx_stack, by_stack, bz_stack), axis=-1).astype(
        np.float32, copy=False
    )
    blank_identifiers = [
        f"{blank_sections_hdf5_path.as_posix()}::index_{index}"
        for index in range(blank_stack.shape[0])
    ]
    blank_cubes = [blank_stack[index] for index in range(blank_stack.shape[0])]
    return blank_identifiers, blank_cubes


def load_blank_noise_cubes_from_directory(
    blank_directory_path: Path,
) -> tuple[list[str], list[np.ndarray]]:
    """Load blank noise cubes from individual ``.npy`` files."""

    if not blank_directory_path.exists():
        raise FileNotFoundError(
            f"Blank directory does not exist: {blank_directory_path.as_posix()}"
        )

    blank_paths = sorted(blank_directory_path.glob("*.npy"))
    if not blank_paths:
        raise FileNotFoundError(
            f"No blank .npy files found under {blank_directory_path.as_posix()}"
        )

    blank_cubes = [np.load(blank_path) for blank_path in blank_paths]
    blank_identifiers = [path.as_posix() for path in blank_paths]
    return blank_identifiers, blank_cubes


def load_blank_noise_sources(
    blank_sections_hdf5_path: Path | None,
    blank_directory_path: Path | None,
) -> tuple[list[str], list[np.ndarray]]:
    """Return blank noise cubes sourced from an HDF5 archive or fallback directory."""

    if blank_sections_hdf5_path is not None:
        if blank_sections_hdf5_path.exists():
            logging.info(
                "Loading blank sections from HDF5 archive at %s.",
                blank_sections_hdf5_path.as_posix(),
            )
            return load_blank_noise_cubes_from_hdf5(blank_sections_hdf5_path)
        logging.warning(
            "Requested blank HDF5 archive %s was not found. Falling back to directory blanks.",
            blank_sections_hdf5_path.as_posix(),
        )

    if blank_directory_path is not None:
        logging.info(
            "Loading blank sections from directory %s.",
            blank_directory_path.as_posix(),
        )
        return load_blank_noise_cubes_from_directory(blank_directory_path)

    raise FileNotFoundError(
        "No blank noise source provided. Supply --blank-hdf5-path or --blank-directory."
    )


def apply_root_attrs(hf: h5py.File, attrs: dict, overwrite: bool = False) -> None:
    """Write metadata to HDF5 root attributes."""

    if not attrs:
        return
    for key, value in attrs.items():
        if value is None:
            continue
        if not overwrite and key in hf.attrs:
            continue
        hf.attrs[key] = value


def load_source_metadata(source_dataset_path: Path) -> dict:
    """Pull reproducibility metadata from the source dataset, if available."""

    meta = {"source_dataset": str(source_dataset_path)}
    h5_path = source_dataset_path / "dataset.h5"
    if not h5_path.exists():
        return meta
    with h5py.File(h5_path, "r") as hf:
        for key in (
            "scale",
            "order",
            "centering",
            "distancing",
            "distancing_parameter",
            "standoff",
            "mapheight",
            "dipoles",
            "seed",
        ):
            if key in hf.attrs:
                meta[key] = hf.attrs[key]
    return meta


def build_snr_metadata(
    *,
    base_meta: dict,
    snr_value: float,
    seed: int | None,
    blank_source: Path | None,
    extra_channel_noise_probability: float,
    extra_channel_noise_multiplier_choices: Sequence[int],
) -> dict:
    meta = dict(base_meta) if base_meta else {}
    meta["snr"] = float(snr_value)
    if "seed" not in meta and seed is not None:
        meta["seed"] = int(seed)
    if seed is not None:
        meta["snr_seed"] = int(seed)
    if blank_source is not None:
        meta["blank_source"] = str(blank_source)
    meta["extra_channel_noise_probability"] = float(extra_channel_noise_probability)
    meta["extra_channel_noise_multiplier_choices"] = list(extra_channel_noise_multiplier_choices)
    return meta


def maybe_apply_extra_channel_noise(
    noise: np.ndarray,
    *,
    probability: float,
    multiplier_choices: Sequence[int] = DEFAULT_EXTRA_CHANNEL_NOISE_MULTIPLIERS,
    random_number_generator: np.random.Generator = _GLOBAL_RANDOM_GENERATOR,
) -> tuple[np.ndarray, int | None, int | None]:
    """Randomly up-scale a single component's noise standard deviation.

    Parameters
    ----------
    noise
        Noise cube with Bx/By/Bz components.
    probability
        Bernoulli probability of enabling the extra noisy channel.
    multiplier_choices
        Discrete noise multipliers to sample when the event triggers.

    Returns
    -------
    noise
        The (possibly) modified noise cube. The input is mutated in place.
    int | None
        Index of the noisy component when applied, otherwise ``None``.
    int | None
        Multiplier sampled for the noisy component when applied, otherwise ``None``.
    """

    if probability <= 0.0:
        return noise, None, None

    trigger_value = random_number_generator.random()
    if trigger_value >= probability:
        return noise, None, None

    if noise.ndim != 3 or noise.shape[2] < 3:
        raise ValueError(
            "Noise array must have shape (H, W, >=3) to align with Bx/By/Bz components."
        )

    noisy_channel_index = int(random_number_generator.integers(0, noise.shape[2]))
    noise_multiplier = int(random_number_generator.choice(multiplier_choices))
    noise[:, :, noisy_channel_index] *= noise_multiplier
    return noise, noisy_channel_index, noise_multiplier


def calculate_per_component_snr(noise_volume: np.ndarray, signal_volume: np.ndarray) -> np.ndarray:
    """Return the SNR (σ_signal/σ_noise, dimensionless) for each vector component."""

    if noise_volume.shape != signal_volume.shape:
        raise ValueError(
            "Noise and signal volumes must share the same shape to compute per-component SNR."
        )

    flattened_noise = noise_volume.reshape(-1, noise_volume.shape[-1])
    flattened_signal = signal_volume.reshape(-1, signal_volume.shape[-1])

    noise_std = np.std(flattened_noise, axis=0)
    signal_std = np.std(flattened_signal, axis=0)

    safe_noise_std = np.maximum(noise_std, _MIN_NOISE_STD)
    return signal_std / safe_noise_std  # dimensionless ratio per component


def calculate_per_component_snr_from_cached_stats(
    *,
    signal_component_standard_deviation: np.ndarray,
    noise_component_standard_deviation: np.ndarray,
    noise_scale: float,
) -> np.ndarray:
    """Return per-component SNR using cached standard deviations."""

    scaled_noise_std = np.maximum(
        noise_component_standard_deviation * noise_scale, _MIN_NOISE_STD
    )
    safe_signal_std = np.maximum(signal_component_standard_deviation, _MIN_NOISE_STD)
    return safe_signal_std / scaled_noise_std


def rotate_blank_about_z(blank: np.ndarray, k: int) -> np.ndarray:
    """Apply a genuine z-axis rotation of ``90 * k`` degrees to a blank noise cube.

    ``blank`` is a ``(H, W, 3)`` array whose last axis holds the ``(Bx, By, Bz)``
    field components. A true rotation of the sample about the out-of-plane z-axis
    rotates *both* the spatial ``(H, W)`` grid **and** the in-plane ``(Bx, By)``
    vector components by the same angle; ``Bz`` is unchanged. Rotating only the
    pixels (a bare ``np.rot90``) leaves the component values pointing the old way,
    which is not a valid vector-field rotation.

    The component update matching ``np.rot90(..., axes=(0, 1))`` is
    ``(Bx, By) -> (By, -Bx)`` per 90 deg step (verified so that an azimuthal
    swirl field is preserved rather than reflected). Only the blank noise is
    rotated here; the dipole signal and the m/r labels are intentionally left
    untouched, so this is a noise-orientation augmentation, not a rotation of the
    underlying dipole measurement.
    """

    k = int(k) % 4
    if k == 0:
        return np.array(blank, copy=True)

    rotated = np.rot90(blank, k=k, axes=(0, 1))
    bx = rotated[..., 0].copy()
    by = rotated[..., 1].copy()
    # k applications of the 90-degree in-plane rotation (Bx, By) -> (By, -Bx).
    if k == 1:
        new_bx, new_by = by, -bx
    elif k == 2:
        new_bx, new_by = -bx, -by
    else:  # k == 3
        new_bx, new_by = -by, bx

    out = np.array(rotated, copy=True)
    out[..., 0] = new_bx
    out[..., 1] = new_by
    return out


def format_bytes(byte_count: int) -> str:
    """Return a human-readable byte count."""

    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(byte_count)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{byte_count} B"


def get_available_disk_space_bytes(target_path: Path) -> int:
    """Return free bytes on the filesystem that backs ``target_path``."""

    probe_path = target_path.resolve()
    while not probe_path.exists():
        if probe_path.parent == probe_path:
            break
        probe_path = probe_path.parent
    return shutil.disk_usage(probe_path).free


def estimate_required_dataset_bytes(sample_count: int) -> int:
    """Estimate bytes required to store the provided number of dataset samples."""

    return int(sample_count * _BYTES_PER_DATASET_SAMPLE)


def get_dataset_row_count(dataset_local_path: Path) -> int:
    """Return the current number of image samples stored in ``dataset.h5``."""

    if not dataset_local_path.exists():
        return 0

    with h5py.File(dataset_local_path, "r") as hf:
        if "I" not in hf:
            raise KeyError(f"Dataset file is missing the 'I' dataset: {dataset_local_path}")
        return len(hf["I"])


def survey_target_dataset_counts(
    *,
    dataset_name_prefix: str,
    snr_values: Sequence[float],
    target_size: int,
) -> tuple[dict[float, int], list[float]]:
    """Inspect destination datasets and return current counts plus unfinished SNRs."""

    snr_image_counts: dict[float, int] = {}
    pending_snr_values: list[float] = []

    for snr_value in snr_values:
        dataset_directory_name = build_snr_dataset_directory_name(
            dataset_name_prefix,
            snr_value,
        )
        dataset_local_path = Path(dataset_directory_name) / "dataset.h5"
        current_image_count = get_dataset_row_count(dataset_local_path)
        snr_image_counts[snr_value] = current_image_count
        if current_image_count < target_size:
            pending_snr_values.append(snr_value)

    return snr_image_counts, pending_snr_values


def ensure_disk_capacity_for_samples(
    *,
    target_path: Path,
    sample_count: int,
    context: str,
) -> None:
    """Raise a clear error before writes when the target drive is nearly full."""

    if sample_count <= 0:
        return

    required_bytes = estimate_required_dataset_bytes(sample_count)
    required_with_margin = int(
        required_bytes * _DISK_SPACE_SAFETY_MULTIPLIER
    ) + _DISK_SPACE_SAFETY_MARGIN_BYTES
    available_bytes = get_available_disk_space_bytes(target_path)
    if available_bytes >= required_with_margin:
        return

    raise RuntimeError(
        f"Insufficient disk space for {context}. Target path: {target_path}. "
        f"Available: {format_bytes(available_bytes)}. "
        f"Estimated required: {format_bytes(required_bytes)} plus a "
        f"{format_bytes(_DISK_SPACE_SAFETY_MARGIN_BYTES)} safety buffer."
    )


# helper class that manages and tracks the directories of all files related to a dataset
class DirectoryManager:
    """
    Manages directories for datasets categorized by SNR values and provides visualization of data distribution.


    Attributes:
    - snr_image_counts: Dictionary mapping SNR values to the number of images in each dataset.
    - snr_loggers: Dictionary mapping SNR values to their respective loggers.
    - app: The main Qt application instance.
    - win: The main window for displaying the histogram.
    - plot: The plot area for the histogram.
    - bar_graph: The bar graph item for the histogram.
    - estimate_label: Label item for displaying estimated completion time.
    - directory_list: List of SNR values sorted in descending order.
    Methods:
    - refresh(): Refreshes the dataset information by checking the size of each dataset stored in HDF5 files.
    - calculate_average_images(): Calculates the average number of images across all datasets.
    - create_snr_directory(snr_directory): Ensures the dataset structure exists and initializes the HDF5 storage.
    - update_histogram(): Updates the bar chart visualization of the dataset sizes per SNR value.
    - update_estimate_label(start_time, progress): Updates the estimated time to completion based on processing progress.
    - visualize_data_distribution(): Runs the PyQt event loop to display the histogram.
    Note:
    - The class uses PyQt5 and pyqtgraph for visualization.
    - The datasets are assumed to be stored in HDF5 files named 'dataset.h5' within directories named according to their SNR values.
    - The SNR values are expected to be logarithmically spaced for better visualization.
    - The class initializes the dataset directories if they do not exist and creates empty HDF5 files with predefined datasets.
    - The class provides real-time updates to the histogram and estimated completion time during data processing.

    """    
    def __init__(self):
        self.snr_image_counts = {}  # Stores the number of images per SNR dataset
        self.snr_loggers = {}  # Stores loggers for each dataset
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(show=True, title="SNR Data Distribution")
        self.win.resize(800, 600)
        self.win.setWindowTitle("Data Distribution by SNR")
        self.plot = self.win.addPlot(title="Number of Images per SNR Value", logX=True)
        self.bar_graph = None
        self.estimate_label = pg.LabelItem(justify="right")
        self.win.addItem(self.estimate_label, row=1, col=0)

        # Process each SNR dataset
        for snr in desired_snr_values:
            dataset_directory_name = build_snr_dataset_directory_name(
                dataset_name_prefix,
                snr,
            )
            snr_directory = Path("../data/datasets") / dataset_directory_name
            self.create_snr_directory(snr_directory)

            logging.info(f"Initialized dataset at {snr_directory}")

            # Read dataset size from the HDF5 file
            h5_path = snr_directory / "dataset.h5"
            if h5_path.exists():
                with h5py.File(h5_path, "r") as hf:
                    self.snr_image_counts[snr] = len(hf["I"])
            else:
                self.snr_image_counts[snr] = 0

        # Initialize dataset list and visualization
        self.directory_list = sorted(self.snr_image_counts.keys(), reverse=True)
        self.update_histogram()

    def refresh(self):
        """
        Refreshes the dataset information by checking the size of each dataset stored in HDF5 files.
        """
        for snr in self.snr_image_counts.keys():
            dataset_directory_name = build_snr_dataset_directory_name(
                dataset_name_prefix,
                snr,
            )
            h5_path = Path("../data/datasets") / dataset_directory_name / "dataset.h5"
            if h5_path.exists():
                with h5py.File(h5_path, "r") as hf:
                    self.snr_image_counts[snr] = len(hf["I"])
            else:
                self.snr_image_counts[snr] = 0

        self.directory_list = sorted(self.snr_image_counts.keys(), reverse=True)
        logging.info("Directory information refreshed.")
        self.update_histogram()
 
    def calculate_average_images(self):
        """
        Calculates the average number of images across all datasets.
        """
        avg = np.mean(list(self.snr_image_counts.values()))
        logging.info(f"Average number of images across datasets: {avg}")
        return avg

    def create_snr_directory(self, snr_directory):
        """
        Ensures the dataset structure exists and initializes the HDF5 storage.
        """
        snr_directory.mkdir(parents=True, exist_ok=True)

        # Ensure the HDF5 file exists
        h5_path = snr_directory / "dataset.h5"
        if not h5_path.exists():
            with h5py.File(h5_path, "w") as hf:
                create_vecmagnet_datasets(hf, image_shape=(0, 100, 100, 3))
            logging.info(f"Initialized new HDF5 dataset at {h5_path}")

    def update_histogram(self):
        """
        Updates the bar chart visualization of the dataset sizes per SNR value.
        """
        snr_keys = list(self.snr_image_counts.keys())
        snr_values = list(self.snr_image_counts.values())

        if self.bar_graph is not None:
            self.plot.removeItem(self.bar_graph)

        self.bar_graph = pg.BarGraphItem(x=snr_keys, height=snr_values, width=0.6, brush="b")
        self.plot.addItem(self.bar_graph)
        QtWidgets.QApplication.processEvents()

    def update_estimate_label(self, start_time, progress):
        """
        Updates the estimated time to completion based on processing progress.
        """
        elapsed_time = time.time() - start_time
        if progress > 0:
            estimated_total_time = elapsed_time / progress
            remaining_time = estimated_total_time - elapsed_time
            eta = datetime.datetime.now() + datetime.timedelta(seconds=remaining_time)
            self.estimate_label.setText(f"Estimated completion time: {eta.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            self.estimate_label.setText("Calculating estimated completion time...")
        QtWidgets.QApplication.processEvents()

    def visualize_data_distribution(self):
        """
        Runs the PyQt event loop to display the histogram.
        """
        QtWidgets.QApplication.instance().exec_()


class NoiseGenerator:
    def __init__(
        self,
        blank_image_paths,
        h5_data_path,
        *,
        blank_images=None,
        dipole_images=None,
        mnets=None,
        r0s=None,
        extra_channel_noise_probability: float = extra_channel_noise_probability,
        extra_channel_noise_multiplier_choices: Sequence[int] = DEFAULT_EXTRA_CHANNEL_NOISE_MULTIPLIERS,
    ):
        """Create a noise generator with cached source data.

        Parameters
        ----------
        blank_image_paths
            Iterable of paths to blank images used for noise injection.
        h5_data_path
            Path to the source HDF5 dataset. Only used when cached data
            are not provided.
        blank_images, dipole_images, mnets, r0s
            Optional preloaded arrays used to avoid repeatedly reading
            from disk. When omitted the data are loaded on demand.
        """

        self.blank_image_paths = list(blank_image_paths)

        if blank_images is None:
            # Load blanks lazily only when cache not provided
            self.blank_images = [np.load(path) for path in self.blank_image_paths]
        else:
            # Reuse supplied blank cache without c2opying data unnecessarily
            self.blank_images = [np.array(img, copy=False) for img in blank_images]

        if dipole_images is None or mnets is None or r0s is None:
            h5_data = H5Dataset(h5_data_path)
            self.dipole_images = h5_data.get_data("I")[:]  # Read once when no cache
            self.mnets = h5_data.get_data("m")[:]  # Same for m vectors
            self.r0s = h5_data.get_data("r")[:]  # And for r vectors
        else:
            self.dipole_images = dipole_images  # Accept cached dipole maps
            self.mnets = mnets  # Accept cached m vectors
            self.r0s = r0s  # Accept cached r vectors

        self.extra_channel_noise_probability = extra_channel_noise_probability
        self.extra_channel_noise_multiplier_choices = tuple(
            extra_channel_noise_multiplier_choices
        )
        self.latest_extra_noise_metadata: dict[str, object] = {
            "applied": False,
            "noisy_channel_index": None,
            "noise_multiplier": None,
        }

        logging.info("NoiseGenerator initialized with cached data.")

    def inject(self, desired_snr):
        time = datetime.datetime.now()
        blank_index = np.random.randint(0, len(self.blank_images))
        blank_path = self.blank_image_paths[blank_index]
        blank = self.blank_images[blank_index].copy()
        blank_shape = blank.shape
        print(f"Blank shape: {blank_shape}")
        if use_gaussian_noise:
            blank = np.random.normal(np.mean(blank), np.std(blank), blank.shape)
        if test_dataset:
            # NOTE: inject() is currently unused; the live generation path is the
            # inline loop in main(). Kept in sync via the shared helper: rotate the
            # blank as a genuine z-axis rotation (spatial + in-plane components).
            blank = rotate_blank_about_z(blank, int(np.random.choice([1, 2, 3])))
        rand_idx = np.random.randint(0, len(self.dipole_images))
        original_dipole_map = self.dipole_images[rand_idx]
        dipole_map_projection = np.min(original_dipole_map, axis=2)
        shape_original = original_dipole_map.shape
        mnet = self.mnets[rand_idx]
        r0 = self.r0s[rand_idx]


        dipole_projection_scale = float(np.mean(dipole_map_projection))
        dipole_projection_scale_abs = abs(dipole_projection_scale)
        blank = blank * dipole_projection_scale_abs
        per_component_snr = calculate_per_component_snr(blank, original_dipole_map)
        logging.debug(
            "Initial per-component SNR prior to scaling: %s",
            per_component_snr,
        )
        if desired_snr == 0:
            desired_snr = 0.0001
        safe_component_snr = np.maximum(per_component_snr, _MIN_NOISE_STD)
        broadcast_shape = (1,) * (blank.ndim - 1) + (blank.shape[-1],)
        scale_factors = desired_snr / safe_component_snr

        scale_blank = blank / scale_factors.reshape(broadcast_shape)
        scale_blank, noisy_channel_index, noise_multiplier = maybe_apply_extra_channel_noise(
            scale_blank,
            probability=self.extra_channel_noise_probability,
            multiplier_choices=self.extra_channel_noise_multiplier_choices,
        )
        self.latest_extra_noise_metadata = {
            "applied": noisy_channel_index is not None,
            "noisy_channel_index": noisy_channel_index,
            "noise_multiplier": noise_multiplier,
        }

        if noisy_channel_index is not None and noise_multiplier is not None:
            logging.debug(
                "Amplified noise channel %d by a factor of %d before blending.",
                noisy_channel_index,
                noise_multiplier,
            )

        actual_snr_components = calculate_per_component_snr(
            scale_blank, original_dipole_map
        )
        average_actual_snr = float(np.mean(actual_snr_components))
        logging.debug(
            "Actual per-component SNR after scaling: %s",
            actual_snr_components,
        )
        try:
            image = original_dipole_map + scale_blank
        except ValueError as e:
            assert shape_original == original_dipole_map.shape, f"Shape mismatch: {blank.shape} vs {original_dipole_map.shape}"
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(original_dipole_map, cmap='viridis')  # Using 'viridis' color scale
            axes[0].set_title(f"Original Dipole Map\nShape: {original_dipole_map.shape}")
            axes[0].axis("off")


            axes[1].imshow(scale_blank, cmap='viridis')  # Using 'viridis' color scale
            axes[1].set_title(
                f"Generated Image with mean SNR: {average_actual_snr:.2f}\nShape: {image.shape}"
            )
            axes[1].axis("off")

            plt.tight_layout()
            plt.show()
            logging.error(f"ValueError: {e}")

        info = [
            scale_blank,
            image,
            mnet,
            r0,
            blank_path,
            actual_snr_components,
            rand_idx,
        ]
        return info


    def calculate_per_component_snr(self, blank_volume, signal_volume):
        """Proxy to :func:`calculate_per_component_snr` for backward compatibility."""

        return calculate_per_component_snr(blank_volume, signal_volume)
    
        
def main():
    """
    Main function to generate and blend noisy images at specified SNR levels, storing them in HDF5 datasets.
    This function performs the following steps:
    1. Loads a source HDF5 dataset containing dipole images, mnet vectors, and r0 position vectors.
    2. Initializes a noise generator using blank images and source data.
    3. Iterates over a set of desired SNR values, generating noisy images for each SNR.
    4. For each SNR, creates or opens an HDF5 file (dataset.h5) to store images, mnet vectors, and r0 vectors.
       - The HDF5 file contains three datasets:
         - 'I': Images, shape (N, 100, 100, 3), dtype float32
         - 'm': mnet vectors, shape (N, 3), dtype float32
         - 'r': r0 position vectors, shape (N, 3), dtype float32
       # HDF5 is a hierarchical data format suitable for large, structured datasets.
    5. Generates noisy images by blending dipole maps with blank images, scaling noise to achieve the desired SNR.
    6. Checks if the generated image meets the SNR threshold, and if so, adds it to a batch for writing.
    7. Writes batches of images and vectors to the HDF5 file, resizing datasets as needed.
    8. Tracks progress using a tqdm progress bar and logs timing information.
    9. Repeats the process up to a maximum number of attempts, or until all datasets reach the required size.
    The function ensures efficient data generation and storage, preventing infinite loops and providing progress feedback.
    """
    source_dataset_path = Path(map_source_path)
    logging.info("Starting main process.")
    start_total = time.perf_counter()

    snr_image_counts, pending_snr_values = survey_target_dataset_counts(
        dataset_name_prefix=dataset_name_prefix,
        snr_values=desired_snr_values,
        target_size=data_size,
    )
    completed_snr_values = [
        snr for snr in desired_snr_values if snr not in set(pending_snr_values)
    ]
    if completed_snr_values:
        logging.info(
            "Skipping %d already-complete SNR datasets: %s",
            len(completed_snr_values),
            completed_snr_values,
        )
    if not pending_snr_values:
        logging.info(
            "All target SNR datasets already satisfy the data size limit of %d samples. Nothing to do.",
            data_size,
        )
        print("All requested SNR datasets already meet the size limit; skipping generation.")
        return

    logging.info(
        "Need to generate %d unfinished SNR datasets: %s",
        len(pending_snr_values),
        pending_snr_values,
    )

    # Load the source HDF5 dataset only after confirming work remains.
    h5_data = H5Dataset(source_dataset_path / "dataset.h5")
    dipole_images = h5_data.get_data("I")[:]  # Cache dipole maps once up front
    mnets = h5_data.get_data("m")[:]  # Cache m vectors once up front
    r0s = h5_data.get_data("r")[:]  # Cache r vectors once up front
    logging.info(
        "Cached source dataset in memory: %d images, %d m vectors, %d r vectors.",
        len(dipole_images),
        len(mnets),
        len(r0s),
    )

    blank_source_identifiers, cached_blank_arrays = load_blank_noise_sources(
        selected_blank_hdf5_path, selected_blank_directory
    )
    logging.info(
        "Prepared %d blank sections for noise injection.", len(cached_blank_arrays)
    )

    cached_blank_arrays = [
        np.array(blank, dtype=np.float32, copy=False) for blank in cached_blank_arrays
    ]

    if dipole_images.ndim != 4 or dipole_images.shape[-1] != 3:
        raise ValueError(
            "Dipole dataset must have shape (N, H, W, 3) for per-component statistics."
        )

    logging.info("Precomputing dipole statistics to speed up SNR scaling.")
    dipole_flat = dipole_images.reshape(dipole_images.shape[0], -1, dipole_images.shape[-1])
    dipole_component_standard_deviations = np.std(dipole_flat, axis=1)
    dipole_projection_means = np.mean(np.min(dipole_images, axis=3), axis=(1, 2))
    dipole_projection_scales = np.abs(dipole_projection_means)

    logging.info("Precomputing blank statistics to speed up SNR scaling.")
    blank_component_standard_deviations = np.array(
        [
            np.std(blank.reshape(-1, blank.shape[-1]), axis=0)
            for blank in cached_blank_arrays
        ],
        dtype=np.float32,
    )

    t0 = time.perf_counter()

    base_meta = load_source_metadata(source_dataset_path)
    blank_source = selected_blank_hdf5_path or selected_blank_directory

    # Initialize the noise generator with blank images and source data
    noise_generator = NoiseGenerator(
        blank_source_identifiers,
        h5_data_path=Path(map_source_path) / "dataset.h5",
        blank_images=cached_blank_arrays,
        dipole_images=dipole_images,
        mnets=mnets,
        r0s=r0s,
        extra_channel_noise_probability=extra_channel_noise_probability,
        extra_channel_noise_multiplier_choices=DEFAULT_EXTRA_CHANNEL_NOISE_MULTIPLIERS,
    )
    print(f"Init time: {time.perf_counter() - t0:.2f} s")
    # Initialize counters and progress bar for tracking processing progress
    processed_image_count = 0  # Total number of images processed across all SNR datasets

    # Create a tqdm progress bar to visualize progress for all SNR datasets
    # The total reflects only unfinished samples across unfinished SNR datasets.
    progress_bar = tqdm(
        total=sum(max(data_size - snr_image_counts[snr], 0) for snr in pending_snr_values)
    )

    # Set the maximum number of repeat attempts to fill each SNR dataset
    # This prevents infinite loops if some SNRs are hard to fill
    max_repeats = 10

    print(f"Setup time: {time.perf_counter() - t0:.2f} s")

    start_time = time.time()

    # Repeat the process to fill each SNR dataset up to the required size
    blank_buffer = np.empty_like(cached_blank_arrays[0])
    for rep in range(max_repeats):
        rep_start = time.perf_counter()
        progress_made = False

        # Iterate over each desired SNR value
        for desired_snr in pending_snr_values:
            # Start timing for this SNR value
            snr_start = time.perf_counter()
            logging.info(f"Processing SNR: {desired_snr}")

            # Construct the directory path for the current SNR dataset
            dataset_directory_name = build_snr_dataset_directory_name(
                dataset_name_prefix,
                desired_snr,
            )
            dset_directory = Path(dataset_directory_name)

            # Ensure the directory exists for storing the dataset
            os.makedirs(dset_directory, exist_ok=True)
            logging.info(f"Using dataset directory: {dset_directory}")

            dataset_local_path = dset_directory / "dataset.h5"
            meta = build_snr_metadata(
                base_meta=base_meta,
                snr_value=desired_snr,
                seed=args.seed,
                blank_source=blank_source,
                extra_channel_noise_probability=extra_channel_noise_probability,
                extra_channel_noise_multiplier_choices=DEFAULT_EXTRA_CHANNEL_NOISE_MULTIPLIERS,
            )

            # Create HDF5 file if it doesn't exist
            if not os.path.exists(dataset_local_path):
                with h5py.File(dataset_local_path, 'w') as hf:
                    create_vecmagnet_datasets(hf, image_shape=(0, 100, 100, 3), include_hash=False)

            with h5py.File(dataset_local_path, "a") as hf:
                apply_root_attrs(hf, meta)
                current_image_count = len(hf["I"])
                snr_image_counts[desired_snr] = current_image_count

                remaining_samples = max(data_size - current_image_count, 0)
                ensure_disk_capacity_for_samples(
                    target_path=dataset_local_path,
                    sample_count=remaining_samples,
                    context=(
                        f"SNR {desired_snr} dataset completion "
                        f"({remaining_samples} samples remaining)"
                    ),
                )

                # Skip if dataset already has enough images
                if current_image_count >= data_size:
                    logging.info(f"SNR {desired_snr} reached data size limit, skipping.")
                    print(f"SNR {desired_snr} skipped in {time.perf_counter() - snr_start:.2f} s")
                    continue

                batch_I, batch_m, batch_r = [], [], []
                BATCH_SIZE = 32
                REPORT_EVERY = 1000
                attempts = 0
                accepted = 0
                compute_time = 0.0
                write_time = 0.0
                last_report = time.perf_counter()

                # Generate and add images until the dataset reaches the required size
                while current_image_count < data_size:
                    attempts += 1
                    attempt_start = time.perf_counter()
                    rand_idx = _GLOBAL_RANDOM_GENERATOR.integers(0, len(dipole_images))
                    original_dipole_map = dipole_images[rand_idx]
                    mnet = mnets[rand_idx]
                    r0 = r0s[rand_idx]
                    dipole_projection_scale = dipole_projection_scales[rand_idx]
                    dipole_component_standard_deviation = dipole_component_standard_deviations[
                        rand_idx
                    ]

                    blank_idx = _GLOBAL_RANDOM_GENERATOR.integers(0, len(cached_blank_arrays))
                    base_blank = cached_blank_arrays[blank_idx]
                    blank_component_standard_deviation = blank_component_standard_deviations[
                        blank_idx
                    ]
                    if test_dataset:
                        # Test datasets get a genuine z-axis rotation of the noise
                        # blank only (spatial + in-plane components); the dipole
                        # signal and m/r labels are left untouched.
                        rotation_k = int(_GLOBAL_RANDOM_GENERATOR.choice(rotation_k_choices))
                        base_blank = rotate_blank_about_z(base_blank, rotation_k)
                        # The rotation mixes Bx/By, so the precomputed per-component
                        # std no longer matches this blank; recompute it so the SNR
                        # scaling below stays correct.
                        blank_component_standard_deviation = np.std(
                            base_blank.reshape(-1, base_blank.shape[-1]), axis=0
                        )
                    if use_gaussian_noise:
                        gaussian_mean = float(np.mean(base_blank))
                        gaussian_std = float(np.std(base_blank))
                        blank_buffer[:] = _GLOBAL_RANDOM_GENERATOR.normal(
                            gaussian_mean,
                            gaussian_std,
                            size=base_blank.shape,
                        )
                        blank_buffer *= dipole_projection_scale
                    else:
                        np.multiply(
                            base_blank,
                            dipole_projection_scale,
                            out=blank_buffer,
                            casting="unsafe",
                        )

                    if use_gaussian_noise:
                        per_component_snr = calculate_per_component_snr(
                            blank_buffer, original_dipole_map
                        )
                    else:
                        per_component_snr = calculate_per_component_snr_from_cached_stats(
                            signal_component_standard_deviation=dipole_component_standard_deviation,
                            noise_component_standard_deviation=blank_component_standard_deviation,
                            noise_scale=dipole_projection_scale,
                        )
                    broadcast_shape = (1,) * (blank_buffer.ndim - 1) + (blank_buffer.shape[-1],)
                    safe_per_component_snr = np.maximum(per_component_snr, _MIN_NOISE_STD)
                    scale_factors = np.maximum(
                        desired_snr / safe_per_component_snr, _MIN_NOISE_STD
                    )
                    np.divide(blank_buffer, scale_factors.reshape(broadcast_shape), out=blank_buffer)
                    (
                        scaled_blank,
                        noisy_channel_index,
                        noise_multiplier,
                    ) = maybe_apply_extra_channel_noise(
                        blank_buffer,
                        probability=extra_channel_noise_probability,
                        multiplier_choices=DEFAULT_EXTRA_CHANNEL_NOISE_MULTIPLIERS,
                    )
                    if noisy_channel_index is not None and noise_multiplier is not None:
                        logging.debug(
                            "Amplified noise channel %d by a factor of %d during dataset write.",
                            noisy_channel_index,
                            noise_multiplier,
                        )
                    image = original_dipole_map + scaled_blank

                    actual_snr_components = calculate_per_component_snr(
                        scaled_blank, original_dipole_map
                    )
                    component_error = np.abs(actual_snr_components - desired_snr)
                    threshold = max(desired_snr * RELATIVE_SNR_TOLERANCE, _MIN_NOISE_STD)
                    component_within_tolerance = component_error < threshold
                    if noisy_channel_index is not None:
                        component_within_tolerance[noisy_channel_index] = True

                    if np.all(component_within_tolerance):
                        batch_I.append(image.reshape(100, 100, 3))
                        batch_m.append(mnet)
                        batch_r.append(r0)
                        current_image_count += 1
                        snr_image_counts[desired_snr] = current_image_count
                        progress_made = True
                        progress_bar.update(1)
                        accepted += 1

                    compute_time += time.perf_counter() - attempt_start

                    # Write batch to HDF5 file when batch is full or dataset is complete
                    if len(batch_I) >= BATCH_SIZE or (current_image_count >= data_size and batch_I):
                        write_start = time.perf_counter()
                        ensure_disk_capacity_for_samples(
                            target_path=dataset_local_path,
                            sample_count=len(batch_I),
                            context=(
                                f"batched write for SNR {desired_snr} "
                                f"({len(batch_I)} samples)"
                            ),
                        )
                        try:
                            hf["I"].resize((hf["I"].shape[0] + len(batch_I)), axis=0)
                            hf["I"][-len(batch_I):] = np.stack(batch_I)
                            hf["m"].resize((hf["m"].shape[0] + len(batch_m)), axis=0)
                            hf["m"][-len(batch_m):] = np.stack(batch_m)
                            hf["r"].resize((hf["r"].shape[0] + len(batch_r)), axis=0)
                            hf["r"][-len(batch_r):] = np.stack(batch_r)
                        except OSError as error:
                            if getattr(error, "errno", None) == 28:
                                available_bytes = get_available_disk_space_bytes(
                                    dataset_local_path
                                )
                                raise RuntimeError(
                                    f"Disk filled up while writing SNR {desired_snr} "
                                    f"to {dataset_local_path}. Available space at failure: "
                                    f"{format_bytes(available_bytes)}. The HDF5 file may be "
                                    "partially written; free space and rerun this dataset."
                                ) from error
                            raise
                        batch_I, batch_m, batch_r = [], [], []
                        write_time += time.perf_counter() - write_start

                    if attempts % REPORT_EVERY == 0:
                        report_now = time.perf_counter()
                        report_elapsed = report_now - last_report
                        last_report = report_now
                        acceptance_rate = (accepted / attempts) * 100 if attempts else 0.0
                        attempt_rate = REPORT_EVERY / report_elapsed if report_elapsed > 0 else 0.0
                        avg_compute = compute_time / attempts if attempts else 0.0
                        avg_write = write_time / attempts if attempts else 0.0
                        logging.info(
                            "SNR %s stats: attempts=%d accepted=%d (%.2f%%) "
                            "attempts/sec=%.2f avg_compute=%.4fs/attempt avg_write=%.4fs/attempt",
                            desired_snr,
                            attempts,
                            accepted,
                            acceptance_rate,
                            attempt_rate,
                            avg_compute,
                            avg_write,
                        )

                if attempts:
                    acceptance_rate = (accepted / attempts) * 100
                    avg_compute = compute_time / attempts
                    avg_write = write_time / attempts
                    logging.info(
                        "SNR %s summary: attempts=%d accepted=%d (%.2f%%) "
                        "avg_compute=%.4fs/attempt avg_write=%.4fs/attempt total_write=%.2fs",
                        desired_snr,
                        attempts,
                        accepted,
                        acceptance_rate,
                        avg_compute,
                        avg_write,
                        write_time,
                    )

        # If no progress was made, exit early to avoid unnecessary repeats
        if not progress_made:
            logging.warning("No progress made in this iteration. Exiting loop to prevent unnecessary repeats.")
            break

        # Exit if all datasets have reached the required size
        if all(snr_image_counts[snr] >= data_size for snr in desired_snr_values):
            break

        print(f"Repeat {rep + 1} done in {time.perf_counter() - rep_start:.2f} s")

    progress_bar.close()
    logging.info("Main process completed.")
    print(f"Total runtime: {time.perf_counter() - start_total:.2f} s")

if __name__ == "__main__":
    main()

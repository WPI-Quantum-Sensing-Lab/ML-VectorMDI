import configparser
import datetime
import logging
import multiprocessing as mp
import shutil
import warnings
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import SeedSequence
import pandas as pd
import scipy.constants as const

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*pynvml package is deprecated.*",
)

import torch
import tqdm
from scipy.stats import poisson
import h5py
import hashlib
from scipy.io import savemat, loadmat

import sys

_SRC_DIR = Path(__file__).resolve().parents[2]
if __package__ in (None, "") and str(_SRC_DIR) not in sys.path:
    # Allow running this file directly as a script (outside `python -m`).
    sys.path.insert(0, str(_SRC_DIR))

from data_creation.utils.dataset_conversion import H5Dataset
from data_creation.utils.hdf5_storage import create_vecmagnet_datasets, image_dataset_kwargs, vector_dataset_kwargs
import argparse

# Define constants
DEFAULT_CONFIG_PATH = _SRC_DIR.parent / "configs" / "generate_data.ini"
config = configparser.ConfigParser()
config.read(DEFAULT_CONFIG_PATH)

parser = argparse.ArgumentParser(description="Magnetic Moment Dataset Generator")
parser.add_argument(
    "--config",
    type=str,
    help=f"Config file with a [map_generation] section (default: {DEFAULT_CONFIG_PATH})",
)
parser.add_argument("--distancing", type=str, help="Override distancing parameter from config")
parser.add_argument(
    "--distancing_parameter",
    type=float,
    help="Override distancing_parameter from config",
)
parser.add_argument("--dipoleLimit", type=int, help="Override dipole_Limit from config")
parser.add_argument("--dataset_name", type=str, help="Override dataset_name from config")
parser.add_argument("--centering", type=str, help="Override centering from config")
parser.add_argument("--seed", type=int, help="Optional random seed for reproducible generation")


def parse_cli_arguments(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Return parsed CLI arguments for dataset generation overrides."""

    return parser.parse_args(list(argv) if argv is not None else None)


def apply_config_overrides(
    *,
    distancing: Optional[str] = None,
    distancing_parameter: Optional[float] = None,
    dipole_limit: Optional[int] = None,
    dataset_name: Optional[str] = None,
    centering: Optional[str] = None,
) -> None:
    """Override config defaults when running the dataset generator programmatically."""

    if distancing is not None:
        config.set("map_generation", "distancing", distancing)
    if distancing_parameter is not None:
        config.set("map_generation", "distancing_parameter", str(distancing_parameter))
    if dipole_limit is not None:
        config.set("map_generation", "dipole_Limit", str(dipole_limit))
    if dataset_name is not None:
        config.set("map_generation", "dataset_name", dataset_name)
    if centering is not None:
        config.set("map_generation", "centering", str(centering))


def _pool_worker_seed_initializer(worker_seeds):
    """Initialize unique NumPy and torch seeds for each Pool worker."""
    worker = mp.current_process()
    worker_idx = worker._identity[0] - 1 if worker._identity else 0
    if worker_idx >= len(worker_seeds):
        worker_idx = len(worker_seeds) - 1

    np_seed, torch_seed = worker_seeds[worker_idx]
    np.random.seed(int(np_seed))
    torch_seed = int(torch_seed)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)


def main(**kwargs):
    if kwargs.get("distancing_parameter") is not None:
        config.set(
            "map_generation",
            "distancing_parameter",
            str(kwargs.get("distancing_parameter")),
        )
    if kwargs.get("centering") is not None:
        config.set("map_generation", "centering", str(kwargs.get("centering")))
    iniState = stateInitializer(config)
    scale = iniState.scale
    data_N = iniState.data_n
    dipole_Limit = kwargs.get("dipoleLimit") or config.getint("map_generation", "dipole_Limit")
    mapheight = iniState.mapheight
    order = iniState.order
    distancing_parameter = iniState.distancing_parameter
    distancing = kwargs.get("distancing") or config.get("map_generation", "distancing")
    centering = iniState.centering
    dataset_name = kwargs.get("name") or config.get("map_generation", "dataset_name")
    test = iniState.test
    complex_sort = iniState.complex_sort
    seed = kwargs.get("seed")

    if seed is not None:
        seed = int(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if test == "True":
        print("Test mode is on. Test will override 'test' dataset.")
        dataset_name = "test"
        test_dataset_path = Path("../data/datasets/test")
        if test_dataset_path.exists():
            print("Test dataset already exists. Overwriting.")
            for file in test_dataset_path.glob("**/*"):
                file.unlink()
        else:
            print("Test dataset does not exist. Creating.")

    dataset_name = (
        dataset_name + "_" + str(datetime.datetime.now().strftime("%Y-%m-%d"))
    )
    dataset_name = dataset_name.replace(" ", "_")
    dataset_name = dataset_name.replace(":", "-")

    if order != 0:
        dataset_name = dataset_name + "_order" + str(order)

    generator = Generator(
        dipole_Limit, scale, mapheight, order, distancing, centering, distancing_parameter, r_scale=1)

    path = Path("../data/datasets") / dataset_name
    path.mkdir(parents=True, exist_ok=True)
    # (path / "images").mkdir(exist_ok=True)
    # (path / "mnettargets").mkdir(exist_ok=True)
    # (path / "r0targets").mkdir(exist_ok=True)
    iniState.write_log(path / "stats.txt")


    meta = {
        "scale": int(scale),
        "order": int(order),
        "centering": bool(centering),
        "distancing": bool(distancing),
        "distancing_parameter": float(distancing_parameter),
        "standoff": float(mapheight),
        "mapheight": float(mapheight),
        "dipoles": int(dipole_Limit),
    }
    if seed is not None:
        meta["seed"] = seed

    existing_count = 0
    h5_path = path / "dataset.h5"
    if h5_path.exists():
        with h5py.File(h5_path, "r") as hf:
            if "m" in hf:
                existing_count = int(hf["m"].shape[0])
    remaining = max(0, data_N - existing_count)
    if existing_count:
        print(f"Resuming: {existing_count} samples already saved, generating {remaining} more.")
    if remaining == 0:
        print("Dataset already complete.")
        return dataset_name

    dataSaver = DataSaver(0, 0, path, meta=meta)
    data_pool = []
    num_workers = max(1, mp.cpu_count() // 2)
    base_seed_seq = SeedSequence(seed) if seed is not None else SeedSequence()
    worker_seeds = [
        tuple(int(s) for s in child_seq.generate_state(2, dtype=np.uint32))
        for child_seq in base_seed_seq.spawn(num_workers)
    ]

    with mp.Pool(
        num_workers,
        initializer=_pool_worker_seed_initializer,
        initargs=(worker_seeds,),
    ) as p:
        results = [
            p.apply_async(
                main_generate,
                args=(dataSaver, generator),
            )
            for i in range(remaining)
        ]
        for r in tqdm.tqdm(results):
            data_pool.append(r.get())

    print("Data generation complete. Saving.")
    if data_pool:
        I_batch = np.stack([e[0] for e in data_pool], axis=0).astype(np.float32)
        m_batch = np.stack([e[1] for e in data_pool], axis=0).astype(np.float32)
        r_batch = np.stack([e[2] for e in data_pool], axis=0).astype(np.float32)
        hashes  = np.array(
            [hashlib.md5(e[0].tobytes()).hexdigest() for e in data_pool], dtype="S32"
        )
        new_total = existing_count + len(data_pool)
        with h5py.File(path / "dataset.h5", "a") as hf:
            dataSaver.apply_root_attrs(hf)
            if "m" not in hf:
                create_vecmagnet_datasets(hf, image_shape=(0, 100, 100, 3))
            for name, arr in [("I", I_batch), ("m", m_batch), ("r", r_batch), ("hash", hashes)]:
                hf[name].resize((new_total,) + hf[name].shape[1:])
                hf[name][existing_count:new_total] = arr
        print(f"Saved {len(data_pool)} samples. Total in dataset: {new_total}")



    if iniState.equalizer_bools[0] == "True" and order != 0:
        equal_set = Equal(iniState.equalizer_bools[0], dataset_name)
        if iniState.equalizer_bools[1] == "True":
            remove_unequal_data(dataset_name)

    # Handle complexity sort if enabled
    if complex_sort:
        complexity_bins = {round(i * 0.1, 1): [] for i in range(1, 7)}  # Bins: 0.1-0.6
        filled_bins = set()  # To keep track of fully filled bins

        # Batch queue for files to be processed
        batch_queue = []
        max_attempts = 5000
        queue_size =  50

        for bin_key, _ in complexity_bins.items():

            entries = list(path.glob(f"complexity_{bin_key}/*"))

            if bin_key in filled_bins or len(entries) >= data_N:
                print(f"Skipping bin {bin_key}, as it is already filled.")
                if bin_key not in filled_bins:
                    filled_bins.add(bin_key)
                continue  # Skip already-filled bins

            print(f"Processing bin {bin_key}...")

            attempts = 0
            while attempts < max_attempts:
                entries = list(path.glob(f"complexity_{bin_key}/*"))
                print(f"Bin {bin_key} now has {len(entries)} entries.")
                # Check if the bin is filled after processing
                if len(entries) >= data_N:
                    filled_bins.add(bin_key)
                    print(f"Bin {bin_key} is now filled with {len(entries)} entries.")
                    break
                attempts += 1

                # Scale distributions based on the progress of the bin filling
                fill_ratio = len(filled_bins) + int(len(entries) / data_N)
                print(f"Fill ratio: {fill_ratio}")
                dipole_scale = 1 + fill_ratio * 0.5  # Adjust scale factor as needed
                distancing_scale = 0.001 + fill_ratio * 0.01  # Slightly increase with progress

                generator.dipole_Limit = poisson.rvs(dipole_scale, size=1)[0] + 1
                generator.distancing_parameter = np.random.uniform(0, 1) * distancing_scale + poisson.rvs(dipole_scale, size=1)[0] * 0.01


                # Generate new data and add files to the batch queue
                additional_data = generate_additional_data(dataSaver, generator, path, queue_size)
                batch_queue.extend(additional_data)

                print(f"Attempt {attempts}: Added {len(additional_data)} files to the queue. Queue size is now {len(batch_queue)}.")

                # If the queue is full, process it
                if len(batch_queue) >= queue_size:
                    print(f"Queue reached {queue_size}. Processing batch...")
                    process_images_in_batches(path, batch_size=queue_size, max_complexity=0.6, filled_bins=filled_bins, max_capacity = data_N)
                    print(f"Processed batch of {queue_size} files.")
                    batch_queue = []  # Reset the queue

            # Final processing of any remaining files in the queue
            if batch_queue:
                print(f"Processing final batch of {len(batch_queue)} files for bin {bin_key}...")
                process_images_in_batches(path, batch_size=len(batch_queue), max_complexity=0.6, filled_bins=filled_bins, max_capacity = data_N)
                print(f"Processed final batch of {len(batch_queue)} files.")
                batch_queue = []  # Clear the queue

            if attempts == max_attempts:
                print(f"Warning: Bin {bin_key} could not be filled after {max_attempts} attempts.")

        print("All bins processed.")
    return dataset_name


def process_images_in_batches(path, batch_size=2500, max_complexity=0.6, filled_bins=set(), max_capacity=100):
    """
    Process images in batches, calculate their complexities, and move them to appropriate bins.

    Args:
        path (Path): Path to the directory containing the images.
        batch_size (int): Number of images to process in each batch.
        max_complexity (float): Maximum complexity threshold for binning.
        log_file (str): Path to the log file that tracks processed files.
        filled_bins (set): Set of complexity bins that are already filled.
    """

    # Get all unprocessed file paths
    all_files = list(path.glob("images/*"))
    # Get the names of processed files
    processed_files = {f.stem for f in path.glob("matlab/*")}
    # print(f"Processed files {processed_files}")
    print(f"Found {len(processed_files)} processed files.")

    # Find files in 'images' that are not in 'matlab'
    unprocessed_files = [f for f in all_files if f.stem not in processed_files]
    # print(f"Unprocessed files {unprocessed_files}")
    print(f"Found {len(unprocessed_files)} unprocessed files.")
    # Process in batches
    for i in range(0, len(unprocessed_files), batch_size):
        batch_files = unprocessed_files[i:i + batch_size]

        # Calculate complexities for the batch
        complexity_df = calculate_complexity(batch_files)

        # Move files to appropriate bins
        for _, row in complexity_df.iterrows():
            file = Path(row["file"])
            complexity = row["complexity"]

            # Round complexity to the nearest bin (e.g., 0.1, 0.2, ...)
            bin_key = round(max(complexity, 0) + 0.05, 1)

            # Skip files with bin_key greater than max_complexity
            if bin_key > max_complexity:
                continue

            # Skip files for filled bins
            if bin_key in filled_bins:
                continue

            # Create the directory for the complexity bin
            bin_dir = path / f"complexity_{bin_key}"
            bin_dir.mkdir(parents=True, exist_ok=True)

            # Copy the file to the bin
            if len(list(bin_dir.glob("*.npy"))) >= batch_size:
                filled_bins.add(bin_key)
                print(f"Bin {bin_key} is now filled.")
                continue
            else:
                target_path = bin_dir / file.stem
                shutil.copy(file, target_path)

                processed_files.add(file.stem)

        print(f"Processed batch {i // batch_size + 1} of {len(unprocessed_files) // batch_size + 1}")

    print("Processing complete.")




def calculate_complexity(file_paths):
    """
    Calculate the complexity for a list of file paths using the make_guess_series function.

    Args:
        file_paths (list): List of file paths (str or Path).

    Returns:
        pd.DataFrame: DataFrame containing the file paths and their corresponding complexity values.
    """
    try:
        from fit_source import make_guess_series, process_py_file
        file_list = [str(file) for file in file_paths]
        original_file_list = file_list.copy()

        # Process files
        file_list = [process_py_file(file) for file in file_list]

        # Call make_guess_series without saving results
        results_df = make_guess_series(file_list, save_results=False, nRuns = 10)

        # Extract only file and complexity columns and explicitly create a copy
        complexity_df = results_df[["file", "complexity"]].copy()

        # Replace file column with original file paths
        complexity_df.loc[:, "file"] = original_file_list

        return complexity_df

    except Exception as e:
        logging.error(f"Error calculating complexity: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on failure



def generate_additional_data(dataSaver, generator, path, n=2500):
    """
    Generate additional data using the given generator.

    Returns:
        list: List of additional data.
    """
    additional_data = []
    for i in tqdm.tqdm(range(n), desc="Generating additional data"):
        data = main_generate(dataSaver, generator)
        additional_data.append(data)
    dataSaver.data = data
    with h5py.File(path / "dataset.h5", "a") as hf:
        if "m" in hf:
            num_entries = len(hf["m"])
        else:
            num_entries = 0
        print(f"Number of entries: {num_entries}")
    dataSaver.index = num_entries
    dataSaver.save_data()


    return additional_data

def main_generate(dataSaver, generator):
    """
    Generate data using the given generator and save it using the dataSaver.

    Args:
        dataSaver: An instance of the DataSaver class used to save the generated data.
        generator: An instance of the Generator class used to generate the data.
        i: An integer representing the current index.

    Returns:
        None
    """
    # Generate data
    data = generator.generateData()
    return data


def remove_unequal_data(dataset_name):
    """
    Remove unequal data subsets from the specified dataset and rename the equalized ones.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        None
    """
    # eliminate images, mnettargets, and r0targets subsets and then rename the equalized ones
    images_path = Path("../data/datasets") / dataset_name / "images"
    mnettargets_path = Path("../data/datasets") / dataset_name / "mnettargets"
    r0targets_path = Path("../data/datasets") / dataset_name / "r0targets"

    # Remove files and directories in images_path
    for file in images_path.glob("**/*"):
        file.unlink()
    images_path.rmdir()

    # Remove files and directories in mnettargets_path
    for file in mnettargets_path.glob("**/*"):
        file.unlink()
    mnettargets_path.rmdir()

    # Remove files and directories in r0targets_path
    for file in r0targets_path.glob("**/*"):
        file.unlink()
    r0targets_path.rmdir()

    # Rename equalized directories
    (images_path.parent / (images_path.name + "_equalized")).rename(images_path)
    (mnettargets_path.parent / (mnettargets_path.name + "_equalized")).rename(
        mnettargets_path
    )
    (r0targets_path.parent / (r0targets_path.name + "_equalized")).rename(
        r0targets_path
    )

    return


def var_dump(obj):
    print(obj)
    return


class autoBin:
    def __init__(self, entry_limit_n):
        self.entry_limit_n = entry_limit_n
        self.container = {}

    def add(self, entry, entry_index):
        if len(self.container) < self.entry_limit_n:
            self.container[entry_index] = entry


class Equal:
    """
    Class representing the equalization of data.

    Attributes:
    - equal_bool (bool): Flag indicating whether equalization should be performed.
    - use_hdf5 (bool): Whether to load/save from HDF5.
    - path (Path): Path to the dataset.
    - images, mnettargets, r0targets (list): Data lists if using .npy.
    """

    def __init__(self, equal_bool, path, use_hdf5=True):
        self.equal_bool = equal_bool
        self.use_hdf5 = use_hdf5
        self.data_path = Path("../data/datasets") / path
        self.images = []
        self.mnettargets = []
        self.r0targets = []

        # Load data based on format
        self.load_data()
        self.equalize_on_n()

    def load_data(self):
        """
        Load data from HDF5 if enabled, else from directories.
        """
        if self.use_hdf5:
            self.hdf5_dataset = H5Dataset(self.data_path / "dataset.h5")
            self.images = self.hdf5_dataset.get_data("I")
            self.mnettargets = self.hdf5_dataset.get_data("m")
            self.r0targets = self.hdf5_dataset.get_data("r")
        else:
            self.load_npy_data()

    def load_npy_data(self):
        """
        Load data from .npy files in the directory structure.
        """
        images_path = self.data_path / "images"
        mnettargets_path = self.data_path / "mnettargets"
        r0targets_path = self.data_path / "r0targets"

        for file in images_path.iterdir():
            self.images.append(np.load(file))
        for file in mnettargets_path.iterdir():
            self.mnettargets.append(np.load(file))
        for file in r0targets_path.iterdir():
            self.r0targets.append(np.load(file))

    def equalize_on_n(self):
        """
        Perform equalization on the target magnetic moments based on the norm.
        """
        if not self.equal_bool:
            return

        indices_to_keep = []
        n_log = {i: np.linalg.norm(m) for i, m in enumerate(self.mnettargets)}
        sorted_n_log = sorted(n_log.items(), key=lambda x: x[1])

        _, bins = np.histogram([n[1] for n in sorted_n_log], bins=50)
        avg_bin_count = len(self.mnettargets) / 50
        bin_bin = []

        for i in range(len(bins) - 1):
            auto_bin = []
            for idx, norm in sorted_n_log:
                if bins[i] <= norm < bins[i + 1]:
                    auto_bin.append(idx)
            bin_bin.append(auto_bin)

        for bin_list in bin_bin:
            indices_to_keep.extend(bin_list[:int(avg_bin_count)])

        print(f"Number of elements after equalization: {len(indices_to_keep)}")

        # Filter datasets
        self.images = [self.images[i] for i in indices_to_keep]
        self.mnettargets = [self.mnettargets[i] for i in indices_to_keep]
        self.r0targets = [self.r0targets[i] for i in indices_to_keep]

        # Save new dataset
        if self.use_hdf5:
            self.save_to_hdf5()
        else:
            self.save_to_npy()

    def save_to_npy(self):
        """
        Save the equalized dataset in .npy format.
        """
        eq_path = self.data_path.parent / f"{self.data_path.stem}_equalized"
        eq_path.mkdir(exist_ok=True)

        (eq_path / "images").mkdir(exist_ok=True)
        (eq_path / "mnettargets").mkdir(exist_ok=True)
        (eq_path / "r0targets").mkdir(exist_ok=True)

        for i in range(len(self.images)):
            np.save(eq_path / "images" / f"{i}.npy", self.images[i])
            np.save(eq_path / "mnettargets" / f"{i}.npy", self.mnettargets[i])
            np.save(eq_path / "r0targets" / f"{i}.npy", self.r0targets[i])

        print(f"Saved equalized dataset in .npy format at {eq_path}")

    def save_to_hdf5(self):
        """
        Save the equalized dataset in an HDF5 file.
        """
        h5_path = self.data_path.parent / f"{self.data_path.stem}_equalized.h5"
        image_array = np.array(self.images, dtype=np.float32)
        m_array = np.array(self.mnettargets, dtype=np.float32)
        r_array = np.array(self.r0targets, dtype=np.float32)
        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset(
                "m",
                data=m_array,
                dtype=np.float32,
                **vector_dataset_kwargs(shape=m_array.shape),
            )
            hf.create_dataset(
                "r",
                data=r_array,
                dtype=np.float32,
                **vector_dataset_kwargs(shape=r_array.shape),
            )
            hf.create_dataset(
                "I",
                data=image_array,
                dtype=np.float32,
                **image_dataset_kwargs(shape=image_array.shape),
            )

        print(f"Saved equalized dataset in HDF5 format at {h5_path}")

    def show_data(self):
        """
        Display histograms of the target magnetic moments.
        """
        x_log = [x[0] for x in self.mnettargets]
        plt.hist(x_log, bins=50, alpha=0.5, color="r")
        plt.title("Distribution of the x components of the target magnetic moments")
        plt.xlabel("Magnetic moment component (2Amu^2)")
        plt.show()

        y_log = [y[1] for y in self.mnettargets]
        plt.hist(y_log, bins=50, alpha=0.5, color="g")
        plt.title("Distribution of the y components of the target magnetic moments")
        plt.xlabel("Magnetic moment component (2Amu^2)")
        plt.show()

        z_log = [z[2] for z in self.mnettargets]
        plt.hist(z_log, bins=50, alpha=0.5, color="b")
        plt.title("Distribution of the z components of the target magnetic moments")
        plt.xlabel("Magnetic moment component (2Amu^2)")
        plt.show()

        net_log = [np.linalg.norm(net) for net in self.mnettargets]
        plt.hist(net_log, bins=50, alpha=0.5, color="b")
        plt.title("Distribution of the net magnetic moments")
        plt.xlabel("Magnetic moment component (2Amu^2)")
        plt.show()


class stateInitializer:
    """
    Class for initializing the state with configuration parameters.

    Args:
        config (dict): A dictionary containing the configuration parameters.

    Attributes:
        scale (int): The scale parameter.
        data_n (int): The data_n parameter.
        dipoleLimit (int): The dipoleLimit parameter.
        mapheight (float): The mapheight parameter.
        order (int): The order parameter.
        distancing (bool): The distancing parameter.
        centering (bool): The centering parameter.
        equalizer_bools (list): A list of boolean values for equalizer.
        test (str): The test parameter.
        dataset_name (str): The dataset_name parameter.
    """

    def __init__(self, config):
        self.scale = int(config["map_generation"]["scale"])
        self.data_n = int(config["map_generation"]["quota"])
        self.dipoleLimit = int(config["map_generation"]["dipole_limit"])
        self.mapheight = float(config["map_generation"]["mapheight"])
        self.order = int(config["map_generation"]["order"])
        self.distancing = config["map_generation"]["distancing"]
        self.centering = config["map_generation"]["centering"]
        self.equalizer_bools = [
            config["map_generation"]["equalize"],
            config["map_generation"]["remove_unequal"],
        ]
        self.test = config["map_generation"]["test"]
        self.distancing_parameter = float(config["map_generation"]["distancing_parameter"])
        self.complex_sort = config["map_generation"]["complex_sort"]

        if self.distancing == "True":
            self.distancing = True
        else:
            self.distancing = False

        if self.centering == "True":
            self.centering = True
        else:
            self.centering = False

        if self.complex_sort == "True":
            self.complex_sort = True
        else:
            self.complex_sort = False

        self.dataset_name = str(config["map_generation"]["dataset_name"])

    def __str__(self):
        return f"""Scale (px x px): {self.scale} x {self.scale}
        Amount of Data Generated: {self.data_n}
        Magnetic Sources per Map: {self.dipoleLimit}
        Map Height (z - microns): ~{self.mapheight*5}
        Number of Dipoles Summed per Source: {self.order+1}
        Distancing applied: {self.distancing} (up to {int(self.distancing_parameter*100*2)} px)
        Centering applied: {self.centering}
        Dataset Name: {self.dataset_name}
        """

    def write_log(self, path):
        path = Path(path)
        with path.open("w") as f:
            f.write(str(self))
        return

    def make_dir(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        print("Created directory: " + str(path))
        return


class Dipole:
    def __init__(self, position, moment):
        self.position = position
        self.moment = moment

    def __str__(self):
        """
        Returns a string representation of the Dipole object.

        The string includes the position and moment of the Dipole.

        Returns:
            str: A string representation of the Dipole object.
        """
        return "Dipole at " + str(self.position) + " with moment " + str(self.moment)

    def scaleMoment(self, scale):
        self.moment = self.moment * scale

    def tattle(self):
        print("Dipole at " + str(self.position) + " with moment " + str(self.moment))


class bField:
    """
    Represents a magnetic field generated by a list of dipoles.

    Attributes:
        dipoleList (list): List of Dipole objects representing the dipoles.
        size (int): Size of the magnetic field grid.
        mapheight (float): Height of the magnetic field map.

    Methods:
        calculateB(): Calculates the magnetic field at every point in the grid.
        dipoleField(position, dipole): Calculates the magnetic field at a specific position due to a dipole.
    """

    def __init__(self, dipoleList, size, mapheight=0.1, device="cpu", save_all_bfields=False):
        self.dipoleList = dipoleList
        self.size = size
        self.B = np.zeros((size, size, 3))
        self.B111 = np.zeros((size, size))
        self.mapheight = mapheight
        self.save_all_bfields = save_all_bfields
        
        if device == "cpu":
            self.device = torch.device("cpu")
            self.calculateB()
        else:
            self.device = torch.device("cuda")
            self.calculateB_opto()

    def calculateB(self):
        """
        Calculates the magnetic field at every point in the grid.
        """
        n111 = np.array([1, 1, 1]) / np.linalg.norm([1, 1, 1])
        x = np.linspace(0, 1, self.size)
        y = np.linspace(0, 1, self.size)

        # calculate Bfield at every point
        for i in range(0, self.size):
            for j in range(0, self.size):
                for dipole in self.dipoleList:
                    self.B[i, j] = self.B[i, j] + self.dipoleField(
                        [x[i], y[j], self.mapheight], dipole
                    )
                # Calculate B111 by using the n111 vector
                self.B111[i, j] = np.dot(self.B[i, j], n111)
        if self.save_all_bfields:
            B_mat = {
                'Bx': self.B[:, :, 0] - np.mean(self.B[:, :, 0]),
                'By': self.B[:, :, 1] - np.mean(self.B[:, :, 1]),
                'Bz': self.B[:, :, 2] - np.mean(self.B[:, :, 2])
            }
            savemat("../data/Bfields.mat", B_mat)

    def calculateB_opto(self):
        """
        Vectorized GPU field calculation: all dipoles computed in one batched operation.
        Shapes: positions/moments (D,3), grid (H,W,3), displacement r (H,W,D,3).
        """
        dev = self.device
        n111 = torch.tensor([1, 1, 1], dtype=torch.float32, device=dev)
        n111 = n111 / torch.norm(n111)

        x = torch.linspace(0, 1, self.size, device=dev)
        y = torch.linspace(0, 1, self.size, device=dev)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        grid = torch.stack([grid_x, grid_y, torch.full_like(grid_x, self.mapheight)], dim=-1)  # (H, W, 3)

        positions = torch.from_numpy(
            np.array([d.position for d in self.dipoleList], dtype=np.float32)
        ).to(device=dev)  # (D, 3)
        moments = torch.from_numpy(
            np.array([d.moment for d in self.dipoleList], dtype=np.float32)
        ).to(device=dev)  # (D, 3)

        # r[i,j,d] = grid point (i,j) minus dipole d — shape (H, W, D, 3)
        r     = grid[:, :, None, :] - positions[None, None, :, :]
        r_mag = torch.norm(r, dim=-1, keepdim=True)          # (H, W, D, 1)
        r_hat = r / r_mag                                      # (H, W, D, 3)

        mu = torch.tensor(const.mu_0 / (4 * np.pi), dtype=torch.float32, device=dev)
        m_dot_rhat = (moments[None, None, :, :] * r_hat).sum(-1, keepdim=True)  # (H, W, D, 1)
        B_each = mu * (3 * m_dot_rhat * r_hat - moments[None, None, :, :]) / r_mag**3
        B = B_each.sum(dim=2)  # sum over dipoles → (H, W, 3)

        if self.save_all_bfields:
            B_cpu = B.cpu().numpy()
            savemat("../data/Bfields.mat", {
                'Bx': B_cpu[:, :, 0], 'By': B_cpu[:, :, 1], 'Bz': B_cpu[:, :, 2]
            })
        self.B111 = torch.einsum('ijk,k->ij', B, n111).cpu().numpy()
        self.B = B.cpu().numpy()


    def dipoleField(self, position, dipole):
        """
        Calculates the magnetic field at a specific position due to a dipole.

        Args:
            position (list): Position of the point where the magnetic field is calculated.
            dipole (Dipole): Dipole object representing the dipole.

        Returns:
            numpy.ndarray: Magnetic field vector at the given position.
        """
        r = np.array(position)
        r0 = np.array(dipole.position)
        m = np.array(dipole.moment)
        r = r - r0
        r_mag = np.linalg.norm(r)
        r_hat = r / r_mag

        B = (const.mu_0 / (4 * np.pi)) * (3 * np.dot(m, r_hat) * r_hat - m) / r_mag**3
        return B


class Generator:
    """
    A class that generates magnetic field data based on given parameters.

    Args:
        dipoleLimit (int): The maximum number of dipoles to generate.
        scale (float): The scaling factor for the magnetic field.
        mapheight (float): The height of the magnetic field map.
        order (int): The order of the magnetic field generation.
        distancing (bool): Whether to perturb the positions of the dipoles.
        centering (bool): Whether to center the dipoles at (0.5, 0.5, 0).
        r_scale (float): The range of the net moment of the dipoles.
        offset (float, optional): The offset value. Defaults to 0.

    Methods:
        generateData(): Generates the magnetic field data.
        netDipoleMoment(dipolelist): Calculates the net dipole moment.
        netDipolePosition(dipolelist): Calculates the net dipole position.
        generateDipoles(): Generates the dipoles.
        perturbPositions(dipoleList): Perturbs the positions of the dipoles.
    """

    def __init__(
        self,
        dipoleLimit,
        scale,
        mapheight,
        order,
        distancing,
        centering,
        distancing_parameter,
        r_scale,
        offset=0,
        staged=False,
    ):
        self.dipoleLimit = dipoleLimit
        self.scale = scale
        self.mapheight = mapheight
        self.order = order
        self.distancing = distancing
        self.centering = centering
        self.r_scale = r_scale
        self.offset = offset
        self.distancing_parameter = distancing_parameter
        self.staging_bool = staged
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def generateData(self):
        # Get random Dipoles
        dipoleList = self.generateDipoles()
        # if higher order, generate second set of dipoles
        order = self.order
        while order > 0:
            dipoleList_sub = self.generateDipoles()
            for dipole in dipoleList_sub:
                dipole.scaleMoment(1 / 2**order)
            dipoleList = dipoleList_sub + dipoleList
            order -= 1

        # if distancing true, perturb the positions a little bit
        # print(self.distancing)
        # if self.distancing:
        #     dipoleList = self.perturbPositions(dipoleList)

        # Create BField
        if self.staging_bool:
            B = bField(dipoleList, self.scale, self.mapheight, device=self.device, save_all_bfields=True)
        else:
            B = bField(dipoleList, self.scale, self.mapheight, device=self.device)

        netDipoleMoment = self.netDipoleMoment(dipoleList)
        netDipolePosition = self.netDipolePosition(dipoleList)
        return B.B, netDipoleMoment, netDipolePosition

    def netDipoleMoment(self, dipolelist):
        m_net = np.array([0, 0, 0])
        for dipole in dipolelist:
            m_net = m_net + dipole.moment
        return m_net

    def netDipolePosition(self, dipolelist):
        r_net = np.array([0, 0, 0])
        for dipole in dipolelist:
            r_net = r_net + dipole.position
        r_net = r_net / len(dipolelist)
        return r_net

    def generateDipoles(self):
        dipoleList = []
        full_range = self.r_scale
        if self.centering:
            # generate random moments but positions are all 0.5 0.5 0
            for i in range(0, self.dipoleLimit):
                position = np.array([0.5, 0.5, 0])
                net_moment = np.random.uniform(0, full_range)
                if self.staging_bool:
                    inclination = np.radians(54.7)
                    declination = np.random.choice(np.radians([45,225]))
                else: 
                    inclination = np.random.uniform(0, np.pi)
                    declination = np.random.uniform(0, 2 * np.pi)
                moment = np.array(
                    [
                        net_moment * np.sin(inclination) * np.cos(declination),
                        net_moment * np.sin(inclination) * np.sin(declination),
                        net_moment * np.cos(inclination),
                    ]
                )
                dipoleList.append(Dipole(position, moment))
            return dipoleList
        else:
            for i in range(0, self.dipoleLimit):
                distance = self.distancing_parameter
                position = np.random.uniform(-1 * distance + 0.5, distance + 0.5, (2))
                position = np.append(position, 0)
                net_moment = np.random.uniform(0, full_range)
                inclination = np.random.uniform(0, np.pi)
                declination = np.random.uniform(0, 2 * np.pi)
                moment = np.array(
                    [
                        net_moment * np.sin(inclination) * np.cos(declination),
                        net_moment * np.sin(inclination) * np.sin(declination),
                        net_moment * np.cos(inclination),
                    ]
                )
                dipoleList.append(Dipole(position, moment))
            return dipoleList

    def perturbPositions(self, dipoleList):
        # ensure dipole list exists and is not empty
        if len(dipoleList) == 0:
            raise ValueError("Dipole list is empty")
        # perturb positions by a small amount
        distancing_parameter = self.distancing_parameter
        if distancing_parameter is None:
            distancing_parameter = 0.1
        for dipole in dipoleList:
            dipole.position = dipole.position + np.random.uniform(
                -1 * distancing_parameter, distancing_parameter, (3)
            )
        return dipoleList

class DataSaver:
    """
    A class for saving data to specified paths, with support for HDF5 format.

    Args:
        data (list): The data to be saved (expected: [I_data, m_data, r_data]).
        index (int): The index of the data.
        path (str or Path): The path where the data will be saved.
        use_hdf5 (bool): Whether to store data in an HDF5 file (default: False).

    Attributes:
        data (list): The data to be saved.
        index (int): The index of the data.
        path (Path): The path where the data will be saved.
        use_hdf5 (bool): Whether to save in HDF5 format.
    """

    def __init__(self, data, index, path, use_hdf5=True, meta: Optional[dict] = None):
        self.data = data  # Expected: [I_data, m_data, r_data]
        self.index = index
        self.path = Path(path)
        self.use_hdf5 = use_hdf5
        self.meta = meta or {}
        self.path_set_up()

    def save_data(self):
        """
        Saves the data to the specified path in either .npy format or HDF5.

        Returns:
            None
        """
        i = self.index
        I_data, m_data, r_data = self.data  # Extract components

        if self.use_hdf5:
            self.save_to_hdf5(I_data, m_data, r_data)
        else:
            self.save_to_npy(i, I_data, m_data, r_data)

    def save_to_npy(self, i, I_data, m_data, r_data):
        """
        Saves data in .npy format.
        """
        np.save(self.path / "images" / f"{i}.npy", I_data)
        np.save(self.path / "mnettargets" / f"{i}.npy", m_data)
        np.save(self.path / "r0targets" / f"{i}.npy", r_data)
        print(f"\nSaved data №{i} in .npy format.")

    def save_to_hdf5(self, I_data, m_data, r_data):
        """
        Saves data in HDF5 format.
        """
        h5_path = self.path / "dataset.h5"
        file_hash = self.compute_hash(I_data)  # Compute hash for tracking

        with h5py.File(h5_path, "a") as hf:
            self.apply_root_attrs(hf)
            # Create datasets if they don't exist
            if "m" not in hf:
                create_vecmagnet_datasets(hf, image_shape=(0, 100, 100, 3))

            # Append new data
            self.append_to_hdf5(hf, "m", m_data)
            self.append_to_hdf5(hf, "r", r_data)
            assert I_data.shape == (100, 100, 3), f"Unexpected I_data shape: {I_data.shape}"
            self.append_to_hdf5(hf, "I", I_data)
            self.append_to_hdf5(hf, "hash", np.array([file_hash], dtype="S32"))


    def append_to_hdf5(self, hf, dataset_name, new_data):
        """
        Appends new data to an existing HDF5 dataset.
        """
        dset = hf[dataset_name]
        new_size = dset.shape[0] + 1
        dset.resize((new_size,) + dset.shape[1:])  # Expand dataset
        dset[-1] = new_data  # Insert new data

    def apply_root_attrs(self, hf):
        """
        Write metadata to HDF5 root attributes once per file.
        """
        if not self.meta:
            return
        for key, value in self.meta.items():
            if value is None:
                continue
            if key in hf.attrs:
                continue
            hf.attrs[key] = value

    def compute_hash(self, data):
        """
        Computes an MD5 hash of the given numpy array.
        """
        return hashlib.md5(data.tobytes()).hexdigest()

    def path_set_up(self):
        """
        Sets up the path for saving the data.

        If HDF5 is not used, it ensures directory structure exists.
        """
        if not self.use_hdf5:
            self.path.mkdir(parents=True, exist_ok=True)
            (self.path / "images").mkdir(parents=True, exist_ok=True)
            (self.path / "mnettargets").mkdir(parents=True, exist_ok=True)
            (self.path / "r0targets").mkdir(parents=True, exist_ok=True)

# Function to display four images with scaling based on the mean � 2 std of the first image
def display_images(data_list, save_path=None):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()

    # Load the first image to determine the scaling
    first_img = np.load(data_list[0])
    img_mean = np.mean(first_img)
    img_std = np.std(first_img)
    vmin = img_mean - 2 * img_std
    vmax = img_mean + 2 * img_std

    for i, ax in enumerate(axs):
        if i < len(data_list):
            # Load the image
            img = np.load(data_list[i])

            # Display the image with jet colormap and scaling based on the first image
            im = ax.imshow(img, cmap="jet", vmin=vmin, vmax=vmax)
            ax.set_title(f"Image {i+1}")
            ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.clf()
    plt.close()


if __name__ == "__main__":
    cli_args = parse_cli_arguments()
    if cli_args.config:
        config.read(cli_args.config)
    apply_config_overrides(
        distancing=cli_args.distancing,
        distancing_parameter=cli_args.distancing_parameter,
        dipole_limit=cli_args.dipoleLimit,
        dataset_name=cli_args.dataset_name,
        centering=cli_args.centering,
    )
    main(
        dipoleLimit=cli_args.dipoleLimit,
        distancing=cli_args.distancing,
        name=cli_args.dataset_name,
        distancing_parameter=cli_args.distancing_parameter,
        centering=cli_args.centering,
        seed=cli_args.seed,
    )

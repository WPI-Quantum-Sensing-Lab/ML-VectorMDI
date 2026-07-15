import configparser
import subprocess as supro
import sys
from pathlib import Path

# src/ directory; the generator module is spawned with this as cwd so
# `-m data_creation...` resolves and relative output paths land under src/.
_SRC_DIR = Path(__file__).resolve().parents[2]


def _load_scale():
    config = configparser.ConfigParser()
    config_path = _SRC_DIR.parent / "configs" / "generate_data.ini"
    if config_path.exists():
        config.read(config_path)
        return config.getint("map_generation", "scale", fallback=100)
    return 100


def create_dataset(distance_px, ds_name, dipole_limit, *, centering=False, scale=100):
    distancing_parameter = float(distance_px) / float(scale)
    cmd = [
        sys.executable, "-m", "data_creation.generation.magnetic_moment_gen_mp",
        "--distancing", "True",
        "--distancing_parameter", str(distancing_parameter),
        "--centering", str(centering),
        "--dataset_name", ds_name,
        "--dipoleLimit", str(dipole_limit),
    ]
    supro.run(cmd, check=True, cwd=_SRC_DIR)


if __name__ == "__main__":
    scale = _load_scale()
    for dipole_number in range(2, 11):
        if dipole_number <= 6:
            distance_px = 5 + 2 * (dipole_number - 2)
        else:
            distance_px = round(13 + 1.75 * (dipole_number - 6))
        ds_name = rf"multidipoleSOURCEdistanceFixUpdate_{dipole_number}"
        dipole_limit = dipole_number
        create_dataset(distance_px, ds_name, dipole_limit, centering=False, scale=scale)

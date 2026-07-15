"""INI-config → argv bridge for the task entry scripts.

Each task script (train_model.py, train_ensemble.py, calibrate.py, ...) is a
thin wrapper around an argparse-based module in the ``model`` package.  This
helper reads the task's config file and turns each ``key = value`` pair in the
task section into ``--key value`` CLI flags:

  - ``key = true``            → ``--key``            (store_true flag)
  - ``key = false`` / empty   → omitted
  - anything else             → ``--key <value>``

Extra flags passed on the command line are appended after the config-derived
ones, so they override the config (argparse last-one-wins).  A ``--config
PATH`` argument selects an alternative config file.
"""
from __future__ import annotations

import configparser
import sys
from pathlib import Path
from typing import List

_SRC = Path(__file__).resolve().parent
CONFIGS_DIR = _SRC.parent / "configs"

# Only literal booleans toggle flags; numerics like 0/1 pass through as values.
_TRUE_VALUES = {"true", "yes", "on"}
_FALSE_VALUES = {"false", "no", "off", "none", ""}


def _section_to_flags(config_path: Path, section: str) -> List[str]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    parser = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    parser.read(config_path)
    if section not in parser:
        raise KeyError(f"Config file {config_path} has no [{section}] section")

    flags: List[str] = []
    for key, value in parser[section].items():
        value = value.strip()
        if value.lower() in _TRUE_VALUES:
            flags.append(f"--{key}")
        elif value.lower() in _FALSE_VALUES:
            continue
        else:
            flags.extend([f"--{key}", value])
    return flags


def build_argv(task: str, *, section: str | None = None) -> List[str]:
    """Return a new sys.argv for the wrapped module.

    Reads ``configs/<task>.ini`` (or the file given by a ``--config PATH``
    command-line argument), converts the ``[<section>]`` items to CLI flags,
    and appends any remaining command-line arguments so they take precedence.
    """
    section = section or task
    cli = list(sys.argv[1:])

    config_path = CONFIGS_DIR / f"{task}.ini"
    if "--config" in cli:
        idx = cli.index("--config")
        try:
            config_path = Path(cli[idx + 1])
        except IndexError:
            raise SystemExit("--config requires a path argument")
        del cli[idx : idx + 2]

    return [task] + _section_to_flags(config_path, section) + cli

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
SYS = copy.deepcopy(CONFIG["system"])
RL = copy.deepcopy(CONFIG["training"])
UNSPEC = copy.deepcopy(CONFIG["environment"])
SCENARIO = copy.deepcopy(CONFIG["scenario"])


def apply_method(method: str) -> None:
    validate_runtime()
    validate_slot_duration()
    if method not in CONFIG["methods"]:
        raise ValueError(f"unknown method: {method}")
    SYS.clear()
    SYS.update(copy.deepcopy(CONFIG["system"]))
    UNSPEC.clear()
    UNSPEC.update(copy.deepcopy(CONFIG["environment"]))
    UNSPEC.update(copy.deepcopy(CONFIG["methods"][method]))


def load_config(path: str | Path = ROOT / "config.json") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_slot_duration() -> None:
    if float(CONFIG["system"]["slot_duration_s"]) != 1.0:
        raise ValueError("Only 1-second slots are supported")


def validate_runtime() -> None:
    import numpy as np

    if np.__version__ != "1.26.3":
        warnings.warn(
            f"Reference simulations use NumPy 1.26.3; found {np.__version__}. "
            "Simulation results may differ.",
            RuntimeWarning,
            stacklevel=2,
        )

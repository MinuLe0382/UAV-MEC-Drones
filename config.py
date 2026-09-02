from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
SYS = copy.deepcopy(CONFIG["system"])
RL = copy.deepcopy(CONFIG["training"])
UNSPEC = copy.deepcopy(CONFIG["environment"])
SCENARIO = copy.deepcopy(CONFIG["scenario"])


def apply_method(method: str) -> None:
    validate_runtime()
    if method not in CONFIG["methods"]:
        raise ValueError(f"unknown method: {method}")
    SYS.clear()
    SYS.update(copy.deepcopy(CONFIG["system"]))
    UNSPEC.clear()
    UNSPEC.update(copy.deepcopy(CONFIG["environment"]))
    UNSPEC.update(copy.deepcopy(CONFIG["methods"][method]))


def load_config(path: str | Path = ROOT / "config.json") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_runtime() -> None:
    import numpy as np

    if np.__version__ != "1.26.3":
        raise RuntimeError(f"NumPy 1.26.3 is required; found {np.__version__}")

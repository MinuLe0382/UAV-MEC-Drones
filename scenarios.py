from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ARRAYS = (
    "gu_positions",
    "gu_generation_probabilities",
    "cluster_centers",
    "cluster_labels",
    "activation_schedule",
    "task_arrival_indicators",
    "task_sizes_bits",
    "initial_uav_positions",
    "initial_energy_budgets",
)


def load_scenario(path: str | Path) -> dict:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in ("metadata_json", *ARRAYS) if name not in archive]
        if missing:
            raise ValueError(f"scenario is missing fields: {missing}")
        result = {"metadata": json.loads(str(archive["metadata_json"].item()))}
        result.update({name: archive[name].copy() for name in ARRAYS})
    validate_scenario(result)
    return result


def validate_scenario(scenario: dict) -> None:
    metadata = scenario["metadata"]
    users = int(metadata["num_gus"])
    uavs = int(metadata["num_uavs"])
    horizon = int(metadata["mission_slots"])
    clusters = len(scenario["cluster_centers"])
    shapes = {
        "gu_positions": (users, 2),
        "gu_generation_probabilities": (users,),
        "cluster_centers": (clusters, 2),
        "cluster_labels": (users,),
        "activation_schedule": (horizon, clusters),
        "task_arrival_indicators": (horizon, users),
        "task_sizes_bits": (horizon, users),
        "initial_uav_positions": (uavs, 2),
        "initial_energy_budgets": (uavs,),
    }
    for name, shape in shapes.items():
        value = np.asarray(scenario[name])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"invalid {name}: expected finite array of shape {shape}")
    labels = scenario["cluster_labels"]
    if not np.issubdtype(labels.dtype, np.integer) or np.any((labels < 0) | (labels >= clusters)):
        raise ValueError("cluster labels are out of range")
    probabilities = scenario["gu_generation_probabilities"]
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("generation probabilities must lie in [0, 1]")
    length = float(metadata["mission_side_length_m"])
    for name in ("gu_positions", "cluster_centers", "initial_uav_positions"):
        if np.any((scenario[name] < 0) | (scenario[name] > length)):
            raise ValueError(f"{name} lies outside the mission area")
    if np.any(scenario["task_sizes_bits"] < 0) or np.any(scenario["initial_energy_budgets"] < 0):
        raise ValueError("workloads and initial operating budgets must be nonnegative")


def scenario_files(split: str = "test", root: str | Path | None = None) -> list[Path]:
    base = Path(root) if root is not None else Path(__file__).resolve().parent
    paths = sorted((base / "data" / "scenarios" / split).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no scenarios found for split {split!r}")
    return paths

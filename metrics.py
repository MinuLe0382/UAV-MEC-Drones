from __future__ import annotations

import numpy as np


class PatrolTracker:
    def __init__(self, env):
        self.horizon = int(env.T)
        occupancy = np.bincount(
            np.asarray(env.sd_cell, dtype=np.int64), minlength=int(env._n_cells)
        )
        self.occupied = occupancy > 0
        self.last_visit = np.full(len(occupancy), -1, dtype=np.int64)
        self.completed_gaps = np.zeros(self.horizon + 1, dtype=np.int64)
        self.steps = 0

    def after_step(self, env, slot: int) -> None:
        if int(env.t) != slot + 1 or slot != self.steps:
            raise ValueError("patrol observations must follow consecutive slots")
        visited = self.occupied & np.isclose(
            np.asarray(env.cell_last_seen, dtype=np.float64), float(slot)
        )
        for cell in np.flatnonzero(visited):
            previous = int(self.last_visit[cell])
            if previous >= 0:
                self.completed_gaps[slot - previous] += 1
            self.last_visit[cell] = slot
        self.steps += 1

    def finalize(self, env) -> dict:
        if self.steps != self.horizon:
            raise ValueError(f"expected {self.horizon} slots, observed {self.steps}")
        discovered = np.asarray(env.discovered_user_cells, dtype=bool)
        discovery = float(discovered[self.occupied].mean()) if self.occupied.any() else 1.0
        last = self.last_visit[self.occupied]
        tail = np.bincount(self.horizon - last[last >= 0], minlength=self.horizon + 1)
        gaps = self.completed_gaps + tail
        samples = np.repeat(np.arange(len(gaps)), gaps)
        return {
            "discovery": discovery,
            "revisit_p95": float(np.percentile(samples, 95)) if len(samples) else None,
        }


def _jain(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    denominator = len(values) * float(np.square(values).sum())
    return float(values.sum() ** 2 / denominator) if denominator > 0 else 0.0


def service_metrics(env) -> tuple[float, float, float]:
    coverage = float(np.asarray(env.visited_global, dtype=bool).mean())
    generated = np.asarray(env.generated_per_sd, dtype=np.float64)
    served = np.minimum(np.asarray(env.served_per_sd, dtype=np.float64), generated)
    valid_users = generated > 1e-9
    if valid_users.any():
        reach = float((served[valid_users] > 1e-9).mean())
        fairness = _jain(served[valid_users] / generated[valid_users])
    else:
        reach = fairness = 0.0
    ecf = reach * fairness
    min_cluster = float("nan")
    if getattr(env, "sd_cluster", None) is not None and env.n_clusters > 0:
        cluster_generated = np.zeros(env.n_clusters, dtype=np.float64)
        cluster_served = np.zeros(env.n_clusters, dtype=np.float64)
        np.add.at(cluster_generated, env.sd_cluster, generated)
        np.add.at(cluster_served, env.sd_cluster, served)
        valid_clusters = cluster_generated > 1e-9
        if valid_clusters.any():
            min_cluster = float(
                np.min(cluster_served[valid_clusters] / cluster_generated[valid_clusters])
            )
    return coverage, ecf, min_cluster


def episode_metrics(env, movement: np.ndarray) -> dict[str, float]:
    summary = env.summary()
    coverage, ecf, min_cluster = service_metrics(env)
    return {
        "e2e": float(summary["end_to_end_success_rate"]),
        "collection": float(summary["collection_success_rate"]),
        "ecf": ecf,
        "min_cluster": min_cluster,
        "coverage": coverage,
        "movement_per_uav": float(np.asarray(movement, dtype=np.float64).mean()),
    }

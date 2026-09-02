from __future__ import annotations

from typing import Any

import numpy as np


GOVERNOR_ID = "remaining_budget_empirical_v1"
PLAN_SAFETY_FACTOR = 0.9


FREQUENCY_FRACTION_POINTS = np.asarray([0.05, 0.10, 0.20, 0.35, 0.60, 1.00], dtype=np.float64)
ENERGY_RATE_POINTS = np.asarray([11.7, 12.0, 13.9, 20.4, 42.9, 97.4], dtype=np.float64) / 320.0


def remaining_budget_frequency_fraction(
    env: Any,
    *,
    safety: float = PLAN_SAFETY_FACTOR,
) -> np.ndarray:

    if not 0.0 < float(safety) <= 1.0:
        raise ValueError("governor safety must lie in (0, 1]")
    remaining_slots = max(int(env.T) - int(env.t), 1)
    target_rate = (
        float(safety)
        * np.maximum(np.asarray(env.energy_remaining, dtype=np.float64), 0.0)
        / remaining_slots
    )
    fractions = np.interp(target_rate, ENERGY_RATE_POINTS, FREQUENCY_FRACTION_POINTS)
    queue = np.asarray(env.uav_queue, dtype=np.float64)
    fractions = np.where(queue <= 1e-6, 0.0, fractions)
    return np.clip(fractions, 0.0, 1.0).astype(np.float32)


def policy_action_dimension(
    frequency_rule: str,
    environment_action_dim: int,
    *,
    separate_governor: bool,
) -> int:

    if frequency_rule not in {"learned", "fmax", "plan"}:
        raise ValueError(f"unsupported frequency rule: {frequency_rule}")
    if environment_action_dim < 3:
        raise ValueError("the UAV-MEC environment requires three action channels")
    if separate_governor and frequency_rule in {"fmax", "plan"}:
        return environment_action_dim - 1
    return environment_action_dim


def compose_environment_actions(
    policy_actions: np.ndarray,
    env: Any,
    frequency_rule: str,
    *,
    safety: float = PLAN_SAFETY_FACTOR,
) -> np.ndarray:

    actions = np.asarray(policy_actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] != int(env.M):
        raise ValueError(f"policy actions must have shape (M,D); got {actions.shape}")
    if frequency_rule == "learned":
        if actions.shape[1] != int(env.act_dim):
            raise ValueError("learned frequency requires the full environment action dimension")
        return actions.copy()
    if frequency_rule not in {"fmax", "plan"}:
        raise ValueError(f"unsupported frequency rule: {frequency_rule}")
    if actions.shape[1] not in {int(env.act_dim) - 1, int(env.act_dim)}:
        raise ValueError(
            "governed policy actions must contain movement only or legacy full actions"
        )

    full = np.zeros((int(env.M), int(env.act_dim)), dtype=np.float32)
    full[:, :2] = actions[:, :2]
    if frequency_rule == "fmax":
        full[:, 2] = 1.0
    else:
        fractions = remaining_budget_frequency_fraction(env, safety=safety)
        full[:, 2] = 2.0 * fractions - 1.0
    return full


def mean_frequency_fraction(environment_actions: np.ndarray) -> float:

    actions = np.asarray(environment_actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] < 3:
        raise ValueError("environment actions must include the frequency channel")
    return float(np.mean((np.clip(actions[:, 2], -1.0, 1.0) + 1.0) * 0.5))

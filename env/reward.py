from __future__ import annotations

import numpy as np


def movement_reward(
    *,
    completed_bits: np.ndarray,
    new_coarse_cells: np.ndarray,
    active_at_slot_start: np.ndarray,
    boundary_hit: np.ndarray,
    collision_hit: np.ndarray,
    energy_remaining_after_slot: np.ndarray,
    reward_scale: float,
    gamma_e: float,
    idle_penalty: float,
    boundary_collision_penalty: float,
    energy_overrun_base: float,
    energy_overrun_per_excess: float,
) -> np.ndarray:
    completed = np.asarray(completed_bits, dtype=np.float32)
    cells = np.asarray(new_coarse_cells, dtype=np.float32)
    active = np.asarray(active_at_slot_start, dtype=bool)
    boundary = np.asarray(boundary_hit, dtype=np.float32)
    collision = np.asarray(collision_hit, dtype=np.float32)
    remaining = np.asarray(energy_remaining_after_slot, dtype=np.float32)
    rewards = completed / float(reward_scale)
    rewards = rewards + float(gamma_e) * cells
    rewards = rewards - float(idle_penalty) * (completed <= 0).astype(np.float32)
    rewards = rewards - float(boundary_collision_penalty) * boundary
    rewards = rewards - float(boundary_collision_penalty) * collision
    excess = np.maximum(-remaining, 0.0).astype(np.float32)
    rewards = rewards - float(energy_overrun_base) * (excess > 0)
    rewards = rewards - float(energy_overrun_per_excess) * excess
    return np.where(active, rewards, 0.0).astype(np.float32)

from __future__ import annotations

from itertools import permutations

import numpy as np


def _targets_to_action(env, targets: np.ndarray) -> np.ndarray:
    target_positions = env._cell_centers[np.asarray(targets, dtype=int)]
    delta = target_positions - env.uav_pos
    distance = np.minimum(np.linalg.norm(delta, axis=1), env.d_max)
    heading = np.mod(np.arctan2(delta[:, 1], delta[:, 0]), 2.0 * np.pi)
    return np.stack([2.0 * distance / env.d_max - 1.0, heading / np.pi - 1.0], axis=1).astype(
        np.float32
    )


def _assign(env, score: np.ndarray) -> np.ndarray:
    distance = np.linalg.norm(env.uav_pos[:, None, :] - env._cell_centers[None, :, :], axis=2)
    utility = score[None, :] / np.maximum(distance / env.d_max, 1.0)
    return _assign_matrix(utility)


def _assign_matrix(utility: np.ndarray) -> np.ndarray:
    targets = np.full(utility.shape[0], -1, dtype=int)
    claimed: set[int] = set()
    cell_count = utility.shape[1]
    for index in np.argsort(utility.ravel())[::-1]:
        uav, cell = divmod(int(index), cell_count)
        if targets[uav] < 0 and cell not in claimed:
            targets[uav] = cell
            claimed.add(cell)
        if (targets >= 0).all():
            break
    return targets


def _full_information_score(env) -> np.ndarray:
    demand = np.zeros(env._n_cells, dtype=np.float32)
    np.add.at(demand, env.sd_cell, env.queue)
    return demand


def _belief_mpc(
    env, stock_bits: np.ndarray, horizon: int = 10, candidate_count: int = 8, discount: float = 0.85
) -> np.ndarray:
    centers = env._cell_centers
    pool = list(np.argsort(stock_bits)[::-1][:candidate_count])
    if len([cell for cell in pool if stock_bits[cell] > 0]) < env.M:
        nearest = np.argsort(
            np.linalg.norm(centers[None, :, :] - env.uav_pos[:, None, :], axis=2).min(axis=0)
        )
        for cell in nearest:
            if int(cell) not in pool:
                pool.append(int(cell))
            if len(pool) >= candidate_count + env.M:
                break
    pool_array = np.asarray(pool[: candidate_count + env.M], dtype=int)
    candidates = np.asarray(list(permutations(range(len(pool_array)), env.M)), dtype=int)
    candidate_total = len(candidates)
    positions = np.broadcast_to(env.uav_pos, (candidate_total, env.M, 2)).copy()
    stock = np.broadcast_to(stock_bits[pool_array], (candidate_total, len(pool_array))).copy()
    pool_positions = centers[pool_array]
    targets = centers[pool_array[candidates]]
    score = np.zeros(candidate_total, dtype=np.float64)
    weight = 1.0
    for _ in range(horizon):
        delta = targets - positions
        distance = np.linalg.norm(delta, axis=2, keepdims=True)
        step = np.minimum(distance, env.d_max)
        positions += np.where(distance > 1e-9, delta / np.maximum(distance, 1e-9) * step, 0.0)
        squared = np.square(positions[:, :, None, :] - pool_positions[None, None, :, :]).sum(axis=3)
        efficiency = np.log2(
            1.0 + env.ch_gamma0 / np.power(squared + env.ch_H * env.ch_H, env.ch_alpha / 2.0)
        )
        eligible = (efficiency >= env.ch_eta_min) & (stock[:, None, :] > 0)
        weights = np.where(eligible, stock[:, None, :], 0.0)
        denominator = weights.sum(axis=2, keepdims=True)
        potential = (
            np.where(
                denominator > 0,
                weights / np.maximum(denominator, 1e-12),
                0.0,
            )
            * env.ch_link_scale
            * efficiency
        )
        drained = np.minimum(stock, np.where(eligible, potential, 0.0).sum(axis=1))
        score += weight * drained.sum(axis=1)
        stock -= drained
        weight *= discount
    return pool_array[candidates[int(np.argmax(score))]]


class RuleController:
    def __init__(self, method: str, env, exploration_weight: float = 0.5):
        if method not in {"memory", "anticipatory", "belief_mpc", "full_information"}:
            raise ValueError(f"unknown rule method: {method}")
        self.method = method
        self.env = env
        self.exploration_weight = float(exploration_weight)
        self.period = int(env.burst_period)
        self.memory_sum = np.zeros(env._n_cells, dtype=np.float32)
        self.memory_count = np.zeros(env._n_cells, dtype=np.float32)
        self.phase_sum = np.zeros((env._n_cells, self.period), dtype=np.float32)
        self.phase_count = np.zeros((env._n_cells, self.period), dtype=np.float32)

    def action(self) -> np.ndarray:
        env = self.env
        mean = np.divide(
            self.memory_sum,
            np.maximum(self.memory_count, 1.0),
            out=np.zeros_like(self.memory_sum),
        )
        stale = np.clip((env.t - env.cell_last_seen) / max(env.T, 1), 0.0, 1.0)
        if self.method == "full_information":
            targets = _assign(env, _full_information_score(env))
        elif self.method == "memory":
            score = mean / max(env.grid_density_norm, 1.0)
            score += self.exploration_weight * stale
            targets = _assign(env, score)
        elif self.method == "belief_mpc":
            stock = mean + self.exploration_weight * stale * max(env.grid_density_norm, 1.0)
            targets = _belief_mpc(env, stock.astype(np.float64))
        else:
            distance = np.linalg.norm(
                env.uav_pos[:, None, :] - env._cell_centers[None, :, :], axis=2
            )
            travel = np.maximum(distance / env.d_max, 1.0)
            phase_mean = np.divide(
                self.phase_sum,
                np.maximum(self.phase_count, 1.0),
                out=np.zeros_like(self.phase_sum),
            )
            arrival_phase = ((env.t + np.ceil(travel)).astype(int)) % self.period
            rows = np.broadcast_to(np.arange(env._n_cells), arrival_phase.shape)
            prediction = phase_mean[rows, arrival_phase]
            prediction = np.where(
                self.phase_count[rows, arrival_phase] > 0,
                prediction,
                mean[None, :],
            )
            utility = (
                prediction / max(env.grid_density_norm, 1.0)
                + self.exploration_weight * stale[None, :]
            ) / travel
            targets = _assign_matrix(utility)
        return _targets_to_action(env, targets)

    def update(self, observed_slot: int) -> None:
        env = self.env
        cells = np.flatnonzero(env.cell_last_seen == float(observed_slot))
        self.memory_sum[cells] += env.cell_last_density[cells]
        self.memory_count[cells] += 1.0
        self.phase_sum[cells, observed_slot % self.period] += env.cell_last_density[cells]
        self.phase_count[cells, observed_slot % self.period] += 1.0

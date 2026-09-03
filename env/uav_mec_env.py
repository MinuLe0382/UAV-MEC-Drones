from __future__ import annotations
from env.reward import movement_reward
import copy
from typing import Any, Mapping
import numpy as np
from config import SYS, UNSPEC, SCENARIO


class UAVMECEnv:
    RNG_STREAM_NAMES = (
        "scenario",
        "cluster_center",
        "gu_position",
        "cluster_assignment",
        "phase",
        "arrival",
        "task_size",
        "uav_initial_state",
        "initial_budget",
        "miscellaneous",
    )

    def __init__(
        self,
        alpha_dist: str = "uniform",
        alpha_params: tuple | None = None,
        seed: int | None = None,
        scenario: str | None = None,
        use_grid_belief: bool | None = None,
        scenario_artifact: Mapping[str, Any] | None = None,
    ):
        self.M = SYS["num_uavs"]
        self.K = SYS["num_sds"]
        self.l_max = float(SYS["l_max"])
        self.T = int(SYS["T"])
        self.r_min = float(SYS["r_min"])
        self.r_max = float(SYS["r_max"])
        self.d_max = float(SYS["d_max"])
        self.v_uav = float(SYS["v_uav"])
        self.e_uav = float(SYS["e_uav"])
        self.task_mean = float(SYS["task_mean"])
        self.task_std = float(SYS["task_std"])
        self.l_queue_max = float(SYS["l_queue_max"])
        self.f_max = float(SYS["f_max"])
        self.f_min = float(SYS["f_min"])
        self.penalty = float(SYS["penalty"])
        self.operating_cost = {key: float(value) for key, value in UNSPEC["operating_cost"].items()}
        self.uav_init = np.array(SYS["uav_init_positions"], dtype=np.float32)
        self.reward_scale = float(UNSPEC["reward_scale_N_divisor"])
        self.energy_overrun_base = float(UNSPEC.get("energy_overrun_penalty_base", 1.0))
        self.energy_overrun_per_excess = float(UNSPEC.get("energy_overrun_penalty_per_excess", 0.0))
        self.use_uav_queue = bool(UNSPEC.get("use_uav_queue", False))
        self.slot_capacity_mode = str(UNSPEC.get("slot_capacity_mode", "identity"))
        _uav_q_max = UNSPEC.get("uav_queue_max", 0)
        self.uav_queue_max = float(_uav_q_max) if _uav_q_max else float("inf")
        self.uav_buffer_mode = str(UNSPEC.get("uav_buffer_mode", "backpressure"))
        if self.uav_buffer_mode not in {"backpressure", "legacy_drop"}:
            raise ValueError(f"Unknown uav_buffer_mode: {self.uav_buffer_mode}")
        self.alpha_dist = alpha_dist
        self.alpha_params = alpha_params or (0.5, 0.2)
        self.scenario = scenario or str(SCENARIO.get("default", "uniform"))
        self._hetero_cfg = dict(SCENARIO.get("hetero", {}))
        self._hetero_cfg.update(UNSPEC.get("hetero_cfg", {}))
        self.gu_max_resampling_rounds = int(UNSPEC.get("gu_max_resampling_rounds", 1000))
        if self.gu_max_resampling_rounds <= 0:
            raise ValueError("gu_max_resampling_rounds must be positive")
        self._scenario_artifact = (
            copy.deepcopy(dict(scenario_artifact)) if scenario_artifact is not None else None
        )
        self._k_range = UNSPEC.get("k_range")
        self.scenario_active = self.scenario
        self.gamma_e = float(UNSPEC.get("gamma_e", 0.25))
        self.use_energy_obs = bool(UNSPEC.get("use_energy_obs", False))
        self.hard_energy_constraint = bool(UNSPEC.get("hard_energy_constraint", False))
        requested_energy_mode = UNSPEC.get("energy_init_mode")
        legacy_energy_range = UNSPEC.get("energy_init_random")
        self.energy_init_levels = None
        self.energy_init_range = None
        self.energy_init_fraction_range = None
        if requested_energy_mode is None:
            if legacy_energy_range:
                self.energy_init_mode = "iid_uniform_fraction_legacy"
                values = np.asarray(legacy_energy_range, dtype=np.float64).reshape(-1)
                if (
                    values.size != 2
                    or not np.isfinite(values).all()
                    or values[0] <= 0
                    or (values[1] < values[0])
                ):
                    raise ValueError("energy_init_random must be two positive ordered fractions")
                self.energy_init_fraction_range = values.astype(np.float32)
            else:
                self.energy_init_mode = "fixed"
        else:
            self.energy_init_mode = str(requested_energy_mode)
            if legacy_energy_range:
                raise ValueError(
                    "energy_init_random is legacy-only and cannot be combined with energy_init_mode"
                )
            if self.energy_init_mode == "balanced_permutation":
                values = np.asarray(UNSPEC.get("energy_init_levels"), dtype=np.float64).reshape(-1)
                if values.size != self.M:
                    raise ValueError(
                        "balanced_permutation requires exactly one energy_init_levels value per UAV"
                    )
                if (
                    not np.isfinite(values).all()
                    or np.any(values <= 0)
                    or float(values.max()) > self.e_uav
                ):
                    raise ValueError(
                        "energy_init_levels must be finite, positive, and no greater than e_uav"
                    )
                if np.unique(values).size < 2:
                    raise ValueError("balanced_permutation requires at least two distinct levels")
                self.energy_init_levels = values.astype(np.float32)
                if UNSPEC.get("energy_init_range") is not None:
                    raise ValueError("energy_init_range is incompatible with balanced_permutation")
            elif self.energy_init_mode == "iid_uniform":
                values = np.asarray(UNSPEC.get("energy_init_range"), dtype=np.float64).reshape(-1)
                if (
                    values.size != 2
                    or not np.isfinite(values).all()
                    or values[0] <= 0
                    or (values[1] < values[0])
                    or (values[1] > self.e_uav)
                ):
                    raise ValueError(
                        "iid_uniform energy_init_range must contain two positive ordered absolute energies no greater than e_uav"
                    )
                self.energy_init_range = values.astype(np.float32)
                if UNSPEC.get("energy_init_levels") is not None:
                    raise ValueError("energy_init_levels is incompatible with iid_uniform")
            elif self.energy_init_mode == "fixed":
                if (
                    UNSPEC.get("energy_init_levels") is not None
                    or UNSPEC.get("energy_init_range") is not None
                ):
                    raise ValueError("fixed energy initialization cannot define levels or a range")
            else:
                raise ValueError(
                    "energy_init_mode must be one of 'fixed', 'iid_uniform', or 'balanced_permutation'"
                )
        self.arrival_mode = str(UNSPEC.get("arrival_mode", "steady"))
        self.burst_period = int(UNSPEC.get("burst_period", 10))
        self.on_duty = float(UNSPEC.get("on_duty", 0.4))
        self.onoff_preserve_load = bool(UNSPEC.get("onoff_preserve_load", True))
        self.deadline_ttl = int(UNSPEC.get("deadline_ttl", 0))
        self.use_deadline = self.deadline_ttl > 0
        self.sd_deadline_ttl = int(UNSPEC.get("sd_deadline_ttl", 0))
        self.use_sd_deadline = self.sd_deadline_ttl > 0
        self.use_channel = bool(UNSPEC.get("use_channel", False))
        self.ch_gamma0 = float(UNSPEC.get("ch_gamma0", 2000.0))
        self.ch_alpha = float(UNSPEC.get("ch_alpha", 2.0))
        self.ch_H = float(UNSPEC.get("ch_H", SYS.get("altitude", 100.0)))
        self.ch_link_scale = float(UNSPEC.get("ch_link_scale", 110000.0))
        self.ch_eta_min = float(UNSPEC.get("ch_eta_min", 0.5))
        self.ch_alloc = str(UNSPEC.get("ch_alloc", "A2"))
        self.idle_penalty = float(UNSPEC.get("idle_penalty", 0.0))
        self.uav_init_jitter = float(UNSPEC.get("uav_init_jitter", 0.0))
        self.uav_init_random = bool(UNSPEC.get("uav_init_random", False))
        self.patch_g = int(UNSPEC.get("patch_grid_g", 5))
        gc = (np.arange(self.patch_g) + 0.5) * (self.l_max / self.patch_g)
        cx, cy = np.meshgrid(gc, gc)
        self._cell_centers = np.stack([cx.ravel(), cy.ravel()], axis=1).astype(np.float32)
        if use_grid_belief is None:
            use_grid_belief = bool(UNSPEC.get("use_grid_belief", False))
        self.use_grid_belief = bool(use_grid_belief)
        self._n_cells = self.patch_g * self.patch_g
        self.use_grid_c4 = self.use_grid_belief and bool(UNSPEC.get("grid_belief_c4", False))
        _gdn = UNSPEC.get("grid_density_norm", 0)
        self.grid_density_norm = float(_gdn) if _gdn else self.l_queue_max
        self.assign_target_mode = UNSPEC.get("assign_target_mode")
        if self.assign_target_mode not in {None, "cell", "cluster", "external"}:
            raise ValueError(
                "assign_target_mode must be one of None, 'cell', 'cluster', or 'external'"
            )
        self.assignment_target_mode_canonical = (
            "cell" if self.assign_target_mode == "cluster" else self.assign_target_mode
        )
        self.assignment_target_unit = "coarse_cell" if self.assign_target_mode is not None else None
        self.ego_patch = bool(UNSPEC.get("ego_patch", False))
        self.ego_half = float(UNSPEC.get("ego_half", 20.0))
        self.ego_p = int(UNSPEC.get("ego_p", 8))
        self.ego_sense = float(UNSPEC.get("ego_sense", 15.0))
        self.drop_global_belief = bool(UNSPEC.get("drop_global_belief", False))
        self._assign_target = None
        self._external_target = None
        self.goal_offreg = float(UNSPEC.get("goal_offreg", 1.0))
        self.goal_region_r = float(UNSPEC.get("goal_region_r", 25.0))
        self.goal_attribution_mode = str(UNSPEC.get("goal_attribution_mode", "processing_proxy"))
        if self.goal_attribution_mode not in {"processing_proxy", "split_admission_completion"}:
            raise ValueError(
                "goal_attribution_mode must be 'processing_proxy' or 'split_admission_completion'"
            )
        self.depletion_mask_mode = str(UNSPEC.get("depletion_mask_mode", "legacy"))
        if self.depletion_mask_mode not in {"legacy", "agent_terminal_v1"}:
            raise ValueError("depletion_mask_mode must be 'legacy' or 'agent_terminal_v1'")
        self.goal_source = str(UNSPEC.get("goal_source", "memory"))
        self.goal_obs_ttl = bool(UNSPEC.get("goal_obs_ttl", False))
        self._goal_credit_w = None
        self._goal_onregion_admitted = None
        self.seed = None if seed is None else int(seed)
        self._initialize_rng_streams(seed)
        self.gu_rejection_stats = {
            "proposal_count": 0,
            "rejection_count": 0,
            "rejection_rate": 0.0,
            "max_attempts_for_single_gu": 0,
        }
        self._task_arrival_trace: np.ndarray | None = None
        self._task_size_trace: np.ndarray | None = None
        if self.use_grid_belief:
            sd_block = (4 if self.use_grid_c4 else 3) * self._n_cells
        else:
            sd_block = self.K
        self.obs_dim = 2 + sd_block + self.M + (self.M - 1) + self.M
        if self.use_uav_queue:
            self.obs_dim += self.M
        if self.use_energy_obs:
            self.obs_dim += 2 * self.M
        if self.depletion_mask_mode == "agent_terminal_v1":
            self.obs_dim += self.M
        if self.use_deadline:
            self.obs_dim += self.M
        if self.drop_global_belief and self.use_grid_belief:
            self.obs_dim -= sd_block
        if self.assign_target_mode:
            self.obs_dim += 3
            if self.goal_obs_ttl:
                self.obs_dim += 2
        if self.ego_patch:
            self.obs_dim += self.ego_p**2
        self.act_dim = 3
        self._other_mask = ~np.eye(self.M, dtype=bool)
        self._uav_ids = np.eye(self.M, dtype=np.float32)
        self.reset()

    def _initialize_rng_streams(self, seed: int | None) -> None:
        master = np.random.SeedSequence(seed)
        children = master.spawn(len(self.RNG_STREAM_NAMES))
        self._rng_streams = {
            name: np.random.default_rng(child)
            for name, child in zip(self.RNG_STREAM_NAMES, children)
        }
        self.rng = self._rng_streams["miscellaneous"]
        self.rng_bit_generator = type(self.rng.bit_generator).__name__

    def _rng(self, name: str) -> np.random.Generator:
        try:
            return self._rng_streams[name]
        except KeyError as error:
            raise KeyError(f"unknown RNG stream: {name}") from error

    def _load_scenario_artifact_state(self, artifact: Mapping[str, Any]) -> None:
        metadata = dict(artifact.get("metadata", {}))
        declared_shape = {
            "num_uavs": int(self.M),
            "num_gus": int(self.K),
            "mission_slots": int(self.T),
            "mission_side_length_m": float(self.l_max),
        }
        for key, expected_value in declared_shape.items():
            if key not in metadata or float(metadata[key]) != float(expected_value):
                raise ValueError(
                    f"scenario artifact {key} mismatch: expected {expected_value}, got {metadata.get(key)!r}"
                )
        expected = {
            "gu_positions",
            "gu_generation_probabilities",
            "cluster_centers",
            "cluster_labels",
            "activation_schedule",
            "task_arrival_indicators",
            "task_sizes_bits",
            "initial_uav_positions",
            "initial_energy_budgets",
        }
        missing = sorted(expected - set(artifact))
        if missing:
            raise ValueError(f"scenario artifact missing arrays: {missing}")
        self.scenario_active = str(metadata.get("scenario", self.scenario))
        self.sd_pos = np.asarray(artifact["gu_positions"], dtype=np.float32).copy()
        self.alpha = np.asarray(artifact["gu_generation_probabilities"], dtype=np.float32).copy()
        if self.sd_pos.shape != (self.K, 2) or self.alpha.shape != (self.K,):
            raise ValueError(
                f"scenario artifact GU shapes disagree with environment: positions={self.sd_pos.shape}, alpha={self.alpha.shape}, expected={(self.K, 2)} and {(self.K,)}"
            )
        if (
            not np.isfinite(self.alpha).all()
            or np.any(self.alpha < 0.0)
            or np.any(self.alpha > 1.0)
        ):
            raise ValueError("scenario artifact contains invalid GU generation probabilities")
        if (
            not np.isfinite(self.sd_pos).all()
            or np.any(self.sd_pos < 0.0)
            or np.any(self.sd_pos > self.l_max)
        ):
            raise ValueError("scenario artifact contains an out-of-area GU")
        at_boundary = np.isclose(self.sd_pos, 0.0, atol=0.0, rtol=0.0) | np.isclose(
            self.sd_pos, self.l_max, atol=0.0, rtol=0.0
        )
        if np.any(at_boundary):
            raise ValueError(
                "rejection-sampled scenario artifact contains a GU projected exactly onto the mission boundary"
            )
        centers = np.asarray(artifact["cluster_centers"], dtype=np.float32)
        labels = np.asarray(artifact["cluster_labels"], dtype=np.int64)
        if labels.size == 0:
            self.sd_cluster = None
            self.n_clusters = 0
            self.cluster_centers = None
            self.cluster_sizes = None
            self.cluster_is_large = None
        else:
            if labels.shape != (self.K,):
                raise ValueError("scenario artifact cluster-label shape mismatch")
            self.sd_cluster = labels.copy()
            self.n_clusters = int(centers.shape[0])
            if centers.shape != (self.n_clusters, 2):
                raise ValueError("scenario artifact cluster-centre shape mismatch")
            if not np.isfinite(centers).all():
                raise ValueError("scenario artifact contains a non-finite cluster centre")
            if np.any(labels < 0) or np.any(labels >= self.n_clusters):
                raise ValueError("scenario artifact contains an invalid cluster label")
            margin = float(metadata.get("cluster_center_margin_m", 0.0))
            if np.any(centers < margin) or np.any(centers > self.l_max - margin):
                raise ValueError("scenario artifact cluster centre violates the declared margin")
            minimum_separation = float(metadata.get("cluster_center_min_separation_m", 0.0))
            if self.n_clusters > 1:
                delta = centers[:, None, :] - centers[None, :, :]
                distances = np.linalg.norm(delta, axis=2)
                np.fill_diagonal(distances, np.inf)
                if float(distances.min()) + 1e-05 < minimum_separation:
                    raise ValueError(
                        "scenario artifact cluster centres violate the declared minimum separation"
                    )
            self.cluster_centers = centers.copy()
            self.cluster_sizes = np.bincount(labels, minlength=self.n_clusters).astype(int).tolist()
            large = metadata.get("cluster_is_large")
            self.cluster_is_large = (
                [bool(value) for value in large] if large is not None else [False] * self.n_clusters
            )
        sigma = metadata.get("gu_sigma_m")
        if sigma is not None:
            self.cluster_sigma = float(sigma)
            self.cluster_radius = float(sigma)
        self.activation_schedule = np.asarray(artifact["activation_schedule"], dtype=bool).copy()
        self._task_arrival_trace = np.asarray(
            artifact["task_arrival_indicators"], dtype=bool
        ).copy()
        self._task_size_trace = np.asarray(artifact["task_sizes_bits"], dtype=np.float32).copy()
        if self.activation_schedule.shape != (self.T, self.n_clusters):
            raise ValueError("scenario artifact activation-schedule shape mismatch")
        if self._task_arrival_trace.shape != (self.T, self.K) or self._task_size_trace.shape != (
            self.T,
            self.K,
        ):
            raise ValueError("scenario artifact workload-trace shape mismatch")
        if (
            not np.isfinite(self._task_size_trace).all()
            or np.any(self._task_size_trace < 0.0)
            or np.any(self._task_size_trace > self.l_queue_max)
        ):
            raise ValueError("scenario artifact contains an invalid task size")
        initial_positions = np.asarray(artifact["initial_uav_positions"], dtype=np.float32)
        initial_budgets = np.asarray(artifact["initial_energy_budgets"], dtype=np.float32)
        if initial_positions.shape != (self.M, 2):
            raise ValueError("scenario artifact initial-UAV-position shape mismatch")
        if initial_budgets.shape != (self.M,):
            raise ValueError("scenario artifact initial-budget shape mismatch")
        if np.any(initial_positions < 0.0) or np.any(initial_positions > self.l_max):
            raise ValueError("scenario artifact contains an invalid initial UAV position")
        if not np.isfinite(initial_budgets).all() or np.any(initial_budgets <= 0.0):
            raise ValueError("scenario artifact contains an invalid initial budget")
        rejection = metadata.get("gu_rejection_stats")
        if rejection is not None:
            loaded_rejection = {
                "proposal_count": int(rejection["proposal_count"]),
                "rejection_count": int(rejection["rejection_count"]),
                "rejection_rate": float(rejection["rejection_rate"]),
                "max_attempts_for_single_gu": int(rejection["max_attempts_for_single_gu"]),
            }
            proposals = loaded_rejection["proposal_count"]
            rejections = loaded_rejection["rejection_count"]
            expected_rate = rejections / proposals if proposals else 0.0
            if (
                proposals < self.K
                or rejections != proposals - self.K
                or (
                    not np.isclose(
                        loaded_rejection["rejection_rate"], expected_rate, rtol=0.0, atol=1e-12
                    )
                )
                or (loaded_rejection["max_attempts_for_single_gu"] < 1)
            ):
                raise ValueError("scenario artifact contains inconsistent GU rejection statistics")
            self.gu_rejection_stats = loaded_rejection

    def export_scenario_state(self, *, scenario_id: str, scenario_seed: int) -> dict[str, Any]:
        if self._task_arrival_trace is None or self._task_size_trace is None:
            raise RuntimeError("scenario artifacts require env_v2 pre-generated task traces")
        centers = (
            np.empty((0, 2), dtype=np.float32)
            if self.cluster_centers is None
            else np.asarray(self.cluster_centers, dtype=np.float32)
        )
        labels = (
            np.empty((0,), dtype=np.int64)
            if self.sd_cluster is None
            else np.asarray(self.sd_cluster, dtype=np.int64)
        )
        return {
            "metadata": {
                "scenario_seed": int(scenario_seed),
                "scenario": str(self.scenario_active),
                "gu_position_distribution": "bivariate_isotropic_gaussian",
                "gu_location_parameter": "cluster_center",
                "gu_sigma_m": float(getattr(self, "cluster_sigma", 0.0)),
                "gu_position_support": "[0,L]^2",
                "gu_boundary_handling": "reject_and_resample_full_2d_proposal",
                "gu_boundary_projection": False,
                "gu_max_resampling_rounds": self.gu_max_resampling_rounds,
                "cluster_center_margin_m": float(self._hetero_cfg.get("center_margin", 0.0)),
                "cluster_center_min_separation_m": float(
                    self._hetero_cfg.get("min_center_dist", 0.0)
                ),
                "rng_bit_generator": self.rng_bit_generator,
                "cluster_is_large": (
                    None
                    if self.cluster_is_large is None
                    else [bool(value) for value in self.cluster_is_large]
                ),
                "gu_rejection_stats": copy.deepcopy(self.gu_rejection_stats),
                "num_uavs": int(self.M),
                "num_gus": int(self.K),
                "mission_slots": int(self.T),
                "mission_side_length_m": float(self.l_max),
            },
            "gu_positions": self.sd_pos.copy(),
            "gu_generation_probabilities": self.alpha.copy(),
            "cluster_centers": centers.copy(),
            "cluster_labels": labels.copy(),
            "activation_schedule": self.activation_schedule.copy(),
            "task_arrival_indicators": self._task_arrival_trace.copy(),
            "task_sizes_bits": self._task_size_trace.copy(),
            "initial_uav_positions": self.uav_pos.copy(),
            "initial_energy_budgets": self.energy_init.copy(),
        }

    def reset(self) -> list[np.ndarray]:
        self.gu_rejection_stats = {
            "proposal_count": 0,
            "rejection_count": 0,
            "rejection_rate": 0.0,
            "max_attempts_for_single_gu": 0,
        }
        self._task_arrival_trace = None
        self._task_size_trace = None
        if self._scenario_artifact is not None:
            self._load_scenario_artifact_state(self._scenario_artifact)
        else:
            scen = self.scenario
            if scen == "mixed":
                scen = str(self._rng("scenario").choice(["uniform", "hetero"]))
            self.scenario_active = scen
            if scen == "hetero":
                self.sd_pos, self.alpha = self._place_hetero()
            else:
                self.sd_pos, self.alpha = self._place_uniform()
        ci = np.clip((self.sd_pos / (self.l_max / self.patch_g)).astype(int), 0, self.patch_g - 1)
        self.sd_cell = (ci[:, 1] * self.patch_g + ci[:, 0]).astype(np.int64)
        self.cell_last_density = np.zeros(self._n_cells, dtype=np.float32)
        self.cell_last_seen = np.full(self._n_cells, -1000000000.0, dtype=np.float32)
        self.cell_obs_sum = np.zeros(self._n_cells, dtype=np.float32)
        self.cell_obs_cnt = np.zeros(self._n_cells, dtype=np.float32)
        if self._scenario_artifact is None:
            self._build_activation_schedule()
            self._build_task_trace()
        if self._scenario_artifact is not None:
            self.uav_pos = np.asarray(
                self._scenario_artifact["initial_uav_positions"], dtype=np.float32
            ).copy()
        elif self.uav_init_random:
            margin = 0.05 * self.l_max
            for _ in range(200):
                cand = (
                    self._rng("uav_initial_state")
                    .uniform(margin, self.l_max - margin, size=(self.M, 2))
                    .astype(np.float32)
                )
                dd = np.linalg.norm(cand[:, None, :] - cand[None, :, :], axis=2)
                np.fill_diagonal(dd, np.inf)
                if dd.min() > 2.0 * self.r_min:
                    break
            self.uav_pos = cand
        elif self.uav_init_jitter > 0:
            jit = (
                self._rng("uav_initial_state")
                .uniform(-self.uav_init_jitter, self.uav_init_jitter, size=self.uav_init.shape)
                .astype(np.float32)
            )
            self.uav_pos = np.clip(self.uav_init + jit, 0.0, self.l_max).astype(np.float32)
        else:
            self.uav_pos = self.uav_init.copy()
        if self._scenario_artifact is not None:
            self.energy_remaining = np.asarray(
                self._scenario_artifact["initial_energy_budgets"], dtype=np.float32
            ).copy()
        elif self.energy_init_mode == "balanced_permutation":
            self.energy_remaining = (
                self._rng("initial_budget").permutation(self.energy_init_levels).astype(np.float32)
            )
        elif self.energy_init_mode == "iid_uniform":
            lo, hi = self.energy_init_range
            self.energy_remaining = (
                self._rng("initial_budget")
                .uniform(float(lo), float(hi), size=self.M)
                .astype(np.float32)
            )
        elif self.energy_init_mode == "iid_uniform_fraction_legacy":
            lo, hi = self.energy_init_fraction_range
            fractions = self._rng("initial_budget").uniform(float(lo), float(hi), size=self.M)
            self.energy_remaining = (fractions * self.e_uav).astype(np.float32)
        else:
            self.energy_remaining = np.full(self.M, self.e_uav, dtype=np.float32)
        self.energy_init = self.energy_remaining.copy()
        self.energy_total_used = np.zeros(self.M, dtype=np.float32)
        self.active = np.ones(self.M, dtype=bool)
        self.coordination_leader_index = 0 if self.M > 0 else None
        self.coordination_leader_handover_count = 0
        self.coordination_leader_handover_slots: list[int] = []
        self.energy_violated = np.zeros(self.M, dtype=bool)
        self.violation_slot = -np.ones(self.M, dtype=np.int32)
        self.cumulative_excess = np.zeros(self.M, dtype=np.float32)
        self.peak_excess = np.zeros(self.M, dtype=np.float32)
        self.queue = np.zeros(self.K, dtype=np.float32)
        if self.use_sd_deadline:
            self.sd_queue_ttl = np.zeros((self.K, self.sd_deadline_ttl), dtype=np.float32)
            self.sd_missed = np.zeros(self.K, dtype=np.float32)
        self.uav_queue = np.zeros(self.M, dtype=np.float32)
        self.uav_queue_peak = np.zeros(self.M, dtype=np.float32)
        if self.use_deadline:
            self.uav_queue_ttl = np.zeros((self.M, self.deadline_ttl), dtype=np.float32)
            self.missed_bits = np.zeros(self.M, dtype=np.float32)
        self.Z = np.zeros((self.M, self.K), dtype=np.float32)
        self.total_bits = 0.0
        self.bits_per_uav = np.zeros(self.M, dtype=np.float32)
        self.generated_per_sd = np.zeros(self.K, dtype=np.float32)
        self.sd_overflow_dropped_per_sd = np.zeros(self.K, dtype=np.float32)
        self.radio_per_sd = np.zeros(self.K, dtype=np.float32)
        self.admitted_per_sd = np.zeros(self.K, dtype=np.float32)
        self.buffer_dropped_per_sd = np.zeros(self.K, dtype=np.float32)
        self.served_per_sd = np.zeros(self.K, dtype=np.float32)
        self.visited_global = np.zeros(self.patch_g * self.patch_g, dtype=bool)
        self.pioneer_cum = np.zeros(self.M, dtype=np.float32)
        self.discovered_user_cells = np.zeros(self._n_cells, dtype=bool)
        self.t = 0
        return self._all_obs()

    def step(self, actions: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, bool, dict]:
        actions = np.asarray(actions, dtype=np.float32).reshape(self.M, self.act_dim)
        self._generate_tasks()
        gen_queue = self.queue.copy()
        d, theta, f = self._decode_actions(actions)
        start_pos = self.uav_pos.copy()
        alive_start = self.active.copy() if self.hard_energy_constraint else None
        if self.hard_energy_constraint:
            d = np.where(alive_start, d, 0.0).astype(np.float32)
        new_pos, boundary_hit = self._move_uavs(d, theta)
        self.uav_pos = new_pos
        cell_d = np.linalg.norm(self.uav_pos[:, None, :] - self._cell_centers[None, :, :], axis=2)
        covered = cell_d <= self.r_max
        if self.hard_energy_constraint:
            covered &= alive_start[:, None]
        new_cells = covered.any(axis=0) & ~self.visited_global
        dC = np.zeros(self.M, dtype=np.float32)
        if new_cells.any():
            owner = np.where(covered, cell_d, np.inf).argmin(axis=0)
            np.add.at(dC, owner[new_cells], 1.0)
            self.visited_global |= new_cells
        self.pioneer_cum += dC
        team_covered = covered.any(axis=0)
        if team_covered.any():
            cell_density = np.zeros(self._n_cells, dtype=np.float32)
            np.add.at(cell_density, self.sd_cell, self.queue)
            true_user_cells = np.bincount(self.sd_cell, minlength=self._n_cells) > 0
            self.discovered_user_cells[team_covered] |= true_user_cells[team_covered]
            self.cell_last_density[team_covered] = cell_density[team_covered]
            self.cell_last_seen[team_covered] = float(self.t)
            if self.use_grid_c4 or self.assign_target_mode:
                self.cell_obs_sum[team_covered] += cell_density[team_covered]
                self.cell_obs_cnt[team_covered] += 1.0
        collision_hit = self._collision_mask(self.uav_pos, alive_start)
        z = self._service_assignment(self.uav_pos)
        if self.hard_energy_constraint:
            z[~alive_start] = False
        radio_per_sd_delta = np.zeros(self.K, dtype=np.float32)
        admitted_per_sd_delta = np.zeros(self.K, dtype=np.float32)
        buffer_dropped_per_sd_delta = np.zeros(self.K, dtype=np.float32)
        if not self.use_uav_queue:
            served_load = np.where(z, self.queue[None, :], 0.0)
            N = served_load.sum(axis=1).astype(np.float32)
            served_active_mask = z & (N[:, None] > 0) & (self.queue[None, :] > 0)
            served_any = served_active_mask.any(axis=0)
            served_per_sd_delta = np.where(served_any, self.queue, 0.0).astype(np.float32)
            radio_per_sd_delta = served_per_sd_delta.copy()
            admitted_per_sd_delta = served_per_sd_delta.copy()
            self.queue = np.where(served_any, 0.0, self.queue)
        else:
            finite_buffer = np.isfinite(self.uav_queue_max) and (not self.use_deadline)
            free_buffer = (
                np.maximum(self.uav_queue_max - self.uav_queue, 0.0) if finite_buffer else None
            )
            if self.use_channel:
                receive_capacity = free_buffer if self.uav_buffer_mode == "backpressure" else None
                collected_mk, delta_k = self._collect_channel(
                    start_pos, self.uav_pos, alive_start, receive_capacity
                )
                radio_arriving = collected_mk.sum(axis=1).astype(np.float32)
                if finite_buffer and self.uav_buffer_mode == "legacy_drop":
                    admit_scale = np.where(
                        radio_arriving > free_buffer,
                        free_buffer / np.maximum(radio_arriving, 1e-12),
                        1.0,
                    )
                    admitted_mk = (collected_mk * admit_scale[:, None]).astype(np.float32)
                else:
                    admitted_mk = collected_mk
                arriving = admitted_mk.sum(axis=1).astype(np.float32)
                if (
                    self.assign_target_mode
                    and self.goal_offreg < 1.0
                    and (self._assign_target is not None)
                ):
                    ctr = self._cell_centers[self._assign_target]
                    d2 = ((self.sd_pos[None, :, :] - ctr[:, None, :]) ** 2).sum(axis=2)
                    onregion = d2 <= self.goal_region_r**2
                    w = np.where(onregion, 1.0, self.goal_offreg)
                    cred = (admitted_mk * w).sum(axis=1)
                    self._goal_credit_w = np.where(
                        arriving > 1e-09, cred / np.maximum(arriving, 1e-09), 1.0
                    ).astype(np.float32)
                    self._goal_onregion_admitted = (
                        (admitted_mk * onregion.astype(np.float32)).sum(axis=1).astype(np.float32)
                    )
                else:
                    self._goal_credit_w = None
                    self._goal_onregion_admitted = None
                served_per_sd_delta = delta_k
                served_active_mask = collected_mk > 0
                served_any = delta_k > 1e-09
                radio_per_sd_delta = delta_k
                admitted_per_sd_delta = admitted_mk.sum(axis=0).astype(np.float32)
                buffer_dropped_per_sd_delta = np.maximum(
                    radio_per_sd_delta - admitted_per_sd_delta, 0.0
                ).astype(np.float32)
                if self.use_sd_deadline:
                    rem = delta_k.copy()
                    for r in range(self.sd_deadline_ttl):
                        take = np.minimum(self.sd_queue_ttl[:, r], rem)
                        self.sd_queue_ttl[:, r] -= take
                        rem -= take
                    self.queue = self.sd_queue_ttl.sum(axis=1).astype(np.float32)
                else:
                    self.queue = np.maximum(self.queue - delta_k, 0.0).astype(np.float32)
            else:
                transfer = np.where(z, self.queue[None, :], 0.0).astype(np.float32)
                radio_arriving = transfer.sum(axis=1).astype(np.float32)
                if finite_buffer:
                    admit_scale = np.where(
                        radio_arriving > free_buffer,
                        free_buffer / np.maximum(radio_arriving, 1e-12),
                        1.0,
                    )
                else:
                    admit_scale = np.ones(self.M, dtype=np.float32)
                admitted_mk = (transfer * admit_scale[:, None]).astype(np.float32)
                if self.uav_buffer_mode == "backpressure":
                    transfer = admitted_mk
                    radio_arriving = transfer.sum(axis=1).astype(np.float32)
                arriving = admitted_mk.sum(axis=1).astype(np.float32)
                radio_per_sd_delta = transfer.sum(axis=0).astype(np.float32)
                admitted_per_sd_delta = admitted_mk.sum(axis=0).astype(np.float32)
                buffer_dropped_per_sd_delta = np.maximum(
                    radio_per_sd_delta - admitted_per_sd_delta, 0.0
                ).astype(np.float32)
                served_per_sd_delta = radio_per_sd_delta
                served_any = radio_per_sd_delta > 1e-09
                self.queue = np.maximum(self.queue - radio_per_sd_delta, 0.0).astype(np.float32)
                served_active_mask = transfer > 0
            slot_capacity = self._slot_capacity(f)
            if self.hard_energy_constraint:
                slot_capacity = np.where(alive_start, slot_capacity, 0.0)
            if self.use_deadline:
                self.uav_queue_ttl[:, -1] += arriving
                cum_before = np.cumsum(self.uav_queue_ttl, axis=1) - self.uav_queue_ttl
                drain = np.clip(slot_capacity[:, None] - cum_before, 0.0, self.uav_queue_ttl)
                N = drain.sum(axis=1).astype(np.float32)
                self.uav_queue_ttl = (self.uav_queue_ttl - drain).astype(np.float32)
                self.missed_bits += self.uav_queue_ttl[:, 0]
                self.uav_queue_ttl[:, :-1] = self.uav_queue_ttl[:, 1:]
                self.uav_queue_ttl[:, -1] = 0.0
                self.uav_queue = self.uav_queue_ttl.sum(axis=1).astype(np.float32)
            else:
                self.uav_queue = np.minimum(self.uav_queue + arriving, self.uav_queue_max).astype(
                    np.float32
                )
                N = np.minimum(self.uav_queue, slot_capacity).astype(np.float32)
                self.uav_queue = self.uav_queue - N
            self.uav_queue_peak = np.maximum(self.uav_queue_peak, self.uav_queue)
        if self.use_sd_deadline:
            if not self.use_channel:
                rem = radio_per_sd_delta.copy()
                for r in range(self.sd_deadline_ttl):
                    take = np.minimum(self.sd_queue_ttl[:, r], rem)
                    self.sd_queue_ttl[:, r] -= take
                    rem -= take
            self.sd_missed += self.sd_queue_ttl[:, 0]
            self.sd_queue_ttl[:, :-1] = self.sd_queue_ttl[:, 1:]
            self.sd_queue_ttl[:, -1] = 0.0
            self.queue = self.sd_queue_ttl.sum(axis=1).astype(np.float32)
        E_total = self._energy_consumption(d, f, N)
        if self.hard_energy_constraint:
            E_total = np.where(alive_start, E_total, 0.0).astype(np.float32)
        prev_remaining = self.energy_remaining.copy()
        self.energy_remaining = self.energy_remaining - E_total
        self.energy_total_used += E_total
        slot_excess = np.maximum(E_total - np.maximum(prev_remaining, 0.0), 0.0).astype(np.float32)
        new_violators = ~self.energy_violated & (slot_excess > 0)
        self.violation_slot = np.where(new_violators, self.t, self.violation_slot)
        self.energy_violated = self.energy_violated | (slot_excess > 0)
        self.cumulative_excess += slot_excess
        self.peak_excess = np.maximum(self.peak_excess, slot_excess)
        excess = np.maximum(-self.energy_remaining, 0.0).astype(np.float32)
        if self.hard_energy_constraint:
            self.active = alive_start & (self.energy_remaining > 0)
        else:
            self.active = self.energy_remaining > 0
        coordination_active = (
            self.active if self.hard_energy_constraint else np.ones(self.M, dtype=bool)
        )
        self._ensure_coordination_leader(coordination_active)
        self.Z += served_active_mask.astype(np.float32)
        self.served_per_sd += served_per_sd_delta
        self.radio_per_sd += radio_per_sd_delta
        self.admitted_per_sd += admitted_per_sd_delta
        self.buffer_dropped_per_sd += buffer_dropped_per_sd_delta
        self.total_bits += float(N.sum())
        self.bits_per_uav += N
        F_sd = self._fairness_sd()
        F_uav = self._fairness_uav()
        rewards = movement_reward(
            completed_bits=N,
            new_coarse_cells=dC,
            active_at_slot_start=alive_start,
            boundary_hit=boundary_hit,
            collision_hit=collision_hit,
            energy_remaining_after_slot=self.energy_remaining,
            reward_scale=self.reward_scale,
            gamma_e=self.gamma_e,
            idle_penalty=self.idle_penalty,
            boundary_collision_penalty=self.penalty,
            energy_overrun_base=self.energy_overrun_base,
            energy_overrun_per_excess=self.energy_overrun_per_excess,
        )
        self.t += 1
        done = self.t >= self.T
        info = {
            "N": N.copy(),
            "F_sd": F_sd.copy(),
            "F_uav": float(F_uav),
            "energy_remaining": self.energy_remaining.copy(),
            "energy_total_used": self.energy_total_used.copy(),
            "energy_violated": self.energy_violated.copy(),
            "violation_slot": self.violation_slot.copy(),
            "cumulative_excess": self.cumulative_excess.copy(),
            "peak_excess": self.peak_excess.copy(),
            "slot_excess": slot_excess.copy(),
            "active": self.active.copy(),
            "coordination_leader_index": self.coordination_leader_index,
            "coordination_leader_handover_count": self.coordination_leader_handover_count,
            "boundary_hit": boundary_hit.copy(),
            "collision_hit": collision_hit.copy(),
            "total_bits": self.total_bits,
            "uav_queue": self.uav_queue.copy(),
        }
        return (self._all_obs(), rewards, done, info)

    def _place_uniform(self) -> tuple[np.ndarray, np.ndarray]:
        self.cluster_centers = None
        self.cluster_sizes = None
        self.cluster_is_large = None
        self.sd_cluster = None
        self.n_clusters = 0
        sd_pos = (
            self._rng("gu_position").uniform(0, self.l_max, size=(self.K, 2)).astype(np.float32)
        )
        if self.alpha_dist == "normal":
            mu, sigma = self.alpha_params
            alpha = np.clip(
                self._rng("cluster_assignment").normal(mu, sigma, size=self.K), 0.001, 1.0
            )
        else:
            alpha = self._rng("cluster_assignment").uniform(0.0, 1.0, size=self.K)
        return (sd_pos, alpha.astype(np.float32))

    def _sample_gu_positions(self, *, center: np.ndarray, count: int, sigma: float) -> np.ndarray:
        count = int(count)
        sigma = float(sigma)
        if count < 0 or sigma <= 0:
            raise ValueError("GU count must be nonnegative and sigma positive")
        points = np.empty((count, 2), dtype=np.float64)
        pending = np.arange(count, dtype=np.int64)
        attempts = np.zeros(count, dtype=np.int32)
        proposal_count = 0
        generator = self._rng("gu_position")
        for _round in range(self.gu_max_resampling_rounds):
            if pending.size == 0:
                break
            candidates = generator.normal(
                loc=np.asarray(center, dtype=np.float64), scale=sigma, size=(pending.size, 2)
            ).astype(np.float32)
            attempts[pending] += 1
            proposal_count += int(pending.size)
            valid = np.all((candidates > 0.0) & (candidates < self.l_max), axis=1)
            points[pending[valid]] = candidates[valid]
            pending = pending[~valid]
        if pending.size:
            raise RuntimeError(
                f"GU rejection sampling did not terminate after {self.gu_max_resampling_rounds} rounds; remaining={pending.size}, center={np.asarray(center).tolist()}, sigma={sigma}, l_max={self.l_max}"
            )
        rejection_count = proposal_count - count
        stats = self.gu_rejection_stats
        stats["proposal_count"] += int(proposal_count)
        stats["rejection_count"] += int(rejection_count)
        stats["max_attempts_for_single_gu"] = max(
            int(stats["max_attempts_for_single_gu"]), int(attempts.max(initial=0))
        )
        stats["rejection_rate"] = (
            float(stats["rejection_count"]) / float(stats["proposal_count"])
            if stats["proposal_count"]
            else 0.0
        )
        return points.astype(np.float32)

    def _place_hetero(self) -> tuple[np.ndarray, np.ndarray]:
        cfg = self._hetero_cfg
        n_clusters = int(cfg.get("n_clusters", 4))
        if self._k_range is not None:
            n_clusters = int(
                self._rng("cluster_assignment").integers(
                    int(self._k_range[0]), int(self._k_range[1]) + 1
                )
            )
        large_count = int(cfg.get("large_count", 1))
        sds_large = int(cfg.get("sds_large", 20))
        sds_small = int(cfg.get("sds_small", 10))
        rem = self.K - sds_large * large_count
        n_small = max(n_clusters - large_count, 1)
        if sds_small * (n_small - 1) >= rem:
            sds_small = max(1, rem // n_small)
        sigma_clu = float(cfg.get("cluster_radius", 12.0))
        min_dist = float(cfg.get("min_center_dist", 40.0))
        margin = float(cfg.get("center_margin", 18.0))
        rho_large = float(cfg.get("rho_large", 2.0))
        rho_small = float(cfg.get("rho_small", 0.5))
        counts = [sds_large] * large_count + [sds_small] * (n_clusters - large_count)
        counts[-1] += self.K - int(np.sum(counts))
        if counts[-1] < 1 or any((c < 1 for c in counts)):
            raise ValueError(f"hetero cluster counts must be >=1 and sum to K={self.K}: {counts}")
        centers = self._sample_cluster_centers(n_clusters, min_dist, margin)
        pos_blocks, alpha_blocks, cluster_blocks = ([], [], [])
        for ci, (center, n) in enumerate(zip(centers, counts)):
            pts = self._sample_gu_positions(center=center, count=n, sigma=sigma_clu)
            pos_blocks.append(pts)
            rho = rho_large if ci < large_count else rho_small
            a = rho * self.f_max / (n * self.task_mean)
            alpha_blocks.append(np.full(n, np.clip(a, 0.001, 1.0), dtype=np.float32))
            cluster_blocks.append(np.full(n, ci, dtype=np.int64))
        sd_pos = np.concatenate(pos_blocks, axis=0).astype(np.float32)
        alpha = np.concatenate(alpha_blocks, axis=0).astype(np.float32)
        self.sd_cluster = np.concatenate(cluster_blocks).astype(np.int64)
        self.n_clusters = int(n_clusters)
        self.cluster_centers = centers
        self.cluster_sizes = list(counts)
        self.cluster_is_large = [ci < large_count for ci in range(n_clusters)]
        self.cluster_sigma = sigma_clu
        self.cluster_radius = sigma_clu
        return (sd_pos, alpha)

    def _sample_cluster_centers(self, n: int, min_dist: float, margin: float) -> np.ndarray:
        lo, hi = (margin, self.l_max - margin)
        generator = self._rng("cluster_center")
        for _restart in range(100):
            centers: list[np.ndarray] = []
            for _ in range(10000):
                if len(centers) == n:
                    return np.stack(centers).astype(np.float32)
                cand = generator.uniform(lo, hi, size=2)
                if all((np.linalg.norm(cand - c) >= min_dist for c in centers)):
                    centers.append(cand)
        raise RuntimeError(
            f"cluster-centre rejection sampling could not satisfy the configured geometry: n={n}, margin={margin}, min_dist={min_dist}, l_max={self.l_max}"
        )

    def _periodic_active(self) -> np.ndarray:
        nc = int(self.n_clusters)
        c = np.arange(nc)
        phase = (c * self.burst_period / max(nc, 1)).astype(np.float32)
        on_len = self.on_duty * self.burst_period
        return (self.t + phase) % self.burst_period < on_len

    def _build_activation_schedule(self) -> None:
        nc = int(self.n_clusters)
        T = self.T
        if nc == 0:
            self.activation_schedule = np.zeros((T, 0), dtype=bool)
            return
        ts = np.arange(T)[:, None]
        c = np.arange(nc)[None, :]
        phase = c * self.burst_period / max(nc, 1)
        on_len = self.on_duty * self.burst_period
        periodic = (ts + phase) % self.burst_period < on_len
        sched = periodic
        self.activation_schedule = sched.astype(bool)

    def _cluster_active(self) -> np.ndarray:
        if int(self.n_clusters) == 0:
            return np.zeros(0, dtype=bool)
        return self.activation_schedule[min(self.t, self.T - 1)]

    def _arrival_probability_for_slot(self, slot: int) -> np.ndarray:
        alpha = np.asarray(self.alpha, dtype=np.float64)
        if self.arrival_mode == "onoff" and self.sd_cluster is not None:
            active = self.activation_schedule[min(int(slot), self.T - 1)][self.sd_cluster].astype(
                np.float64
            )
            if self.onoff_preserve_load and self.on_duty > 0:
                alpha = np.minimum(alpha / self.on_duty, 1.0) * active
            else:
                alpha = alpha * active
        return np.asarray(alpha, dtype=np.float64)

    def _build_task_trace(self) -> None:
        probabilities = np.stack(
            [self._arrival_probability_for_slot(slot) for slot in range(self.T)], axis=0
        )
        uniforms = self._rng("arrival").random((self.T, self.K))
        self._task_arrival_trace = uniforms < probabilities
        self._task_size_trace = np.clip(
            self._rng("task_size").normal(self.task_mean, self.task_std, size=(self.T, self.K)),
            0.0,
            self.l_queue_max,
        ).astype(np.float32)

    def _generate_tasks(self) -> None:
        if self._task_arrival_trace is not None:
            slot = min(int(self.t), self.T - 1)
            gen = self._task_arrival_trace[slot]
            sizes = self._task_size_trace[slot]
            new_load = np.where(gen, sizes, 0.0).astype(np.float32)
            self.generated_per_sd += new_load
            if self.use_sd_deadline:
                self.sd_queue_ttl[:, -1] += new_load
                self.queue = self.sd_queue_ttl.sum(axis=1).astype(np.float32)
            else:
                before = self.queue.copy()
                self.queue = np.minimum(before + new_load, self.l_queue_max)
                admitted = self.queue - before
                self.sd_overflow_dropped_per_sd += np.maximum(new_load - admitted, 0.0)
            return
        alpha = self.alpha
        if self.arrival_mode == "onoff" and self.sd_cluster is not None:
            amask = self._cluster_active()[self.sd_cluster].astype(np.float32)
            if self.onoff_preserve_load and self.on_duty > 0:
                alpha = np.minimum(alpha / self.on_duty, 1.0) * amask
            else:
                alpha = alpha * amask
        gen = self._rng("arrival").uniform(size=self.K) < alpha
        sizes = np.clip(
            self._rng("task_size").normal(self.task_mean, self.task_std, size=self.K),
            0.0,
            self.l_queue_max,
        ).astype(np.float32)
        new_load = np.where(gen, sizes, 0.0).astype(np.float32)
        self.generated_per_sd += new_load
        if self.use_sd_deadline:
            self.sd_queue_ttl[:, -1] += new_load
            self.queue = self.sd_queue_ttl.sum(axis=1).astype(np.float32)
        else:
            before = self.queue.copy()
            self.queue = np.minimum(before + new_load, self.l_queue_max)
            admitted = self.queue - before
            self.sd_overflow_dropped_per_sd += np.maximum(new_load - admitted, 0.0)

    def _decode_actions(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.clip(actions, -1.0, 1.0)
        u = (a + 1.0) * 0.5
        d = u[:, 0] * self.d_max
        theta = u[:, 1] * 2.0 * np.pi
        f = self.f_min + u[:, 2] * (self.f_max - self.f_min)
        return (d, theta, f)

    def _move_uavs(self, d: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        delta = np.stack([d * np.cos(theta), d * np.sin(theta)], axis=1).astype(np.float32)
        tentative = self.uav_pos + delta
        clipped = np.clip(tentative, 0.0, self.l_max)
        boundary_hit = np.any(np.abs(clipped - tentative) > 1e-06, axis=1)
        return (clipped.astype(np.float32), boundary_hit)

    def _collision_mask(self, positions: np.ndarray, alive: np.ndarray | None = None) -> np.ndarray:
        diff = positions[:, None, :] - positions[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        if alive is not None:
            dead = ~alive
            dists[dead, :] = np.inf
            dists[:, dead] = np.inf
        return np.any(dists < self.r_min, axis=1)

    def _service_assignment(self, positions: np.ndarray) -> np.ndarray:
        diff = positions[:, None, :] - self.sd_pos[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        in_range = dist <= self.r_max
        has_task = self.queue > 0
        in_range &= has_task[None, :]
        masked_dist = np.where(in_range, dist, np.inf)
        best_uav = np.argmin(masked_dist, axis=0)
        sd_served = np.min(masked_dist, axis=0) < np.inf
        z = np.zeros((self.M, self.K), dtype=bool)
        served_idx = np.where(sd_served)[0]
        z[best_uav[served_idx], served_idx] = True
        return z

    def _spectral_eff(self, positions: np.ndarray) -> np.ndarray:
        diff = positions[:, None, :] - self.sd_pos[None, :, :]
        d2 = (diff * diff).sum(axis=2)
        gain = self.ch_gamma0 / np.power(d2 + self.ch_H * self.ch_H, self.ch_alpha / 2.0)
        return np.log2(1.0 + gain).astype(np.float32)

    def _ensure_coordination_leader(self, active: np.ndarray) -> int | None:
        live = np.flatnonzero(np.asarray(active, dtype=bool))
        if live.size == 0:
            self.coordination_leader_index = None
            return None
        current = self.coordination_leader_index
        if current is not None and 0 <= int(current) < self.M and active[int(current)]:
            return int(current)
        replacement = int(live[0])
        if current != replacement:
            self.coordination_leader_handover_count += 1
            self.coordination_leader_handover_slots.append(int(self.t))
        self.coordination_leader_index = replacement
        return replacement

    def _memory_assign(self, w_explore: float = 0.5, source: str = "memory"):
        if source == "checkin_eq":
            ttl = float(self.sd_deadline_ttl) if self.use_sd_deadline else 8.0
            known_site = self.discovered_user_cells.astype(np.float32)
            stale_ttl = np.clip((self.t - self.cell_last_seen) / max(ttl, 1.0), 0.0, 1.0)
            score = known_site * (0.25 + 0.75 * stale_ttl) + 0.5 * (self.cell_obs_cnt <= 0)
        else:
            meanpast = np.where(
                self.cell_obs_cnt > 0, self.cell_obs_sum / np.maximum(self.cell_obs_cnt, 1.0), 0.0
            )
            stale = np.clip((self.t - self.cell_last_seen) / max(self.T, 1), 0.0, 1.0)
            score = meanpast / max(self.grid_density_norm, 1.0) + w_explore * stale
        dist = np.linalg.norm(self.uav_pos[:, None, :] - self._cell_centers[None, :, :], axis=2)
        travel = np.maximum(dist / self.d_max, 1.0)
        util = score[None, :] / travel
        active = self.active if self.hard_energy_constraint else np.ones(self.M, dtype=bool)
        self._ensure_coordination_leader(active)
        targets = (
            np.linalg.norm(self.uav_pos[:, None, :] - self._cell_centers[None, :, :], axis=2)
            .argmin(axis=1)
            .astype(int)
        )
        targets[active] = -1
        claimed = set()
        nc = util.shape[1]
        for idx in np.argsort(util.ravel())[::-1]:
            m, c = (idx // nc, idx % nc)
            if active[m] and targets[m] < 0 and (c not in claimed):
                targets[m] = c
                claimed.add(c)
            if (targets[active] >= 0).all():
                break
        if np.unique(targets[active]).size != int(active.sum()):
            raise RuntimeError("coordination invariant violated: active UAV targets are not unique")
        return (targets, score)

    def set_assignment(self, targets) -> None:
        self._external_target = np.asarray(targets, dtype=int).reshape(self.M)

    def hrl_state(self) -> np.ndarray:
        belief = self._grid_belief()[0]
        pos = (self.uav_pos / self.l_max).reshape(-1)
        cur = np.zeros(self.M * self._n_cells, dtype=np.float32)
        if self._external_target is not None:
            for m in range(self.M):
                cur[m * self._n_cells + int(self._external_target[m])] = 1.0
        return np.concatenate([belief, pos, cur]).astype(np.float32)

    def _goal_block(self) -> np.ndarray:
        targets, score = self._memory_assign(source=self.goal_source)
        if self.assign_target_mode == "external" and self._external_target is not None:
            targets = self._external_target
        self._assign_target = targets
        goal_vec = (self._cell_centers[targets] - self.uav_pos) / self.l_max
        region_dem = (score[targets] / max(float(score.max()), 1e-09)).reshape(self.M, 1)
        blocks = [goal_vec, region_dem]
        if self.goal_obs_ttl:
            ttl = float(self.sd_deadline_ttl) if self.use_sd_deadline else 8.0
            stale_ttl = np.clip(
                (self.t - self.cell_last_seen[targets]) / max(ttl, 1.0), 0.0, 1.0
            ).reshape(self.M, 1)
            dist = np.linalg.norm(self._cell_centers[targets] - self.uav_pos, axis=1)
            travel = np.clip(dist / max(self.d_max, 1e-09) / 10.0, 0.0, 1.0).reshape(self.M, 1)
            blocks += [stale_ttl.astype(np.float32), travel.astype(np.float32)]
        return np.concatenate(blocks, axis=1).astype(np.float32)

    def _ego_patch_block(self) -> np.ndarray:
        P, H = (self.ego_p, self.ego_half)
        R2 = self.ego_sense * self.ego_sense
        q = self.queue.astype(np.float32)
        patch = np.zeros((self.M, P, P), dtype=np.float32)
        for m in range(self.M):
            if self.hard_energy_constraint and (not self.active[m]):
                continue
            rel = self.sd_pos - self.uav_pos[m]
            sensed = ((rel * rel).sum(axis=1) <= R2) & (q > 0.0)
            if not sensed.any():
                continue
            gx = np.clip(((rel[sensed, 0] + H) / (2.0 * H) * P).astype(int), 0, P - 1)
            gy = np.clip(((rel[sensed, 1] + H) / (2.0 * H) * P).astype(int), 0, P - 1)
            np.add.at(patch[m], (gx, gy), q[sensed])
        patch = patch.reshape(self.M, P * P) / max(self.l_queue_max, 1.0)
        return np.clip(patch, 0.0, 1.0).astype(np.float32)

    def _sd_soonest_ttl(self) -> np.ndarray:
        if not self.use_sd_deadline:
            return np.ones(self.K, dtype=np.float32)
        has = self.sd_queue_ttl > 0
        idx = np.where(has.any(axis=1), has.argmax(axis=1), self.sd_deadline_ttl - 1)
        return (idx + 1).astype(np.float32)

    def _collect_channel(self, start_pos, end_pos, alive_start, receive_capacity=None):
        eta_s = self._spectral_eff(start_pos)
        eta_e = self._spectral_eff(end_pos)
        eta_bar = 0.5 * (eta_s + eta_e)
        served = (eta_e >= self.ch_eta_min) & (self.queue[None, :] > 0)
        if alive_start is not None:
            served &= alive_start[:, None]
        if self.ch_alloc == "A3":
            w = self.queue / np.maximum(self._sd_soonest_ttl(), 1.0)
        else:
            w = self.queue
        w_mk = np.where(served, w[None, :], 0.0).astype(np.float32)
        denom = w_mk.sum(axis=1, keepdims=True)
        a_mk = np.where(denom > 0, w_mk / np.maximum(denom, 1e-12), 0.0)
        pot = a_mk * self.ch_link_scale * eta_bar
        pot = np.where(served, pot, 0.0)
        total_pot = pot.sum(axis=0)
        scale = np.where(total_pot > self.queue, self.queue / np.maximum(total_pot, 1e-12), 1.0)
        collected_mk = (pot * scale[None, :]).astype(np.float32)
        if receive_capacity is not None:
            cap = np.maximum(np.asarray(receive_capacity, dtype=np.float32), 0.0)
            per_uav = collected_mk.sum(axis=1)
            row_scale = np.where(per_uav > cap, cap / np.maximum(per_uav, 1e-12), 1.0)
            collected_mk = (collected_mk * row_scale[:, None]).astype(np.float32)
        delta_k = collected_mk.sum(axis=0).astype(np.float32)
        return (collected_mk, delta_k)

    def _slot_capacity(self, f: np.ndarray) -> np.ndarray:
        if self.slot_capacity_mode == "identity":
            return f.astype(np.float32)
        raise ValueError(f"Unknown slot_capacity_mode: {self.slot_capacity_mode}")

    def _energy_consumption(self, d: np.ndarray, f: np.ndarray, N: np.ndarray) -> np.ndarray:
        cost = self.operating_cost
        transmission = cost["transmission_coefficient"] * N / cost["reference_rate"]
        rate_scale = np.power(self.f_max, max(cost["processing_exponent"] - 2.0, 0.0))
        processing = (
            cost["processing_coefficient"]
            * np.power(f, cost["processing_exponent"] - 1.0)
            * N
            * cost["workload_conversion"]
            / rate_scale
        )
        transmission_time = N / cost["reference_rate"]
        processing_time = N * cost["workload_conversion"] / np.maximum(f, 1e-09)
        movement_time = d / self.v_uav
        duration = transmission_time + processing_time + movement_time * cost["movement_multiplier"]
        operating = 0.5 * cost["operating_weight"] * self.v_uav**2 * duration
        return ((transmission + processing + operating) / cost["normalization"]).astype(np.float32)

    def _fairness_sd(self) -> np.ndarray:
        s1 = self.Z.sum(axis=1)
        s2 = (self.Z**2).sum(axis=1)
        denom = self.K * s2 + 1e-09
        F = np.where(s2 > 0, s1**2 / denom, 0.0)
        return F.astype(np.float32)

    @staticmethod
    def _jain(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        s2 = (x**2).sum()
        if s2 <= 0:
            return 0.0
        return float(x.sum() ** 2 / (len(x) * s2))

    def _fairness_uav(self) -> float:
        per_uav = self.Z.sum(axis=1)
        s1 = per_uav.sum()
        s2 = (per_uav**2).sum()
        if s2 <= 0:
            return 0.0
        return float(s1**2 / (self.M * s2 + 1e-09))

    def _grid_belief(self) -> np.ndarray:
        c1 = self.visited_global.astype(np.float32)
        c2 = np.clip(self.cell_last_density / self.grid_density_norm, 0.0, 1.0)
        c3 = np.clip((self.t - self.cell_last_seen) / max(self.T, 1), 0.0, 1.0)
        chans = [c1, c2, c3]
        if self.use_grid_c4:
            c4mean = np.where(
                self.cell_obs_cnt > 0, self.cell_obs_sum / np.maximum(self.cell_obs_cnt, 1.0), 0.0
            )
            chans.append(np.clip(c4mean / self.grid_density_norm, 0.0, 1.0))
        grid = np.concatenate(chans).astype(np.float32)
        return np.broadcast_to(grid, (self.M, grid.shape[0])).copy()

    def _all_obs(self) -> list[np.ndarray]:
        per_uav_total = self.Z.sum(axis=1)
        diff = self.uav_pos[:, None, :] - self.uav_pos[None, :, :]
        all_dists = np.linalg.norm(diff, axis=2).astype(np.float32)
        other_dists = all_dists[self._other_mask].reshape(self.M, self.M - 1)
        denom_t = max(self.t, 1)
        coords_n = self.uav_pos / self.l_max
        share = np.broadcast_to(per_uav_total / denom_t / self.K, (self.M, self.M))
        other_n = other_dists / self.l_max
        if self.depletion_mask_mode == "agent_terminal_v1":
            other_active = np.broadcast_to(self.active[None, :], (self.M, self.M))[
                self._other_mask
            ].reshape(self.M, self.M - 1)
            other_n = other_n * other_active.astype(np.float32)
        if self.use_grid_belief:
            sd_block = None if self.drop_global_belief else self._grid_belief()
        else:
            sd_block = self.Z / denom_t
        inter_uav_block = other_n
        blocks = [coords_n]
        if sd_block is not None:
            blocks.append(sd_block)
        blocks += [share, inter_uav_block, self._uav_ids]
        if self.depletion_mask_mode == "agent_terminal_v1":
            active_n = np.broadcast_to(self.active.astype(np.float32), (self.M, self.M)).copy()
            blocks.append(active_n)
        if self.use_uav_queue:
            visible_queue = (
                np.where(self.active, self.uav_queue, 0.0)
                if self.depletion_mask_mode == "agent_terminal_v1"
                else self.uav_queue
            )
            uq_norm = np.broadcast_to(
                visible_queue / max(self.l_queue_max, 1.0), (self.M, self.M)
            ).astype(np.float32)
            blocks.append(uq_norm)
        if self.use_energy_obs:
            used_ratio = np.broadcast_to(
                np.clip(self.energy_total_used / max(self.e_uav, 1.0), 0.0, 2.0), (self.M, self.M)
            ).astype(np.float32)
            rem_ratio = np.broadcast_to(
                np.clip(self.energy_remaining / max(self.e_uav, 1.0), 0.0, 1.0), (self.M, self.M)
            ).astype(np.float32)
            blocks.append(used_ratio)
            blocks.append(rem_ratio)
        if self.use_deadline:
            tot = self.uav_queue_ttl.sum(axis=1)
            urg = np.where(tot > 1e-09, self.uav_queue_ttl[:, 0] / np.maximum(tot, 1e-09), 0.0)
            urg_n = np.broadcast_to(urg.astype(np.float32), (self.M, self.M)).astype(np.float32)
            blocks.append(urg_n)
        if self.assign_target_mode:
            blocks.append(self._goal_block())
        if self.ego_patch:
            blocks.append(self._ego_patch_block())
        obs_block = np.concatenate(blocks, axis=1).astype(np.float32)
        if self.depletion_mask_mode == "agent_terminal_v1":
            obs_block[~self.active] = 0.0
        return [obs_block[m] for m in range(self.M)]

    def mass_balance(self) -> dict:
        generated = float(self.generated_per_sd.sum(dtype=np.float64))
        radio = float(self.radio_per_sd.sum(dtype=np.float64))
        sd_queued = float(self.queue.sum(dtype=np.float64))
        sd_missed = float(self.sd_missed.sum(dtype=np.float64)) if self.use_sd_deadline else 0.0
        sd_overflow = float(self.sd_overflow_dropped_per_sd.sum(dtype=np.float64))
        admitted = float(self.admitted_per_sd.sum(dtype=np.float64))
        buffered = float(self.uav_queue.sum(dtype=np.float64)) if self.use_uav_queue else 0.0
        proc_missed = float(self.missed_bits.sum(dtype=np.float64)) if self.use_deadline else 0.0
        completed = float(self.total_bits)
        return {
            "sd_mass_error": generated - (sd_queued + sd_missed + radio + sd_overflow),
            "uav_mass_error": admitted - (completed + buffered + proc_missed),
        }

    def summary(self) -> dict:
        generated = float(self.generated_per_sd.sum())
        radio = float(self.radio_per_sd.sum())
        admitted = float(self.admitted_per_sd.sum())
        dropped = float(self.buffer_dropped_per_sd.sum())
        balance = self.mass_balance()
        return {
            "scenario": self.scenario_active,
            "gu_boundary_handling": "reject_and_resample_full_2d_proposal",
            "gu_rejection_stats": copy.deepcopy(self.gu_rejection_stats),
            "computation_bits": self.total_bits,
            "bits_per_uav": self.bits_per_uav.tolist(),
            "pioneer_cum": self.pioneer_cum.tolist(),
            "patches_visited": int(self.visited_global.sum()),
            "F_sd_final": self._fairness_sd().tolist(),
            "F_uav_final": self._fairness_uav(),
            "coordination_leader_index": self.coordination_leader_index,
            "coordination_leader_handover_count": int(self.coordination_leader_handover_count),
            "coordination_leader_handover_slots": list(self.coordination_leader_handover_slots),
            "energy_init_mode": self.energy_init_mode,
            "energy_init": self.energy_init.tolist(),
            "team_initial_energy": float(self.energy_init.sum()),
            "energy_remaining": self.energy_remaining.tolist(),
            "energy_total_used": self.energy_total_used.tolist(),
            "energy_violated": self.energy_violated.tolist(),
            "n_violators": int(self.energy_violated.sum()),
            "violation_slot": self.violation_slot.tolist(),
            "cumulative_excess": self.cumulative_excess.tolist(),
            "peak_excess": self.peak_excess.tolist(),
            "uav_queue_final": self.uav_queue.tolist(),
            "uav_queue_peak": self.uav_queue_peak.tolist(),
            "arrival_mode": self.arrival_mode,
            "onoff_preserve_load": self.onoff_preserve_load,
            "deadline_ttl": self.deadline_ttl,
            "uav_buffer_mode": self.uav_buffer_mode,
            "generated_total": generated,
            "collected_total": radio,
            "radio_collected_total": radio,
            "admitted_total": admitted,
            "buffer_dropped_total": dropped,
            "buffer_drop_rate": dropped / max(radio, 1.0),
            "sd_overflow_dropped_total": float(self.sd_overflow_dropped_per_sd.sum()),
            "completed_total": float(self.total_bits),
            "sd_deadline_ttl": self.sd_deadline_ttl,
            "sd_missed_total": float(self.sd_missed.sum()) if self.use_sd_deadline else 0.0,
            "collection_success_rate": radio / max(generated, 1.0),
            "admission_success_rate": admitted / max(generated, 1.0),
            "processing_success_rate": float(self.total_bits) / max(admitted, 1.0),
            "radio_to_completion_rate": float(self.total_bits) / max(radio, 1.0),
            "end_to_end_success_rate": float(self.total_bits) / max(generated, 1.0),
            "sd_mass_error": balance["sd_mass_error"],
            "uav_mass_error": balance["uav_mass_error"],
            "missed_bits": self.missed_bits.tolist() if self.use_deadline else None,
            "missed_total": float(self.missed_bits.sum()) if self.use_deadline else 0.0,
            "miss_rate": (
                float(self.missed_bits.sum())
                / max(self.total_bits + float(self.missed_bits.sum()), 1.0)
                if self.use_deadline
                else 0.0
            ),
        }

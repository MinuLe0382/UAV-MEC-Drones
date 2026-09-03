from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from config import CONFIG, ROOT, SYS, UNSPEC, apply_method
from env.uav_mec_env import UAVMECEnv
from scenarios import ARRAYS, load_scenario, validate_scenario


SETTINGS = json.loads((ROOT / "scenario_settings.json").read_text(encoding="utf-8"))
CONDITIONS = {row["id"]: row for row in SETTINGS["conditions"]}


class ScenarioGenerator(UAVMECEnv):
    def _place_hetero(self):
        cfg = self._hetero_cfg
        if "cluster_size_vector" not in cfg:
            return super()._place_hetero()
        counts = np.asarray(cfg["cluster_size_vector"], dtype=np.int64)
        if counts.ndim != 1 or np.any(counts < 1) or int(counts.sum()) != self.K:
            raise ValueError("cluster populations must be positive and sum to K")
        if cfg.get("randomize_population_labels", False):
            counts = self._rng("cluster_assignment").permutation(counts)
        sigma = float(cfg["cluster_radius"])
        centers = self._sample_cluster_centers(
            len(counts), float(cfg["min_center_dist"]), float(cfg["center_margin"])
        )
        if cfg["population_load_mode"] == "equal_cluster_load":
            alphas = [
                np.clip(
                    float(cfg["rho_per_cluster"]) * self.f_max / (int(n) * self.task_mean),
                    1e-3,
                    1.0,
                )
                for n in counts
            ]
        elif cfg["population_load_mode"] == "equal_per_gu_load":
            alpha = np.clip(
                float(cfg["total_rho"]) * self.f_max / (self.K * self.task_mean), 1e-3, 1.0
            )
            alphas = [float(alpha)] * len(counts)
        else:
            raise ValueError("unknown population load mode")
        positions, probabilities, labels = [], [], []
        for label, (center, count, alpha) in enumerate(zip(centers, counts, alphas)):
            positions.append(
                self._sample_gu_positions(center=center, count=int(count), sigma=sigma)
            )
            probabilities.append(np.full(int(count), float(alpha), dtype=np.float32))
            labels.append(np.full(int(count), label, dtype=np.int64))
        self.sd_cluster = np.concatenate(labels)
        self.n_clusters = len(counts)
        self.cluster_centers = centers.astype(np.float32)
        self.cluster_sizes = counts.tolist()
        self.cluster_is_large = [bool(n > np.median(counts)) for n in counts]
        self.cluster_sigma = self.cluster_radius = sigma
        return np.concatenate(positions).astype(np.float32), np.concatenate(probabilities).astype(
            np.float32
        )


def generate(seed: int, condition_id: str = "nominal") -> dict:
    condition = CONDITIONS[condition_id]
    apply_method("dpp_gcmarl")
    SYS["num_sds"] = int(condition.get("num_ground_users", SYS["num_sds"]))
    SYS["r_max"] = float(condition.get("sensing_radius_m", SYS["r_max"]))
    UNSPEC["ego_sense"] = SYS["r_max"]
    UNSPEC["patch_grid_g"] = int(condition.get("coordination_grid_g", UNSPEC["patch_grid_g"]))
    UNSPEC["sd_deadline_ttl"] = int(
        condition.get("collection_deadline_slots", UNSPEC["sd_deadline_ttl"])
    )
    cfg = UNSPEC["hetero_cfg"]
    cfg.update(condition.get("hetero_overrides", {}))
    scale = float(condition.get("workload_multiplier", 1.0))
    for key in ("rho_large", "rho_small"):
        cfg[key] = float(cfg[key]) * scale
    for key in (
        "cluster_size_vector",
        "population_load_mode",
        "randomize_population_labels",
        "rho_per_cluster",
        "total_rho",
    ):
        if key in condition:
            cfg[key] = condition[key]
    if "cluster_size_vector" in condition:
        cfg["n_clusters"] = len(condition["cluster_size_vector"])
    energy_scale = float(condition.get("energy_budget_multiplier", 1.0))
    SYS["e_uav"] = float(SYS["e_uav"]) * energy_scale
    UNSPEC["energy_init_levels"] = [float(x) * energy_scale for x in UNSPEC["energy_init_levels"]]
    # Construction performs the first seeded reset; a second reset advances the streams.
    env = ScenarioGenerator(seed=int(seed), scenario="hetero", use_grid_belief=True)
    result = {
        "metadata": {
            "scenario_seed": int(seed),
            "condition_id": condition_id,
            "num_uavs": int(env.M),
            "num_gus": int(env.K),
            "mission_slots": int(env.T),
            "slot_duration_s": 1.0,
            "mission_side_length_m": float(env.l_max),
            "coordination_grid_g": int(env.patch_g),
            "sensing_radius_m": float(env.r_max),
            "ego_sense_radius_m": float(env.ego_sense),
            "collection_deadline_slots": int(env.sd_deadline_ttl),
            "nominal_energy_ceiling": float(env.e_uav),
            "cluster_center_margin_m": float(cfg["center_margin"]),
            "cluster_center_min_separation_m": float(cfg["min_center_dist"]),
            "gu_sigma_m": float(env.cluster_sigma),
            "gu_boundary_handling": "reject_and_resample_full_2d_proposal",
            "realized_cluster_population": list(env.cluster_sizes),
        },
        "gu_positions": env.sd_pos.copy(),
        "gu_generation_probabilities": env.alpha.copy(),
        "cluster_centers": env.cluster_centers.copy(),
        "cluster_labels": env.sd_cluster.copy(),
        "activation_schedule": env.activation_schedule.copy(),
        "task_arrival_indicators": env._task_arrival_trace.copy(),
        "task_sizes_bits": env._task_size_trace.copy(),
        "initial_uav_positions": env.uav_pos.copy(),
        "initial_energy_budgets": env.energy_init.copy(),
    }
    validate_scenario(result)
    return result


def released_files(root: Path = ROOT):
    for split in ("validation", "development", "test"):
        for path in sorted((root / "data/scenarios" / split).glob("*.npz")):
            yield split, "nominal", path
    for condition in CONDITIONS:
        for path in sorted((root / "data/robustness" / condition).glob("*.npz")):
            yield "robustness", condition, path


def write_manifest(root: Path = ROOT) -> int:
    fields = (
        "split",
        "condition",
        "file",
        "scenario_seed",
        "U",
        "K",
        "C",
        "Cclu",
        "Dcol",
        "sensing_radius_m",
        "cluster_populations",
        "initial_budgets",
        "workload_multiplier",
        "energy_multiplier",
        "slot_duration_s",
    )
    rows = []
    for split, condition_id, path in released_files(root):
        scenario = load_scenario(path)
        m = scenario["metadata"]
        condition = CONDITIONS[condition_id]
        rows.append(
            dict(
                zip(
                    fields,
                    (
                        split,
                        condition_id,
                        path.relative_to(root).as_posix(),
                        int(m["scenario_seed"]),
                        int(m["num_uavs"]),
                        int(m["num_gus"]),
                        int(m.get("coordination_grid_g", CONFIG["environment"]["patch_grid_g"]))
                        ** 2,
                        len(scenario["cluster_centers"]),
                        int(
                            m.get(
                                "collection_deadline_slots",
                                CONFIG["environment"]["sd_deadline_ttl"],
                            )
                        ),
                        float(m.get("sensing_radius_m", CONFIG["system"]["r_max"])),
                        ";".join(map(str, np.bincount(scenario["cluster_labels"]).tolist())),
                        ";".join(map(str, scenario["initial_energy_budgets"].tolist())),
                        condition.get("workload_multiplier", 1.0),
                        condition.get("energy_budget_multiplier", 1.0),
                        1.0,
                    ),
                )
            )
        )
    if not rows:
        raise ValueError("no scenario files found")
    with (root / "data/scenario_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def verify_existing(root: Path = ROOT) -> int:
    count = 0
    group = None
    for split, condition, path in released_files(root):
        if (split, condition) != group:
            print(f"verifying {split}/{condition}", flush=True)
            group = (split, condition)
        expected = load_scenario(path)
        actual = generate(int(expected["metadata"]["scenario_seed"]), condition)
        for name in ARRAYS:
            if actual[name].dtype != expected[name].dtype or not np.array_equal(
                actual[name], expected[name]
            ):
                raise ValueError(f"generated array differs: {path.name}, {condition}, {name}")
        count += 1
    if count != 750:
        raise ValueError(f"expected 750 released scenarios; found {count}")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", choices=("validation", "development", "test", "robustness"), default="test"
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-existing", action="store_true")
    mode.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    if args.verify_existing:
        print(f"verified {verify_existing()} scenarios")
        return
    if args.write_manifest:
        print(f"listed {write_manifest()} scenarios")
        return
    if args.output is None:
        parser.error("--output is required for generation")
    seeds = args.seeds or SETTINGS["splits"]["test" if args.split == "robustness" else args.split]
    conditions = list(CONDITIONS) if args.split == "robustness" else ["nominal"]
    targets = [
        (
            condition,
            seed,
            (
                args.output / condition / f"scenario_{i:03d}.npz"
                if args.split == "robustness"
                else args.output / f"scenario_{i:03d}.npz"
            ),
        )
        for condition in conditions
        for i, seed in enumerate(seeds)
    ]
    if any(path.exists() for _, _, path in targets):
        raise FileExistsError("output already contains scenario files")
    for condition, seed, path in targets:
        scenario = generate(seed, condition)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez_compressed(
                stream,
                metadata_json=np.asarray(json.dumps(scenario["metadata"], sort_keys=True)),
                **{name: scenario[name] for name in ARRAYS},
            )
    print(f"wrote {len(targets)} scenarios to {args.output}")


if __name__ == "__main__":
    main()

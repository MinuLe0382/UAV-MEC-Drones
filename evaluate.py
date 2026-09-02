from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from config import CONFIG, SYS, UNSPEC, apply_method
from controllers.frequency_governor import compose_environment_actions
from metrics import PatrolTracker, episode_metrics
from rule_methods import RuleController
from scenarios import load_scenario, scenario_files


ROOT = Path(__file__).resolve().parent
LEARNED_METHODS = ("dpp_gcmarl", "mappo", "memory_gcmarl")
RULE_METHODS = ("memory", "anticipatory", "belief_mpc", "full_information")
ALL_METHODS = (*LEARNED_METHODS, "without_gcm", *RULE_METHODS)


def make_environment(method: str, scenario: dict):
    profile = method if method in LEARNED_METHODS else "dpp_gcmarl"
    apply_method(profile)
    metadata = scenario["metadata"]
    SYS["num_sds"] = int(metadata.get("num_gus", SYS["num_sds"]))
    SYS["r_max"] = float(metadata.get("sensing_radius_m", SYS["r_max"]))
    UNSPEC["ego_sense"] = float(metadata.get("ego_sense_radius_m", UNSPEC["ego_sense"]))
    UNSPEC["patch_grid_g"] = int(metadata.get("coordination_grid_g", UNSPEC["patch_grid_g"]))
    UNSPEC["sd_deadline_ttl"] = int(
        metadata.get("collection_deadline_slots", UNSPEC["sd_deadline_ttl"])
    )
    energy_ceiling = float(metadata.get("nominal_energy_ceiling", SYS["e_uav"]))
    energy_scale = energy_ceiling / float(SYS["e_uav"])
    SYS["e_uav"] = energy_ceiling
    if UNSPEC.get("energy_init_levels") is not None:
        UNSPEC["energy_init_levels"] = [
            float(value) * energy_scale for value in UNSPEC["energy_init_levels"]
        ]
    from env.uav_mec_env import UAVMECEnv

    return UAVMECEnv(
        seed=int(scenario["metadata"]["scenario_seed"]),
        scenario="hetero",
        use_grid_belief=True,
        scenario_artifact=scenario,
    )


def load_agent(method: str, checkpoint: Path, env):
    from agents.mappo import MAPPO

    agent = MAPPO(
        env.M,
        env.obs_dim,
        2,
        device="cpu",
        environment_action_dim=env.act_dim,
        frequency_rule="plan",
    )
    agent.load(str(checkpoint))
    return agent


def target_following_action(env) -> np.ndarray:
    targets, _ = env._memory_assign(
        source=str(env.goal_source),
    )
    env._assign_target = np.asarray(targets, dtype=np.int64).copy()
    centers = env._cell_centers[np.asarray(targets, dtype=int)]
    delta = centers - env.uav_pos
    distance = np.minimum(np.linalg.norm(delta, axis=1), env.d_max)
    heading = np.mod(np.arctan2(delta[:, 1], delta[:, 0]), 2.0 * np.pi)
    return np.stack([2.0 * distance / env.d_max - 1.0, heading / np.pi - 1.0], axis=1).astype(
        np.float32
    )


def run_episode(method: str, scenario: dict, checkpoint: Path | None = None) -> dict:
    env = make_environment(method, scenario)
    observation = env.reset()
    tracker = PatrolTracker(env)
    movement = np.zeros(env.M, dtype=np.float64)
    agent = load_agent(method, checkpoint, env) if checkpoint is not None else None
    rule = (
        RuleController(method, env, CONFIG["evaluation"]["exploration_weight"])
        if method in RULE_METHODS
        else None
    )

    for _ in range(env.T):
        previous_position = env.uav_pos.copy()
        observed_slot = int(env.t)
        if agent is not None:
            policy_action = agent.select_actions(observation, noise=False)
        elif method == "without_gcm":
            policy_action = target_following_action(env)
        elif rule is not None:
            policy_action = rule.action()
        else:
            raise ValueError(f"method requires a checkpoint: {method}")
        action = compose_environment_actions(policy_action, env, frequency_rule="plan")
        observation, _, done, _ = env.step(action)
        tracker.after_step(env, observed_slot)
        movement += np.linalg.norm(env.uav_pos - previous_position, axis=1)
        if rule is not None:
            rule.update(observed_slot)
        if done:
            break

    result = episode_metrics(env, movement)
    result.update(tracker.finalize(env))
    result["scenario_seed"] = int(scenario["metadata"]["scenario_seed"])
    return result


def checkpoint_files(method: str, root: Path = ROOT) -> list[Path]:
    paths = sorted((root / "checkpoints" / method).glob("seed_*.pt"))
    if len(paths) != 4:
        raise FileNotFoundError(
            f"expected four checkpoints in checkpoints/{method}; found {len(paths)}"
        )
    return paths


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        summary[method] = {
            metric: float(np.mean([row[metric] for row in selected]))
            for metric in (
                "e2e",
                "collection",
                "ecf",
                "min_cluster",
                "coverage",
                "movement_per_uav",
                "discovery",
                "revisit_p95",
            )
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=("validation", "development", "test"))
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=list(ALL_METHODS))
    parser.add_argument("--output", default="results/evaluation.json")
    args = parser.parse_args()

    scenarios = [load_scenario(path) for path in scenario_files(args.split, ROOT)]
    rows = []
    for method in args.methods:
        checkpoints = checkpoint_files(method) if method in LEARNED_METHODS else [None]
        for checkpoint in checkpoints:
            training_seed = None
            if checkpoint is not None:
                training_seed = int(checkpoint.stem.split("_")[-1])
            for scenario in scenarios:
                row = run_episode(method, scenario, checkpoint)
                row.update({"method": method, "training_seed": training_seed})
                rows.append(row)
        print(f"evaluated {method}: {len(checkpoints)} policy set(s) x {len(scenarios)} scenarios")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"split": args.split, "summary": summarize(rows), "rows": rows}
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output.resolve()}")


if __name__ == "__main__":
    main()

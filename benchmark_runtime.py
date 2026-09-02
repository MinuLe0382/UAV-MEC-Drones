from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from config import CONFIG, validate_runtime
from controllers.frequency_governor import compose_environment_actions
from evaluate import (
    LEARNED_METHODS,
    RULE_METHODS,
    checkpoint_files,
    load_agent,
    target_following_action,
)
from rule_methods import RuleController
from train import make_environment


ROOT = Path(__file__).resolve().parent
METHODS = (
    "memory",
    "anticipatory",
    "belief_mpc",
    "mappo",
    "memory_gcmarl",
    "without_gcm",
    "dpp_gcmarl",
    "full_information",
)


def measure(function, warmup: int, repetitions: int) -> tuple[dict, np.ndarray]:
    if warmup < 0 or repetitions < 2:
        raise ValueError("invalid timing repetition counts")
    for _ in range(warmup):
        function()
    samples = np.empty(repetitions, dtype=np.int64)
    for index in range(repetitions):
        started = time.perf_counter_ns()
        function()
        samples[index] = time.perf_counter_ns() - started
    microseconds = samples.astype(np.float64) / 1000.0
    return {
        "median_us": float(np.median(microseconds)),
        "p95_us": float(np.percentile(microseconds, 95)),
        "mean_us": float(np.mean(microseconds)),
    }, samples


def prepare_learned(method: str, rollout_slots: int, seed: int):
    env = make_environment(method, seed)
    observation = env.reset()
    if not 0 < rollout_slots < env.T:
        raise ValueError("representative rollout must end before mission termination")
    agent = load_agent(method, checkpoint_files(method)[0], env)
    for _ in range(rollout_slots):
        movement = agent.select_actions(observation, noise=False)
        action = compose_environment_actions(movement, env, "plan")
        observation, _, done, _ = env.step(action)
        if done:
            raise RuntimeError("representative rollout terminated early")
    return env, agent, observation


def prepare_rules(rollout_slots: int, seed: int):
    env = make_environment("dpp_gcmarl", seed)
    env.reset()
    if not 0 < rollout_slots < env.T:
        raise ValueError("representative rollout must end before mission termination")
    memory = RuleController("memory", env, CONFIG["evaluation"]["exploration_weight"])
    for _ in range(rollout_slots):
        slot = int(env.t)
        action = compose_environment_actions(memory.action(), env, "plan")
        _, _, done, _ = env.step(action)
        memory.update(slot)
        if done:
            raise RuntimeError("representative rollout terminated early")
    rules = {}
    for method in RULE_METHODS:
        rule = RuleController(method, env, memory.exploration_weight)
        for name in ("memory_sum", "memory_count", "phase_sum", "phase_count"):
            setattr(rule, name, getattr(memory, name).copy())
        rules[method] = rule
    return env, rules


def learned_decision(env, agent):
    movement = agent.select_actions(env._all_obs(), noise=False)
    return compose_environment_actions(movement, env, "plan")


def component_functions(env, agent, observations) -> dict:
    tensor = torch.as_tensor(np.asarray(observations, dtype=np.float32))
    actions = agent.select_actions(observations, noise=False)

    def actor_forward():
        with torch.no_grad():
            return agent.actor(tensor)

    return {
        "dpp_assignment": lambda: env._memory_assign(source=env.goal_source),
        "observation_builder": env._all_obs,
        "actor_network_only": actor_forward,
        "actor_interface": lambda: agent.select_actions(observations, noise=False),
        "plan_governor": lambda: compose_environment_actions(actions, env, "plan"),
        "online_decision_total": lambda: learned_decision(env, agent),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--repetitions", type=int, default=10000)
    parser.add_argument("--rollout-slots", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--components", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results/runtime.json")
    args = parser.parse_args()
    validate_runtime()
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(variable) != "1":
            raise RuntimeError(f"set {variable}=1 for the benchmark")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    learned = {}
    needed = set(args.methods) & set(LEARNED_METHODS)
    if args.components:
        needed.add("dpp_gcmarl")
    for method in sorted(needed):
        learned[method] = prepare_learned(method, args.rollout_slots, args.seed)
    env = rules = None
    if set(args.methods) - set(LEARNED_METHODS):
        env, rules = prepare_rules(args.rollout_slots, args.seed)
    functions = {}
    for method in args.methods:
        if method in learned:
            e, agent, _ = learned[method]
            functions[method] = lambda e=e, agent=agent: learned_decision(e, agent)
        elif method == "without_gcm":
            functions[method] = lambda: compose_environment_actions(
                target_following_action(env), env, "plan"
            )
        else:
            rule = rules[method]
            functions[method] = lambda rule=rule: compose_environment_actions(
                rule.action(), rule.env, "plan"
            )
    raw, results = {}, {}
    for method, function in functions.items():
        results[method], raw[method] = measure(function, args.warmup, args.repetitions)
        print(
            f"{method:22s} median={results[method]['median_us']:.3f} us p95={results[method]['p95_us']:.3f} us",
            flush=True,
        )
    components = {}
    if args.components:
        for name, function in component_functions(*learned["dpp_gcmarl"]).items():
            components[name], raw[f"component_{name}"] = measure(
                function, args.warmup, args.repetitions
            )
            print(f"{name:22s} median={components[name]['median_us']:.3f} us", flush=True)
    payload = {
        "runtime": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "threads": 1,
        },
        "measurement": {
            "warmup_iterations": args.warmup,
            "measurement_iterations": args.repetitions,
            "generator_seed": args.seed,
            "rollout_slots": args.rollout_slots,
            "rule_rollout_method": "memory",
        },
        "methods": results,
        "components": components,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.output.with_suffix(".npz"), **raw)
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()

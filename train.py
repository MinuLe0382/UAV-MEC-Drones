from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from config import CONFIG, apply_method
from controllers.frequency_governor import compose_environment_actions


def make_environment(method: str, seed: int):
    apply_method(method)
    from env.uav_mec_env import UAVMECEnv

    return UAVMECEnv(
        seed=seed,
        scenario="hetero",
        use_grid_belief=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("dpp_gcmarl", "mappo", "memory_gcmarl"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=CONFIG["training"]["episodes"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = make_environment(args.method, args.seed)
    from agents.mappo import MAPPO

    settings = CONFIG["training"]
    agent = MAPPO(
        env.M,
        env.obs_dim,
        2,
        device=args.device,
        clip=settings["clip"],
        ppo_epochs=settings["ppo_epochs"],
        num_minibatches=settings["num_minibatches"],
        gae_lambda=settings["gae_lambda"],
        entropy_coef=settings["entropy_coef"],
        value_coef=settings["value_coef"],
        lr=settings["lr"],
        log_std_init=settings["log_std_init"],
        environment_action_dim=env.act_dim,
        frequency_rule="plan",
    )
    output = Path(args.output or f"runs/{args.method}/seed_{args.seed}")
    output.mkdir(parents=True, exist_ok=True)
    log_file = (output / "training.jsonl").open("w", encoding="utf-8")
    returns: list[float] = []
    episode = 0
    started = time.time()

    while episode < args.episodes:
        agent.buffer.clear()
        last_observation = None
        batch_size = min(settings["rollout_episodes"], args.episodes - episode)
        for _ in range(batch_size):
            observation = env.reset()
            episode_return = 0.0
            for _ in range(env.T):
                active = env.active.copy()
                policy_action, log_probability, value = agent.act(observation)
                policy_action[~active] = 0.0
                log_probability[~active] = 0.0
                action = compose_environment_actions(policy_action, env, frequency_rule="plan")
                next_observation, reward, done, info = env.step(action)
                agent.buffer.add(
                    np.asarray(observation, dtype=np.float32),
                    policy_action,
                    log_probability,
                    value,
                    reward.astype(np.float32),
                    float(done),
                    alive=active,
                    next_alive=np.asarray(info["active"], dtype=np.float32),
                )
                episode_return += float(reward.sum())
                observation = next_observation
                last_observation = observation
                if done:
                    break
            episode += 1
            returns.append(episode_return)
            summary = env.summary()
            record = {
                "episode": episode,
                "return": episode_return,
                "e2e": float(summary["end_to_end_success_rate"]),
                "collection": float(summary["collection_success_rate"]),
            }
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
            if episode % settings["log_every"] == 0:
                recent = np.mean(returns[-settings["log_every"] :])
                print(
                    f"episode={episode} return={recent:.3f} "
                    f"e2e={record['e2e']:.4f} elapsed={time.time() - started:.0f}s",
                    flush=True,
                )
            if episode % settings["save_every"] == 0:
                agent.save(str(output / f"checkpoint_episode_{episode}.pt"))
        agent.update(last_observation)

    agent.save(str(output / "final.pt"))
    log_file.close()
    print(f"wrote {output.resolve()}")


if __name__ == "__main__":
    main()

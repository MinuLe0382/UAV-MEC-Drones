from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from evaluate import run_episode
from scenarios import load_scenario, scenario_files


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("dpp_gcmarl", "mappo", "memory_gcmarl"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-directory", required=True)
    args = parser.parse_args()
    candidates = sorted(
        Path(args.run_directory).glob("checkpoint_episode_*.pt"),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    if not candidates:
        raise FileNotFoundError("no checkpoint candidates were found")
    scenarios = [load_scenario(path) for path in scenario_files("validation", ROOT)]
    scores = []
    for candidate in candidates:
        score = float(
            np.mean(
                [run_episode(args.method, scenario, candidate)["e2e"] for scenario in scenarios]
            )
        )
        scores.append((score, candidate))
        print(f"{candidate.name}: validation E2E={100.0 * score:.4f}%")
    score, selected = max(scores, key=lambda item: item[0])
    target = ROOT / "checkpoints" / args.method / f"seed_{args.seed}.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected, target)
    print(f"selected {selected.name}: E2E={100.0 * score:.4f}%")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()

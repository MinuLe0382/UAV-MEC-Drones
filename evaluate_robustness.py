from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate import checkpoint_files, run_episode
from scenarios import load_scenario


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/robustness_evaluation.json")
    args = parser.parse_args()
    rows = []
    data_root = ROOT / "data" / "robustness"
    for condition in sorted(path for path in data_root.iterdir() if path.is_dir()):
        scenarios = [load_scenario(path) for path in sorted(condition.glob("*.npz"))]
        for method in ("dpp_gcmarl", "mappo"):
            for checkpoint in checkpoint_files(method):
                training_seed = int(checkpoint.stem.split("_")[-1])
                for scenario in scenarios:
                    row = run_episode(method, scenario, checkpoint)
                    row.update(
                        {
                            "condition": condition.name,
                            "method": method,
                            "training_seed": training_seed,
                        }
                    )
                    rows.append(row)
        means = {}
        for method in ("dpp_gcmarl", "mappo"):
            selected = [
                row["e2e"]
                for row in rows
                if row["condition"] == condition.name and row["method"] == method
            ]
            means[method] = 100.0 * float(np.mean(selected))
        print(
            f"{condition.name:22s} DPP-GCMARL={means['dpp_gcmarl']:.2f}% "
            f"MAPPO={means['mappo']:.2f}%"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output.resolve()}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats
from config import validate_runtime


DISPLAY = {
    "dpp_gcmarl": "DPP-GCMARL",
    "mappo": "MAPPO",
    "memory_gcmarl": "Memory-GCMARL",
    "without_gcm": "w/o GCM",
    "memory": "Memory",
    "anticipatory": "Anticipatory",
    "belief_mpc": "Belief-MPC",
    "full_information": "Full-information heuristic",
}
ROOT = Path(__file__).resolve().parent
METRICS = (
    "e2e",
    "collection",
    "ecf",
    "min_cluster",
    "coverage",
    "movement_per_uav",
    "discovery",
    "revisit_p95",
)
TABLE3_METRICS = ("e2e", "collection", "min_cluster", "ecf")
HOLM_FAMILIES = {
    "memory_gcmarl": (
        "e2e",
        "min_cluster",
        "ecf",
        "movement_per_uav",
        "discovery",
        "revisit_p95",
    ),
    "without_gcm": ("e2e", "min_cluster", "ecf", "movement_per_uav"),
}


def scenario_values(rows: list[dict], method: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    selected = [row for row in rows if row["method"] == method]
    if not selected:
        raise ValueError(f"no observations for {method}")
    identities = [(int(row["scenario_seed"]), row.get("training_seed")) for row in selected]
    if len(set(identities)) != len(identities):
        raise ValueError(f"duplicate scenario/policy observations for {method}")
    seeds = np.asarray(sorted({int(row["scenario_seed"]) for row in selected}), dtype=int)
    values = np.asarray(
        [
            np.mean([float(row[metric]) for row in selected if int(row["scenario_seed"]) == seed])
            for seed in seeds
        ]
    )
    counts = [sum(int(row["scenario_seed"]) == seed for row in selected) for seed in seeds]
    if len(set(counts)) != 1 or not np.isfinite(values).all():
        raise ValueError(f"incomplete or non-finite observations for {method}.{metric}")
    return seeds, values


def confidence_interval(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("confidence intervals require at least two finite observations")
    mean = float(values.mean())
    half = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
    return mean - half, mean + half


def comparison(rows: list[dict], first: str, second: str, metric: str) -> dict:
    seeds_a, a = scenario_values(rows, first, metric)
    seeds_b, b = scenario_values(rows, second, metric)
    if not np.array_equal(seeds_a, seeds_b):
        raise ValueError(f"scenario sets differ: {first} and {second}")
    difference = a - b
    low, high = confidence_interval(difference)
    standard_deviation = float(difference.std(ddof=1))
    p_value = (
        float(stats.ttest_rel(a, b).pvalue)
        if standard_deviation > 0
        else float(difference.mean() == 0)
    )
    favorable = -difference if metric in {"movement_per_uav", "revisit_p95"} else difference
    return {
        "n_scenarios": len(difference),
        "difference": float(difference.mean()),
        "confidence_interval": [low, high],
        "p_value": p_value,
        "cohen_dz": (
            float(difference.mean() / standard_deviation) if standard_deviation > 0 else None
        ),
        "wins_ties_losses": [
            int((favorable > 1e-12).sum()),
            int((np.abs(favorable) <= 1e-12).sum()),
            int((favorable < -1e-12).sum()),
        ],
    }


def absolute(rows: list[dict], method: str, metric: str) -> dict:
    _, values = scenario_values(rows, method, metric)
    low, high = confidence_interval(values)
    return {
        "n_scenarios": len(values),
        "mean": float(values.mean()),
        "confidence_interval": [low, high],
    }


def add_holm_adjustment(records: dict[str, dict]) -> None:
    ordered = sorted(records, key=lambda metric: records[metric]["p_value"])
    adjusted = 0.0
    total = len(ordered)
    for rank, metric in enumerate(ordered):
        adjusted = max(adjusted, (total - rank) * records[metric]["p_value"])
        records[metric]["adjusted_p_value"] = float(min(adjusted, 1.0))


def analyze_main(rows: list[dict]) -> dict:
    methods = sorted({row["method"] for row in rows})
    result = {
        "absolute": {
            method: {
                metric: absolute(rows, method, metric)
                for metric in METRICS
                if all(row.get(metric) is not None for row in rows if row["method"] == method)
            }
            for method in methods
        },
        "comparisons": {},
    }
    pairs = (
        ("dpp_gcmarl", "mappo"),
        ("dpp_gcmarl", "memory_gcmarl"),
        ("dpp_gcmarl", "without_gcm"),
    )
    for first, second in pairs:
        if first not in methods or second not in methods:
            continue
        key = f"{first}_minus_{second}"
        family = HOLM_FAMILIES.get(second, TABLE3_METRICS)
        result["comparisons"][key] = {
            metric: comparison(rows, first, second, metric) for metric in family
        }
        if second in HOLM_FAMILIES:
            add_holm_adjustment(result["comparisons"][key])
    return result


def analyze_robustness(rows: list[dict]) -> dict:
    result = {"conditions": {}, "holm_family": []}
    for condition in sorted({row["condition"] for row in rows}):
        selected = [row for row in rows if row["condition"] == condition]
        result["conditions"][condition] = {
            "dpp_gcmarl": absolute(selected, "dpp_gcmarl", "e2e"),
            "mappo": absolute(selected, "mappo", "e2e"),
            "comparison": comparison(selected, "dpp_gcmarl", "mappo", "e2e"),
        }
    family = {
        condition: value["comparison"]
        for condition, value in result["conditions"].items()
        if condition != "nominal"
    }
    add_holm_adjustment(family)
    result["holm_family"] = list(family)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data/reported_results.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/statistical_analysis.json")
    args = parser.parse_args()
    validate_runtime()
    rows = json.loads(args.input.read_text(encoding="utf-8"))["rows"]
    robustness = any("condition" in row for row in rows)
    if robustness and not all("condition" in row for row in rows):
        raise ValueError("mixed main-comparison and robustness observations")
    result = analyze_robustness(rows) if robustness else analyze_main(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if robustness:
        for condition, record in result["conditions"].items():
            paired = record["comparison"]
            lo, hi = paired["confidence_interval"]
            print(
                f"{condition:22s} difference={100 * paired['difference']:+.2f} pp [{100 * lo:+.2f}, {100 * hi:+.2f}]"
            )
    else:
        for method, metrics in result["absolute"].items():
            print(f"{DISPLAY.get(method, method):28s} E2E={100 * metrics['e2e']['mean']:.2f}%")
    print(f"wrote {output.resolve()}")


if __name__ == "__main__":
    main()

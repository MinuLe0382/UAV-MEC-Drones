from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REPLICATES = 100_000
SEED = 20260909
BATCH_SIZE = 512

TABLE3_METHODS = (
    "memory",
    "anticipatory",
    "belief_mpc",
    "mappo",
    "dpp_gcmarl",
    "full_information",
)
TABLE3_METRICS = ("e2e", "collection", "min_cluster", "ecf")
TABLE4_METRICS = (
    "e2e",
    "min_cluster",
    "ecf",
    "movement_per_uav",
    "discovery",
    "revisit_p95",
)
TABLE5_METRICS = ("e2e", "min_cluster", "ecf", "movement_per_uav")
MAIN_METHODS = TABLE3_METHODS + ("memory_gcmarl", "without_gcm")
SCALES = {
    "e2e": 100.0,
    "collection": 100.0,
    "min_cluster": 1.0,
    "ecf": 1.0,
    "movement_per_uav": 1.0,
    "discovery": 100.0,
    "revisit_p95": 1.0,
}
ROBUSTNESS_CONDITIONS = (
    "nominal",
    "gu_30",
    "gu_70",
    "clusters_2",
    "clusters_6",
    "population_moderate",
    "population_strong",
    "grid_6",
    "grid_10",
    "sensing_10m",
    "sensing_30m",
    "deadline_4",
    "deadline_12",
    "workload_0p75",
    "workload_1p25",
    "energy_0p8",
    "energy_1p2",
)
COST_CONDITIONS = (
    "nominal",
    "a1_0p8",
    "a1_1p2",
    "a2_0p8",
    "a2_1p2",
    "a3_0p8",
    "a3_1p2",
    "a4_0p8",
    "a4_1p2",
)


def read_rows(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8")).get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} does not contain a non-empty rows array")
    return rows


def method_matrix(
    rows: list[dict], method: str, metric: str, condition: str | None = None
) -> np.ndarray:
    selected = [
        row
        for row in rows
        if row.get("method") == method
        and (condition is None or row.get("condition") == condition)
    ]
    if not selected:
        raise ValueError(f"no rows for {method}.{metric}")

    scenarios = sorted({int(row["scenario_seed"]) for row in selected})
    runs = sorted(
        {
            int(row["training_seed"])
            for row in selected
            if row.get("training_seed") is not None
        }
    )
    run_keys: list[int | None] = runs if runs else [None]
    lookup: dict[tuple[int | None, int], float] = {}
    for row in selected:
        run = row.get("training_seed")
        key = (None if run is None else int(run), int(row["scenario_seed"]))
        if key in lookup:
            raise ValueError(f"duplicate row for {method}: {key}")
        value = float(row[metric])
        if not np.isfinite(value):
            raise ValueError(f"non-finite value for {method}.{metric}: {key}")
        lookup[key] = value

    expected = {(run, scenario) for run in run_keys for scenario in scenarios}
    if set(lookup) != expected:
        raise ValueError(f"incomplete matrix for {method}.{metric}")
    return np.asarray(
        [[lookup[(run, scenario)] for scenario in scenarios] for run in run_keys],
        dtype=np.float64,
    )


def percentile_interval(values: np.ndarray) -> list[float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def holm(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def mean_samples(
    matrix: np.ndarray, replicates: int, seed: int, scale: float
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    runs, scenarios = matrix.shape
    samples = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, BATCH_SIZE):
        size = min(BATCH_SIZE, replicates - start)
        scenario_index = rng.integers(0, scenarios, size=(size, scenarios))
        if runs == 1:
            values = matrix[0, scenario_index].mean(axis=1)
        else:
            run_index = rng.integers(0, runs, size=(size, runs))
            values = matrix[
                run_index[:, :, None], scenario_index[:, None, :]
            ].mean(axis=(1, 2))
        samples[start : start + size] = values * scale
    return samples


def absolute_record(
    matrix: np.ndarray, replicates: int, seed: int, scale: float
) -> dict:
    samples = mean_samples(matrix, replicates, seed, scale)
    run_means = matrix.mean(axis=1) * scale
    return {
        "mean": float(matrix.mean() * scale),
        "ci95": percentile_interval(samples),
        "run_means": [float(value) for value in run_means]
        if len(run_means) > 1
        else None,
        "run_sd": float(run_means.std(ddof=1)) if len(run_means) > 1 else None,
    }


def comparison_record(
    first: np.ndarray,
    second: np.ndarray,
    replicates: int,
    seed: int,
    scale: float,
) -> dict:
    if first.shape[1] != second.shape[1]:
        raise ValueError("compared methods do not share the same scenarios")

    rng = np.random.default_rng(seed)
    first_runs, scenarios = first.shape
    second_runs = second.shape[0]
    first_samples = np.empty(replicates, dtype=np.float64)
    second_samples = np.empty(replicates, dtype=np.float64)
    difference_samples = np.empty(replicates, dtype=np.float64)

    for start in range(0, replicates, BATCH_SIZE):
        size = min(BATCH_SIZE, replicates - start)
        scenario_index = rng.integers(0, scenarios, size=(size, scenarios))
        if first_runs == 1:
            first_values = first[0, scenario_index].mean(axis=1)
        else:
            first_index = rng.integers(0, first_runs, size=(size, first_runs))
            first_values = first[
                first_index[:, :, None], scenario_index[:, None, :]
            ].mean(axis=(1, 2))
        if second_runs == 1:
            second_values = second[0, scenario_index].mean(axis=1)
        else:
            second_index = rng.integers(0, second_runs, size=(size, second_runs))
            second_values = second[
                second_index[:, :, None], scenario_index[:, None, :]
            ].mean(axis=(1, 2))

        first_values *= scale
        second_values *= scale
        first_samples[start : start + size] = first_values
        second_samples[start : start + size] = second_values
        difference_samples[start : start + size] = first_values - second_values

    first_mean = float(first.mean() * scale)
    second_mean = float(second.mean() * scale)
    difference = first_mean - second_mean
    centered = difference_samples - difference
    extreme = int(np.count_nonzero(np.abs(centered) >= abs(difference)))
    return {
        "first": {"mean": first_mean, "ci95": percentile_interval(first_samples)},
        "second": {"mean": second_mean, "ci95": percentile_interval(second_samples)},
        "difference": difference,
        "difference_ci95": percentile_interval(difference_samples),
        "p_value": float((extreme + 1) / (replicates + 1)),
    }


def add_holm(records: dict[str, dict]) -> None:
    adjusted = holm([record["p_value"] for record in records.values()])
    for record, value in zip(records.values(), adjusted):
        record["p_holm"] = float(value)


def analysis_metadata(replicates: int) -> dict:
    return {
        "analysis": "training-run and scenario bootstrap",
        "bootstrap_replicates": replicates,
        "bootstrap_seed": SEED,
        "interval": "percentile 95% confidence interval",
        "test": "two-sided centered bootstrap approximate test",
        "multiple_testing": "Holm adjustment within each table family",
    }


def analyze_main(rows: list[dict], replicates: int = REPLICATES) -> dict:
    requirements = {method: TABLE3_METRICS for method in TABLE3_METHODS}
    requirements["dpp_gcmarl"] = tuple(SCALES)
    requirements["memory_gcmarl"] = TABLE4_METRICS
    requirements["without_gcm"] = TABLE5_METRICS
    matrices = {
        method: {
            metric: method_matrix(rows, method, metric) for metric in metrics
        }
        for method, metrics in requirements.items()
    }

    absolute: dict[str, dict] = {}
    for method_index, method in enumerate(MAIN_METHODS):
        absolute[method] = {}
        for metric, matrix in matrices[method].items():
            absolute[method][metric] = absolute_record(
                matrix,
                replicates,
                SEED + 10_000 + 100 * method_index,
                SCALES[metric],
            )

    def compare(first: str, second: str, metrics: tuple[str, ...], seed: int) -> dict:
        records = {
            metric: comparison_record(
                matrices[first][metric],
                matrices[second][metric],
                replicates,
                seed,
                SCALES[metric],
            )
            for metric in metrics
        }
        add_holm(records)
        return records

    return {
        **analysis_metadata(replicates),
        "tables": {
            "table3": {
                "absolute": {
                    method: {
                        metric: absolute[method][metric]
                        for metric in TABLE3_METRICS
                    }
                    for method in TABLE3_METHODS
                },
                "dpp_gcmarl_minus_mappo": compare(
                    "dpp_gcmarl", "mappo", TABLE3_METRICS, SEED + 20_000
                ),
                "training_run_e2e": {
                    method: absolute[method]["e2e"]
                    for method in ("mappo", "memory_gcmarl", "dpp_gcmarl")
                },
            },
            "table4": {
                "absolute": {
                    method: {
                        metric: absolute[method][metric]
                        for metric in TABLE4_METRICS
                    }
                    for method in ("memory_gcmarl", "dpp_gcmarl")
                },
                "dpp_gcmarl_minus_memory_gcmarl": compare(
                    "dpp_gcmarl",
                    "memory_gcmarl",
                    TABLE4_METRICS,
                    SEED + 30_000,
                )
            },
            "table5": {
                "absolute": {
                    method: {
                        metric: absolute[method][metric]
                        for metric in TABLE5_METRICS
                    }
                    for method in ("without_gcm", "dpp_gcmarl")
                },
                "dpp_gcmarl_minus_without_gcm": compare(
                    "dpp_gcmarl",
                    "without_gcm",
                    TABLE5_METRICS,
                    SEED + 40_000,
                )
            },
        },
    }


def analyze_conditions(
    rows: list[dict], conditions: tuple[str, ...], replicates: int, seed_step: int
) -> dict:
    observed = tuple(dict.fromkeys(row["condition"] for row in rows))
    if observed != conditions:
        raise ValueError(f"condition order mismatch: {observed}")

    records = {}
    for index, condition in enumerate(conditions):
        records[condition] = comparison_record(
            method_matrix(rows, "dpp_gcmarl", "e2e", condition),
            method_matrix(rows, "mappo", "e2e", condition),
            replicates,
            SEED + 50_000 + seed_step * index,
            100.0,
        )

    family = {key: value for key, value in records.items() if key != "nominal"}
    add_holm(family)
    return {
        **analysis_metadata(replicates),
        "holm_family": list(family),
        "conditions": records,
    }


def analyze_robustness(rows: list[dict], replicates: int = REPLICATES) -> dict:
    return analyze_conditions(rows, ROBUSTNESS_CONDITIONS, replicates, 10)


def analyze_cost_sensitivity(rows: list[dict], replicates: int = REPLICATES) -> dict:
    return analyze_conditions(rows, COST_CONDITIONS, replicates, 1)


def analyze(rows: list[dict], replicates: int = REPLICATES) -> dict:
    if "condition" not in rows[0]:
        return analyze_main(rows, replicates)
    conditions = {row["condition"] for row in rows}
    if conditions == set(COST_CONDITIONS):
        return analyze_cost_sensitivity(rows, replicates)
    return analyze_robustness(rows, replicates)


def print_summary(result: dict) -> None:
    if "tables" in result:
        table3 = result["tables"]["table3"]
        for method, metrics in table3["absolute"].items():
            record = metrics["e2e"]
            print(
                f"{method:24s} {record['mean']:.2f} "
                f"[{record['ci95'][0]:.2f}, {record['ci95'][1]:.2f}]"
            )
        return
    for condition, record in result["conditions"].items():
        low, high = record["difference_ci95"]
        print(
            f"{condition:24s} {record['difference']:+.2f} "
            f"[{low:+.2f}, {high:+.2f}]"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=ROOT / "data/reported_results.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/statistical_analysis.json"
    )
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    args = parser.parse_args()
    if args.replicates < 1_000:
        raise ValueError("at least 1,000 bootstrap replicates are required")

    result = analyze(read_rows(args.input), args.replicates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print_summary(result)
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()

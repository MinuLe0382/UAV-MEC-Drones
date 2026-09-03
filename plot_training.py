from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np


METHODS = {
    "dpp": {
        "label": "DPP-GCMARL",
        "color": "#D55E00",
        "relative": "dpp_gcmarl",
    },
    "mappo": {
        "label": "MAPPO",
        "color": "#0072B2",
        "relative": "mappo",
    },
}
METRICS = (
    ("end_to_end_success_rate", "(a)", "E2E completed workload (%)"),
    ("collection_success_rate", "(b)", "Collected workload (%)"),
)


def _read_run(path: Path) -> dict[str, np.ndarray]:
    episodes: list[int] = []
    values = {metric: [] for metric, _, _ in METRICS}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        for record in csv.DictReader(stream):
            episodes.append(int(record["episode"]))
            for metric in values:
                values[metric].append(
                    float(
                        record[
                            {
                                "end_to_end_success_rate": "e2e",
                                "collection_success_rate": "collection",
                            }[metric]
                        ]
                    )
                )
    expected = np.arange(1, 50001, dtype=np.int64)
    observed = np.asarray(episodes, dtype=np.int64)
    if not np.array_equal(observed, expected):
        raise RuntimeError(f"expected episodes 1..50000 in {path}")
    return {
        "episode": observed,
        **{key: np.asarray(item, dtype=np.float64) for key, item in values.items()},
    }


def _window(values: dict[str, np.ndarray], width: int) -> dict[str, np.ndarray]:
    centers: list[float] = []
    output = {metric: [] for metric, _, _ in METRICS}
    for start in range(1, 50001, width):
        end = min(start + width - 1, 50000)
        mask = (values["episode"] >= start) & (values["episode"] <= end)
        centers.append(0.5 * (start + end))
        for metric in output:
            output[metric].append(float(values[metric][mask].mean()))
    return {
        "episode": np.asarray(centers, dtype=np.float64),
        **{key: np.asarray(item, dtype=np.float64) for key, item in output.items()},
    }


def _aggregate(root: Path, width: int) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for method_id, method in METHODS.items():
        per_seed = []
        for seed in range(4):
            path = root / method["relative"] / f"seed_{seed}.csv.gz"
            if not path.is_file():
                raise FileNotFoundError(path)
            per_seed.append(_window(_read_run(path), width))
        episodes = per_seed[0]["episode"]
        if any(not np.array_equal(row["episode"], episodes) for row in per_seed):
            raise RuntimeError(f"window centers differ for {method_id}")
        aggregated: dict[str, np.ndarray] = {"episode": episodes}
        for metric, _, _ in METRICS:
            matrix = np.vstack([row[metric] for row in per_seed]) * 100.0
            aggregated[f"{metric}_mean"] = matrix.mean(axis=0)
            aggregated[f"{metric}_sd"] = matrix.std(axis=0, ddof=1)
            aggregated[f"{metric}_per_seed"] = matrix
        result[method_id] = aggregated
    return result


def _nice_limits(low: float, high: float) -> tuple[float, float, float]:
    span = max(high - low, 1.0)
    raw_step = span / 5.0
    magnitude = 10.0 ** np.floor(np.log10(raw_step))
    normalized = raw_step / magnitude
    step = (1.0 if normalized <= 1 else 2.0 if normalized <= 2 else 5.0) * magnitude
    ymin = float(np.floor((low - 0.06 * span) / step) * step)
    ymax = float(np.ceil((high + 0.06 * span) / step) * step)
    return ymin, ymax, float(step)


def _render(data: dict[str, dict[str, np.ndarray]], output: Path) -> None:
    width, height = 2200, 900
    top, bottom = 150, 210
    panel_width = 900
    panel_gap = 180
    panel_lefts = (130, 130 + panel_width + panel_gap)
    plot_height = height - top - bottom
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.tick{font-size:29px}.label{font-size:34px}.panel{font-size:34px;font-weight:bold}.legend{font-size:31px}</style>",
    ]

    legend_y = 55
    legend_items = list(METHODS.values())
    legend_start = 690
    for index, method in enumerate(legend_items):
        x = legend_start + index * 440
        svg.extend(
            [
                f'<line x1="{x}" y1="{legend_y}" x2="{x+95}" y2="{legend_y}" stroke="{method["color"]}" stroke-width="7"/>',
                f'<text class="legend" x="{x+115}" y="{legend_y+11}">{html.escape(method["label"])}</text>',
            ]
        )

    for panel_index, (metric, panel, ylabel) in enumerate(METRICS):
        x0 = panel_lefts[panel_index]
        x1 = x0 + panel_width
        y0 = top
        y1 = top + plot_height
        lows = []
        highs = []
        for method_id in METHODS:
            mean = data[method_id][f"{metric}_mean"]
            sd = data[method_id][f"{metric}_sd"]
            lows.extend((mean - sd).tolist())
            highs.extend((mean + sd).tolist())
        ymin, ymax, ystep = _nice_limits(min(lows), max(highs))

        def x_map(episode: float) -> float:
            return x0 + episode / 50000.0 * panel_width

        def y_map(value: float) -> float:
            return y1 - (value - ymin) / (ymax - ymin) * plot_height

        tick = ymin
        while tick <= ymax + 1e-9:
            y = y_map(tick)
            svg.extend(
                [
                    f'<line x1="{x0}" y1="{y:.2f}" x2="{x1}" y2="{y:.2f}" stroke="#D9D9D9" stroke-width="1.5"/>',
                    f'<text class="tick" x="{x0-18}" y="{y+10:.2f}" text-anchor="end">{tick:.0f}</text>',
                ]
            )
            tick += ystep
        for xtick in range(0, 51, 10):
            x = x_map(xtick * 1000.0)
            svg.extend(
                [
                    f'<line x1="{x:.2f}" y1="{y1}" x2="{x:.2f}" y2="{y1+10}" stroke="#222" stroke-width="2"/>',
                    f'<text class="tick" x="{x:.2f}" y="{y1+45}" text-anchor="middle">{xtick}</text>',
                ]
            )
        svg.extend(
            [
                f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#222" stroke-width="2.5"/>',
                f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#222" stroke-width="2.5"/>',
                f'<text class="label" x="{(x0+x1)/2:.2f}" y="{height-80}" text-anchor="middle">Training episode (×10³)</text>',
                f'<text class="panel" x="{(x0+x1)/2:.2f}" y="{height-25}" text-anchor="middle">{panel}</text>',
                f'<text class="label" x="{x0-92}" y="{(y0+y1)/2:.2f}" text-anchor="middle" transform="rotate(-90 {x0-92} {(y0+y1)/2:.2f})">{html.escape(ylabel)}</text>',
            ]
        )

        for method_id, method in METHODS.items():
            row = data[method_id]
            episodes = row["episode"]
            mean = row[f"{metric}_mean"]
            sd = row[f"{metric}_sd"]
            upper = [f"{x_map(x):.2f},{y_map(y):.2f}" for x, y in zip(episodes, mean + sd)]
            lower = [f"{x_map(x):.2f},{y_map(y):.2f}" for x, y in zip(episodes, mean - sd)]
            svg.append(
                f'<polygon points="{" ".join(upper + list(reversed(lower)))}" fill="{method["color"]}" fill-opacity="0.18"/>'
            )
        for method_id, method in METHODS.items():
            row = data[method_id]
            points = " ".join(
                f"{x_map(x):.2f},{y_map(y):.2f}"
                for x, y in zip(row["episode"], row[f"{metric}_mean"])
            )
            svg.append(
                f'<polyline points="{points}" fill="none" stroke="{method["color"]}" stroke-width="6" stroke-linejoin="round" stroke-linecap="round"/>'
            )

    svg.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    svg_path = output.with_suffix(".svg")
    svg_path.write_text("\n".join(svg) + "\n", encoding="utf-8", newline="\n")
    if output.suffix.lower() == ".svg":
        return
    magick = shutil.which("magick")
    if magick is None:
        raise RuntimeError("ImageMagick executable 'magick' was not found")
    subprocess.run(
        [
            magick,
            "-background",
            "white",
            "-density",
            "300",
            str(svg_path),
            "-alpha",
            "remove",
            str(output),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path(__file__).resolve().parent / "data/training"
    )
    parser.add_argument("--output", type=Path, default=Path("results/figure3.svg"))
    parser.add_argument("--png", action="store_true")
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--summary", type=Path, default=Path("results/training_summary.json"))
    args = parser.parse_args()
    if args.window <= 0 or 50000 % args.window:
        raise ValueError("window must be a positive divisor of 50000")
    data = _aggregate(args.input.resolve(), args.window)
    _render(data, args.output.resolve().with_suffix(".png" if args.png else ".svg"))
    if args.summary:
        summary = {
            "window": args.window,
            "training_seeds": [0, 1, 2, 3],
            "band": "pointwise sample standard deviation across four run-level window means",
            "methods": {},
        }
        for method_id, method in METHODS.items():
            summary["methods"][method["label"]] = {}
            for metric, _, _ in METRICS:
                per_seed = data[method_id][f"{metric}_per_seed"]
                tail = per_seed[:, -10000 // args.window :].mean(axis=1)
                summary["methods"][method["label"]][metric] = {
                    "final_10000_episode_per_seed_percent": tail.tolist(),
                    "mean_percent": float(tail.mean()),
                    "sd_percent": float(tail.std(ddof=1)),
                }
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()

# Reproduction guide

## Setup

Use Python 3.10–3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -v
```

## Reproduce the statistical tables

Recompute the statistical tables:

```bash
python analyze_results.py
python analyze_results.py --input data/reported_robustness.json --output results/robustness_statistics.json
```

## Evaluate the selected policies

The selected checkpoints are included in:

```text
checkpoints/dpp_gcmarl/seed_0.pt ... seed_3.pt
checkpoints/mappo/seed_0.pt ... seed_3.pt
checkpoints/memory_gcmarl/seed_0.pt ... seed_3.pt
```

Then run:

```bash
python evaluate.py --split test
```

The rule-based methods do not require learned checkpoints:

```bash
python evaluate.py --split test --methods memory anticipatory belief_mpc full_information without_gcm --output results/rule_evaluation.json
```

All methods receive the same fixed realization for each scenario. Each learned
method is evaluated with four independently trained policies, and the analysis
first averages those four values within each scenario.

The eight-axis robustness evaluation uses the same selected policies:

```bash
python evaluate_robustness.py
python analyze_results.py --input results/robustness_evaluation.json --output results/robustness_statistics.json
```

## Train from scratch

Run four independent seeds for every learned method:

```bash
python train.py --method dpp_gcmarl --seed 0
python train.py --method mappo --seed 0
python train.py --method memory_gcmarl --seed 0
```

Repeat each command with seeds 1, 2, and 3. Training uses 50,000 episodes and
saves checkpoint candidates every 2,000 episodes. The validation scenarios are
stored under `data/scenarios/validation` and are separate from the development
and test scenarios.

Select the checkpoint with the highest mean validation E2E for each run:

```bash
python select_checkpoint.py --method dpp_gcmarl --seed 0 \
  --run-directory runs/dpp_gcmarl/seed_0
```

Repeat this selection for every method and training seed before evaluation.

## Figure 3

```bash
python plot_training.py
```

Reads eight compressed logs from `data/training`. Saves an SVG figure and
the final 10,000-episode means. Curves use non-overlapping 1,000-episode
window means; bands show one sample SD across four runs. Add `--png` to
also produce PNG using ImageMagick.

## Scenario files

Seeds and parameters: `data/scenario_manifest.csv` and `scenario_settings.json`.

```bash
python generate_scenarios.py --split test --output results/scenarios/test
python generate_scenarios.py --split robustness --output results/scenarios/robustness
python generate_scenarios.py --verify-existing
```

Generation uses the listed seeds and settings, without loading saved arrays.
Verification compares all nine generated arrays with the 750 supplied files.
Existing files are not overwritten.

## Runtime benchmark

Use one CPU thread for all numerical libraries:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python benchmark_runtime.py --components
```

The benchmark uses 1,000 warm-up calls followed by 10,000 measured calls after
160 rollout slots from generator seed 0. Each learned policy generates its own
rollout. Rule methods share the state obtained from a Memory rollout. The
component measurements cover DPP assignment, observation construction, actor
forward pass, actor interface, and processing-rate governor. Per-call timings
are saved in an NPZ file alongside the summary.

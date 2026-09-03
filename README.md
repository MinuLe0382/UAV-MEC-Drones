# DPP-GCMARL

Python: 3.10–3.12.

Reference environment: Linux x86-64, Intel Core i9-10940X,
Python 3.10.13, NumPy 1.26.3, PyTorch 2.2.0.

Simulation results may vary across platforms and NumPy builds because
equal-utility assignments can be ordered differently. Statistical tables can
be recalculated directly from the supplied result files.

## Install

```bash
pip install -r requirements.txt
```

## Commands

```bash
python -m unittest discover -v
python analyze_results.py
python evaluate.py --split test
```

Training, robustness, runtime: [RUNBOOK.md](RUNBOOK.md).

Parameters: [PAPER_CONFIG.md](PAPER_CONFIG.md).

Code, data, and checkpoints: [MIT License](LICENSE).

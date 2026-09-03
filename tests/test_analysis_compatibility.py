from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

import analyze_results
from config import CONFIG, SYS, apply_method


ROOT = Path(__file__).resolve().parents[1]


class AnalysisCompatibilityTests(unittest.TestCase):
    def test_analysis_cli_does_not_check_simulation_numpy_version(self):
        inputs = ("reported_results.json", "reported_robustness.json")
        for name in inputs:
            with self.subTest(input=name), tempfile.TemporaryDirectory() as directory:
                source = ROOT / "data" / name
                output = Path(directory) / "statistics.json"
                rows = json.loads(source.read_text(encoding="utf-8"))["rows"]
                analyze = (
                    analyze_results.analyze_robustness
                    if "condition" in rows[0]
                    else analyze_results.analyze_main
                )
                expected = analyze(rows)
                argv = ["analyze_results.py", "--input", str(source), "--output", str(output)]
                with (
                    patch.object(np, "__version__", "2.4.4"),
                    patch(
                        "config.validate_runtime",
                        side_effect=AssertionError("version guard called"),
                    ),
                    patch("sys.argv", argv),
                    redirect_stdout(io.StringIO()),
                ):
                    analyze_results.main()
                actual = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(actual, expected)

    def test_method_setup_continues_after_version_warning(self):
        with patch.object(np, "__version__", "2.4.4"):
            with self.assertWarnsRegex(RuntimeWarning, "Simulation results may differ"):
                apply_method("dpp_gcmarl")
        self.assertEqual(SYS, CONFIG["system"])


if __name__ == "__main__":
    unittest.main()

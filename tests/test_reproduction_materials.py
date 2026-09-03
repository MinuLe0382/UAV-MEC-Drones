from __future__ import annotations

import csv
import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from config import CONFIG, ROOT, validate_slot_duration
from env.uav_mec_env import UAVMECEnv
from generate_scenarios import SETTINGS
from plot_training import _aggregate, _window
from rule_methods import _assign, RuleController


class ReproductionMaterialsTests(unittest.TestCase):
    def test_slot_duration_is_explicit_and_fixed(self):
        self.assertEqual(CONFIG["system"]["slot_duration_s"], 1.0)
        validate_slot_duration()
        with patch.dict(CONFIG["system"], {"slot_duration_s": 2.0}):
            with self.assertRaises(ValueError):
                validate_slot_duration()

    def test_assignment_uses_distance_per_slot(self):
        env = SimpleNamespace(
            d_max=20.0,
            v_uav=7.0,
            uav_pos=np.array([[0.0, 0.0]]),
            _cell_centers=np.array([[10.0, 0.0], [80.0, 0.0]]),
        )
        with patch("rule_methods._assign_matrix") as assign:
            _assign(env, np.array([1.0, 2.0]))
        np.testing.assert_array_equal(assign.call_args.args[0], [[1.0, 0.5]])
        self.assertNotIn("v_uav", inspect.getsource(UAVMECEnv._memory_assign))
        self.assertNotIn("v_uav", inspect.getsource(RuleController.action))
        self.assertIn("self.v_uav", inspect.getsource(UAVMECEnv._energy_consumption))

    def test_nominal_travel_denominator_is_unchanged(self):
        self.assertEqual(CONFIG["system"]["v_uav"], CONFIG["system"]["d_max"])
        distances = np.linspace(0, 300, 1000, dtype=np.float32)
        np.testing.assert_array_equal(
            np.maximum(distances / CONFIG["system"]["v_uav"], 1.0),
            np.maximum(distances / CONFIG["system"]["d_max"], 1.0),
        )

    def test_nonoverlapping_training_windows(self):
        values = {
            "episode": np.arange(1, 50001),
            "end_to_end_success_rate": np.arange(50000, dtype=float),
            "collection_success_rate": np.arange(50000, dtype=float),
        }
        result = _window(values, 1000)
        self.assertEqual(len(result["episode"]), 50)
        self.assertEqual(result["episode"][0], 500.5)
        self.assertEqual(result["end_to_end_success_rate"][1], 1499.5)

    def test_training_data_reproduce_paper_means_and_sample_sd(self):
        data = _aggregate(ROOT / "data/training", 1000)
        expected = {
            "dpp": (39.20423447982576, 39.43017004338535),
            "mappo": (35.24016156929058, 35.4480468749948),
        }
        for method, target in expected.items():
            for metric, mean in zip(("end_to_end_success_rate", "collection_success_rate"), target):
                matrix = data[method][metric + "_per_seed"]
                self.assertEqual(matrix.shape, (4, 50))
                self.assertAlmostEqual(float(matrix[:, -10:].mean()), mean, places=10)
                np.testing.assert_array_equal(
                    data[method][metric + "_sd"], matrix.std(axis=0, ddof=1)
                )

    def test_manifest_lists_every_released_scenario(self):
        with (ROOT / "data/scenario_manifest.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 750)
        files = {row["file"] for row in rows}
        expected = {
            p.relative_to(ROOT).as_posix()
            for folder in ("scenarios", "robustness")
            for p in (ROOT / "data" / folder).glob("*/*.npz")
        }
        self.assertEqual(files, expected)
        for split, seeds in SETTINGS["splits"].items():
            self.assertEqual(
                [int(row["scenario_seed"]) for row in rows if row["split"] == split], seeds
            )


if __name__ == "__main__":
    unittest.main()

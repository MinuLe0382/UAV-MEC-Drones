from __future__ import annotations

import inspect
import json
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from analyze_results import (
    COST_CONDITIONS,
    TABLE4_METRICS,
    TABLE5_METRICS,
    analyze_cost_sensitivity,
    analyze_main,
    analyze_robustness,
    holm,
    method_matrix,
)
from config import validate_runtime
from evaluate import make_environment
from metrics import PatrolTracker, service_metrics
from scenarios import load_scenario, scenario_files, validate_scenario


ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = json.loads((ROOT / "data/reported_results.json").read_text())["rows"]

    def test_environment_dimensions_match_the_reported_methods(self):
        scenario = load_scenario(scenario_files("test", ROOT)[0])
        expected = {"dpp_gcmarl": 91, "mappo": 86, "memory_gcmarl": 91}
        for method, dimension in expected.items():
            self.assertEqual(make_environment(method, scenario).obs_dim, dimension)

    def test_selected_checkpoints_load_and_produce_finite_outputs(self):
        import torch
        from evaluate import LEARNED_METHODS, checkpoint_files, load_agent

        scenario = load_scenario(scenario_files("test", ROOT)[0])
        for method in LEARNED_METHODS:
            env = make_environment(method, scenario)
            observations = np.asarray(env.reset(), dtype=np.float32)
            checkpoints = checkpoint_files(method)
            self.assertEqual(len(checkpoints), 4)
            for checkpoint in checkpoints:
                with self.subTest(method=method, checkpoint=checkpoint.name):
                    agent = load_agent(method, checkpoint, env)
                    actions = agent.select_actions(observations, noise=False)
                    self.assertEqual(actions.shape, (env.M, 2))
                    self.assertTrue(np.isfinite(actions).all())
                    self.assertTrue((np.abs(actions) <= 1).all())
                    with torch.no_grad():
                        values = agent.critic(torch.as_tensor(observations.reshape(1, -1)))
                    self.assertEqual(tuple(values.shape), (1, env.M))
                    self.assertTrue(torch.isfinite(values).all().item())

    def test_reward_signature_matches_the_reported_definition(self):
        from env.reward import movement_reward

        parameters = tuple(inspect.signature(movement_reward).parameters)
        self.assertEqual(
            parameters,
            (
                "completed_bits",
                "new_coarse_cells",
                "active_at_slot_start",
                "boundary_hit",
                "collision_hit",
                "energy_remaining_after_slot",
                "reward_scale",
                "gamma_e",
                "idle_penalty",
                "boundary_collision_penalty",
                "energy_overrun_base",
                "energy_overrun_per_excess",
            ),
        )

    def test_primary_result(self):
        result = json.loads((ROOT / "data/main_statistics.json").read_text())
        record = result["tables"]["table3"]["dpp_gcmarl_minus_mappo"]["e2e"]
        self.assertAlmostEqual(record["difference"], 3.6339661459058306)
        np.testing.assert_allclose(
            record["difference_ci95"],
            [2.601030599393475, 4.633024493442783],
            rtol=0,
            atol=0,
        )

    def test_reward_arithmetic(self):
        from env.reward import movement_reward

        actual = movement_reward(
            completed_bits=np.array([2e6, 0, 1e6, 0]),
            new_coarse_cells=np.array([1, 2, 0, 1]),
            active_at_slot_start=np.array([True, True, True, False]),
            boundary_hit=np.array([1, 0, 0, 1]),
            collision_hit=np.array([0, 1, 0, 1]),
            energy_remaining_after_slot=np.array([1, -2, 0, -4]),
            reward_scale=1e6,
            gamma_e=0.1,
            idle_penalty=1,
            boundary_collision_penalty=1,
            energy_overrun_base=3,
            energy_overrun_per_excess=0.5,
        )
        np.testing.assert_allclose(actual, [1.1, -5.8, 1.0, 0.0], atol=1e-6)

    def test_zero_generation_users_and_clusters_are_excluded(self):
        env = SimpleNamespace(
            visited_global=np.array([True, False]),
            generated_per_sd=np.array([4.0, 4.0, 0.0]),
            served_per_sd=np.array([2.0, 0.0, 100.0]),
            sd_cluster=np.array([0, 1, 2]),
            n_clusters=3,
        )
        self.assertEqual(service_metrics(env), (0.5, 0.25, 0.0))
        env.generated_per_sd[:] = 0
        _, ecf, minimum = service_metrics(env)
        self.assertEqual(ecf, 0.0)
        self.assertTrue(np.isnan(minimum))

    def test_collection_ratios_are_capped_at_generated_workload(self):
        env = SimpleNamespace(
            visited_global=np.array([True]),
            generated_per_sd=np.array([1.0, 2.0]),
            served_per_sd=np.array([3.0, 4.0]),
            sd_cluster=np.array([0, 1]),
            n_clusters=2,
        )
        self.assertEqual(service_metrics(env), (1.0, 1.0, 1.0))

    def test_revisit_includes_terminal_tail_not_first_discovery_wait(self):
        env = SimpleNamespace(
            T=4,
            _n_cells=3,
            sd_cell=np.array([0, 1, 2]),
            t=0,
            cell_last_seen=np.full(3, -100.0),
            discovered_user_cells=np.zeros(3, dtype=bool),
        )
        tracker = PatrolTracker(env)
        for slot, cells in enumerate(([0], [1], [0], [])):
            env.cell_last_seen[cells] = slot
            env.discovered_user_cells[cells] = True
            env.t = slot + 1
            tracker.after_step(env, slot)
        result = tracker.finalize(env)
        self.assertAlmostEqual(result["discovery"], 2 / 3)
        self.assertAlmostEqual(result["revisit_p95"], 2.9)

    def test_revisit_rejects_incomplete_mission(self):
        env = SimpleNamespace(T=4, _n_cells=1, sd_cell=np.array([0]))
        with self.assertRaises(ValueError):
            PatrolTracker(env).finalize(env)

    def test_scenario_loader_shapes_and_copies(self):
        path = scenario_files("test", ROOT)[0]
        a, b = load_scenario(path), load_scenario(path)
        a["gu_positions"][0, 0] = -1
        self.assertGreaterEqual(b["gu_positions"][0, 0], 0)
        with self.assertRaises(ValueError):
            validate_scenario(a)
        b["task_sizes_bits"] = b["task_sizes_bits"][:-1]
        with self.assertRaises(ValueError):
            validate_scenario(b)

    def test_all_released_scenarios_load(self):
        paths = list((ROOT / "data/scenarios").rglob("*.npz")) + list(
            (ROOT / "data/robustness").rglob("*.npz")
        )
        self.assertEqual(len(paths), 750)
        for path in paths:
            load_scenario(path)

    def test_reference_numpy_version_is_silent(self):
        with patch.object(np, "__version__", "1.26.3"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                validate_runtime()
        self.assertEqual(caught, [])

    def test_other_numpy_versions_warn_without_blocking(self):
        with patch.object(np, "__version__", "2.4.4"):
            with self.assertWarnsRegex(RuntimeWarning, "Simulation results may differ"):
                validate_runtime()

    def test_holm_algorithm(self):
        self.assertEqual(holm([0.03, 0.01, 0.04]), [0.06, 0.03, 0.06])

    def test_holm_families_match_the_tables(self):
        result = analyze_main(self.rows, replicates=1_000)["tables"]
        table4 = result["table4"]["dpp_gcmarl_minus_memory_gcmarl"]
        table5 = result["table5"]["dpp_gcmarl_minus_without_gcm"]
        self.assertEqual(tuple(table4), TABLE4_METRICS)
        self.assertEqual(tuple(table5), TABLE5_METRICS)
        revisit = table4["revisit_p95"]
        self.assertAlmostEqual(revisit["difference"], 0.75, delta=0.01)
        self.assertIn("p_holm", revisit)

    def test_duplicate_policy_row_is_rejected(self):
        with self.assertRaises(ValueError):
            method_matrix([*self.rows, self.rows[0]], self.rows[0]["method"], "e2e")

    def test_nominal_is_excluded_from_holm(self):
        rows = json.loads((ROOT / "data/reported_robustness.json").read_text())["rows"]
        result = analyze_robustness(rows, replicates=1_000)
        self.assertEqual(len(result["holm_family"]), 16)
        self.assertNotIn("nominal", result["holm_family"])
        self.assertNotIn("p_holm", result["conditions"]["nominal"])

    def test_timing_warmup_is_excluded_from_samples(self):
        from benchmark_runtime import measure

        function = Mock()
        with patch("benchmark_runtime.time.perf_counter_ns", side_effect=[0, 1000, 2000, 5000]):
            result, samples = measure(function, warmup=3, repetitions=2)
        self.assertEqual(function.call_count, 5)
        np.testing.assert_array_equal(samples, [1000, 3000])
        self.assertEqual(result, {"median_us": 2.0, "p95_us": 2.9, "mean_us": 2.0})

    def test_component_benchmark_uses_distinct_code_paths(self):
        import torch
        from benchmark_runtime import component_functions

        observations = np.zeros((3, 91), dtype=np.float32)
        actions = np.zeros((3, 2), dtype=np.float32)
        env = SimpleNamespace(
            goal_source="dpp", _memory_assign=Mock(), _all_obs=Mock(return_value=observations)
        )
        agent = SimpleNamespace(
            select_actions=Mock(return_value=actions),
            actor=Mock(side_effect=lambda tensor: torch.is_grad_enabled()),
        )
        with patch("benchmark_runtime.compose_environment_actions") as governor:
            functions = component_functions(env, agent, observations)
            self.assertEqual(
                set(functions),
                {
                    "dpp_assignment",
                    "observation_builder",
                    "actor_network_only",
                    "actor_interface",
                    "plan_governor",
                    "online_decision_total",
                },
            )
            agent.select_actions.reset_mock()
            self.assertFalse(functions["actor_network_only"]())
            agent.select_actions.assert_not_called()
            functions["dpp_assignment"]()
            env._memory_assign.assert_called_once_with(source="dpp")
            functions["observation_builder"]()
            functions["actor_interface"]()
            functions["plan_governor"]()
            functions["online_decision_total"]()
            self.assertEqual(env._all_obs.call_count, 2)
            self.assertEqual(agent.select_actions.call_count, 2)
            self.assertEqual(governor.call_count, 2)

    def test_no_active_uavs_does_not_end_the_mission_early(self):
        scenario = load_scenario(scenario_files("test", ROOT)[0])
        env = make_environment("dpp_gcmarl", scenario)
        env.reset()
        env.energy_remaining[:] = 0
        env.active[:] = False
        for slot in range(env.T):
            _, reward, done, _ = env.step(np.zeros((env.M, env.act_dim), dtype=np.float32))
            self.assertEqual(done, slot == env.T - 1)
            np.testing.assert_array_equal(reward, np.zeros(env.M, dtype=np.float32))

    def test_table6_recomputed_from_rows(self):
        rows = json.loads((ROOT / "data/reported_robustness.json").read_text())["rows"]
        expected = json.loads((ROOT / "data/robustness_statistics.json").read_text())
        actual = analyze_robustness(rows)
        self.assertEqual(len(rows), 5440)
        self.assertEqual(len(actual["holm_family"]), 16)
        self.assertNotIn("nominal", actual["holm_family"])
        self.assertEqual(actual, expected)

    def test_table7_recomputed_from_rows(self):
        rows = json.loads((ROOT / "data/operating_cost_sensitivity.json").read_text())["rows"]
        expected = json.loads((ROOT / "data/operating_cost_statistics.json").read_text())
        actual = analyze_cost_sensitivity(rows)
        self.assertEqual(len(rows), 2880)
        self.assertEqual(tuple(actual["conditions"]), COST_CONDITIONS)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()

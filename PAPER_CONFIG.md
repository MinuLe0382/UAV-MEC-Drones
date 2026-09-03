# Paper configuration

System, workload, reward, and training settings are in `config.json`.
Scenario variations are in `scenario_settings.json`. File-level settings and
seeds are in `data/scenario_manifest.csv`.

| Parameter | Value | Code location |
| --- | --- | --- |
| DPP occupancy weight | 0.25 | `env/uav_mec_env.py`, `_memory_assign` |
| DPP staleness weight | 0.75 | `env/uav_mec_env.py`, `_memory_assign` |
| DPP unobserved-cell weight | 0.5 | `env/uav_mec_env.py`, `_memory_assign` |
| Target-distance horizon, Tg | 10 slots | `env/uav_mec_env.py`, `_goal_block` |
| Budget-pacing factor, kappa | 0.9 | `controllers/frequency_governor.py`, `PLAN_SAFETY_FACTOR` |
| Governor rate fractions | 0.05, 0.10, 0.20, 0.35, 0.60, 1.00 | `FREQUENCY_FRACTION_POINTS` |
| Governor cost per slot | (11.7, 12.0, 13.9, 20.4, 42.9, 97.4) / 320 | `ENERGY_RATE_POINTS` |
| Belief-MPC horizon | 10 slots | `rule_methods.py`, `_belief_mpc` |
| Belief-MPC base candidate count | 8 | `rule_methods.py`, `_belief_mpc` |
| Belief-MPC discount | 0.85 | `rule_methods.py`, `_belief_mpc` |

The governor interpolates between the listed cost and rate-fraction points.
Belief-MPC may add candidates up to a total of 8 + U (11 for the nominal team).
Travel time is measured in slots using `distance / d_max`. The operating-cost
calculation separately uses `v_uav`. Only 1-second slots are supported.

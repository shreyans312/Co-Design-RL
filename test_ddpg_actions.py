import json
import unittest

import numpy as np
import torch

from diagnose_ddpg_actions import collect_episode, probe_battery
from rl_config import get_rl_config
from train_rl import CoDesignTrainer, generate_dataset, get_default_config


class ActionDiagnosticTests(unittest.TestCase):
    def test_physical_limits_block_empty_discharge_and_full_charge(self):
        config = get_default_config()
        config["design"]["battery_capacity_kwh"] = 4.5
        for action, soc in [(0.5, 0.0), (-0.5, 4.5)]:
            result = probe_battery(action, soc, config, 0.1, np.array([-1.0, 1.0]))
            self.assertGreater(result["requested_std_kw"], 0.0)
            self.assertEqual(result["applied_std_kw"], 0.0)
            self.assertEqual(result["blocked_request_fraction"], 1.0)

    def test_feasible_probe_preserves_power_variation(self):
        config = get_default_config()
        result = probe_battery(0.0, 4.0, config, 0.1, np.array([-1.0, 1.0]))
        self.assertAlmostEqual(result["requested_std_kw"], result["applied_std_kw"])
        self.assertEqual(result["power_limited_fraction"], 0.0)

    def test_replay_matches_validation_and_probes_do_not_change_trajectory(self):
        torch.set_num_threads(1)
        system = get_default_config()
        system["simulation"]["episode_hours"] = 24
        config = get_rl_config("ddpg_lognormal_improved")
        config["algorithm"]["hidden_sizes"] = (16, 16)
        trainer = CoDesignTrainer(system, generate_dataset(year=2026, seed=42), config)
        before_weights = [p.detach().clone() for p in trainer.agent.actor.parameters()]
        before_rng = json.dumps(trainer.agent.rng.bit_generator.state)
        design = np.array([6.6, 4.5], dtype=np.float32)
        seed = 1_000_042
        steps, probes = collect_episode(trainer, design, seed, [0.0, 0.3], probe_draws=4)
        repeat, _ = collect_episode(trainer, design, seed, [0.4], probe_draws=8)
        np.testing.assert_array_equal(steps.battery_power_kw, repeat.battery_power_kw)
        expected = trainer._run_episode(design, "validation", seed, False, False)
        self.assertAlmostEqual(steps.total_cost_chf.sum(), expected["cost_chf"])
        self.assertAlmostEqual(steps.battery_power_limited.mean(), expected["battery_power_limited_fraction"])
        self.assertAlmostEqual(steps.iloc[0].battery_soc_before_kwh, 2.25)
        np.testing.assert_allclose(steps.battery_soc_before_kwh.iloc[1:], steps.battery_soc_kwh.iloc[:-1])
        self.assertLess(steps.power_balance_error_kw.abs().max(), 1e-8)
        self.assertEqual(len(probes), 48)
        self.assertEqual(len(trainer.agent.replay), 0)
        self.assertEqual(before_rng, json.dumps(trainer.agent.rng.bit_generator.state))
        for before, after in zip(before_weights, trainer.agent.actor.parameters()):
            torch.testing.assert_close(before, after)


if __name__ == "__main__":
    unittest.main()

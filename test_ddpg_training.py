import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from rl_config import get_rl_config, validate_rl_config
from train_rl import CoDesignTrainer, generate_dataset, get_default_config, run_experiment


class DDPGTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.data = generate_dataset(year=2026, days=365, seed=42)

    def trainer(self, seed=42):
        system = get_default_config()
        system["simulation"]["episode_hours"] = 24
        config = get_rl_config("ddpg_lognormal_improved")
        config["algorithm"].update(hidden_sizes=(16, 16), minibatch_size=8, learning_starts=8)
        config["training"].update(
            iterations=2, episodes_per_iteration=4, scenarios_per_design=2,
            validation_episodes=2, matched_validation=True,
            validation_seed=1_000_042, reference_designs=[[6.6, 4.5]],
        )
        return CoDesignTrainer(system, self.data, config, seed=seed)

    def test_matched_evaluation_repeats_without_changing_training_rng(self):
        trainer = self.trainer()
        generator_before = trainer.design_distribution.generator.get_state().clone()
        agent_rng_before = json.dumps(trainer.agent.rng.bit_generator.state)
        torch_rng_before = torch.get_rng_state().clone()
        first = trainer.evaluate_matched()
        first_rows = pd.DataFrame(trainer.validation_records)
        trainer.completed_iterations = 50
        second = trainer.evaluate_matched()
        second_rows = pd.DataFrame(trainer.validation_records[len(first_rows):])
        for key in first:
            if key != "iteration":
                self.assertEqual(first[key], second[key])
        pd.testing.assert_frame_equal(
            first_rows.drop(columns="iteration"), second_rows.drop(columns="iteration").reset_index(drop=True),
        )
        torch.testing.assert_close(generator_before, trainer.design_distribution.generator.get_state())
        torch.testing.assert_close(torch_rng_before, torch.get_rng_state())
        self.assertEqual(agent_rng_before, json.dumps(trainer.agent.rng.bit_generator.state))
        self.assertEqual(len(trainer.agent.replay), 0)

    def test_custom_entropy_update_increases_spread_for_equal_returns(self):
        trainer = self.trainer()
        trainer.rl_config["design_distribution"]["entropy_weight"] = 1.0
        trainer.rl_config["training"]["design_warmup_fraction"] = 0.0
        before = trainer.design_distribution.component_log_stds.detach().clone()
        batch = trainer.design_distribution.sample(40000)
        metrics = trainer._update_design(batch, np.zeros(40000), 0)
        self.assertTrue(torch.all(trainer.design_distribution.component_log_stds.detach() > before))
        self.assertGreater(metrics["design_gradient_norm"], 0.0)

    def test_actor_update_leaves_critic_gradients_from_critic_loss_only(self):
        trainer = self.trainer()
        agent = trainer.agent
        for index in range(8):
            state = np.full(11, index / 10, dtype=np.float32)
            agent.observe(state, np.array([0.2, -0.1]), -1.0, state + 0.01, False)
        calls = []
        handle = next(agent.critic.parameters()).register_hook(lambda grad: calls.append(grad.clone()))
        before = [p.detach().clone() for p in agent.actor.parameters()]
        metrics = agent.update()
        handle.remove()
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(p.requires_grad for p in agent.critic.parameters()))
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, agent.actor.parameters())))
        self.assertGreater(metrics["actor_gradient_norm"], 0)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_horizon_bootstrap_uses_post_action_energy_and_next_hour(self):
        trainer = self.trainer()
        trainer.system_config["simulation"]["episode_hours"] = 168
        trainer._run_episode(np.array([6.6, 4.5]), "train", 42, True, True)
        state, action, reward, next_state, terminated = trainer.agent.replay.storage[-1]
        self.assertFalse(terminated)
        self.assertEqual(float(state[0]), 1.0)
        self.assertEqual(float(next_state[0]), 0.0)
        self.assertGreater(float(next_state[1]), float(state[1]))
        self.assertTrue(0 <= next_state[2] <= 1)

    def test_training_saves_matched_pairs_and_final_progress(self):
        trainer = self.trainer()
        trainer.rl_config["training"]["design_warmup_fraction"] = 0.5
        trainer.rl_config["algorithm"]["exploration_noise_std"] = 0.30
        trainer.rl_config["algorithm"]["final_exploration_noise_std"] = 0.05
        with TemporaryDirectory() as folder:
            path = Path(folder)
            history = trainer.train(output_directory=path, save_every=5, validation_every=5)
            self.assertEqual(len(history), 2)
            np.testing.assert_allclose(history["exploration_noise_std"], [0.30, 0.05])
            self.assertTrue(history["validation_mode_cost_chf"].notna().all())
            self.assertTrue(np.isnan(history.iloc[0]["design_loss"]))
            self.assertTrue(np.isfinite(history.iloc[1]["design_loss"]))
            self.assertEqual(len(pd.read_csv(path / "training.csv")), 2)
            self.assertEqual(len(pd.read_csv(path / "training_samples.csv")), 8)
            rows = pd.read_csv(path / "validation_episodes.csv")
            self.assertEqual(set(rows["iteration"]), {0, 1, 2})
            self.assertLess(rows["max_power_balance_error_kw"].max(), 1e-8)
            best = json.loads((path / "best_validation.json").read_text())
            pair = torch.load(path / "best_checkpoint.pt", weights_only=False)
            self.assertEqual(best["iteration"], pair["completed_iterations"])
            self.assertIn("actor", pair["agent"])
            self.assertIn("design_distribution", pair)
            self.assertFalse(pair["exact_resume_supported"])
            expected = rows[rows["evaluation"] == "mode"].groupby("iteration")["cost_chf"].mean().min()
            self.assertAlmostEqual(best["mean_cost_chf"], expected)
            self.assertFalse(list(path.glob("*.tmp")))

    def test_frozen_scoring_is_repeatable_and_does_not_train(self):
        trainer = self.trainer()
        trainer._run_episode(np.array([6.6, 4.5]), "train", 42, True, True)
        batch = trainer.design_distribution.sample(2)
        batch["designs"][1] = batch["designs"][0]
        replay = deepcopy(trainer.agent.replay.storage)
        modules = [trainer.agent.actor, trainer.agent.critic, trainer.agent.target_actor, trainer.agent.target_critic]
        states = [deepcopy(module.state_dict()) for module in modules]
        agent_rng = json.dumps(trainer.agent.rng.bit_generator.state)
        torch_rng = torch.get_rng_state().clone()
        design_rng = trainer.design_distribution.generator.get_state().clone()
        with patch.object(trainer.agent, "update", side_effect=AssertionError("scoring must not update")):
            first = trainer._score_designs(batch, 1)
            second = trainer._score_designs(batch, 1)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first[0], first[1])
        self.assertTrue(trainer.agent.actor.training)
        self.assertEqual(agent_rng, json.dumps(trainer.agent.rng.bit_generator.state))
        torch.testing.assert_close(torch_rng, torch.get_rng_state(), rtol=0, atol=0)
        torch.testing.assert_close(design_rng, trainer.design_distribution.generator.get_state(), rtol=0, atol=0)
        self.assertEqual(len(replay), len(trainer.agent.replay.storage))
        for old, current in zip(replay, trainer.agent.replay.storage):
            for left, right in zip(old, current):
                np.testing.assert_array_equal(left, right)
        for module, state in zip(modules, states):
            for name, value in module.state_dict().items():
                torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        rows = pd.DataFrame(trainer.design_scores[:4])
        self.assertEqual(set(rows.split), {"train"})
        self.assertTrue((rows.action_noise_std == 0).all())
        seeds = rows.groupby("design_index").scenario_seed.apply(list)
        self.assertEqual(seeds.iloc[0], seeds.iloc[1])
        training_seeds = {trainer._episode_seed(i, s, "train") for i in range(2) for s in range(2)}
        self.assertTrue(training_seeds.isdisjoint(rows.scenario_seed))

    def test_frozen_scoring_updates_design_from_logged_clean_returns_after_warmup(self):
        trainer = self.trainer()
        trainer.rl_config["training"].update(
            design_scoring="frozen_actor", iterations=3, design_warmup_fraction=1 / 3, fixed_design_fraction=1 / 3,
        )
        with TemporaryDirectory() as folder:
            path = Path(folder)
            with patch.object(trainer, "_score_designs", wraps=trainer._score_designs) as scoring:
                with patch.object(trainer, "_update_design", wraps=trainer._update_design) as update:
                    history = trainer.train(output_directory=path, save_every=1, validation_every=1)
            self.assertEqual(scoring.call_count, 1)
            rows = pd.read_csv(path / "design_scores.csv")
            self.assertEqual(len(rows), 4)
            self.assertEqual(set(rows.iteration), {2})
            expected = rows.groupby("design_index").reward_chf.mean().to_numpy()
            np.testing.assert_allclose(update.call_args_list[1].args[1], expected)
            self.assertEqual(history.design_scoring_episodes.tolist(), [0, 4, 0])
            self.assertAlmostEqual(history.iloc[1].design_score_mean_cost_chf, -expected.mean())
            self.assertEqual(len(trainer.agent.replay.storage), 3 * 4 * 24)
            trained = pd.DataFrame(trainer.training_samples)
            for index, group in rows.groupby("design_index"):
                original = trained[(trained.iteration == 2) & (trained.design_index == index)].iloc[0]
                np.testing.assert_allclose(group.pv_capacity_kwp, original.pv_capacity_kwp)
                np.testing.assert_allclose(group.battery_capacity_kwh, original.battery_capacity_kwh)
            saved = json.loads((path / "configuration.json").read_text())
            self.assertEqual(saved["rl"]["training"]["design_scoring"], "frozen_actor")

    def test_default_and_old_config_do_not_score_designs(self):
        trainer = self.trainer()
        trainer.rl_config["training"].pop("design_scoring")
        with patch.object(trainer, "_score_designs", side_effect=AssertionError("default must not score")):
            trainer.train(validation_every=2)
        self.assertEqual(trainer.design_scores, [])

    def test_frozen_scoring_restores_actor_mode_on_failure(self):
        trainer = self.trainer()
        batch = trainer.design_distribution.sample(2)
        with patch.object(trainer, "_run_episode", side_effect=RuntimeError("scoring failed")):
            with self.assertRaisesRegex(RuntimeError, "scoring failed"):
                trainer._score_designs(batch, 0)
        self.assertTrue(trainer.agent.actor.training)

    def test_frozen_scoring_rejects_ppo_and_invalid_modes(self):
        config = get_rl_config("ppo_gmm")
        config["training"]["design_scoring"] = "frozen_actor"
        with self.assertRaisesRegex(ValueError, "DDPG experiment"):
            validate_rl_config(config)
        config = get_rl_config("ddpg_lognormal_improved")
        config["training"]["design_scoring"] = "unknown"
        with self.assertRaisesRegex(ValueError, "design_scoring"):
            validate_rl_config(config)

    def test_experiment_entry_point_records_scoring_protocol(self):
        with TemporaryDirectory() as folder:
            trainer, history, _ = run_experiment(
                "ddpg_lognormal_improved", seed=43, data_seed=42, episode_hours=24,
                iterations=1, episodes_per_iteration=4, validation_episodes=2,
                output_directory=Path(folder), save_every=1, design_scoring="frozen_actor",
            )
            saved = json.loads((Path(folder) / "configuration.json").read_text())
            self.assertEqual(saved["rl"]["training"]["design_scoring"], "frozen_actor")
            self.assertFalse(saved["rl"]["provenance"]["paper_faithful"])
            self.assertIn("fresh common training scenarios", saved["rl"]["provenance"]["design_scoring"])
            self.assertEqual(history.iloc[0].design_scoring_episodes, 4)
            self.assertEqual(len(trainer.design_scores), 4)


if __name__ == "__main__":
    unittest.main()

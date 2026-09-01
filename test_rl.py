import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).with_name("stochastic-model")))

from config import get_default_config
from data import generate_dataset
from design_distributions import create_design_distribution
from rl_algorithms import StateEncoder, create_agent
from rl_config import get_rl_config
from train_rl import CoDesignTrainer, build_system_config


class RLConfigTests(unittest.TestCase):
    def test_model_variants_select_the_intended_algorithm_and_distribution(self):
        expected = {
            "ppo_gmm": ("ppo", "gaussian"),
            "ddpg_lognormal": ("ddpg", "lognormal"),
            "ppo_lognormal": ("ppo", "lognormal"),
        }

        for name, pair in expected.items():
            config = get_rl_config(name)
            self.assertEqual((config["algorithm"]["name"], config["design_distribution"]["name"]), pair)

    def test_returned_configuration_is_independent(self):
        first = get_rl_config("ppo_gmm")
        first["training"]["iterations"] = 99

        second = get_rl_config("ppo_gmm")
        self.assertEqual(second["training"]["iterations"], 500)

    def test_only_the_schaff_based_ppo_variants_decay_learning_rates(self):
        self.assertTrue(get_rl_config("ppo_gmm")["training"]["linear_learning_rate_decay"])
        self.assertTrue(get_rl_config("ppo_lognormal")["training"]["linear_learning_rate_decay"])
        self.assertFalse(get_rl_config("ddpg_lognormal")["training"]["linear_learning_rate_decay"])


class DesignDistributionTests(unittest.TestCase):
    def test_gaussian_mixture_samples_feasible_designs_and_score_gradients(self):
        distribution = create_design_distribution(get_rl_config("ppo_gmm"), seed=7)
        batch = distribution.sample(64)

        self.assertEqual(tuple(batch["designs"].shape), (64, 2))
        self.assertTrue(torch.all(batch["designs"] >= 0.0))
        self.assertTrue(torch.isfinite(batch["log_prob"]).all())

        loss = -(batch["log_prob"] * torch.linspace(-1.0, 1.0, 64)).mean()
        loss.backward()
        self.assertGreater(distribution.component_means.grad.abs().sum().item(), 0.0)

    def test_lognormal_mixture_is_positive_and_has_expected_weight_policy(self):
        ddpg_distribution = create_design_distribution(get_rl_config("ddpg_lognormal"), seed=8)
        ppo_distribution = create_design_distribution(get_rl_config("ppo_lognormal"), seed=8)

        self.assertTrue(torch.all(ddpg_distribution.sample(64)["designs"] > 0.0))
        self.assertTrue(ddpg_distribution.mixture_logits.requires_grad)
        self.assertFalse(ppo_distribution.mixture_logits.requires_grad)

    def test_pruning_keeps_the_higher_scoring_half(self):
        distribution = create_design_distribution(get_rl_config("ppo_gmm"), seed=9)
        distribution.prune(torch.arange(8, dtype=torch.float32))

        self.assertEqual(distribution.active_components, 4)
        self.assertEqual(
            distribution.active_mask.nonzero(as_tuple=False).flatten().tolist(),
            [4, 5, 6, 7],
        )

    def test_a_specific_component_can_be_sampled_for_pruning_evaluation(self):
        distribution = create_design_distribution(get_rl_config("ppo_gmm"), seed=10)
        batch = distribution.sample_component(component=3, batch_size=5)

        self.assertEqual(tuple(batch["designs"].shape), (5, 2))
        self.assertEqual(batch["components"].tolist(), [3, 3, 3, 3, 3])

    def test_evaluation_sampling_does_not_change_the_training_sample_stream(self):
        first = create_design_distribution(get_rl_config("ppo_gmm"), seed=11)
        second = create_design_distribution(get_rl_config("ppo_gmm"), seed=11)
        evaluation_generator = torch.Generator().manual_seed(99)

        first.sample(5, generator=evaluation_generator)
        first_training_batch = first.sample(5)
        second_training_batch = second.sample(5)

        torch.testing.assert_close(first_training_batch["designs"], second_training_batch["designs"])


class RLAlgorithmTests(unittest.TestCase):
    def test_state_encoder_maps_the_paper_state_and_design_to_fixed_scales(self):
        encoder = StateEncoder(get_default_config(), get_rl_config("ppo_gmm"))
        state = {
            "timestamp": pd.Timestamp("2026-01-28 12:00:00"),
            "hour": 12,
            "battery_soc_kwh": 4.0,
            "pv_power_kw": 3.0,
            "load_kw": 2.0,
            "import_price": 0.5,
            "export_price": 0.0,
            "ev_present": True,
            "ev_soc_kwh": 40.0,
        }

        encoded = encoder.encode(state, np.array([6.0, 8.0]))
        expected = np.array(
            [12 / 23, 27 / 364, 0.5, 0.25, 0.2, 0.5, 0.0, 1.0, 0.5, 0.5, 0.4],
            dtype=np.float32,
        )
        np.testing.assert_allclose(encoded, expected, rtol=0.0, atol=1e-6)

    def test_ppo_returns_bounded_actions_and_updates_from_a_rollout(self):
        config = get_rl_config("ppo_gmm")
        config["algorithm"]["hidden_sizes"] = (16,)
        config["algorithm"]["minibatch_size"] = 4
        config["algorithm"]["epochs"] = 2
        agent = create_agent(config, seed=10)

        observations = np.zeros((8, 11), dtype=np.float32)
        decisions = [agent.act(observation) for observation in observations]
        actions = np.stack([decision["action"] for decision in decisions])
        self.assertTrue(np.all(actions >= -1.0))
        self.assertTrue(np.all(actions <= 1.0))

        before = [parameter.detach().clone() for parameter in agent.network.parameters()]
        metrics = agent.update({
            "observations": observations,
            "latent_actions": np.stack([decision["latent_action"] for decision in decisions]),
            "log_probs": np.array([decision["log_prob"] for decision in decisions], dtype=np.float32),
            "values": np.array([decision["value"] for decision in decisions], dtype=np.float32),
            "rewards": np.linspace(-1.0, 1.0, 8, dtype=np.float32),
            "dones": np.array([False] * 7 + [True]),
            "next_values": np.zeros(8, dtype=np.float32),
        })

        self.assertTrue(np.isfinite(metrics["policy_loss"]))
        self.assertTrue(any(not torch.equal(old, new.detach()) for old, new in zip(before, agent.network.parameters())))

    def test_ddpg_returns_bounded_actions_and_learns_from_replay(self):
        config = get_rl_config("ddpg_lognormal")
        config["algorithm"]["hidden_sizes"] = (16,)
        config["algorithm"]["minibatch_size"] = 4
        config["algorithm"]["learning_starts"] = 4
        agent = create_agent(config, seed=11)

        observation = np.zeros(11, dtype=np.float32)
        action = agent.act(observation, explore=True)
        self.assertTrue(np.all(action >= -1.0))
        self.assertTrue(np.all(action <= 1.0))

        for index in range(4):
            agent.observe(
                observation=np.full(11, index / 10, dtype=np.float32),
                action=np.array([0.25, -0.25], dtype=np.float32),
                reward=float(index),
                next_observation=np.full(11, (index + 1) / 10, dtype=np.float32),
                done=index == 3,
            )

        before = [parameter.detach().clone() for parameter in agent.critic.parameters()]
        metrics = agent.update()
        self.assertIsNotNone(metrics)
        self.assertTrue(np.isfinite(metrics["critic_loss"]))
        self.assertTrue(any(not torch.equal(old, new.detach()) for old, new in zip(before, agent.critic.parameters())))


class RLTrainingTests(unittest.TestCase):
    def test_design_is_applied_to_a_copy_of_the_stochastic_model_config(self):
        baseline = get_default_config()
        configured = build_system_config(baseline, np.array([4.5, 7.5]))

        self.assertEqual(configured["design"]["pv_capacity_kwp"], 4.5)
        self.assertEqual(configured["design"]["battery_capacity_kwh"], 7.5)
        self.assertEqual(baseline["design"]["pv_capacity_kwp"], 6.0)
        self.assertEqual(baseline["design"]["battery_capacity_kwh"], 8.0)

    def test_all_variants_train_and_validate_on_the_existing_stochastic_model(self):
        dataset = generate_dataset(year=2026, days=365, seed=21)

        for offset, model_name in enumerate(("ppo_gmm", "ddpg_lognormal", "ppo_lognormal")):
            with self.subTest(model=model_name):
                system_config = get_default_config()
                system_config["simulation"]["episode_hours"] = 6
                rl_config = get_rl_config(model_name)
                rl_config["training"]["iterations"] = 1
                rl_config["training"]["episodes_per_iteration"] = 2
                rl_config["training"]["validation_episodes"] = 1
                rl_config["algorithm"]["hidden_sizes"] = (8,)
                if rl_config["algorithm"]["name"] == "ppo":
                    rl_config["algorithm"]["epochs"] = 1
                    rl_config["algorithm"]["minibatch_size"] = 4
                else:
                    rl_config["algorithm"]["minibatch_size"] = 4
                    rl_config["algorithm"]["learning_starts"] = 4

                trainer = CoDesignTrainer(system_config, dataset, rl_config, seed=30 + offset)
                training = trainer.train()
                validation = trainer.evaluate(episodes=1)

                self.assertEqual(len(training), 1)
                self.assertEqual(training.iloc[0]["model"], model_name)
                samples = pd.DataFrame(trainer.training_samples)
                self.assertEqual(len(samples), 2)
                self.assertEqual(samples["scenario_seed"].nunique(), 2)
                np.testing.assert_allclose(
                    samples[
                        ["reward_chf", "cost_chf", "pv_capacity_kwp", "battery_capacity_kwh"]
                    ].mean().to_numpy(dtype=float),
                    training.iloc[0][
                        [
                            "mean_reward_chf",
                            "mean_cost_chf",
                            "mean_pv_capacity_kwp",
                            "mean_battery_capacity_kwh",
                        ]
                    ].to_numpy(dtype=float),
                )
                distribution_history = pd.DataFrame(trainer.distribution_history)
                expected_rows = 2 * rl_config["design_distribution"]["components"]
                self.assertEqual(len(distribution_history), expected_rows)
                self.assertEqual(sorted(distribution_history["iteration"].unique().tolist()), [0, 1])
                weight_sums = distribution_history.groupby("iteration")["weight"].sum()
                np.testing.assert_allclose(weight_sums, 1.0, atol=1e-6)
                self.assertTrue(np.isfinite(training.iloc[0]["mean_reward_chf"]))
                self.assertLess(training.iloc[0]["max_power_balance_error_kw"], 1e-9)
                self.assertEqual(validation["split"], "validation")
                self.assertTrue(np.isfinite(validation["mean_reward_chf"]))
                self.assertLess(validation["max_power_balance_error_kw"], 1e-9)


if __name__ == "__main__":
    unittest.main()

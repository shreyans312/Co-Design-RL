import unittest
import hashlib
from unittest.mock import patch
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).with_name("stochastic-model")))

from config import get_default_config
from data import generate_dataset
from design_distributions import create_design_distribution
from rl_algorithms import StateEncoder, create_agent
from rl_config import get_rl_config
from stochastic import generate_episode_scenario
from train_rl import CAUZ_DATASET_PATH, CoDesignTrainer, _design_returns, _write_results, build_system_config, load_experiment_dataset


class RLConfigTests(unittest.TestCase):
    def test_ddpg_cuda_request_fails_clearly_without_cuda(self):
        with patch("rl_algorithms.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
                create_agent(get_rl_config("ddpg_lognormal"), seed=11, device="cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_ddpg_cuda_actor_and_critic_update(self):
        config = get_rl_config("ddpg_lognormal")
        config["algorithm"].update(hidden_sizes=(16,), minibatch_size=4, learning_starts=4)
        agent = create_agent(config, seed=11, device="cuda")
        self.assertEqual(agent.actor.network[0].weight.device.type, "cuda")
        for index in range(4):
            observation = np.full(11, index / 10, dtype=np.float32)
            agent.observe(observation, agent.act(observation), -1.0, observation + 0.01, False)
        self.assertTrue(np.isfinite(agent.update()["actor_loss"]))

    def test_cauz_dataset_is_loaded_directly_and_pinned_by_hash(self):
        digest = hashlib.sha256(CAUZ_DATASET_PATH.read_bytes()).hexdigest()
        data = load_experiment_dataset({"dataset": "cauz", "dataset_sha256": digest})
        self.assertEqual(len(data), 8760)
        self.assertEqual(data["timestamp"].iloc[0].year, 2021)
        self.assertEqual(int(data["is_validation"].sum()), 672)
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_experiment_dataset({"dataset": "cauz", "dataset_sha256": "0" * 64})

    def test_model_variants_select_the_intended_algorithm_and_distribution(self):
        expected = {
            "ppo_gmm": ("ppo", "gaussian"),
            "ddpg_lognormal": ("ddpg", "lognormal"),
            "ddpg_lognormal_improved": ("ddpg", "lognormal"),
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
        self.assertFalse(
            get_rl_config("ddpg_lognormal_improved")["training"]["linear_learning_rate_decay"]
        )

    def test_ddpg_uses_the_imported_experimental_parameter_range(self):
        config = get_rl_config("ddpg_lognormal")
        distribution = config["design_distribution"]

        self.assertEqual(distribution["initial_parameter_low"], (0.1823, 0.6931))
        self.assertEqual(distribution["initial_parameter_high"], (2.3795, 2.8904))
        self.assertEqual(distribution["initial_std"], 3.0)
        self.assertEqual(distribution["learning_rate"], 5e-3)
        self.assertFalse(config["training"]["normalize_design_returns"])
        self.assertEqual(config["training"]["design_return_baseline"], "leave_one_out")

    def test_leave_one_out_design_returns_exclude_each_own_return(self):
        training = get_rl_config("ddpg_lognormal")["training"]
        actual = _design_returns(np.array([1.0, 2.0, 4.0]), training)

        torch.testing.assert_close(actual, torch.tensor([-2.0, -0.5, 2.5]))

    def test_improved_ddpg_uses_repeated_scenarios_and_bounded_designs(self):
        config = get_rl_config("ddpg_lognormal_improved")

        self.assertEqual(config["training"]["scenarios_per_design"], 4)
        self.assertTrue(config["training"]["normalize_design_returns"])
        self.assertEqual(config["design_distribution"]["maximum"], (12.0, 20.0))
        self.assertTrue(config["design_distribution"]["stratified_initialization"])
        self.assertTrue(config["design_distribution"]["stratified_sampling"])


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

        batch = ddpg_distribution.sample(64)
        centered_returns = torch.linspace(-1.0, 1.0, 64)
        loss = -(batch["log_prob"] * centered_returns).mean()
        loss.backward()
        self.assertGreater(ddpg_distribution.component_means.grad.abs().sum().item(), 0.0)
        self.assertGreater(ddpg_distribution.component_log_stds.grad.abs().sum().item(), 0.0)
        self.assertGreater(ddpg_distribution.mixture_logits.grad.abs().sum().item(), 0.0)

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

    def test_ddpg_entropy_bonus_increases_distribution_spread(self):
        data = generate_dataset(year=2026, days=365, seed=12)
        trainer = CoDesignTrainer(
            get_default_config(), data, get_rl_config("ddpg_lognormal"), seed=13
        )
        batch = trainer.design_distribution.sample(8192)
        before = trainer.design_distribution.component_log_stds.detach().mean()

        trainer._update_design(batch, np.zeros(8192), iteration=0)

        after = trainer.design_distribution.component_log_stds.detach().mean()
        self.assertGreater(after, before)

    def test_improved_ddpg_spreads_centres_and_preserves_component_exploration(self):
        config = get_rl_config("ddpg_lognormal_improved")
        distribution = create_design_distribution(config, seed=14)
        parameters = distribution.component_parameters()
        medians = np.exp(parameters["locations"])

        for dimension, (low, high) in enumerate(((1.2, 10.8), (2.0, 18.0))):
            strata = np.floor(
                config["design_distribution"]["components"]
                * (medians[:, dimension] - low)
                / (high - low)
            ).astype(int)
            self.assertEqual(sorted(strata.tolist()), list(range(4)))

        with torch.no_grad():
            distribution.mixture_logits.copy_(torch.tensor([20.0, -20.0, -20.0, -20.0]))
        self.assertTrue((distribution.component_parameters()["weights"] >= 0.05 - 1e-6).all())
        designs = distribution.sample(128)["designs"]
        self.assertTrue(torch.all(designs[:, 0] <= 12.0))
        self.assertTrue(torch.all(designs[:, 1] <= 20.0))


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

    def test_ddpg_returns_bounded_actions_and_updates_actor_and_critic(self):
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
                terminated=index == 3,
            )

        actor_before = [parameter.detach().clone() for parameter in agent.actor.parameters()]
        critic_before = [parameter.detach().clone() for parameter in agent.critic.parameters()]
        metrics = agent.update()
        self.assertIsNotNone(metrics)
        self.assertTrue(np.isfinite(metrics["critic_loss"]))
        self.assertTrue(
            any(not torch.equal(old, new.detach()) for old, new in zip(actor_before, agent.actor.parameters()))
        )
        self.assertTrue(
            any(not torch.equal(old, new.detach()) for old, new in zip(critic_before, agent.critic.parameters()))
        )


class RLTrainingTests(unittest.TestCase):
    def test_bootstrap_row_preserves_the_original_episode(self):
        data = generate_dataset(year=2026, days=365, seed=21)
        config = get_default_config()
        regular = generate_episode_scenario(data, config, split="all", seed=22)
        bootstrapped = generate_episode_scenario(
            data,
            config,
            split="all",
            seed=22,
            bootstrap_hours=1,
        )

        pd.testing.assert_frame_equal(regular["episode_data"], bootstrapped["episode_data"])
        pd.testing.assert_frame_equal(regular["ev_schedule"], bootstrapped["ev_schedule"])
        self.assertEqual(len(bootstrapped["bootstrap_data"]), 1)
        self.assertEqual(
            bootstrapped["bootstrap_data"].iloc[0]["timestamp"],
            regular["episode_data"].iloc[-1]["timestamp"] + pd.Timedelta(hours=1),
        )

    def test_design_is_applied_to_a_copy_of_the_stochastic_model_config(self):
        baseline = get_default_config()
        configured = build_system_config(baseline, np.array([4.5, 7.5]))

        self.assertEqual(configured["design"]["pv_capacity_kwp"], 4.5)
        self.assertEqual(configured["design"]["battery_capacity_kwh"], 7.5)
        self.assertEqual(baseline["design"]["pv_capacity_kwp"], 6.0)
        self.assertEqual(baseline["design"]["battery_capacity_kwh"], 8.0)

    def test_fixed_design_is_used_for_training_and_evaluation(self):
        data = generate_dataset(year=2026, days=365, seed=20)
        system_config = get_default_config()
        system_config["simulation"]["episode_hours"] = 2
        rl_config = get_rl_config("ppo_gmm")
        rl_config["training"].update(
            iterations=1,
            episodes_per_iteration=2,
            validation_episodes=1,
            design_warmup_fraction=0.0,
            fixed_design_fraction=1.0,
            prune_interval_fraction=0.0,
            fixed_design_values=[7.5, 6.5],
        )
        rl_config["algorithm"].update(hidden_sizes=(8,), epochs=1, minibatch_size=2)
        trainer = CoDesignTrainer(system_config, data, rl_config, seed=21)

        trainer.train()
        validation = trainer.evaluate(episodes=1, sample_designs=False)

        samples = pd.DataFrame(trainer.training_samples)
        self.assertTrue((samples["pv_capacity_kwp"] == 7.5).all())
        self.assertTrue((samples["battery_capacity_kwh"] == 6.5).all())
        self.assertEqual(validation["mean_pv_capacity_kwp"], 7.5)
        self.assertEqual(validation["mean_battery_capacity_kwh"], 6.5)

    def test_validation_can_run_less_often_than_training(self):
        data = generate_dataset(year=2026, days=365, seed=22)
        system_config = get_default_config()
        system_config["simulation"]["episode_hours"] = 2
        rl_config = get_rl_config("ppo_gmm")
        rl_config["training"].update(
            iterations=2,
            episodes_per_iteration=2,
            validation_episodes=1,
            fixed_design_fraction=1.0,
        )
        rl_config["algorithm"].update(hidden_sizes=(8,), epochs=1, minibatch_size=2)
        trainer = CoDesignTrainer(system_config, data, rl_config, seed=23)

        history = trainer.train(validation_every=2)

        self.assertTrue(np.isnan(history.iloc[0]["validation_mean_cost_chf"]))
        self.assertTrue(np.isfinite(history.iloc[1]["validation_mean_cost_chf"]))

    def test_all_variants_train_and_validate_on_the_existing_stochastic_model(self):
        dataset = generate_dataset(year=2026, days=365, seed=21)

        model_names = (
            "ppo_gmm",
            "ddpg_lognormal",
            "ddpg_lognormal_improved",
            "ppo_lognormal",
        )
        for offset, model_name in enumerate(model_names):
            with self.subTest(model=model_name):
                system_config = get_default_config()
                system_config["simulation"]["episode_hours"] = 6
                rl_config = get_rl_config(model_name)
                rl_config["training"]["iterations"] = 1
                rl_config["training"]["episodes_per_iteration"] = 4
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
                self.assertEqual(len(samples), 4)
                self.assertEqual(samples["scenario_seed"].nunique(), 4)
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

    def test_improved_ddpg_compares_each_design_on_common_scenarios(self):
        data = generate_dataset(year=2026, days=365, seed=29)
        system_config = get_default_config()
        system_config["simulation"]["episode_hours"] = 2
        rl_config = get_rl_config("ddpg_lognormal_improved")
        rl_config["training"].update(iterations=1, episodes_per_iteration=8, validation_episodes=1)
        rl_config["algorithm"].update(hidden_sizes=(8,), minibatch_size=4, learning_starts=4)
        trainer = CoDesignTrainer(system_config, data, rl_config, seed=30)

        trainer.train()

        samples = pd.DataFrame(trainer.training_samples)
        self.assertEqual(samples["design_index"].nunique(), 2)
        self.assertTrue((samples.groupby("design_index").size() == 4).all())
        self.assertEqual(samples["scenario_seed"].nunique(), 4)
        self.assertTrue((samples.groupby("scenario_seed").size() == 2).all())

    def test_ddpg_uses_real_bootstrap_state_and_saves_progress(self):
        data = generate_dataset(year=2026, days=365, seed=31)
        system_config = get_default_config()
        system_config["simulation"]["episode_hours"] = 24
        rl_config = get_rl_config("ddpg_lognormal")
        rl_config["training"].update(iterations=1, episodes_per_iteration=2, validation_episodes=1)
        rl_config["algorithm"].update(hidden_sizes=(8,), minibatch_size=4, learning_starts=4)
        trainer = CoDesignTrainer(system_config, data, rl_config, seed=32)

        with TemporaryDirectory() as directory:
            history = trainer.train(
                output_directory=Path(directory), save_every=1, keep_checkpoints=True
            )
            validation = trainer.evaluate(episodes=1, sample_designs=True)
            _write_results(Path(directory), trainer, history, validation)
            self.assertTrue(
                {
                    "checkpoint.pt",
                    "configuration.json",
                    "design_summary.json",
                    "design_distribution.csv",
                    "final_design_samples.csv",
                    "latest_validation.json",
                    "training.csv",
                    "training_samples.csv",
                    "validation.json",
                }.issubset(path.name for path in Path(directory).iterdir())
            )
            self.assertEqual(
                len(pd.read_csv(Path(directory) / "final_design_samples.csv")), 1000
            )
            self.assertTrue((Path(directory) / "checkpoints" / "iteration_0001.pt").is_file())

        observation, _, _, next_observation, terminated = trainer.agent.replay.storage[23]
        self.assertFalse(terminated)
        self.assertFalse(np.array_equal(observation, next_observation))


if __name__ == "__main__":
    unittest.main()

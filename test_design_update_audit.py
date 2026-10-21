import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from audit_design_updates import expected_weight_gradient, perfect_signal_progress, probe_design_updates, sampling_metrics, score_gradients
from design_distributions import create_design_distribution
from rl_config import get_rl_config
from train_rl import CoDesignTrainer, generate_dataset, get_default_config


class DesignGradientAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_weight_gradient_matches_finite_difference_with_floor(self):
        distribution = create_design_distribution(get_rl_config("ddpg_lognormal_improved"), 43).double()
        returns = np.array([-10.0, -3.0, -6.0, -12.0])
        with torch.no_grad():
            distribution.mixture_logits.copy_(torch.tensor([0.1, -0.2, 0.5, 0.8]))
        analytic = expected_weight_gradient(distribution, returns)
        numerical = []
        delta = 1e-5
        for component in range(4):
            with torch.no_grad():
                distribution.mixture_logits[component] += delta
                high = float((distribution._log_weights().exp() * torch.tensor(returns)).sum())
                distribution.mixture_logits[component] -= 2 * delta
                low = float((distribution._log_weights().exp() * torch.tensor(returns)).sum())
                distribution.mixture_logits[component] += delta
            numerical.append((high - low) / (2 * delta))
        np.testing.assert_allclose(analytic, numerical, rtol=1e-6, atol=1e-8)

    def test_current_update_rewards_good_component_and_improves_gaussian_parameters(self):
        config = get_rl_config("ddpg_lognormal_improved")
        config["design_distribution"].update(name="gaussian", initial_std_fraction=0.02, entropy_weight=0.0)
        system = get_default_config()
        trainer = CoDesignTrainer(system, generate_dataset(year=2026, seed=42), config)
        distribution = trainer.design_distribution
        with torch.no_grad():
            distribution.component_means.copy_(torch.tensor([[1., 1.], [4., 4.], [7., 7.], [10., 10.]]))
            distribution.component_log_stds.fill_(np.log(0.25))
        batch = distribution.sample(65536)
        returns = -((batch["designs"].numpy() - 7.0) ** 2).sum(axis=1)
        logits, locations, scales = score_gradients(trainer, batch, returns, 100)
        self.assertGreater(logits[2], 0)
        self.assertLess(logits[0], 0)
        self.assertTrue(np.all(locations[[0, 1]] > 0))
        self.assertTrue(np.all(locations[3] < 0))
        before = distribution.component_parameters()["weights"].copy()
        trainer._update_design(batch, returns, 100)
        after = distribution.component_parameters()["weights"]
        self.assertGreater(after[2], before[2])
        self.assertLess(after[0], before[0])

    def test_quadratic_reward_contracts_gaussian_spread(self):
        config = get_rl_config("ddpg_lognormal_improved")
        config["design_distribution"].update(name="gaussian", initial_std_fraction=0.02)
        distribution = create_design_distribution(config, 43).double()
        with torch.no_grad():
            distribution.component_means.copy_(torch.tensor([[1., 1.], [4., 4.], [7., 7.], [10., 10.]]))
            distribution.component_log_stds.fill_(np.log(0.25))
        # Gaussian quadrature removes sample noise from this gradient check
        nodes, weights = np.polynomial.hermite.hermgauss(7)
        grid = np.array(np.meshgrid(nodes, nodes)).reshape(2, -1).T * np.sqrt(2)
        quadrature = np.outer(weights, weights).reshape(-1) / np.pi
        samples = torch.cat([
            distribution.component_means[k].detach() + torch.tensor(grid) * 0.25 for k in range(4)
        ])
        probabilities = torch.tensor(np.tile(quadrature / 4, 4))
        rewards = -(samples - 7).square().sum(dim=1)
        log_prob = torch.logsumexp(distribution._component_log_prob(samples) + distribution._log_weights(), dim=1)
        gradient = torch.autograd.grad((probabilities * rewards * log_prob).sum(), distribution.component_log_stds)[0]
        expected = torch.full_like(gradient, -2 * 0.25 ** 2 / 4)
        torch.testing.assert_close(gradient, expected, rtol=1e-5, atol=1e-8)

    def test_perfect_signal_changes_only_disposable_copy(self):
        distribution = create_design_distribution(get_rl_config("ddpg_lognormal_improved"), 43)
        original = distribution.mixture_logits.detach().clone()
        progress = perfect_signal_progress(distribution, 1e-3, 450, 2)
        self.assertGreater(progress[-1]["preferred_weight"], progress[0]["preferred_weight"])
        self.assertLess(progress[-1]["preferred_weight"], 0.85)
        torch.testing.assert_close(distribution.mixture_logits, original)

    def test_both_samplers_follow_weights_with_different_count_variance(self):
        distribution = create_design_distribution(get_rl_config("ddpg_lognormal_improved"), 43)
        with torch.no_grad():
            distribution.mixture_logits.copy_(torch.tensor([0.5, 0.1, -0.1, -0.2]))
        expected = 8 * distribution.component_parameters()["weights"]
        counts = {}
        for stratified in [True, False]:
            distribution.stratified_sampling = stratified
            generator = torch.Generator().manual_seed(901)
            counts[stratified] = np.array([
                torch.bincount(distribution.sample(8, generator)["components"], minlength=4).numpy()
                for _ in range(1024)
            ])
            np.testing.assert_allclose(counts[stratified].mean(axis=0), expected, atol=0.15, rtol=0)
        self.assertTrue(np.all(counts[True] > 0))
        self.assertTrue(np.any(counts[False] == 0))
        self.assertLess(counts[True].var(axis=0).sum(), counts[False].var(axis=0).sum())

    def test_sampling_probes_restore_state_and_repeat_exactly(self):
        config = get_rl_config("ddpg_lognormal_improved")
        trainer = CoDesignTrainer(get_default_config(), generate_dataset(year=2026, seed=42), config)
        distribution = trainer.design_distribution
        batch = distribution.sample(8)
        trainer._update_design(batch, -batch["designs"].numpy().sum(axis=1), 100)
        initial = deepcopy(distribution.state_dict())
        optimizer = deepcopy(trainer.design_optimizer.state_dict())
        generator = torch.Generator().manual_seed(801)
        rng_state = generator.get_state().clone()

        def evaluate(_, batch, scenario_seed, scenarios):
            returns = -((batch["designs"].numpy() - 7) ** 2).sum(axis=1)
            return returns, pd.DataFrame({"cost_chf": -returns, "scenario_seed": scenario_seed})

        with patch("audit_design_updates.frozen_returns", side_effect=evaluate):
            first, _ = probe_design_updates(trainer, 4, 1, 40000000, generator, 100, True)
            second, _ = probe_design_updates(trainer, 4, 1, 40000000, generator, 100, True)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(set(first.sampler), {"stratified", "weighted_random"})
        self.assertTrue((first.groupby(["sampler", "batch"]).sample_count.sum() == 8).all())
        self.assertTrue(distribution.stratified_sampling)
        torch.testing.assert_close(generator.get_state(), rng_state)
        for name, value in distribution.state_dict().items():
            torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
        after = trainer.design_optimizer.state_dict()
        self.assertEqual(after["param_groups"], optimizer["param_groups"])
        for parameter, values in optimizer["state"].items():
            for name, value in values.items():
                torch.testing.assert_close(after["state"][parameter][name], value, rtol=0, atol=0)

    def test_failed_sampling_probe_restores_sampler(self):
        config = get_rl_config("ddpg_lognormal_improved")
        config["design_distribution"]["stratified_sampling"] = False
        trainer = CoDesignTrainer(get_default_config(), generate_dataset(year=2026, seed=42), config)
        with patch("audit_design_updates.frozen_returns", side_effect=RuntimeError("evaluation failed")):
            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                probe_design_updates(trainer, 2, 1, 40000000, torch.Generator(), 100, True)
        self.assertFalse(trainer.design_distribution.stratified_sampling)

    def test_sampling_metrics_use_weights_and_direction_not_cost_level(self):
        rows = []
        for batch in range(2):
            for component in range(2):
                rows.append({
                    "sampler": "stratified", "batch": batch, "component": component,
                    "sample_count": 4, "logit_ascent_gradient": [-1., 1.][component],
                    "adam_weight_change": [-0.01, 0.01][component],
                })
        metrics, summary = sampling_metrics(pd.DataFrame(rows), np.array([-10., -5.]), np.array([-1., 1.]))
        np.testing.assert_allclose(metrics.gradient_reference_cosine, 1)
        np.testing.assert_allclose(metrics.weight_only_predicted_saving_chf, 0.05)
        self.assertEqual(summary.iloc[0].all_components_present_fraction, 1)
        self.assertEqual(summary.iloc[0].logit_gradient_variance_trace, 0)


if __name__ == "__main__":
    unittest.main()

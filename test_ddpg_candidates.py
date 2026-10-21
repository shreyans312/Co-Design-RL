import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from evaluate_ddpg_candidates import component_candidates, evaluate_design, main
from rl_config import get_rl_config
from train_rl import CoDesignTrainer, generate_dataset, get_default_config


class CandidateEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.data = generate_dataset(year=2026, seed=42)

    def trainer(self):
        system = get_default_config()
        system["simulation"]["episode_hours"] = 24
        config = get_rl_config("ddpg_lognormal_improved")
        config["algorithm"]["hidden_sizes"] = (16, 16)
        return CoDesignTrainer(system, self.data, config)

    def test_candidates_match_bounded_component_modes(self):
        trainer = self.trainer()
        candidates = component_candidates(trainer)
        self.assertEqual(len(candidates), 4)
        self.assertTrue(any(np.array_equal(design, trainer._design_mode()) for design in candidates.values()))
        trainer.design_distribution.active_mask[0] = False
        self.assertNotIn("component_0", component_candidates(trainer))
        for design in candidates.values():
            self.assertTrue(np.all(design >= [0, 0]) and np.all(design <= [12, 20]))

    def test_evaluation_is_repeatable_and_does_not_train(self):
        trainer = self.trainer()
        design = trainer._design_mode()
        before = [p.detach().clone() for p in trainer.agent.actor.parameters()]
        first = evaluate_design(trainer, design, 2, 31_000_000)
        second = evaluate_design(trainer, design, 2, 31_000_000)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.scenario_seed.tolist(), [31_000_000, 31_000_001])
        self.assertEqual(len(trainer.agent.replay), 0)
        for old, new in zip(before, trainer.agent.actor.parameters()):
            torch.testing.assert_close(old, new)

    def test_cli_saves_separate_selection_and_assessment(self):
        trainer = self.trainer()
        with TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / "checkpoint.pt"
            output = root / "evaluation"
            trainer.save_checkpoint(checkpoint)
            arguments = ["evaluate_ddpg_candidates.py", "--checkpoint", str(checkpoint),
                         "--output-directory", str(output), "--validation-episodes", "2",
                         "--assessment-episodes", "2"]
            with patch("sys.argv", arguments):
                main()
            selection = json.loads((output / "selection.json").read_text())
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(selection["selected_candidate"], result["selected_candidate"])
            self.assertNotIn("selected_assessment_cost_chf", selection)
            ranking = pd.read_csv(output / "candidate_ranking.csv")
            self.assertEqual(ranking.iloc[0].candidate, result["selected_candidate"])
            selected = pd.read_csv(output / "selection_episodes.csv")
            assessed = pd.read_csv(output / "assessment_episodes.csv")
            self.assertFalse(set(selected.scenario_seed) & set(assessed.scenario_seed))
            self.assertEqual(len(assessed), 4)


if __name__ == "__main__":
    unittest.main()

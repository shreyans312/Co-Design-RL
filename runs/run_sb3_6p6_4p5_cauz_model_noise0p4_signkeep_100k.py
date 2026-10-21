"""6.6 kWp / 4.5 kWh, default hyperparameters (actor 1e-4, critic 1e-3, noise 0.4 with sign-preserving noise, batch 256),
100k steps on the fitted Cauz-like generator (--dataset cauz_model). TensorBoard disabled."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.argv = ["train_sb3_ddpg.py", "--pv-capacity-kwp", "6.6", "--battery-capacity-kwh", "4.5",
            "--dataset", "cauz_model", "--total-timesteps", "100000", "--exploration-noise-std", "0.4", "--sign-preserving-noise",
            "--output-dir", "runs/sb3_ddpg_6p6_4p5_cauz_model_noise0p4_signkeep_100k"]

import train_sb3_ddpg as t  # noqa: E402

_orig = t.DDPGSplitLR.__init__


def _init(self, *args, **kwargs):
    kwargs["tensorboard_log"] = None
    _orig(self, *args, **kwargs)


t.DDPGSplitLR.__init__ = _init
t.main()

"""6.6 kWp / 4.5 kWh, 2,688,000 timesteps, learning rates divided by 10 again vs lr_div10
(actor 1e-6, critic 1e-5). Exploration noise decays exponentially 0.4 -> 0.15 over the run.
TensorBoard disabled."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

TOTAL_TIMESTEPS = 2_688_000
SIGMA0, SIGMA_END = 0.4, 0.15
DECAY_RATE = float(np.log(SIGMA0 / SIGMA_END) / TOTAL_TIMESTEPS)  # sigma_t = SIGMA0*exp(-rate*t)

sys.argv = ["train_sb3_ddpg.py", "--pv-capacity-kwp", "6.6", "--battery-capacity-kwh", "4.5",
            "--actor-learning-rate", "1e-6", "--critic-learning-rate", "1e-5",
            "--total-timesteps", str(TOTAL_TIMESTEPS),
            "--exploration-noise-std", str(SIGMA0),
            "--output-dir", "runs/sb3_ddpg_6p6_4p5_lr_div100_noise0p4to0p15_2p688M"]

import train_sb3_ddpg as t  # noqa: E402
from noise_decay import NoiseDecay  # noqa: E402
from stable_baselines3.common.callbacks import CallbackList  # noqa: E402

_orig_init = t.DDPGSplitLR.__init__
_orig_learn = t.DDPGSplitLR.learn


def _init(self, *args, **kwargs):
    kwargs["tensorboard_log"] = None
    _orig_init(self, *args, **kwargs)


def _learn(self, *args, callback=None, **kwargs):
    callbacks = [NoiseDecay(SIGMA0, DECAY_RATE)] + ([callback] if callback is not None else [])
    return _orig_learn(self, *args, callback=CallbackList(callbacks), **kwargs)


t.DDPGSplitLR.__init__ = _init
t.DDPGSplitLR.learn = _learn
t.main()

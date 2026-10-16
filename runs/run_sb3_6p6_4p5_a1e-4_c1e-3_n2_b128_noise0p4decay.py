"""6.6 kWp / 4.5 kWh, actor lr 1e-4, critic lr 1e-3, 2-step TD, batch 128,
exploration noise 0.4 decaying as 0.4*exp(-1e-5*t). TensorBoard disabled."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

N_STEP, GAMMA = 2, 0.99
SIGMA0, DECAY_RATE = 0.4, 1e-5
sys.argv = ["train_sb3_ddpg.py", "--pv-capacity-kwp", "6.6", "--battery-capacity-kwh", "4.5",
            "--actor-learning-rate", "1e-4", "--critic-learning-rate", "1e-3",
            "--batch-size", "128",
            "--exploration-noise-std", str(SIGMA0),
            "--gamma", str(GAMMA ** N_STEP),  # bootstrap discount gamma^n
            "--output-dir", "runs/sb3_ddpg_6p6_4p5_a1e-4_c1e-3_n2_b128_noise0p4decay"]

import train_sb3_ddpg as t  # noqa: E402
from nstep_buffer import NStepReplayBuffer  # noqa: E402
from noise_decay import NoiseDecay  # noqa: E402
from stable_baselines3.common.callbacks import CallbackList  # noqa: E402

_orig_init = t.DDPGSplitLR.__init__
_orig_learn = t.DDPGSplitLR.learn


def _init(self, *args, **kwargs):
    kwargs["tensorboard_log"] = None
    kwargs["replay_buffer_class"] = NStepReplayBuffer
    kwargs["replay_buffer_kwargs"] = {"n_step": N_STEP, "gamma": GAMMA}
    _orig_init(self, *args, **kwargs)


def _learn(self, *args, callback=None, **kwargs):
    callbacks = [NoiseDecay(SIGMA0, DECAY_RATE)] + ([callback] if callback is not None else [])
    return _orig_learn(self, *args, callback=CallbackList(callbacks), **kwargs)


t.DDPGSplitLR.__init__ = _init
t.DDPGSplitLR.learn = _learn
t.main()

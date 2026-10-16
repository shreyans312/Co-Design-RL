"""6.6 kWp / 4.5 kWh, actor lr 1e-4, critic lr 1e-3, 2-step TD target, batch 128. TensorBoard disabled."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

N_STEP, GAMMA = 2, 0.99
sys.argv = ["train_sb3_ddpg.py", "--pv-capacity-kwp", "6.6", "--battery-capacity-kwh", "4.5",
            "--actor-learning-rate", "1e-4", "--critic-learning-rate", "1e-3",
            "--batch-size", "128",
            "--gamma", str(GAMMA ** N_STEP),  # bootstrap discount gamma^n
            "--output-dir", "runs/sb3_ddpg_6p6_4p5_a1e-4_c1e-3_n2_b128"]

import train_sb3_ddpg as t  # noqa: E402
from nstep_buffer import NStepReplayBuffer  # noqa: E402

_orig = t.DDPGSplitLR.__init__


def _init(self, *args, **kwargs):
    kwargs["tensorboard_log"] = None
    kwargs["replay_buffer_class"] = NStepReplayBuffer
    kwargs["replay_buffer_kwargs"] = {"n_step": N_STEP, "gamma": GAMMA}
    _orig(self, *args, **kwargs)


t.DDPGSplitLR.__init__ = _init
t.main()

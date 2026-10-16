"""6.6 kWp / 4.5 kWh, actor lr 1e-5 (lowered), critic lr 1e-3 (original). TensorBoard disabled."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.argv = ["train_sb3_ddpg.py", "--pv-capacity-kwp", "6.6", "--battery-capacity-kwh", "4.5",
            "--actor-learning-rate", "1e-5", "--critic-learning-rate", "1e-3",
            "--output-dir", "runs/sb3_ddpg_6p6_4p5_a1e-5_c1e-3"]

import train_sb3_ddpg as t  # noqa: E402

_orig = t.DDPGSplitLR.__init__


def _init(self, *args, **kwargs):
    kwargs["tensorboard_log"] = None
    _orig(self, *args, **kwargs)


t.DDPGSplitLR.__init__ = _init
t.main()

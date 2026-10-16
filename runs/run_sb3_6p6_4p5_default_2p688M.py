"""6.6 kWp / 4.5 kWh, default settings (actor 1e-4, critic 1e-3, noise 0.2, batch 256), 2,688,000 steps."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.argv = ["train_sb3_ddpg.py", "--pv-capacity-kwp", "6.6", "--battery-capacity-kwh", "4.5",
            "--output-dir", "runs/sb3_ddpg_6p6_4p5_default_2p688M", "--total-timesteps", "2688000"]

import train_sb3_ddpg as t  # noqa: E402

_orig = t.DDPGSplitLR.__init__


def _init(self, *args, **kwargs):
    kwargs["tensorboard_log"] = None
    _orig(self, *args, **kwargs)


t.DDPGSplitLR.__init__ = _init
t.main()

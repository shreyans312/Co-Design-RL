"""Run train_sb3_ddpg.main() with tensorboard logging disabled (tensorboard is not installed here).

Everything else (refined split LR 1e-4 / 1e-3, 100k steps, noise 0.2) is the script's defaults.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.argv = ["train_sb3_ddpg.py", "--pv-capacity-kwp", "6.6", "--battery-capacity-kwh", "4.5",
            "--output-dir", "runs/sb3_ddpg_6p6_4p5"]

import train_sb3_ddpg as t  # noqa: E402

_orig = t.DDPGSplitLR.__init__


def _init(self, *args, **kwargs):
    kwargs["tensorboard_log"] = None
    _orig(self, *args, **kwargs)


t.DDPGSplitLR.__init__ = _init
t.main()

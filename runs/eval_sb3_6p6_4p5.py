"""Deterministic 100-episode eval (seeds 0-99, dataset seed 21, train split) at 6.6 kWp / 4.5 kWh.

Same protocol as rule_based_baseline_check.py and eval_sb3_refined.py.
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "stochastic-model"))
sys.argv = ["x"]

from stable_baselines3 import DDPG  # noqa: E402

from config import get_default_config  # noqa: E402
from data import generate_dataset  # noqa: E402
from simulator import EnergySystemSimulator  # noqa: E402
from stochastic import generate_episode_scenario  # noqa: E402
from train_rl import build_system_config  # noqa: E402
from train_sb3_ddpg import EnergySystemEnv  # noqa: E402

PV, BAT = 6.6, 4.5


def sb3_eval(path):
    m = DDPG.load(path, device="cpu")
    env = EnergySystemEnv(PV, BAT, split="train", seed=21, episode_seeds=list(range(100)))
    rs = []
    for _ in range(100):
        o, _ = env.reset()
        done, tot = False, 0.0
        while not done:
            a, _ = m.predict(o, deterministic=True)
            o, r, te, tr, _ = env.step(a)
            tot += r
            done = te or tr
        rs.append(tot)
    return np.array(rs)


def rule_eval():
    dataset = generate_dataset(year=2026, days=365, seed=21)
    cfg = build_system_config(get_default_config(), np.array([PV, BAT]))
    rs = []
    for ep in range(100):
        sim = EnergySystemSimulator(cfg)
        state = sim.reset(generate_episode_scenario(data=dataset, config=cfg, split="train", seed=ep))
        while not sim.done:
            net = state["load_kw"] - state["pv_power_kw"]
            ev = float(np.clip(net / 5.0, -1.0, 1.0)) if state["ev_present"] else 0.0
            state, _, _, _ = sim.step(np.array([np.clip(net / 6.0, -1.0, 1.0), ev], dtype=np.float32))
        rs.append(sim.cumulative_reward)
    return np.array(rs)


out = {}
for name, fn in [("rule_based", rule_eval),
                 ("sb3_best", lambda: sb3_eval("runs/sb3_ddpg_6p6_4p5/best_model/best_model.zip")),
                 ("sb3_final", lambda: sb3_eval("runs/sb3_ddpg_6p6_4p5/final_model.zip"))]:
    r = fn()
    out[name] = r
    print(f"{name}: mean {r.mean():.2f} std {r.std():.2f} (n={len(r)})", flush=True)
np.savez("runs/sb3_ddpg_6p6_4p5/eval_100.npz", **out)

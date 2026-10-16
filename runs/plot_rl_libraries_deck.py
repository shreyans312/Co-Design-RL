import re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("runs"); O = R / "rl_libraries_deck/figs"
NAVY, TEAL, ORANGE, RED, GREY = "#1B3050", "#0A7E8C", "#D97706", "#BC5252", "#6B7385"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "axes.titleweight": "bold", "axes.titlecolor": NAVY})

def sb3_eval(d):
    ev = np.load(R / f"sb3_ddpg_{d}/eval_logs/evaluations.npz"); r = ev["results"]
    return ev["timesteps"], r.mean(1), r.std(1)

def sb3_losses(d):
    txt = (R / f"sb3_ddpg_{d}/train.log").read_text(errors="ignore")
    ts, al, cl = [], [], []
    for b in txt.split("-----------"):
        if "ep_rew_mean" in b and "actor_loss" in b:
            ts.append(int(re.search(r"total_timesteps\s*\|\s*(\d+)", b)[1]))
            al.append(float(re.search(r"actor_loss\s*\|\s*([-\d.e+]+)", b)[1]))
            cl.append(float(re.search(r"critic_loss\s*\|\s*([-\d.e+]+)", b)[1]))
    return np.array(ts), np.array(al), np.array(cl)

# A: SB3 eval reward
fig, ax = plt.subplots(figsize=(10, 4.6))
for d, lab, c in [("default_design", "SB3 DDPG, default (100k steps)", RED),
                  ("refined", "SB3 DDPG, refined (100k steps)", TEAL),
                  ("2p7M", "SB3 DDPG, refined, long run (475k steps)", NAVY)]:
    t, m, s = sb3_eval(d); ax.plot(t / 1e3, m, c=c, lw=2, marker="o", ms=3, label=lab)
    ax.fill_between(t / 1e3, m - s, m + s, color=c, alpha=.08)
ax.axhline(-105.7, c=GREY, ls="--", lw=1.2); ax.text(2, -104, "rule-based baseline -105.7", color=GREY, fontsize=10, va="bottom")
ax.set_xlabel("Training steps (thousands)"); ax.set_ylabel("Eval reward (CHF / episode-week)")
ax.set_title("SB3 DDPG: deterministic eval reward, design 6 kWp / 14 kWh", loc="left")
ax.set_ylim(-112, -55); ax.legend(loc="lower right", frameon=False); fig.tight_layout(); fig.savefig(O / "sb3_eval.png", dpi=170)

# B: SB3 losses
fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
for d, lab, c in [("default_design", "default", RED), ("refined", "refined", TEAL)]:
    t, al, cl = sb3_losses(d)
    ax[0].plot(t / 1e3, al, c=c, lw=2, label=lab); ax[1].plot(t / 1e3, cl, c=c, lw=1.3, label=lab)
ax[0].set_title("Actor loss", loc="left"); ax[1].set_title("Critic loss", loc="left")
for a in ax: a.set_xlabel("Training steps (thousands)"); a.legend(frameon=False)
fig.tight_layout(); fig.savefig(O / "sb3_losses.png", dpi=170)

# C: manual DDPG fixed design
def log(p):
    x = re.findall(r"iteration\s+(\d+)/500\s+mean_train_reward=([-\d.]+)", (R / p).read_text())
    return [int(a) for a, _ in x], [float(b) for _, b in x]
fig, ax = plt.subplots(figsize=(10, 4.6))
for p, lab, c in [("fixed_design_actor_check.log", "Baseline (battery 4.99)", GREY),
                  ("fixed_design_actor_check_battery4.5.log", "Battery 4.5, no PER", NAVY),
                  ("fixed_design_actor_check_battery4.5_per_warmstart.log", "Battery 4.5, PER + warm-start", ORANGE)]:
    i, r = log(p); ax.plot(i, r, c=c, lw=1.8, label=lab)
ax.set_xlabel("Training iteration (32 episodes each)"); ax.set_ylabel("Mean training reward (CHF / episode-week)")
ax.set_title("Manual DDPG, fixed design: training reward", loc="left"); ax.set_ylim(-98, -76)
ax.legend(loc="lower right", frameon=False); fig.tight_layout(); fig.savefig(O / "manual_fixed.png", dpi=170)

# D: co-design with libraries
fig, ax = plt.subplots(figsize=(10, 4.6))
for n, lab, c in [("ddpg_lognormal", "Manual DDPG (500 it.)", NAVY), ("sb3_ddpg_codesign_stable_check", "SB3 DDPG, stabilised (152 it.)", TEAL),
                  ("sb3_ddpg_codesign", "SB3 DDPG, first attempt (50 it.)", RED), ("rl4co_ppo_codesign", "RL4CO PPO (78 it.)", ORANGE)]:
    d = pd.read_csv(R / n / "training.csv"); v = d.filter(like="validation_mean").iloc[:, 0]
    ok = v.notna(); ax.plot(d.iteration[ok], v[ok], c=c, lw=1.6, label=lab)
ax.set_ylim(-420, -60); ax.set_xlabel("Training iteration"); ax.set_ylabel("Validation reward (CHF)")
ax.set_title("Co-design runs: validation reward", loc="left"); ax.legend(loc="lower right", frameon=False)
fig.tight_layout(); fig.savefig(O / "codesign.png", dpi=170)

# E: head-to-head at 6.6 / 4.5
fig, ax = plt.subplots(figsize=(6.2, 4.4))
names = ["Manual DDPG\n(train, noisy)", "SB3 DDPG\n(refined, eval)", "Custom PPO\n(validation)"]
vals = [91.9, 86.2, 88.46]; cols = [NAVY, TEAL, ORANGE]
b = ax.bar(names, vals, color=cols, width=.6)
for x, v in zip(b, vals): ax.text(x.get_x() + x.get_width() / 2, v + .4, f"{v:.1f}", ha="center", fontweight="bold")
ax.set_ylim(80, 95); ax.set_ylabel("Cost (CHF / episode-week), lower is better")
ax.set_title("Fixed design 6.6 kWp / 4.5 kWh", loc="left"); ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(O / "head2head.png", dpi=170)

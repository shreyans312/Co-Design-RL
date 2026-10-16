import re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = Path("runs/sb3_ddpg_refined")
ev = np.load(d / "eval_logs/evaluations.npz"); t, r = ev["timesteps"], ev["results"]
txt = (d / "train.log").read_text(errors="ignore")
al = [float(x) for x in re.findall(r"actor_loss\s*\|\s*([-\d.e+]+)", txt)]
cl = [float(x) for x in re.findall(r"critic_loss\s*\|\s*([-\d.e+]+)", txt)]
ts = [int(x) for x in re.findall(r"total_timesteps\s*\|\s*(\d+)", txt)][-len(al):]

c = pd.read_csv("runs/ddpg_lognormal_fixed/seed_20/ddpg_lognormal/training.csv")
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0,0].plot(t, r.mean(1), c="C1"); ax[0,0].fill_between(t, r.mean(1)-r.std(1), r.mean(1)+r.std(1), color="C1", alpha=.2)
ax[0,0].set_title("SB3 DDPG (refined): eval reward vs timesteps (fixed design)")
ax[0,1].plot(c.iteration, c.validation_mean_reward_chf, c="C0", label="validation"); ax[0,1].plot(c.iteration, c.mean_reward_chf, c="C0", alpha=.3, label="train")
ax[0,1].set_ylim(-400, 0); ax[0,1].legend(); ax[0,1].set_title("Custom DDPG (seed 20): reward vs iteration (co-optimised design)")
ax[1,0].plot(ts, al, c="C1", label="SB3"); ax[1,0].set_title("SB3 actor loss"); 
ax[1,1].plot(c.iteration, c.actor_loss, c="C0", label="custom actor"); ax[1,1].set_title("Custom actor loss")
for a in ax.flat: a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("runs/refined_sb3_vs_custom_ddpg.png", dpi=130)
print("custom last10 train reward", c.mean_reward_chf.tail(10).mean(), "val", c.validation_mean_reward_chf.tail(10).mean(), "best val", c.validation_mean_reward_chf.max(), c.validation_mean_reward_chf.idxmax()+1)
print("custom actor/critic loss last", c.actor_loss.iloc[-1], c.critic_loss.iloc[-1])
print("sb3 eval last", r.mean(1)[-1], r.std(1)[-1])

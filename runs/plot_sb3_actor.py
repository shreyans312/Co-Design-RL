import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs = {"default_design": Path("runs/sb3_ddpg_default_design"), "refined": Path("runs/sb3_ddpg_refined")}
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
for name, d in runs.items():
    ev = np.load(d / "eval_logs/evaluations.npz")
    t, r = ev["timesteps"], ev["results"]
    m, s = r.mean(1), r.std(1)
    ax[0, 0].plot(t, m, label=name); ax[0, 0].fill_between(t, m - s, m + s, alpha=0.2)
    txt = (d / "train.log").read_text(errors="ignore")
    blocks = re.findall(r"actor_loss\s*\|\s*([-\d.e+]+).*?critic_loss\s*\|\s*([-\d.e+]+).*?", txt, re.S)
    steps = re.findall(r"total_timesteps\s*\|\s*(\d+)", txt)
    rew = re.findall(r"ep_rew_mean\s*\|\s*([-\d.e+]+)", txt)
    n = min(len(blocks), len(steps))
    # actor/critic loss appear only under eval blocks (train/); align by order
    al = [float(b[0]) for b in blocks]; cl = [float(b[1]) for b in blocks]
    ts = [int(x) for x in steps][-len(al):] if len(al) <= len(steps) else list(range(len(al)))
    ax[0, 1].plot(ts[:len(al)], al, label=name)
    ax[1, 0].plot(ts[:len(cl)], cl, label=name)
    if rew:
        ax[1, 1].plot(np.arange(len(rew)), [float(x) for x in rew], label=name)
    best = m.max(); print(f"{name}: final eval {m[-1]:.2f} +/- {s[-1]:.2f}, best {best:.2f} @ {t[m.argmax()]}, first {m[0]:.2f}")
for a, ti in zip(ax.flat, ["Eval reward (mean±std, 20 eps)", "Actor loss", "Critic loss", "Training ep_rew_mean (log order)"]):
    a.set_title(ti); a.grid(alpha=.3); a.legend()
ax[0, 0].set_xlabel("timesteps")
plt.tight_layout(); plt.savefig("runs/sb3_actor_results.png", dpi=130)

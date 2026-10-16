import re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ev = np.load("runs/sb3_ddpg_refined/eval_logs/evaluations.npz"); t, r = ev["timesteps"], ev["results"]
log = open("runs/fixed_design_actor_check.log").read()
it, rw = zip(*[(int(a), float(b)) for a, b in re.findall(r"iteration\s+(\d+)/500\s+mean_train_reward=([-\d.]+)", log)])
base, sb3_final, cus_final = -105.7, -75.6, -91.9

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1, 0.7]}, sharey=False)
ax[0].plot(t, r.mean(1), c="C1", label="SB3 refined (eval, deterministic)")
ax[0].fill_between(t, r.mean(1)-r.std(1), r.mean(1)+r.std(1), color="C1", alpha=.2)
ax[0].axhline(base, c="gray", ls="--", label="rule-based −105.7"); ax[0].set_xlabel("timesteps"); ax[0].set_title("SB3 DDPG (refined)")
ax[1].plot(it, rw, c="C0", label="custom (training reward, noisy)")
ax[1].axhline(base, c="gray", ls="--", label="rule-based −105.7"); ax[1].set_xlabel("iteration (32 episodes each)"); ax[1].set_title("Custom DDPG, fixed design")
for a in ax[:2]: a.set_ylim(-115, -60); a.grid(alpha=.3); a.legend(loc="lower right"); a.set_ylabel("weekly reward (CHF)")
ax[2].bar(["SB3\nrefined", "Custom\n(train)", "Rule-\nbased"], [sb3_final, cus_final, base], color=["C1", "C0", "gray"])
for i, v in enumerate([sb3_final, cus_final, base]): ax[2].text(i, v-3, f"{v:.1f}", ha="center", va="top", color="black", fontweight="bold")
ax[2].set_ylim(-115, 0); ax[2].set_title("Final reward"); ax[2].grid(axis="y", alpha=.3)
plt.suptitle("Fixed design 6 kWp / 14 kWh: SB3 vs custom DDPG actor"); plt.tight_layout()
plt.savefig("runs/fixed_design_sb3_vs_custom.png", dpi=130)

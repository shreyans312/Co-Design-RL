import sys, numpy as np
sys.argv=["x"]
from train_sb3_ddpg import EnergySystemEnv
from stable_baselines3 import DDPG
m = DDPG.load("runs/sb3_ddpg_refined/best_model/best_model.zip", device="cpu")
for pv, bat in [(6.0,14.0),(6.6,4.5)]:
    env = EnergySystemEnv(pv, bat, split="train", seed=21, episode_seeds=list(range(100)))
    rs=[]
    for _ in range(100):
        o,_ = env.reset(); done=False; tot=0
        while not done:
            a,_ = m.predict(o, deterministic=True)
            o,r,te,tr,_ = env.step(a); tot+=r; done = te or tr
        rs.append(tot)
    rs=np.array(rs); print(f"SB3 refined @ design {pv}/{bat}: mean {rs.mean():.2f} std {rs.std():.2f}")

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "stochastic-model"))

from rl_config import get_rl_config
from rl_algorithms import StateEncoder
from train_rl import build_system_config
from config import get_default_config
from data import generate_dataset
from simulator import EnergySystemSimulator
from stochastic import generate_episode_scenario

dataset = generate_dataset(year=2026, days=365, seed=21)
base_config = get_default_config()
rl_config = get_rl_config("ddpg_lognormal")
FIXED_DESIGN = np.array([6.0, 14.0])
system_config = build_system_config(base_config, FIXED_DESIGN)


def rule_based_action(state):
    # Paper's own rule-based baseline: discharge battery when load exceeds PV,
    # charge it when PV exceeds load. Same logic applied to the EV when present.
    net_load_kw = state["load_kw"] - state["pv_power_kw"]
    battery_action = float(np.clip(net_load_kw / 6.0, -1.0, 1.0))
    if state["ev_present"]:
        ev_action = float(np.clip(net_load_kw / 5.0, -1.0, 1.0))
    else:
        ev_action = 0.0
    return np.array([battery_action, ev_action], dtype=np.float32)


rewards = []
for episode in range(100):
    seed = episode
    scenario = generate_episode_scenario(data=dataset, config=system_config, split="train", seed=seed)
    simulator = EnergySystemSimulator(system_config)
    state = simulator.reset(scenario)
    while not simulator.done:
        action = rule_based_action(state)
        next_state, reward, done, transition = simulator.step(action)
        state = next_state
    rewards.append(simulator.cumulative_reward)

rewards = np.array(rewards)
print(f"Rule-based baseline over 100 episodes, fixed design (6 kWp / 14 kWh):")
print(f"mean reward: {rewards.mean():.3f}  std: {rewards.std():.3f}  min: {rewards.min():.3f}  max: {rewards.max():.3f}")

# Co-Design-RL

Our implementation for jointly optimizing photovoltaic and battery design parameters while learning energy-system control policies on a stochastic simulation model.

The repository compares three co-design variants on the same stochastic environment

| Model | Controller | Design distribution |
|---|---|---|
| `ppo_gmm` | PPO | Gaussian mixture |
| `ddpg_lognormal` | DDPG | Log-normal mixture |
| `ppo_lognormal` | PPO | Log-normal mixture |

The implementation preserves the local stochastic model as the common environment for every controller.

The files under `stochastic-model/` define the stable simulation baseline. The root Python files contain design distributions, PPO, DDPG, training, evaluation, and tests !

## Setup

Python 3.12 is recommended

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```


## Full experiment

Run all three models and save terminal output

```bash
mkdir -p runs
python -u train_rl.py --model all --log-every 10 --output-directory runs/ 2>&1 | tee runs/full.log
```

Models run in this order

1. `ddpg_lognormal`
2. `ppo_gmm`
3. `ppo_lognormal`

Run one model by replacing `all`

```bash
python train_rl.py --model ppo_gmm --log-every 10 --output-directory runs/ppo_gmm
```

The implementation currently runs on CPU. A GPU is not used by the training code.

## Training terminology

| Term | Meaning |
|---|---|
| Timestep | One simulated hour |
| Episode | 168 timesteps or one week |
| Iteration | 32 training episodes, learning updates, and 32 validation episodes |
| Design batch | The 32 designs sampled for the training episodes in one iteration |
| PPO epoch | One pass over the collected PPO rollout |
| PPO minibatch | 64 rollout transitions used for one gradient update |
| DDPG minibatch | 256 transitions sampled from replay memory |

PPO reuses each iteration's rollout for four epochs. DDPG has no epoch setting and attempts one update after every training transition once replay memory contains 256 samples

The default full run contains 500 iterations per model. At iteration 10, a model has completed 320 training episodes and 320 validation episodes

## Shared settings

| Setting | Value |
|---|---:|
| Random seed | 42 |
| Dataset year | 2026 |
| Dataset length | 365 days |
| Episode length | 168 hours |
| Training iterations | 500 |
| Training episodes per iteration | 32 |
| Validation episodes per iteration | 32 |
| Design-distribution learning rate | `1e-3` |
| Gradient clipping norm | `0.5` |

## Controller settings

| Setting | PPO | DDPG |
|---|---:|---:|
| Hidden layers | `128 × 128 × 128` | `256 × 256` |
| Activation | Tanh | ReLU |
| Learning rate | `1e-4` | Actor `1e-4`, critic `1e-3` |
| Discount factor | `0.99` | `0.99` |
| Minibatch size | 64 | 256 |
| Epochs per iteration | 4 | Not applicable |
| GAE lambda | `0.95` | Not applicable |
| PPO clip ratio | `0.2` | Not applicable |
| Value-loss coefficient | `0.5` | Not applicable |
| Entropy coefficient | `0.0` | Not applicable |
| Replay capacity | Not applicable | 100,000 |
| Learning starts | Not applicable | 256 transitions |
| Exploration-noise standard deviation | Not applicable | `0.10` |
| Target update rate | Not applicable | `0.005` |
| Linear learning-rate decay | Yes | No |

## Design-distribution settings

| Model | Components | Initial scale | Trainable weights | Pruning |
|---|---:|---:|---:|---:|
| `ppo_gmm` | 8 | `0.2885` of design range | No | Yes |
| `ddpg_lognormal` | 3 | Log-space standard deviation `1.0` | Yes | No |
| `ppo_lognormal` | 8 | Log-space standard deviation `0.75` | No | Yes |

PPO variants use 10% design warmup, 10% pruning intervals, and a fixed design during the final 10% of training. Each pruning evaluation uses 100 episodes per active component

The DDPG log-normal distribution starts with log-space locations sampled between `0.0` and `1.0`. Its entropy weight starts at `0.01` and decreases to zero halfway through training. PPO design capacities are bounded at 12 kWp PV and 20 kWh battery, while the DDPG log-normal design has no configured upper bound

## Stochastic-model consistency

All models use the same 2026 dataset generator, episode rules, simulator equations, reward calculation, and train-validation split. Matching seeds give each model the same underlying weeks and random events.

Each sampled design is applied to a copy of the baseline configuration, so only PV and battery capacity change. The stochastic model remains unchanged while the controller and design distribution differ between models.

## RL interface

The controller receives an 11-value normalized observation

1. Hour
2. Day of year
3. Battery state of charge
4. PV power
5. Building load
6. Import price
7. Export price
8. EV presence
9. EV state of charge
10. PV design capacity
11. Battery design capacity

The two actions control battery and EV power on `[-1, 1]`. Positive values discharge storage and negative values charge it. The reward is the negative total operating and design cost for the timestep.

## Results notebook

After running the RL models run the results_analysis.ipnb notebook for results inference. The notebook compares training reward, validation cost, PV and battery design trajectories, final validation metrics, and mixture-distribution convergence.

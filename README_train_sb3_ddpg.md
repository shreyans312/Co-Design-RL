# train_sb3_ddpg.py

Trains a Stable-Baselines3 DDPG agent to control the household energy system (PV, battery, EV, grid) for a **fixed PV/battery design**. It is a control-policy baseline, comparable to the "best two-step" scenario: it does not do the design co-optimisation that `CoDesignTrainer` in `train_rl.py` does (see `train_sb3_codesign.py` for that).

## What the script contains

| Piece | Purpose |
|---|---|
| `EnergySystemEnv` | Gymnasium wrapper around `EnergySystemSimulator` (from `stochastic-model/`) for one fixed design. |
| `DDPGSplitLR` | SB3 `DDPG` subclass with separate actor and critic learning rates. SB3 normally has one. |
| `build_env` | Builds an `EnergySystemEnv` wrapped in `Monitor`. |
| `main` | Parses arguments, builds train and eval envs, trains with `EvalCallback`, saves the final model. |

## Environment

- **Observation:** 11 floats, built by `StateEncoder` in `rl_algorithms.py`, scaled using `rl_config.py`:
  hour/23, day/364, battery SoC fraction, PV power/12, load/10, import price, export price, EV present, EV SoC fraction, PV design/12, battery design/20.
  The last two are constant for a given run.
- **Action:** 2 continuous values in [-1, 1].
- **Reward:** the simulator's reward (negative cost; the log's `ep_rew_mean` is around -80 to -90 for good policies on the default setup).
- **Episode:** a fixed 168-hour (one-week) horizon. It ends by truncation only, never termination. The last observation is reused as the bootstrap input at the horizon.
- **Splits:** training episodes are drawn from the `train` split with a random scenario seed each reset. The eval env uses the `validation` split and cycles through fixed seeds (`10000 + seed*1000 + i`), so every evaluation scores the same episodes.

## Usage

Run from `MCD310/`:

```bash
python train_sb3_ddpg.py --pv-capacity-kwp 6.6 --battery-capacity-kwh 4.5 --total-timesteps 100000 --output-dir runs/my_run
```

### Arguments

| Argument | Default | Notes |
|---|---|---|
| `--pv-capacity-kwp` | 6.0 | PV size of the fixed design |
| `--battery-capacity-kwh` | 14.0 | Battery size of the fixed design |
| `--dataset-days` | 365 | Days of generated data |
| `--seed` | 42 | Seeds the dataset, SB3 and the eval episodes |
| `--total-timesteps` | 100000 | 2,688,000 matches the repo's custom DDPG schedule (500 iterations x 32 episodes x 168 h) |
| `--actor-learning-rate` | 1e-4 | |
| `--critic-learning-rate` | 1e-3 | Also what SB3 logs as `learning_rate` |
| `--buffer-size` | 100000 | |
| `--batch-size` | 256 | |
| `--tau` | 0.005 | Target network update rate |
| `--gamma` | 0.99 | |
| `--learning-starts` | 512 | Steps before updates begin |
| `--exploration-noise-std` | 0.20 | Constant Gaussian action noise (`NormalActionNoise`) |
| `--eval-freq` | 5000 | Steps between evaluations |
| `--n-eval-episodes` | 20 | Validation episodes per evaluation |
| `--output-dir` | `runs/sb3_ddpg` | |

Network architecture: MLP policy with `net_arch=[256, 256]`.

## Outputs (in `--output-dir`)

- `best_model/best_model.zip`: best model by mean validation reward.
- `eval_logs/evaluations.npz`: `timesteps` and `results` (per-episode eval rewards), read with `numpy.load`.
- `final_model.zip`: model at the end of training (only written if training finishes).
- `tensorboard/`: TensorBoard logs. The script enables these by default, so TensorBoard must be installed or the run wrapper must pass `tensorboard_log=None` (the scripts in `runs/` do).

Each model zip is about 2.3 MB.

## Not included (in `runs/`)

The script itself has constant exploration noise and 1-step TD. The experiment wrappers in `runs/` patch it for variations:

- `noise_decay.py`: `NoiseDecay` callback, sigma_t = sigma0 * exp(-rate * t).
- `nstep_buffer.py`: `NStepReplayBuffer` for n-step returns (use `--gamma` = gamma^n).
- `run_sb3_*.py`: wrappers that set `sys.argv`, disable TensorBoard and apply the above, one per experiment.

## Notes

- The saved replay buffer is not part of the output, so a stopped run cannot be resumed.
- Run it on CPU: the networks and batches are small and the simulator is the bottleneck, so a GPU is unlikely to help.

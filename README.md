# Co-Design-RL

Our implementation for jointly optimizing photovoltaic and battery design parameters while learning energy-system control policies on a stochastic simulation model.

The repository compares three co-design variants on the same stochastic environment

| Model | Controller | Design distribution |
|---|---|---|
| `ppo_gmm` | PPO | Gaussian mixture |
| `ddpg_lognormal` | DDPG | Log-normal mixture |
| `ddpg_lognormal_improved` | DDPG | Bounded stratified log-normal mixture |
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
python -u train_rl.py --model all --log-every 10 --save-every 10 --output-directory runs/ 2>&1 | tee runs/full.log
```

Models run in this order

1. `ddpg_lognormal`
2. `ddpg_lognormal_improved`
3. `ppo_gmm`
4. `ppo_lognormal`

Run one model by replacing `all`

```bash
python train_rl.py --model ppo_gmm --log-every 10 --save-every 10 --output-directory runs/
```

The implementation currently runs on CPU. A GPU is not used by the training code.

## Training terminology

| Term | Meaning |
|---|---|
| Timestep | One simulated hour |
| Episode | 168 timesteps or one week |
| Iteration | 32 training episodes, learning updates, and 32 validation episodes |
| Design batch | Normally 32 sampled designs; improved DDPG uses 8 designs on 4 common scenarios each |
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
| `ddpg_lognormal` | 3 | Log-space standard deviation `1.5` | Yes | No |
| `ddpg_lognormal_improved` | 4 | Log-space standard deviation `0.55` | Yes | No |
| `ppo_lognormal` | 8 | Log-space standard deviation `0.75` | No | Yes |

PPO variants use 10% design warmup, 10% pruning intervals, and a fixed design during the final 10% of training. Each pruning evaluation uses 100 episodes per active component

The current imported `ddpg_lognormal` preset uses log-space location ranges `(0.1823, 0.6931)` to `(2.3795, 2.8904)`, log-space standard deviation `3.0`, design learning rate `0.005`, and initial entropy weight `5`. These are experimental settings, not verified paper hyperparameters, and differ from the older saved baseline runs. Its unbounded log-normal tail can generate very large capacities. For the controlled custom-DDPG experiments below, use `ddpg_lognormal_improved`, which retains its bounded design initialization

DDPG uses a leave-one-out return baseline for its design update, comparing each design with the other 31 designs in the same iteration. When an output directory is provided, training tables and a model snapshot are saved every 10 iterations by default

DDPG samples designs throughout all 500 iterations. Its final validation also samples from the learned distribution, while `final_design_samples.csv` and `design_summary.json` record 1,000 final samples and their quartiles

The improved DDPG variant is an experimental deviation from the paper baseline. Its component centres are stratified across the full design range, capacities are bounded at 12 kWp and 20 kWh, design advantages are normalized, and every design is evaluated on four common stochastic scenarios. Component weights and standard deviations retain minimum exploration levels while action noise decreases from `0.20` to `0.05`

## Stochastic-model consistency

All models use the same 2026 dataset generator, episode rules, simulator equations, reward calculation, and train-validation split. Matching seeds give each model the same underlying weeks and random events.

Each sampled design is applied to a copy of the baseline configuration, so only PV and battery capacity change. The stochastic model remains unchanged while the controller and design distribution differ between models.

For the separate Cauz-data experiment, pass `--dataset cauz` to `train_rl.py`. This loads `stochastic-model/cauz_dataset.csv` directly instead of generating the default synthetic 2026 year. The CSV is a reconstructed 2021 series, not the original measured dataset. Its 2021 timestamps and existing validation-week flags are retained; the simulator and reward equations are unchanged. The run configuration records the dataset choice and SHA-256 hash. Evaluate its checkpoints with the supplied evaluation scripts, which reload and verify that same CSV. Do not compare its absolute costs directly with runs on the synthetic 2026 environment. The historical synthetic design sweep is preserved under `references/synthetic_2026_frozen_ddpg_grid/`.

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

After running the RL models run the results_analysis.ipynb notebook for results inference. The notebook compares training reward, validation cost, PV and battery design trajectories, final validation metrics, and mixture-distribution convergence.

## Custom DDPG with repeatable validation

This runs our own PyTorch actor, critic, replay buffer and design learner. It does not use SB3. The stochastic model is unchanged. The existing custom entropy-gradient and next-state bootstrap calculations are retained and tested. Critic parameters are frozen during the actor backward pass to avoid computing unused gradients; the actor still receives gradients through the critic's action input.

```bash
python3 -u train_rl.py \
  --model ddpg_lognormal_improved \
  --seed 42 --data-seed 42 \
  --iterations 500 --episodes-per-iteration 32 \
  --design-warmup-fraction 0.10 \
  --exploration-noise-std 0.30 --final-exploration-noise-std 0.05 \
  --matched-validation --validation-episodes 32 --validation-every 5 \
  --save-every 5 --log-every 5 --torch-threads 1 \
  --output-directory runs/custom_ddpg_noise030_seed42
```

The proposed noise experiment increases the initial action-noise standard deviation to `0.30` and decreases it linearly to `0.05`. Design learning is frozen for the first 50 iterations while control learns. Each iteration still uses 8 sampled designs on 4 common training scenarios, for 32 episodes of 168 hours. The capacity bounds, reward, controller learning rates and design initialization remain those of the improved preset.

For a controlled comparison, repeat with initial noise `0.20` and a different output directory. Run both settings for training seeds 42, 43 and 44, keeping `--data-seed 42` and `--validation-seed 1000042` unchanged. Six runs may take longer than one night. All runs use the same generated dataset and validation scenarios; training seeds control network initialization and training sampling. No Optuna search is included yet.

With `--matched-validation`, evaluation is deterministic and uses the same scenario seeds at each checkpoint. It reports:

- `validation_mode_cost_chf`: cost at the current selected design, used for best-checkpoint selection
- `validation_distribution_cost_chf`: cost across the current mixture, using a repeatable sampling stream
- `validation_reference_cost_chf`: average controller cost at the fixed designs `(4, 4.5)`, `(6.6, 4.5)` and `(9, 4.5)` in kWp and kWh

The existing `validation_mean_cost_chf` column means selected-design cost in this protocol. It is not directly comparable to older distribution-validation curves. The reported mode is the existing highest-weighted-component-peak selection, not an exact global optimization of mixture density. Reference costs isolate controller progress because capacities stay fixed; distribution costs also reflect changes in the design learner.

Results are saved inside `<output-directory>/ddpg_lognormal_improved/`:

- `configuration.json`: effective settings, seeds and runtime versions
- `training.csv`: costs, actor/critic losses, gradient norms, Q estimates, TD error, saturation and design statistics
- `training_samples.csv` and `design_distribution.csv`: exact training designs and mixture parameters
- `validation_episodes.csv`: per-scenario costs, actions, applied power and constraint-limiting fractions for each evaluation group
- `initial_checkpoint.pt` and `initial_validation.json`: untrained comparison
- `best_checkpoint.pt` and `best_validation.json`: controller and design saved together when selected-design validation cost improves
- `checkpoint.pt` and `latest_validation.json`: latest saved progress
- `validation.json`: final selected-design, distribution and reference results

Checkpoints are written atomically and include both controller and design parameters. They support evaluation, but not exact training resumption because replay and random-generator states are not saved. Best-checkpoint selection uses validation only; an independent final test is still needed for reporting generalization. Existing non-empty model output directories are rejected to protect earlier runs.

On macOS, prefix the command with `caffeinate -i` to prevent idle sleep while training. Keep the laptop connected to power and its lid open. To capture terminal output, create the parent run directory first and append `2>&1 | tee -a runs/custom_ddpg_noise030_seed42/train.log` to the command.

## Frozen-actor design scoring experiment

`--design-scoring training` is the default and retains the existing design update from noisy controller-training returns. The separate DDPG option `--design-scoring frozen_actor` first trains the controller normally, then holds the post-training actor fixed while re-evaluating the same sampled designs without action noise. The design learner uses the mean return across these additional scenarios. Mixture sampling, return normalisation, entropy and learning rates are unchanged.

Every design receives the same fresh training-scenario seeds in an iteration. Scoring uses the training split, not validation data, and its seed block starts after the planned controller-training seed block. Scenario seeds change each iteration and are recorded. Scoring does not update the controller, add transitions to replay or consume its exploration RNG. No scoring episodes run during design warmup or the final fixed-design phase.

For eight designs and four scenarios each, this adds 32 scoring episodes per active design-update iteration. With 500 iterations and 10% warmup, that is 14,400 additional episodes or 2,419,200 simulator steps, but no extra actor/critic updates. Compare both controller-training steps and total simulator work when reporting results.

To test this separately from the increased-noise experiment:

```bash
python3 -u train_rl.py \
  --model ddpg_lognormal_improved \
  --seed 43 --data-seed 42 \
  --iterations 500 --episodes-per-iteration 32 \
  --design-warmup-fraction 0.10 \
  --exploration-noise-std 0.30 --final-exploration-noise-std 0.05 \
  --design-scoring frozen_actor \
  --matched-validation --validation-episodes 32 --validation-every 5 \
  --save-every 5 --log-every 5 --torch-threads 1 \
  --output-directory runs/custom_ddpg_frozen_scoring_seed43
```

Use a new output directory for each run. The mode and provenance are saved in the configuration and checkpoints. `design_scores.csv` records the clean per-scenario scores, component, capacities and seeds; join it with `training_samples.csv` using iteration and design index to inspect how the scores differ. Those scores use fresh scenarios, so differences reflect scenario variation as well as controller changes and action noise. `training.csv` adds `design_scoring_episodes`, `design_score_mean_cost_chf` and `design_score_std_cost_chf`; the latter is the spread across design-average costs. Terminal output adds `design_score_cost`. Existing training rewards and validation metrics keep their meanings.

For the noise-only experiment, retain `--design-scoring training` and change only the final action-noise standard deviation from `0.05` to `0.10`, with the initial value still `0.30`. Do not combine both interventions in the first comparison.

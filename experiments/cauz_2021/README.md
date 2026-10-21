# Direct reconstructed-Cauz 2021 experiment

This experiment changes the input dataset only. It uses `stochastic-model/cauz_dataset.csv` directly, including its 2021 timestamps and four marked validation weeks. The source series was reconstructed from a published figure and contains interpolated hours; it is not the paper's original measurement file. The fitted `cauz_model.py` generator is **not** used.

The baseline custom co-design DDPG protocol is `ddpg_lognormal_improved`, training seed 43, dataset seed 42 (recorded but not used to regenerate the fixed CSV), 500 iterations, 32 episodes per iteration, 168 hours per episode, 10% design warmup, action-noise standard deviation 0.30 decreasing to 0.05, frozen-actor design scoring, 32 matched validation episodes every 5 iterations, and checkpointing every 5 iterations. This repeats the earlier frozen-scoring settings and changes only the dataset source. There are 8 unique sampled designs per iteration, each evaluated on 4 common training scenarios. Frozen scoring adds 32 further training-split simulations at each active design update. Validation selects checkpoints; it is not an unseen-week test.

The CSV SHA-256 is `e9e9c987dc370a67bfe31db232ad657f6314cd5a4e855775a91a6bd990c77c45`. The training configuration and checkpoints record the selected dataset and hash; evaluators reject a changed CSV. The four validation weeks remain excluded from training episode selection. Costs on this environment must not be compared directly with the old synthetic-2026 grid, which is archived at `references/synthetic_2026_frozen_ddpg_grid/`.

Run from the repository root with the environment's Python interpreter and a fresh output directory:

```bash
python3 -u train_rl.py \
  --dataset cauz --model ddpg_lognormal_improved \
  --seed 43 --data-seed 42 \
  --iterations 500 --episodes-per-iteration 32 \
  --design-warmup-fraction 0.10 \
  --exploration-noise-std 0.30 --final-exploration-noise-std 0.05 \
  --design-scoring frozen_actor \
  --matched-validation --validation-episodes 32 --validation-every 5 \
  --save-every 5 --log-every 5 --torch-threads 1 \
  --output-directory runs/cauz_2021_frozen_scoring_seed43
```

Use a new directory for additional training seeds. For fixed-design SB3 controller-only runs, `train_sb3_ddpg.py --dataset cauz` already uses the same canonical CSV, but those runs are a different algorithmic experiment.

For Windows CUDA runs, first install a PyTorch build compatible with the GPU and confirm `torch.cuda.is_available()`. The custom DDPG command accepts `--device cuda`; only the controller networks and replay minibatches move to CUDA, while the simulator and design distribution stay on CPU. `--design-scoring training` disables the extra frozen-actor scoring pass. To study the original three-component, non-stratified DDPG separately, use `--model ddpg_lognormal`; `--design-initial-std 0.8` is an explicit experimental override of its default `3.0` log-space scale. Keep the unchanged original and overridden runs in different directories and compare them on matched validation scenarios. A larger controller action-noise standard deviation is a different intervention from widening the design distribution.

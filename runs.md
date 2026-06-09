# ProSpero Run Backlog

This file is operational memory for active and recent ProSpero experiments. Update it when launching, fixing, or completing runs.

## Active / Recent Variable-k Runs

### AAV full-train one-hot ridge
- Status: complete
- Output: `outputs/aav_one_hot_ridge_rerun_20260525`
- Budgets: `k=8,16,32,64,128`, seeds `1..5`
- Final metrics:
  - `k=8`: mean max `0.731`, mean perf `0.630`
  - `k=16`: mean max `0.702`, mean perf `0.650`
  - `k=32`: mean max `0.708`, mean perf `0.671`
  - `k=64`: mean max `0.710`, mean perf `0.675`
  - `k=128`: mean max `0.704`, mean perf `0.668`

### AAV full-train frozen CLS ridge
- Status: complete
- Surrogate: `frozen_esm_cls_ridge`
- Output: `outputs/aav_cls_ridge_20260525`
- Budgets: `k=8,16,32,64,128`, seeds `1..5`
- Final metrics:
  - `k=8`: mean max `0.555`, mean perf `0.375`
  - `k=16`: mean max `0.564`, mean perf `0.468`
  - `k=32`: mean max `0.600`, mean perf `0.536`
  - `k=64`: mean max `0.591`, mean perf `0.551`
  - `k=128`: mean max `0.630`, mean perf `0.590`

### AAV top-100 initial-train one-hot ridge
- Status: complete
- Output: `outputs/aav_top100_train_variable_k_20260527/one_hot_ridge`
- Command uses `--initial-train-top-k 100`
- Final metrics:
  - `k=8`: mean max `0.583`, mean perf `0.297`
  - `k=16`: mean max `0.630`, mean perf `0.495`
  - `k=32`: mean max `0.656`, mean perf `0.597`
  - `k=64`: mean max `0.668`, mean perf `0.621`
  - `k=128`: mean max `0.683`, mean perf `0.650`

### AAV top-100 initial-train frozen CLS ridge
- Status: running as of 2026-05-29
- Surrogate: `frozen_esm_cls_ridge`
- Output: `outputs/aav_top100_train_variable_k_20260527/cls_ridge`
- Command uses `--initial-train-top-k 100`
- Completed:
  - `k=8`: mean max `0.419`, mean perf `0.082`
  - `k=16`: mean max `0.499`, mean perf `0.173`
  - `k=32`: mean max `0.550`, mean perf `0.385`
  - `k=64`: mean max `0.584`, mean perf `0.519`
- `k=128`: 1/5 valid seeds so far, interim mean max `0.588`, mean perf `0.554`; seed 2 partial at iteration 2/10 and actively running.

### AAV top-100 layer-2 CLS LoRA ridge
- Status: running as of 2026-05-29 after retry/fix
- Fine-tune output: `outputs/aav_top100_train_variable_k_20260527/layer2_cls_lora_r4/adapter`
- Variable-k output: `outputs/aav_top100_train_variable_k_20260527/layer2_cls_lora_r4/variable_k`
- Surrogate: `frozen_esm_layer_cls_ridge`
- Representation: truncated ESM `hidden_states[2][:, 0, :]`
- LoRA: rank `4`, alpha `8`, target `encoder.layer.1.attention.self.{query,value}`
- Fine-tune validation: best epoch `7`, RMSE `0.1266`, Spearman `0.025`
- Implementation note: initial variable-k failed because shared ESM was LoRA-wrapped repeatedly across in-process seeds; fixed by making LoRA application idempotent.
- Completed:
  - `k=8`: mean max `0.494`, mean perf `0.089`
- In progress: `k=16`, 2/5 valid seeds so far, interim mean max `0.493`, mean perf `0.165`; seed 3 partial at iteration 1/10 and actively running.
- Pending: `k=32,64,128`

### LGK one-hot ridge variable-k
- Status: complete
- Output: `outputs/lgk_one_hot_ridge_variable_k_20260526`
- Final metrics:
  - `k=8`: mean max `0.035`, mean perf `0.030`
  - `k=16`: mean max `0.039`, mean perf `0.037`
  - `k=32`: mean max `0.040`, mean perf `0.038`
  - `k=64`: mean max `0.042`, mean perf `0.041`
  - `k=128`: mean max `0.044`, mean perf `0.042`

### LGK frozen CLS ridge variable-k
- Status: complete
- Output: `outputs/lgk_cls_ridge_variable_k_20260526`
- Completed:
  - `k=8`: mean max `0.035`, mean perf `0.009`
  - `k=16`: mean max `0.037`, mean perf `0.031`
  - `k=32`: mean max `0.040`, mean perf `0.037`
  - `k=64`: mean max `0.042`, mean perf `0.040`
  - `k=128`: mean max `0.043`, mean perf `0.041`

## Static LGK Validation Benchmarks

### LGK CLS ridge static validation
- Status: complete
- Output: `outputs/lgk_cls_ridge_20260525`
- Results:
  - `full_train`: R2 `0.251`, Spearman `0.611`, RMSE `0.182`
  - `top10_train`: R2 `-0.315`, Spearman `-0.047`, RMSE `0.241`
  - `top100_train`: R2 `-0.213`, Spearman `0.193`, RMSE `0.231`
  - `top500_train`: R2 `-0.115`, Spearman `0.422`, RMSE `0.222`

### LGK one-hot ridge static validation
- Status: complete
- Output: `outputs/lgk_one_hot_ridge_20260525_rerun`
- Results:
  - `full_train`: R2 `0.083`, Spearman `0.488`, RMSE `0.201`
  - `top10_train`: R2 `-0.323`, Spearman `-0.071`, RMSE `0.241`
  - `top100_train`: R2 `-0.269`, Spearman `0.006`, RMSE `0.236`
  - `top500_train`: R2 `-0.153`, Spearman `-0.275`, RMSE `0.225`

## Implemented Experiment Infrastructure

### Initial train top-k option
- Added to `src/prospero/runners/run_protein.py`: `--initial_train_top_k`
- Added to `src/prospero/runners/run_variable_k.py`: `--initial-train-top-k`
- Semantics: restrict only the initial training split to top-k by initial train score; validation remains unchanged; newly collected sequences are added normally.

### Layer-2 CLS LoRA infrastructure
- Added `src/prospero/esm_lora.py`
- Added `scripts/finetune_esm_layer_cls_lora.py`
- Added surrogate `frozen_esm_layer_cls_ridge`
- Added variable-k args:
  - `--esm-representation-layer`
  - `--esm-truncate-to-representation-layer / --no-esm-truncate-to-representation-layer`
  - `--esm-lora-adapter-path`
- Hidden-state convention: `hidden_states[0]` is embeddings, `hidden_states[2]` is the repo's layer-2 representation; LoRA defaults to encoder block `hidden_state_layer - 1`.

### Masking/scoring ablation flags
- Added to `src/prospero/runners/run_protein.py`:
  - `--mask_strategy targeted|random`
  - `--sequence_scoring surrogate|zero_shot_logp_delta`
  - `--mask_count_calibration_rounds`
- Added passthroughs to `src/prospero/runners/run_variable_k.py`:
  - `--mask-strategy`
  - `--sequence-scoring`
  - `--mask-count-calibration-rounds`
- Random masking semantics: calibrate selected targeted-mask count stats at seed start using that seed's initial surrogate, then draw rounded clipped Normal counts with calibrated min/max.
- Zero-shot-style scoring semantics: keep surrogate training/retraining when requested, but use cumulative `logP(sampled)-logP(original)` for SMC rollout scores and candidate ranking.

### Zero-shot fixed-budget masking strategies
- Added to `src/prospero/runners/run_zero_shot_protein.py`:
  - `--mask_strategy calibrated_random|random|middle_entropy|seed_grow`
  - `--mask_budget`
  - `--entropy_quantile`
  - `--entropy_sigma`
  - `--seed_grow_alpha`
  - `--seed_grow_beta`
  - `--seed_grow_coupling_tau`
- Fixed-budget `random`: sample exactly `K=--mask_budget` maskable residues uniformly without replacement.
- Fixed-budget `middle_entropy`: compute single-site masked EvoDiff entropy for each position, choose `H*` as an entropy quantile, and sample exactly `K` positions with Gaussian weight around `H*`.
- Fixed-budget `seed_grow`: sample a middle-entropy seed, then grow to exactly `K` positions using `alpha * middle_entropy_score + beta * sequence_proximity_coupling`, where coupling is `exp(-distance/tau)`.
- These routes use no surrogate and no training data for mask selection or scoring; training-set files are only still loaded for oracle bookkeeping/ref filtering.

## AAV Zero-Shot ProSpero / No Surrogate

User proposal:
- Remove trainable/retrained surrogate entirely.
- Randomly mask amino acids using empirical mean/std of number of masked residues from targeted masking runs.
- Run SMC as usual.
- During biologically constrained unmasking and rollout unmasking, score sampled choices by cumulative probability delta: `P(sampled residue) - P(original residue)` for each unmasked position.
- Use final cumulative probability score to rank candidates before oracle scoring.

Status: complete 2026-05-28.

Run details:
- tmux: `aav_zero_shot_20260528`
- GPU: RTX 2080 Ti UUID `GPU-a3d54441-08da-ed66-fa0f-6aaaaf97baea`
- Output: `outputs/aav_zero_shot_prospero_20260528`
- Seeds: `1..5`, sequential in one tmux session
- Candidate budget: `n_queries=128`, `n_iters=10`
- Final metrics across 5 seeds: mean max `0.441`, mean perf `0.369`, diversity `4.110`, novelty `2.936`, WT novelty `3.206`.
- Per-seed final mean perf / best:
  - seed 1: `0.338` / `0.454`
  - seed 2: `0.372` / `0.449`
  - seed 3: `0.377` / `0.439`
  - seed 4: `0.383` / `0.433`
  - seed 5: `0.378` / `0.433`
- Calibrated selected-mask stats by seed had mean count around `3.848` and stds around `1.115-1.160`; all used empirical min `3`, max `10`.

Implementation:
- Added `src/prospero/runners/run_zero_shot_protein.py`
- Added zero-shot random-mask SMC methods to `src/prospero/inference.py`
- No surrogate is trained/retrained during the optimization loop.
- A fast one-hot ridge surrogate is used only before each seed to calibrate selected targeted-mask count stats because historical selected mask counts were not logged.
- Random mask counts are drawn from rounded clipped Normal using calibrated selected-mask `mean/std/min/max`.
- Main-path unmasking samples from the biologically constrained alphabet cluster.
- Rollouts sample from the full 20-AA vocabulary.
- Candidate score is cumulative `logP(sampled residue)-logP(original residue)`.
- Resampling keeps the existing structure using `zero_shot_score * inverse_perplexity`.

## LGK Zero-Shot ProSpero / No Surrogate

Status: complete as of 2026-05-29.

Run details:
- tmux: `lgk_zero_shot_20260528`
- GPU: RTX 2080 Ti UUID `GPU-fcb13561-e5da-20e1-2ff7-a9bdbfa68c26`
- Output: `outputs/lgk_zero_shot_prospero_20260528`
- Seeds: `1..5`, sequential in one tmux session
- Candidate budget: `n_queries=128`, `n_iters=10`

Implementation:
- Same zero-shot runner and SMC logic as AAV zero-shot.
- No surrogate is trained/retrained during the optimization loop.
- A fast one-hot ridge surrogate is used only before each seed to calibrate selected targeted-mask count stats.
- Random mask counts are drawn from rounded clipped Normal using calibrated selected-mask `mean/std/min/max`.
- Main-path unmasking samples from the biologically constrained alphabet cluster.
- Rollouts sample from the full 20-AA vocabulary.
- Candidate score is cumulative `logP(sampled residue)-logP(original residue)`.
- Resampling keeps the existing structure using `zero_shot_score * inverse_perplexity`.
- Final metrics across seeds `1..5`: mean max `0.041`, mean perf `0.040`, diversity `12.151`, novelty `58.380`, WT novelty `58.640`.

## AAV One-Hot Ridge Random-Masking Ablation

Status: running as of 2026-05-29.

Run details:
- tmux: `aav_onehot_random_mask_20260528`
- GPU: RTX 2080 Ti UUID `GPU-a3d54441-08da-ed66-fa0f-6aaaaf97baea`
- Output: `outputs/aav_onehot_random_mask_variable_k_20260528`
- Variable-k budgets: `k=8,16,32,64,128`
- Seeds: `1..5`
- Surrogate: `one_hot_ridge`
- Masking: calibrated random masks, with selected targeted-mask count stats calibrated at each seed start.
- Scoring: standard surrogate UCB scoring.
- Current progress: `k=8`, `k=16`, `k=32`, and `k=64` complete; `k=128` seeds `1..3` complete, seed 4 partial at iteration 2/10.
- `k=8` metrics across 5 seeds: mean max `0.672`, mean perf `0.595`, diversity `8.289`, novelty `8.813`, WT novelty `9.543`.
- `k=16` metrics across 5 seeds: mean max `0.693`, mean perf `0.636`, diversity `6.841`, novelty `11.056`, WT novelty `11.942`.
- `k=32` metrics across 5 seeds: mean max `0.656`, mean perf `0.625`, diversity `5.606`, novelty `12.594`, WT novelty `13.654`.
- `k=64` metrics across 5 seeds: mean max `0.691`, mean perf `0.656`, diversity `4.946`, novelty `13.086`, WT novelty `14.090`.
- `k=128` metrics across 3 complete seeds: mean max `0.686`, mean perf `0.638`, diversity `5.865`, novelty `12.787`, WT novelty `13.847`.

## AAV One-Hot Ridge Targeted-Masking Zero-Score Ablation

Status: running as of 2026-05-29.

Run details:
- tmux: `aav_onehot_targeted_zero_score_20260528`
- GPU: RTX 2080 Ti UUID `GPU-6faf056b-00ab-15fd-e25c-a149c7dcf3d7`
- Output: `outputs/aav_onehot_targeted_zero_score_variable_k_20260528`
- Variable-k budgets: `k=8,16,32,64,128`
- Seeds: `1..5`
- Surrogate: `one_hot_ridge`
- Masking: standard targeted masking selected by the surrogate.
- Scoring: zero-shot-style cumulative `logP(sampled)-logP(original)` for SMC/resampling/ranking.
- Current progress: `k=8`, `k=16`, and `k=32` complete; `k=64` seeds `1..2` complete, seed 3 partial at iteration 9/10.
- `k=8` metrics across 5 seeds: mean max `0.442`, mean perf `0.244`, diversity `5.462`, novelty `3.193`, WT novelty `3.645`.
- `k=16` metrics across 5 seeds: mean max `0.442`, mean perf `0.327`, diversity `4.863`, novelty `3.060`, WT novelty `3.372`.
- `k=32` metrics across 5 seeds: mean max `0.476`, mean perf `0.392`, diversity `4.714`, novelty `3.618`, WT novelty `3.868`.
- `k=64` metrics across 2 complete seeds: mean max `0.485`, mean perf `0.416`, diversity `5.080`, novelty `3.005`, WT novelty `3.245`.
- `k=64` seed 3 partial metrics at iteration 9: max `0.483`, mean perf `0.391`.

## AAV Zero-Shot Fixed-Budget Masking Routes

Status: 128-query route complete as of 2026-05-29; variable-query extension running as of 2026-05-29.

Run details:
- tmux: `aav_zs_fixed_mask_k4_20260529`
- watchdog tmux: `prospero_watchdog_1h_20260529`
- GPU: RTX 2080 Ti UUID `GPU-fcb13561-e5da-20e1-2ff7-a9bdbfa68c26`
- Output root: `outputs/aav_zero_shot_fixed_mask_k4_20260529`
- Script: `scripts/run_aav_zero_shot_fixed_mask_k4_20260529.sh`
- Variable-query script: `scripts/run_aav_zero_shot_fixed_mask_k4_variable_queries_20260529.sh`
- Variable-query tmux: `aav_zs_fixed_mask_varq_a6000_20260529`
- Variable-query GPU: A6000 UUID `GPU-025e10e6-263f-d814-6dd5-added86fc8af`
- Strategies: `random`, `middle_entropy`, `seed_grow`
- Fixed mask budget: `K=4`
- Seeds: `1..5` per strategy, sequential.
- Candidate budget: `n_queries=128`, `n_iters=10`
- Variable-query budgets: `n_queries=8,16,32,64`; completed 128-query results are linked under `n_samples_128`.
- Scoring: zero-shot cumulative `logP(sampled)-logP(original)`.
- Main-path unmasking: constrained cluster softmax.
- Rollouts: full 20-AA softmax.
- One-hour check script: `scripts/check_aav_zero_shot_fixed_mask_routes.sh`; writes `outputs/aav_zero_shot_fixed_mask_k4_20260529/watchdog_1h.log`.
- Completed all strategies and all seeds.
- `random`: mean max `0.450`, mean perf `0.375`, diversity `3.895`, novelty `2.860`, WT novelty `3.030`.
- `middle_entropy`: mean max `0.492`, mean perf `0.386`, diversity `6.198`, novelty `3.140`, WT novelty `3.470`.
- `seed_grow`: mean max `0.478`, mean perf `0.395`, diversity `4.396`, novelty `3.268`, WT novelty `3.522`.
- Quick read: `seed_grow` has best mean performance; `middle_entropy` has best mean max and diversity.
- Variable-query folder layout: `outputs/aav_zero_shot_fixed_mask_k4_20260529/n_samples_<n_queries>/<strategy>/AAV`.
- Current variable-query progress: `n_queries=8` complete for all strategies; `n_queries=16` complete for `random` and `middle_entropy`; running `n_queries=16`, `strategy=seed_grow`, seed 2 at iteration 8/10.
- Variable-query `n_queries=8` metrics:
- `random`: mean max `0.373`, mean perf `0.115`, diversity `5.063`, novelty `3.398`, WT novelty `3.755`.
- `middle_entropy`: mean max `0.409`, mean perf `0.067`, diversity `6.472`, novelty `3.323`, WT novelty `3.788`.
- `seed_grow`: mean max `0.413`, mean perf `0.216`, diversity `4.861`, novelty `3.305`, WT novelty `3.628`.
- Variable-query `n_queries=16` completed metrics:
- `random`: mean max `0.391`, mean perf `0.197`, diversity `4.832`, novelty `3.358`, WT novelty `3.646`.
- `middle_entropy`: mean max `0.435`, mean perf `0.108`, diversity `6.638`, novelty `3.226`, WT novelty `3.774`.
- `seed_grow`: seed 1 complete with max `0.412`, mean perf `0.307`; seed 2 partial at iteration 7 with max `0.440`, mean perf `0.274`.

## AAV Zero-Shot EvoDiff Online Fine-Tuning

Status: running as of 2026-05-30.

Run details:
- Runner: `src/prospero/runners/run_zero_shot_finetune_evodiff.py`
- Fine-tune helper: `src/prospero/evodiff_finetune.py`
- Launch script: `scripts/run_aav_zero_shot_evodiff_ft_variable_k_20260529.sh`
- Output root: `outputs/aav_zero_shot_evodiff_ft_k4_variable_k_20260529`
- Folder layout: `n_samples_<n_queries>/<strategy>/AAV`
- Budgets: `n_queries=8,16,32,64,128`
- Strategies: `random`, `middle_entropy`, `seed_grow`
- Objective: rank-weighted masked-token NLL plus `lambda_kl * KL(p_theta(.|x_t,t) || p_base(.|x_t,t))`
- Rank weighting: normalized oracle-fitness rank, lowest `0`, highest `1`.
- `lambda_kl`: `2`
- Learning rate: `1e-5`
- Fine-tune epochs: `5` after each oracle evaluation round except the final round.
- Replay buffer: all evaluated candidates so far in the seed.
- Training corruption: uniform random fixed-budget masks with `K=4`; generation mask strategy remains the experiment strategy.
- Logged diagnostics per fine-tune epoch: total loss, weighted NLL, KL, mean/max gradient norm, update norm, relative update norm, seconds.
- Plots per seed: fitness curve and fine-tuning diagnostics under `seed_<n>.evodiff_finetune/plots`.

Active sessions:
- `aav_zs_evodiff_ft_seedgrow_gpu0_20260529`: GPU0 RTX 2080 Ti UUID `GPU-fcb13561-e5da-20e1-2ff7-a9bdbfa68c26`, strategy `seed_grow`.
- `aav_zs_evodiff_ft_random_middle_gpu1_20260529`: GPU1 RTX 2080 Ti UUID `GPU-a3d54441-08da-ed66-fa0f-6aaaaf97baea`, strategies `random middle_entropy`.

Smoke test:
- Output: `outputs/_smoke_aav_zs_evodiff_ft`
- Passed: 2 optimization iterations, one 1-epoch fine-tune update, metric JSONL, and plots.
- Smoke fine-tune timing: 2 sequences, 1 epoch, `0.47s` on RTX 2080 Ti.
- Smoke first update: weighted NLL `2.607`, KL `0.0` because model and base are initially identical, grad norm `4.416`, update norm `0.0584`, relative update norm `6.67e-05`.

Current progress:
- Both real runs launched at 2026-05-29 23:22 Europe/Paris.
- Completed `n_queries=8` for `random`, `middle_entropy`, and `seed_grow`.
- Completed `n_queries=16` for `random`, `middle_entropy`, and `seed_grow`.
- Completed `n_queries=32` for `random` and `seed_grow`.
- Completed `n_queries=64` for `seed_grow`.
- Active GPU0: `n_queries=128`, `strategy=seed_grow`, seed 1, iteration 8/10; latest partial mean perf `0.590`, best `0.633`.
- Active GPU1: `n_queries=32`, `strategy=middle_entropy`, seed 1, iteration 3 generation; seed 1 partial after iteration 2 has mean perf `0.072`, best `0.450`.

Completed FT metrics:
- `n=8 random`: mean max `0.406`, mean perf `0.157`.
- `n=8 middle_entropy`: mean max `0.452`, mean perf `0.105`.
- `n=8 seed_grow`: mean max `0.418`, mean perf `0.243`.
- `n=16 random`: mean max `0.423`, mean perf `0.282`.
- `n=16 middle_entropy`: mean max `0.520`, mean perf `0.277`.
- `n=16 seed_grow`: mean max `0.428`, mean perf `0.319`.
- `n=32 random`: mean max `0.430`, mean perf `0.372`.
- `n=32 seed_grow`: mean max `0.473`, mean perf `0.381`.
- `n=64 seed_grow`: mean max `0.515`, mean perf `0.426`.
- `n=128 seed_grow`: seed 1 partial at iteration 8, max `0.633`, mean perf `0.590`, median `0.588`.

Timing and diagnostics:
- Full generation with `batch_size=256` is about `1.9-3.0 min` per optimization round depending on concurrent load.
- Fine-tuning time scales with replay size. Examples: 8 sequences for 5 epochs `~4s`; 256 sequences `~120-185s`; 1024 sequences currently takes several minutes per fine-tune.
- KL rises into a stable `~0.22-0.24` band for larger replay buffers with `lambda_kl=2`.
- Relative update norms remain small so far, roughly `0.003-0.005` in larger runs.

## Status Update 2026-05-30

### Active GPU allocation
- GPU0 RTX 2080 Ti: AAV zero-shot EvoDiff online FT, `seed_grow`, currently `n_queries=128`.
- GPU1 RTX 2080 Ti: AAV zero-shot EvoDiff online FT, currently `middle_entropy`, `n_queries=32`.
- GPU2 RTX 2080 Ti: idle.
- GPU3 A6000: idle.

### Latest active-run status
- AAV top100 one-hot: complete for all budgets, including `k=128`.
- AAV top100 frozen CLS: complete for all budgets, including `k=128`.
- AAV top100 layer2 LoRA CLS: complete for all budgets through `k=128`.
- LGK one-hot variable-k: complete for all budgets.
- LGK frozen CLS variable-k: complete for all budgets.
- AAV zero-shot ProSpero: complete for seeds `1..5`; mean max `0.441`, mean perf `0.369`.
- LGK zero-shot ProSpero: complete for seeds `1..5`; mean max `0.041`, mean perf `0.040`.
- AAV one-hot random-masking variable-k: `k=8`, `k=16`, `k=32`, and `k=64` complete; `k=128` has 3 valid seeds and seed 4 partial at iteration 2/10.
- AAV one-hot targeted-masking zero-score variable-k: complete for all budgets through `k=128`.
- AAV zero-shot fixed-budget K=4 routes: complete for all planned budgets `n_queries=8,16,32,64,128` and all three strategies.
- AAV zero-shot EvoDiff online FT: completed `n=8/16` all strategies, `n=32` random/seed_grow, `n=64` seed_grow; active `n=128 seed_grow` and `n=32 middle_entropy`.
- Latest error scan: no current crash/OOM/killed signatures for active runs.

### Zero-shot ProSpero implementation decisions
- Ranking score: cumulative `logP(sampled)-logP(original)`.
- Original residue probability is computed in the same distribution as the sampling step.
- Mask-count stats use selected targeted masks, calibrated live because old runs did not log the selected mask counts.
- Mask-count sampler uses rounded clipped Normal with empirical min/max.
- Resampling keeps the existing inverse-perplexity factor, replacing the surrogate score with the zero-shot score.

## Additional EvoDiff FT Sweeps 2026-05-30

Status: launched 2026-05-30.

Scope:
- All new sweeps use the online EvoDiff fine-tuning runner with `seed_grow` masking only, fixed mask budget `K=4`, variable budgets `n_queries=8,16,32,64,128`, 5 fine-tune epochs, LR `1e-5`, replay buffer `all`.
- Rationale: `seed_grow` is the strongest zero-shot/FT route so far; running all masking strategies for every task/KL would be much larger.

Launched active runs:
- AAV KL `0.1`: tmux `aav_zs_evodiff_ft_kl01_seedgrow_gpu2_20260530`, GPU2 RTX 2080 Ti UUID `GPU-6faf056b-00ab-15fd-e25c-a149c7dcf3d7`, output `outputs/aav_zero_shot_evodiff_ft_k4_variable_k_kl0p1_20260530`.
- LGK KL `2`: tmux `lgk_zs_evodiff_ft_kl2_seedgrow_a6000_20260530`, A6000 UUID `GPU-025e10e6-263f-d814-6dd5-added86fc8af`, output `outputs/lgk_zero_shot_evodiff_ft_k4_variable_k_kl2_20260530`.
- LGK KL `0.1`: tmux `lgk_zs_evodiff_ft_kl01_seedgrow_a6000_20260530`, A6000 UUID `GPU-025e10e6-263f-d814-6dd5-added86fc8af`, output `outputs/lgk_zero_shot_evodiff_ft_k4_variable_k_kl0p1_20260530`.

Queued runs:
- GFP KL `2`: tmux `gfp_zs_evodiff_ft_kl2_seedgrow_gpu0_queued_20260530`, queued behind `aav_zs_evodiff_ft_seedgrow_gpu0_20260529`, output `outputs/gfp_zero_shot_evodiff_ft_k4_variable_k_kl2_20260530`.
- Pab1 KL `2`: tmux `pab1_zs_evodiff_ft_kl2_seedgrow_gpu1_queued_20260530`, queued behind `aav_zs_evodiff_ft_random_middle_gpu1_20260529`, output `outputs/pab1_zero_shot_evodiff_ft_k4_variable_k_kl2_20260530`.

Initial validation:
- AAV KL `0.1` entered `n_queries=8`, seed 1 generation on GPU2.
- LGK KL `2` and KL `0.1` both loaded the LGK oracle and entered startup on the A6000.
- A6000 memory after launching both LGK runs: about `3.2 GB / 49 GB`; both processes active, sharing compute.

## Immediate GFP/Pab1 Launch 2026-05-30

Status: launched immediately after replacing the queued sessions.

Changes:
- Killed queued sessions `gfp_zs_evodiff_ft_kl2_seedgrow_gpu0_queued_20260530` and `pab1_zs_evodiff_ft_kl2_seedgrow_gpu1_queued_20260530`.
- Launched GFP KL `2` immediately on GPU2 alongside AAV KL `0.1`.
- Launched Pab1 KL `2` immediately on the A6000 alongside the LGK KL sweeps.

Active runs:
- GFP KL `2`: tmux `gfp_zs_evodiff_ft_kl2_seedgrow_gpu2_20260530`, GPU2 RTX 2080 Ti UUID `GPU-6faf056b-00ab-15fd-e25c-a149c7dcf3d7`, output `outputs/gfp_zero_shot_evodiff_ft_k4_variable_k_kl2_20260530`.
- Pab1 KL `2`: tmux `pab1_zs_evodiff_ft_kl2_seedgrow_a6000_20260530`, A6000 UUID `GPU-025e10e6-263f-d814-6dd5-added86fc8af`, output `outputs/pab1_zero_shot_evodiff_ft_k4_variable_k_kl2_20260530`.

Initial validation:
- GFP loaded the GFP oracle and entered startup on GPU2.
- Pab1 loaded the Pab1 oracle and entered iteration 1 generation on A6000.
- GPU2 utilization after launch: about `93%`, memory about `2.5 GB / 11 GB`.
- A6000 utilization after launch: about `99%`, memory about `3.1 GB / 49 GB`.

## 2026-05-31 debrief: online EvoDiff fine-tuning runs

Status check:
- `tmux ls` reported no server running.
- All GPUs idle: 3x RTX 2080 Ti + A6000 at 0% util / 1 MiB used.
- Failure scan over fine-tuning output roots found no traceback/error/OOM/killed signatures.
- All expected fine-tuning pickles reached iteration 10.

Output roots:
- AAV KL=2, all mask strategies: `outputs/aav_zero_shot_evodiff_ft_k4_variable_k_20260529`
- AAV KL=0.1, seed_grow: `outputs/aav_zero_shot_evodiff_ft_k4_variable_k_kl0p1_20260530`
- LGK KL=2, seed_grow: `outputs/lgk_zero_shot_evodiff_ft_k4_variable_k_kl2_20260530`
- LGK KL=0.1, seed_grow: `outputs/lgk_zero_shot_evodiff_ft_k4_variable_k_kl0p1_20260530`
- GFP KL=2, seed_grow: `outputs/gfp_zero_shot_evodiff_ft_k4_variable_k_kl2_20260530`
- Pab1 KL=2, seed_grow: `outputs/pab1_zero_shot_evodiff_ft_k4_variable_k_kl2_20260530`

AAV fixed K=4, online FT final metrics, mean over 5 seeds:
- KL=2 middle_entropy: n=8 perf 0.1053 best 0.4524; n=16 perf 0.2775 best 0.5198; n=32 perf 0.4531 best 0.5551; n=64 perf 0.5510 best 0.6045; n=128 perf 0.5979 best 0.6376.
- KL=2 random: n=8 perf 0.1569 best 0.4064; n=16 perf 0.2816 best 0.4226; n=32 perf 0.3722 best 0.4301; n=64 perf 0.3939 best 0.4408; n=128 perf 0.4415 best 0.5237.
- KL=2 seed_grow: n=8 perf 0.2433 best 0.4178; n=16 perf 0.3193 best 0.4279; n=32 perf 0.3811 best 0.4726; n=64 perf 0.4256 best 0.5146; n=128 perf 0.5784 best 0.6196.
- KL=0.1 seed_grow: n=8 perf 0.1968 best 0.4355; n=16 perf 0.3381 best 0.5609; n=32 perf 0.4969 best 0.5887; n=64 perf 0.5756 best 0.6094; n=128 perf 0.6256 best 0.6567.

AAV no-FT comparison from `outputs/aav_zero_shot_fixed_mask_k4_20260529`:
- seed_grow: n=8 perf 0.2156 best 0.4127; n=16 perf 0.3046 best 0.4176; n=32 perf 0.3692 best 0.4348; n=64 perf 0.3853 best 0.4428.
- middle_entropy: n=8 perf 0.0670 best 0.4090; n=16 perf 0.1080 best 0.4354; n=32 perf 0.2339 best 0.4461; n=64 perf 0.3162 best 0.4825.
- random: n=8 perf 0.1148 best 0.3731; n=16 perf 0.1971 best 0.3914; n=32 perf 0.2560 best 0.3979; n=64 perf 0.3398 best 0.4166.

Other datasets, seed_grow online FT final metrics, mean over 5 seeds:
- LGK KL=2: n=8 perf 0.0101 best 0.0340; n=16 perf 0.0282 best 0.0341; n=32 perf 0.0378 best 0.0403; n=64 perf 0.0405 best 0.0417; n=128 perf 0.0426 best 0.0432.
- LGK KL=0.1: n=8 perf 0.0197 best 0.0313; n=16 perf 0.0287 best 0.0350; n=32 perf 0.0387 best 0.0409; n=64 perf 0.0390 best 0.0402; n=128 perf 0.0420 best 0.0427.
- GFP KL=2: n=8 perf 3.2897 best 3.5843; n=16 perf 3.5711 best 3.5872; n=32 perf 3.5836 best 3.5915; n=64 perf 3.6025 best 3.6057; n=128 perf 3.6082 best 3.6112.
- Pab1 KL=2: n=8 perf 0.6581 best 0.8703; n=16 perf 0.7468 best 0.9146; n=32 perf 0.8626 best 0.9940; n=64 perf 1.0425 best 1.1316; n=128 perf 1.0917 best 1.1459.

Fine-tuning diagnostics:
- AAV KL=2: mean final-epoch FT step about 125s per update bundle; round 9 about 221s; mean KL 0.131 overall, 0.215 at round 9; relative update norm about 0.00243 overall.
- AAV KL=0.1: mean final-epoch FT step about 134s per update bundle; round 9 about 239s; mean KL 1.398 overall, 2.419 at round 9; relative update norm about 0.00243 overall.
- LGK/GFP/Pab1 KL=2 runs had similar relative update magnitudes, about 0.00117 to 0.00240 overall; no obvious gradient explosion in aggregate diagnostics.

Interpretation:
- Online EvoDiff FT materially improves AAV zero-shot relative to no-FT. Best AAV setting so far is KL=0.1 seed_grow at n=128: perf 0.6256, best 0.6567.
- AAV KL=2 middle_entropy also performs strongly at high budgets, n=128 perf 0.5979 / best 0.6376.
- Lower KL lets the model drift more, visible in higher KL diagnostics and better AAV performance in seed_grow, but this needs more sanity checks because it may trade plausibility/diversity for oracle exploitation.
- LGK FT is strong but not obviously sensitive between KL=2 and KL=0.1 at n=128.
- GFP and Pab1 completed cleanly under KL=2 and show monotonic budget scaling.

## 2026-05-31 launched remaining active EvoDiff FT KL=2 seed_grow

Launched remaining non-D-shift CNN benchmark scenarios with the same active fine-tuning config as AAV/LGK/GFP/Pab1:
- Config: zero-shot active EvoDiff FT, `KL=2`, `seed_grow`, fixed mask budget `K=4`, `n_queries=8 16 32 64 128`, seeds `1 2 3 4 5`, `n_iters=10`, `finetune_epochs=5`, `lr=1e-5`, replay `all`.
- `AMIE`: `outputs/amie_zero_shot_evodiff_ft_k4_variable_k_kl2_20260531`, tmux `zsft_amie_kl2_seedgrow`, GPU UUID `GPU-025e10e6-263f-d814-6dd5-added86fc8af`.
- `TEM`: `outputs/tem_zero_shot_evodiff_ft_k4_variable_k_kl2_20260531`, tmux `zsft_tem_kl2_seedgrow`, GPU UUID `GPU-fcb13561-e5da-20e1-2ff7-a9bdbfa68c26`.
- `UBE2I`: `outputs/ube2i_zero_shot_evodiff_ft_k4_variable_k_kl2_20260531`, tmux `zsft_ube2i_kl2_seedgrow`, GPU UUID `GPU-a3d54441-08da-ed66-fa0f-6aaaaf97baea`.
- `E4B`: `outputs/e4b_zero_shot_evodiff_ft_k4_variable_k_kl2_20260531`, tmux `zsft_e4b_kl2_seedgrow`, GPU UUID `GPU-6faf056b-00ab-15fd-e25c-a149c7dcf3d7`.

Also added `scripts/plot_rl_vs_og.py` and launched tmux watcher `rl_vs_og_plot_watcher`; it checks every 15 minutes and regenerates `outputs/rl_vs_og/rl_vs_original_prospero_cnn_variable_k.png` once all four new result sets are complete.

D-shift scenarios were not launched in this batch because matching vanilla CNN variable-k outputs are missing, and prior D-shift CNN variable-k attempts failed while loading the ESMFold oracle.

## 2026-06-01 N=128 trajectory plot

Added `scripts/plot_rl_vs_og_n128_trajectories.py` and generated `outputs/rl_vs_og/rl_vs_original_prospero_cnn_n128_trajectories.png`.
- Plot compares original CNN ProSpero vs active EvoDiff FT `KL=2 seed_grow K=4` at `n_queries=128` across optimization rounds.
- Uses mean top-100 fitness +/- SEM over available seeds at each round.
- Current plot includes partial active-FT trajectories for AMIE/E4B/TEM/UBE2I because those runs are still completing `n=128`.
- Restarted `rl_vs_og_plot_watcher`; when all four new active-FT result sets complete, it regenerates both side-by-side PNGs.

## 2026-06-01 plot title correction and K=8 trajectory plot

Corrected plot titles so `K` denotes the ProSpero query budget rather than EvoDiff mask budget.
- Regenerated `outputs/rl_vs_og/rl_vs_original_prospero_cnn_n128_trajectories.png` with title `K=128 Optimization Trajectories`.
- Added `scripts/plot_rl_vs_og_k8_trajectories.py` and generated `outputs/rl_vs_og/rl_vs_original_prospero_cnn_k8_trajectories.png`.
- Updated `rl_vs_og_plot_watcher` to regenerate side-by-side, K=128 trajectories, and K=8 trajectories when remaining active-FT runs complete.

## 2026-06-01 queued train-set pre-finetune top16 active EvoDiff FT

Implemented initial train-set pre-finetuning in `src/prospero/runners/run_zero_shot_finetune_evodiff.py`:
- New arg: `--pre_finetune_train_top_k`.
- When enabled, before optimization round 1 the runner selects the top-k sequences from the initial training split only, fine-tunes EvoDiff with the same rank-weighted masked-token NLL + KL objective, then runs the existing zero-shot active FT pipeline.
- Pre-finetune uses the same config as online FT: `epochs=5`, `lr=1e-5`, `lambda_kl=2`, `mask_budget=4`, random fixed-budget training masks, frozen original EvoDiff as KL base.
- Pre-finetune data is not inserted into the replay buffer; replay still starts from oracle-evaluated generated sequences.

Queued variable-k runs with `--pre_finetune_train_top_k 16`:
- AAV: `outputs/aav_zero_shot_evodiff_ft_pretrain_top16_k4_variable_k_kl2_20260601`, tmux `zsft_pretrain_top16_aav_kl2_seedgrow`, GPU UUID `GPU-025e10e6-263f-d814-6dd5-added86fc8af`. Queued to start after `zsft_amie_kl2_seedgrow` finishes.
- LGK: `outputs/lgk_zero_shot_evodiff_ft_pretrain_top16_k4_variable_k_kl2_20260601`, tmux `zsft_pretrain_top16_lgk_kl2_seedgrow`, GPU UUID `GPU-6faf056b-00ab-15fd-e25c-a149c7dcf3d7`. Queued to start after `zsft_e4b_kl2_seedgrow` finishes.

Config: `seed_grow`, mask budget 4, `KL=2`, budgets `8 16 32 64 128`, seeds `1..5`, `n_iters=10`, online replay `all`.

## 2026-06-01 launched pre-finetune top16 immediately

User requested immediate launch and LGK on A6000. Replaced the wait-only queued tmux sessions with active runs:
- AAV: tmux `zsft_pretrain_top16_aav_kl2_seedgrow`, GPU UUID `GPU-fcb13561-e5da-20e1-2ff7-a9bdbfa68c26`, output `outputs/aav_zero_shot_evodiff_ft_pretrain_top16_k4_variable_k_kl2_20260601`.
- LGK: tmux `zsft_pretrain_top16_lgk_kl2_seedgrow`, GPU UUID `GPU-025e10e6-263f-d814-6dd5-added86fc8af` (A6000), output `outputs/lgk_zero_shot_evodiff_ft_pretrain_top16_k4_variable_k_kl2_20260601`.

Both use `--pre_finetune_train_top_k 16` with the same active FT settings: `seed_grow`, mask budget 4, `KL=2`, budgets `8 16 32 64 128`, seeds `1..5`, `n_iters=10`.

## 2026-06-01 current run snapshot

`runs.md` tracking is still active. Current status:
- Active tmux sessions: `zsft_amie_kl2_seedgrow`, `zsft_tem_kl2_seedgrow`, `zsft_pretrain_top16_aav_kl2_seedgrow`, `zsft_pretrain_top16_lgk_kl2_seedgrow`, `rl_vs_og_plot_watcher`.
- Finished tmux sessions since previous launch: `zsft_e4b_kl2_seedgrow`, `zsft_ube2i_kl2_seedgrow`.
- GPU state: GPU0 RTX 2080 Ti and A6000 active; GPU1/GPU2 idle at the last check.

Remaining active FT batch:
- E4B: complete for all budgets/seeds.
- UBE2I: complete for all budgets/seeds.
- AMIE: complete through `n=64`; `n=128` is 4/5 complete, seed 5 at iter 7.
- TEM: complete through `n=64`; `n=128` is 4/5 complete, seed 5 at iter 7.

Pre-finetune top16 batch:
- AAV: `n=8` is 4/5 complete, seed 5 at iter 8; larger budgets not started yet.
- LGK: `n=8` is 3/5 complete, seed 4 at iter 9; larger budgets not started yet.

## 2026-06-01 ProteinGym submodule and CNN benchmark notes

Added ProteinGym as a git submodule:
- Path: `external/ProteinGym`
- Remote: `https://github.com/OATML-Markslab/ProteinGym`

ProteinGym benchmark structure:
- DMS substitution/indel assay CSVs are downloaded separately into `PROTEINGYM_CACHE`, default `$HOME/.cache/ProteinGym`.
- Substitution assays are indexed by `external/ProteinGym/reference_files/DMS_substitutions.csv`.
- Each DMS assay CSV has at least `mutant`, `mutated_sequence`, `DMS_score`, and `DMS_score_bin`.
- Zero-shot benchmark expects per-assay model score files under `DMS_output_score_folder_subs/<model_location>/<DMS_id>.csv`, then `proteingym/merge.py` merges them and `proteingym/performance_DMS_benchmarks.py` computes Spearman/AUC/MCC/NDCG/top-recall.
- Supervised benchmark expects per-CV-scheme score files under `<model_scores_location>/<cv_scheme>/<model_location>/<DMS_id>.csv`, with fields matching `config.json` supervised entries. Existing supervised entries use `mutated_sequence`, `predictions_fitness`, and `labels_fitness`.
- Supervised CV schemes for substitutions: `fold_random_5`, `fold_modulo_5`, `fold_contiguous_5`. Indels only use `fold_random_5`.

Likely ProSpero CNN integration route:
- Use ProteinGym supervised substitutions as the first target, not zero-shot, because ProSpero CNN ensembles are supervised regressors.
- Download only the required resources first: `DMS_ProteinGym_substitutions.zip` plus CV folds for substitutions. Full zero-shot model-score downloads are unnecessary unless comparing against all published baselines locally.
- For each DMS assay and CV split, train ProSpero one-hot CNN ensemble on the ProteinGym training fold and score the held-out/all assay variants.
- Emit score CSVs using the ProteinGym supervised schema: `mutated_sequence`, `predictions_fitness`, `labels_fitness`.
- Add a model entry to a local copied config, e.g. `ProSpero_CNN_ensemble`, with `input_score_name=predictions_fitness`, `label_name=labels_fitness`, `location=ProSpero_CNN_ensemble`, `key=mutated_sequence`, `model_type=One-hot CNN`.
- Run `proteingym/merge_supervised.py`, then `proteingym/performance_DMS_supervised_benchmarks.py` to get Spearman/MSE summaries.

Important caveats:
- Full ProteinGym substitution DMS data is ~1GB unzipped; CV folds are ~50MB + ~81MB. This is manageable.
- ProSpero CNN assumes fixed-length sequences; substitution benchmark is OK. Indels need a different model/input handling and should not be first.
- Training a CNN ensemble for every DMS assay and every CV scheme can be expensive but parallelizable across GPUs/CPUs. A pilot on ~5-10 assays should be run first.

## 2026-06-01 ProteinGym substitution-only ProSpero CNN targeted benchmark

Launched supervised substitution-only ProteinGym benchmark for ProSpero-relevant assays using the local ProSpero one-hot CNN ensemble scorer.
- Output: `outputs/proteingym_prospero_cnn_supervised_subs_targeted_20260601`
- Sessions: `proteingym_prospero_targeted_gpu1`, `proteingym_prospero_targeted_gpu2`
- GPUs: `GPU-a3d54441-08da-ed66-fa0f-6aaaaf97baea`, `GPU-6faf056b-00ab-15fd-e25c-a149c7dcf3d7`
- Assays: AMIE, LGK, UBC9/UBE2I, BLAT/TEM variants, UBE4B/E4B variants.
- Config: substitutions only, supervised ProteinGym folds, CV schemes `fold_random_5`, `fold_modulo_5`, `fold_contiguous_5`, ensemble size 5, max epochs 3000, patience 10, max variants 20000, max seq len 1200.
- Notes: AAV/GFP/Pab1 ProteinGym assays are larger than the current 20k variant cap and are intentionally deferred to heavier runs unless requested.

ProteinGym targeted benchmark watcher:
- Session: `proteingym_prospero_targeted_watcher`
- It waits for both benchmark shards to exit, then writes `raw_scores.csv`, `summary_by_cv.csv`, and `aggregate.log` in `outputs/proteingym_prospero_cnn_supervised_subs_targeted_20260601`.

## 2026-06-02 stopped-run debrief

All GPUs are idle and no experiment process is running. The runs that were active on 2026-06-01 completed their expected output files.

ProteinGym substitution-only targeted benchmark:
- Output: `outputs/proteingym_prospero_cnn_supervised_subs_targeted_20260601`
- Completed cleanly for 9 targeted substitution assays and 3 supervised CV schemes.
- Aggregate files: `raw_scores.csv`, `summary_by_cv.csv`, `aggregate.log`.
- Mean Spearman by CV: `fold_random_5=0.3610`, `fold_modulo_5=0.0457`, `fold_contiguous_5=0.0376`.

Pre-finetune top16 active EvoDiff FT, seed_grow, mask budget 4, KL=2:
- AAV output: `outputs/aav_zero_shot_evodiff_ft_pretrain_top16_k4_variable_k_kl2_20260601`; completed through `n=128`, seeds 1-5. Final mean top-100 performance / best: n=8 `0.2314/0.4088`, n=16 `0.3455/0.4207`, n=32 `0.3859/0.4649`, n=64 `0.4589/0.5448`, n=128 `0.5637/0.6191`.
- LGK output: `outputs/lgk_zero_shot_evodiff_ft_pretrain_top16_k4_variable_k_kl2_20260601`; completed through `n=128`, seeds 1-5. Final mean top-100 performance / best: n=8 `0.0218/0.0336`, n=16 `0.0298/0.0347`, n=32 `0.0373/0.0407`, n=64 `0.0408/0.0420`, n=128 `0.0416/0.0424`.

Remaining active FT KL=2 seed_grow runs from 2026-05-31:
- AMIE completed through `n=128`, seeds 1-5. Final mean top-100 performance / best: n=8 `-0.4228/0.2260`, n=16 `0.1531/0.2283`, n=32 `0.2263/0.2373`, n=64 `0.2406/0.2467`, n=128 `0.2562/0.2601`.
- TEM completed through `n=128`, seeds 1-5. Final mean top-100 performance / best: n=8 `0.6527/1.2280`, n=16 `0.9258/1.2279`, n=32 `1.0614/1.2285`, n=64 `1.1597/1.2285`, n=128 `1.2288/1.2316`.
- AMIE/TEM logs end with a shell quote error after final pickle completion; experiment outputs are complete, but the wrapper script has a trailing quote bug.

## 2026-06-02 one-hot ridge task easiness benchmark

Fixed future wrapper fragility in `scripts/run_aav_zero_shot_evodiff_ft_variable_k_20260529.sh` by passing `EXTRA_ARGS` through a bash array instead of raw unquoted expansion. `bash -n` passes.

Added and ran `scripts/run_one_hot_ridge_task_easiness.py`, a CPU-only ProSpero train/valid benchmark using pure flattened positional one-hot features and ridge regression.
- Output: `outputs/one_hot_ridge_task_easiness_20260602`
- Records: `one_hot_ridge_records.csv`
- Summary: `summary.json`
- Budgets: 16, 32, 64, 128, 256, 512, plus full train. Seeds: 1-5 for finite budgets.

Mean validation Spearman by task:
- E4B: 16 `0.0671`, 32 `0.1068`, 64 `0.1605`, 128 `0.2000`, 256 `0.1817`, 512 `0.1778`, full `0.4913`.
- AMIE: 16 `0.0679`, 32 `0.0723`, 64 `0.0943`, 128 `0.1477`, 256 `0.2612`, 512 `0.3859`, full `0.5858`.
- LGK: 16 `0.0303`, 32 `0.0377`, 64 `0.0660`, 128 `0.0775`, 256 `0.1454`, 512 `0.2615`, full `0.4884`.
- Pab1: 16 `0.1172`, 32 `0.2162`, 64 `0.2406`, 128 `0.2901`, 256 `0.2505`, 512 `0.2940`, full `0.6906`.
- TEM: 16 `0.1107`, 32 `0.1338`, 64 `0.2168`, 128 `0.3478`, 256 `0.4709`, 512 `0.5983`, full `0.7805`.
- UBE2I: 16 `0.1505`, 32 `0.2188`, 64 `0.2767`, 128 `0.4055`, 256 `0.5225`, 512 `0.6170`, full `0.7254`.
- GFP: 16 `0.0588`, 32 `0.0784`, 64 `0.1323`, 128 `0.1636`, 256 `0.2120`, 512 `0.2220`, full `0.3489`.
- AAV: 16 `0.0522`, 32 `0.0836`, 64 `0.1183`, 128 `0.1461`, 256 `0.1529`, 512 `0.2136`, full `1.0000`.
- D_SHIFT: 16 `0.4085`, 32 `0.4066`, 64 `0.5233`, 128 `0.5863`, 256 `0.7086`, 512 `0.7650`, full `0.9281`.
- D_SHIFT_SMALL: 16 `0.1746`, 32 `0.1812`, 64 `0.2543`, 128 `0.3693`, full `0.3774`.
- D_SHIFT_HARD: same as D_SHIFT in this dataset loadout.

Existing pure one-hot related artifacts found:
- LGK full/top-k one-hot ridge summaries: `outputs/lgk_one_hot_ridge_20260525/summary.json`, `outputs/lgk_one_hot_ridge_20260525_rerun/summary.json`.
- Script: `scripts/run_lgk_one_hot_ridge.py`.
- AAV/LGK single-mutant one-hot ridge delta test: `outputs/0423_experiments/single_mutant_energy_test_aav_lgk_one_hot_ridge_redo.json`.
- D_SHIFT single-mutant one-hot ridge delta test: `outputs/0426_experiments/single_mutant_energy_test_d_shift_one_hot_ridge.json`.

## 2026-06-02 EvoDiff zero-shot per-task alignment and generated-fitness histograms

Implemented and ran per-task EvoDiff zero-shot predictive scoring on ProSpero validation splits.
- Script: `scripts/score_evodiff_zero_shot_prospero_tasks.py`
- Output: `outputs/evodiff_zero_shot_alignment_20260602/predictive_scores`
- Log: `outputs/evodiff_zero_shot_alignment_20260602/predictive_scores.log`
- GPU: A6000 pinned by UUID `GPU-025e10e6-263f-d814-6dd5-added86fc8af`
- Session: `evodiff_zs_score_all`
- Status: completed cleanly; all GPUs idle afterward.
- Data: collapsed duplicate validation sequences; one CSV plus one summary JSON per task.
- Scores:
  - ProteinGym-style masked marginals: `sum_i log p(mut_i | WT with position i masked) - log p(wt_i | WT with position i masked)`.
  - Generation-path deterministic logP delta: start from WT, apply mutated residues left-to-right, and sum `log p(mut_i | current partial mutant) - log p(wt_i | current partial mutant)`.

Validation Spearman / Pearson:
- AAV: masked `0.0965 / 0.0910`, generation-path `0.1782 / 0.1719`, n=1531.
- AMIE: masked `0.2461 / 0.1634`, generation-path `0.2428 / 0.1571`, n=642.
- D_SHIFT: masked `0.0934 / 0.1492`, generation-path `0.2307 / 0.2275`, n=263.
- D_SHIFT_SMALL: masked `0.0030 / -0.0579`, generation-path `0.0887 / 0.0441`, n=20.
- D_SHIFT_HARD: masked `0.1090 / 0.1536`, generation-path `0.0571 / 0.0498`, n=263.
- E4B: masked `0.3204 / 0.3004`, generation-path `0.3271 / 0.3097`, n=1000.
- GFP: masked `-0.1161 / 0.0300`, generation-path `-0.1360 / 0.0634`, n=1020.
- LGK: masked `0.1064 / 0.1722`, generation-path `0.0679 / 0.1221`, n=764.
- Pab1: masked `-0.0659 / 0.2077`, generation-path `-0.0880 / 0.2071`, n=1000.
- TEM: masked `0.3756 / 0.3704`, generation-path `0.3395 / 0.3324`, n=520.
- UBE2I: masked `0.0492 / -0.0221`, generation-path `0.0190 / -0.0456`, n=303.

Implemented and generated stacked round-by-round fitness histograms for existing zero-shot generated candidates.
- Script: `scripts/plot_zero_shot_round_fitness_histograms.py`
- Output: `outputs/evodiff_zero_shot_alignment_20260602/fitness_histograms`
- Summary: `outputs/evodiff_zero_shot_alignment_20260602/fitness_histograms/histogram_summary.txt`
- Scope: seed_grow, mask budget K=4, KL=2, duplicate generated sequences collapsed per round.
- Budgets plotted: query budgets 8, 64, and 128.
- Tasks plotted: AAV, LGK, GFP, Pab1, AMIE, E4B, TEM, UBE2I.
- Count: 24 PNGs.

Correction for 2026-06-02 histogram plots:
- Initial histogram plotter had a discovery bug: if a root-level `strategy/task` directory existed, it returned before discovering `n_samples_*` directories. This made AAV budget-specific plots render FT-only even though no-FT data existed.
- Fixed `scripts/plot_zero_shot_round_fitness_histograms.py` to include root-level and `n_samples_*` directories, infer root-level query budget from metadata, and emit separate method-specific PNGs under `fitness_histograms/by_method`.
- Regenerated AAV budgets 8, 64, 128. Correct combined n128 file: `outputs/evodiff_zero_shot_alignment_20260602/fitness_histograms/AAV_n128_no_ft_ft_kl2_seed_grow_k4_kl2_histograms.png`.
- Stale FT-only file to avoid: `outputs/evodiff_zero_shot_alignment_20260602/fitness_histograms/AAV_n128_seed_grow_k4_kl2_histograms.png`.

## 2026-06-02 AAV traced zero-shot generation instrumentation

Implemented scalar-only generation/search tracing for zero-shot EvoDiff SMC.
- Trace writer: `src/prospero/debug_trace.py`
- Sampler hooks: `src/prospero/inference.py`
- No-FT runner flag: `--debug_generation_trace` in `src/prospero/runners/run_zero_shot_protein.py`
- FT runner flag: `--debug_generation_trace` in `src/prospero/runners/run_zero_shot_finetune_evodiff.py`
- Launchers:
  - `scripts/run_aav_zero_shot_trace_noft_variable_k_20260602.sh`
  - `scripts/run_aav_zero_shot_trace_ft_variable_k_20260602.sh`

Trace format:
- Per-seed compressed JSONL: `debug_traces/seed_<seed>.events.jsonl.gz`
- Per-seed summary: `debug_traces/seed_<seed>.trace_summary.json`
- Events include `mask_selected`, `smc_step`, `rollout_step`, `candidate`, `resample`, `candidate_selected_for_query`, and `oracle_query`.
- Stored scalar/provenance data only: sequences, mask positions/residues, sampled/original residues, logp_sampled, logp_original, log_delta, full_vocab_logp_sampled, zero_shot_score, log_likelihood, inverse perplexity, resampling parent/weight, selected/query rank, and oracle score. No logits or probability vectors are stored.

Smoke tests:
- No-FT `batch_size=8`, `n_queries=2`, `n_iters=1`: completed, trace summary produced, events valid.
- FT `batch_size=8`, `n_queries=2`, `n_iters=1`: completed, trace summary produced, events valid.
- No-FT `batch_size=256`, `n_queries=8`, `n_iters=1`: completed but slow, ~1m55s for one optimization round; trace size ~178 KB.
- No-FT `batch_size=64`, `n_queries=8`, `n_iters=1`: completed in ~33s; trace size ~47 KB.

Important runtime note:
- Exact previous production batch size was 256. Full AAV traced variable-query reruns at batch 256 would be many hours per method.
- For the first diagnostic pass, launched `batch_size=64` to iterate faster. This changes the SMC particle pool size, so these are diagnostic traces rather than exact production-config reproductions.

Active AAV trace runs launched:
- No-FT output: `outputs/aav_zero_shot_fixed_mask_k4_trace_batch64_20260602`
- No-FT log: `outputs/aav_zero_shot_fixed_mask_k4_trace_batch64_20260602.stdout.log`
- No-FT tmux session: `aav_trace_noft_b64`
- No-FT GPU: `GPU-fcb13561-e5da-20e1-2ff7-a9bdbfa68c26` (RTX 2080 Ti)
- FT KL=2 output: `outputs/aav_zero_shot_evodiff_ft_k4_variable_k_kl2_trace_batch64_20260602`
- FT KL=2 log: `outputs/aav_zero_shot_evodiff_ft_k4_variable_k_kl2_trace_batch64_20260602.stdout.log`
- FT KL=2 tmux session: `aav_trace_ft_b64`
- FT KL=2 GPU: `GPU-025e10e6-263f-d814-6dd5-added86fc8af` (A6000)
- Budgets: 8, 16, 32, 64, 128. Seeds: 1-5. Strategy: seed_grow. Mask budget: K=4. FT config: KL=2, lr=1e-5, 5 epochs, replay=all.

Plot clarification:
- Budget=8 histograms showing `n=40` are expected when aggregating across 5 seeds: 8 queried candidates x 5 seeds = 40 before duplicate collapse. Values like 39 mean a duplicate sequence was collapsed.

## 2026-06-02 trace-by-default policy and standardized-advantage FT

Trace policy update:
- Generation/search traces should be tracked by default on all future zero-shot generation runs unless explicitly disabled for a storage/runtime emergency.
- `run_zero_shot_protein.py` and `run_zero_shot_finetune_evodiff.py` now default `--debug_generation_trace` to true and expose `--no-debug_generation_trace` for explicit opt-out.
- Trace files remain scalar/provenance-only JSONL gzip; do not store logits or full probability vectors by default.

Implemented standardized-advantage EvoDiff FT reward mode.
- Code: `src/prospero/evodiff_finetune.py`, `src/prospero/runners/run_zero_shot_finetune_evodiff.py`
- CLI:
  - `--reward_mode standardized_advantage`
  - `--advantage_baseline moving_starting_sequence`
  - `--advantage_clip 2`
  - `--negative_weight 0.25`
- Baseline semantics: round 1 baseline is WT fitness because the starting sequence is WT; later rounds use the current selected starting sequence fitness.
- Reward: `adv = clip((oracle_score - baseline_fitness) / std(current_round_scores), -clip, clip)`.
- Loss uses signed weights: positive advantages reinforce likelihood; negative advantages penalize likelihood with `negative_weight`.
- Replay stores per-sequence advantages computed at query time so old sequences keep their original round-local baseline.

Smoke test:
- Output: `outputs/_smoke_aav_advantage_ft/seed_grow/AAV`
- Config: AAV, seed_grow K=4, KL=2, n_queries=4, n_iters=2, FT epochs=1, batch_size=32, finetune_batch_size=4.
- Status: completed cleanly with traces and reward metrics.
- Observation: round 1 AAV candidates were all below WT baseline fitness 0.5, so advantages clipped to `-2`; the loss was negative as expected for pure negative-reward learning signal.

Active standardized-advantage AAV run:
- Session: `aav_adv_ft_b64`
- Output: `outputs/aav_zero_shot_evodiff_ft_advantage_k4_variable_k_kl2_trace_batch64_20260602`
- Log: `outputs/aav_zero_shot_evodiff_ft_advantage_k4_variable_k_kl2_trace_batch64_20260602.stdout.log`
- GPU: `GPU-6faf056b-00ab-15fd-e25c-a149c7dcf3d7` (RTX 2080 Ti)
- Config: AAV, seed_grow, K=4, KL=2, lr=1e-5, 5 FT epochs, replay=all, batch_size=64, finetune_batch_size=16, budgets 8/16/32/64/128, seeds 1-5, traces on.
- Note: batch_size=64 is a diagnostic traced run, not an exact batch_size=256 production rerun.

## 2026-06-02 22:07 AAV traced run checkpoint

Status update:
- No-FT traced AAV batch64 run is complete for budgets 8/16/32/64/128, seeds 1-5.
- Rank-reward FT traced AAV batch64 run is still active in tmux session `aav_trace_ft_b64`.
  - Completed artifacts now include budgets 8/16/32/64 all seeds and budget 128 seeds 1-2.
  - Latest log shows budget 128 seed 2 in later optimization rounds; no failure observed.
- Standardized-advantage FT traced AAV batch64 run is still active in tmux session `aav_adv_ft_b64`.
  - Completed artifacts now include budgets 8/16/32/64 all seeds and budget 128 seed 1.
  - Latest log shows budget 128 seed 1 reached iteration 6/10 at 22:04; no failure observed.

Important observation for standardized-advantage AAV:
- The moving-starting-sequence baseline is staying at WT fitness 0.5 for current logged rounds because the selected starting sequence remains WT until a generated candidate exceeds WT.
- Most standardized advantages are negative and often clipped at -2, so the update is dominated by negative likelihood pressure with `negative_weight=0.25`.
- This matches the implemented rule, but if the intended baseline should move to the best generated/query sequence even when below WT, the baseline/update rule needs to be changed explicitly.

## 2026-06-03 AAV GRPO-style EvoDiff FT

Implemented GRPO-style per-round advantage normalization for online EvoDiff FT.
- Code: `src/prospero/evodiff_finetune.py`, `src/prospero/runners/run_zero_shot_finetune_evodiff.py`
- New CLI mode: `--reward_mode grpo_advantage`
- Advantage rule: `adv_i = clip((oracle_score_i - mean(round_scores)) / std(round_scores), -advantage_clip, advantage_clip)`.
- Loss path: same signed-advantage NLL as standardized advantage; positive advantages reinforce likelihood, negative advantages penalize likelihood with `negative_weight`.
- Replay policy: advantages are computed within each original query round and replayed with their original round-local normalization metadata.
- Trace policy remains default-on.

Smoke test:
- Session: `aav_grpo_smoke_20260603` (completed)
- Output: `outputs/_smoke_aav_grpo_ft_tmux/AAV/seed_1.pkl`
- Config: AAV, seed_grow K=4, n_queries=4, n_iters=2, FT epochs=1, KL=2, negative_weight=0.25.
- Result: completed cleanly with traces.
- First FT metadata showed `baseline_mode=group_mean`, mean-zero/std-one advantages, mixed positive and negative advantages (`frac_positive_advantage=0.75`, `frac_negative_advantage=0.25`).

Active run launched:
- Session: `aav_grpo_ft_b64`
- Output: `outputs/aav_zero_shot_evodiff_ft_grpo_k4_variable_k_kl2_trace_batch64_20260603`
- Log: `outputs/aav_zero_shot_evodiff_ft_grpo_k4_variable_k_kl2_trace_batch64_20260603.stdout.log`
- GPU pin: `CUDA_VISIBLE_DEVICES=GPU-a3d54441-08da-ed66-fa0f-6aaaaf97baea` (RTX 2080 Ti)
- Config: AAV, seed_grow, K=4, KL=2, lr=1e-5, 5 FT epochs, replay=all, batch_size=64, finetune_batch_size=16, budgets 8/16/32/64/128, seeds 1-5, traces on.
- Started: 2026-06-03T00:39:48+02:00.

Note:
- Direct PyTorch CUDA checks from the sandbox shell reported no devices, but tmux-launched jobs have working CUDA and the smoke run passed with UUID pinning. Continue using tmux for GPU runs.

## 2026-06-03 AAV mixed explore/exploit masking

Implemented and launched approach 3 masking for AAV.

Implementation:
- Code: `src/prospero/inference.py`, `src/prospero/runners/run_zero_shot_finetune_evodiff.py`
- New mask strategy: `--mask_strategy mixed_explore_exploit`
- For K=4, each particle mask is built as:
  - 2 exploit positions from positive online position-reward memory; before feedback exists, seed-grow/middle-entropy fallback is used.
  - 1 middle-entropy exploration position.
  - 1 anti-collapse random position weighted by `1/sqrt(1 + position_mask_count)`.
- Online position reward is updated after each oracle batch using round-centered fitness assigned to positions mutated relative to that round's starting sequence.
- Trace logging remains default-on, so mask choices, oracle queries, and selected candidates are recorded.

Smoke test:
- Session: `aav_mixed_smoke_20260603` (completed)
- Output: `outputs/_smoke_aav_mixed_ft/AAV/seed_1.pkl`
- Config: AAV, mixed_explore_exploit, K=4, n_queries=4, n_iters=2, rank FT, FT epochs=1.
- Result: completed cleanly with trace writing and FT metrics.

Active runs launched:
- Rank FT mixed masking:
  - Session: `aav_mixed_rank_ft_b64`
  - Output: `outputs/aav_zero_shot_evodiff_ft_rank_mixed_k4_variable_k_kl2_trace_batch64_20260603`
  - Log: `outputs/aav_zero_shot_evodiff_ft_rank_mixed_k4_variable_k_kl2_trace_batch64_20260603.stdout.log`
  - GPU pin: `CUDA_VISIBLE_DEVICES=GPU-fcb13561-e5da-20e1-2ff7-a9bdbfa68c26` (RTX 2080 Ti)
  - Config: AAV, mixed_explore_exploit, K=4, KL=2, lr=1e-5, 5 FT epochs, replay=all, batch_size=64, finetune_batch_size=16, budgets 8/16/32/64/128, seeds 1-5.
- GRPO FT mixed masking:
  - Session: `aav_mixed_grpo_ft_b64`
  - Output: `outputs/aav_zero_shot_evodiff_ft_grpo_mixed_k4_variable_k_kl2_trace_batch64_20260603`
  - Log: `outputs/aav_zero_shot_evodiff_ft_grpo_mixed_k4_variable_k_kl2_trace_batch64_20260603.stdout.log`
  - GPU pin: `CUDA_VISIBLE_DEVICES=GPU-025e10e6-263f-d814-6dd5-added86fc8af` (A6000)
  - Config: AAV, mixed_explore_exploit, K=4, KL=2, lr=1e-5, 5 FT epochs, replay=all, reward_mode=grpo_advantage, advantage_clip=2, negative_weight=0.25, batch_size=64, finetune_batch_size=16, budgets 8/16/32/64/128, seeds 1-5.
- Both started at 2026-06-03T00:50:52+02:00 and reached trace-writing/generation for budget 8 seed 1.

## 2026-06-03 AAV mixed masking status/results checkpoint

Status checkpoint:
- `aav_mixed_rank_ft_b64` completed cleanly at 2026-06-03T11:25:18+02:00.
- `aav_mixed_grpo_ft_b64` is still active on AAV budget 128 seed 5; completed all lower budgets and budget 128 seeds 1-4.
- `aav_grpo_ft_b64` is also still active on AAV seed-grow GRPO budget 128 seed 5; completed all lower budgets and budget 128 seeds 1-4.

AAV final iteration means from logs, batch64 traced runs:
- no-FT seed-grow: n8 0.188, n16 0.273, n32 0.311, n64 0.374, n128 0.404.
- rank FT seed-grow: n8 0.204, n16 0.320, n32 0.410, n64 0.511, n128 0.575.
- rank FT mixed masking: n8 0.164, n16 0.320, n32 0.452, n64 0.543, n128 0.616.
- GRPO FT seed-grow: n8 0.179, n16 0.250, n32 0.361, n64 0.401, n128 0.510 over 4 completed seeds.
- GRPO FT mixed masking: n8 0.143, n16 0.279, n32 0.355, n64 0.470, n128 0.522 over 4 completed seeds.

Interpretation:
- Mixed masking helps the old/rank FT strongly at medium/high budgets: +0.043 at n32, +0.032 at n64, +0.042 at n128 vs rank seed-grow.
- Mixed masking hurts low-budget n8 for both rank and GRPO.
- GRPO remains weaker than rank FT, but mixed masking improves GRPO at n64 and n128 compared with GRPO seed-grow so far.

## 2026-06-03 AAV n128 negative-weight ablation

Implemented bottom-quantile negative reward mode and launched four AAV n=128 mixed-masking ablations in parallel.

Implementation:
- Code: `src/prospero/evodiff_finetune.py`, `src/prospero/runners/run_zero_shot_finetune_evodiff.py`
- New mode: `--reward_mode bottom_quantile_negative`
- Reward: positive normalized-rank weights for most candidates, but bottom `--bottom_quantile` of each oracle batch gets advantage `-1` and is penalized by `--negative_weight`.
- Default bottom quantile used for this ablation: `0.25`.

Active runs:
- `aav_mixed_grpo_neg05_n128`
  - Output: `outputs/aav_zero_shot_evodiff_ft_grpo_mixed_k4_n128_kl2_neg05_trace_batch64_20260603`
  - Log: `outputs/aav_zero_shot_evodiff_ft_grpo_mixed_k4_n128_kl2_neg05_trace_batch64_20260603.stdout.log`
  - GPU: `GPU-fcb13561-e5da-20e1-2ff7-a9bdbfa68c26`
  - Config: AAV, n_queries=128, mixed_explore_exploit, reward_mode=grpo_advantage, negative_weight=0.5.
- `aav_mixed_grpo_neg10_n128`
  - Output: `outputs/aav_zero_shot_evodiff_ft_grpo_mixed_k4_n128_kl2_neg10_trace_batch64_20260603`
  - Log: `outputs/aav_zero_shot_evodiff_ft_grpo_mixed_k4_n128_kl2_neg10_trace_batch64_20260603.stdout.log`
  - GPU: `GPU-a3d54441-08da-ed66-fa0f-6aaaaf97baea`
  - Config: AAV, n_queries=128, mixed_explore_exploit, reward_mode=grpo_advantage, negative_weight=1.0.
- `aav_mixed_bottomq_neg05_n128`
  - Output: `outputs/aav_zero_shot_evodiff_ft_bottomq_mixed_k4_n128_kl2_neg05_trace_batch64_20260603`
  - Log: `outputs/aav_zero_shot_evodiff_ft_bottomq_mixed_k4_n128_kl2_neg05_trace_batch64_20260603.stdout.log`
  - GPU: `GPU-6faf056b-00ab-15fd-e25c-a149c7dcf3d7`
  - Config: AAV, n_queries=128, mixed_explore_exploit, reward_mode=bottom_quantile_negative, bottom_quantile=0.25, negative_weight=0.5.
- `aav_mixed_bottomq_neg10_n128`
  - Output: `outputs/aav_zero_shot_evodiff_ft_bottomq_mixed_k4_n128_kl2_neg10_trace_batch64_20260603`
  - Log: `outputs/aav_zero_shot_evodiff_ft_bottomq_mixed_k4_n128_kl2_neg10_trace_batch64_20260603.stdout.log`
  - GPU: `GPU-025e10e6-263f-d814-6dd5-added86fc8af`
  - Config: AAV, n_queries=128, mixed_explore_exploit, reward_mode=bottom_quantile_negative, bottom_quantile=0.25, negative_weight=1.0.

All four started at 2026-06-03T11:38:00+02:00 and reached trace-writing/generation for seed 1.

## 2026-06-03 AAV full-vocabulary SMC ablation

Implemented and launched the biological-constraint ablation for the best rank-based active FT setup.

Implementation:
- Code: `src/prospero/inference.py`, `src/prospero/runners/run_zero_shot_finetune_evodiff.py`
- New CLI: `--smc_vocab {cluster,full}`
- Default remains `cluster`, matching previous biologically constrained SMC behavior.
- `--smc_vocab full` changes only the main SMC unmasking step to sample from all 20 amino acids. Rollouts already used the full 20-AA vocabulary.
- Trace `smc_step.distribution` records `full_20aa` for this ablation.

Active run:
- Session: `aav_mixed_rank_fullsmc_n128`
- Output: `outputs/aav_zero_shot_evodiff_ft_rank_mixed_fullsmc_k4_n128_kl2_trace_batch64_20260603`
- Log: `outputs/aav_zero_shot_evodiff_ft_rank_mixed_fullsmc_k4_n128_kl2_trace_batch64_20260603.stdout.log`
- GPU pin: `CUDA_VISIBLE_DEVICES=GPU-025e10e6-263f-d814-6dd5-added86fc8af` (A6000)
- Config: AAV, n_queries=128, mixed_explore_exploit, K=4, reward_mode=rank, KL=2, lr=1e-5, 5 FT epochs, replay=all, batch_size=64, finetune_batch_size=16, seeds 1-5, traces on, `--smc_vocab full`.
- Started: 2026-06-03T12:13:35+02:00. Reached trace-writing/generation for seed 1.

## 2026-06-03 AAV no-rollout sequential SMC ablations

Implemented and launched two AAV n=128 ablations to isolate rollout effects from biological-vocabulary constraints.

Implementation:
- Code: `src/prospero/inference.py`, `src/prospero/runners/run_zero_shot_finetune_evodiff.py`
- New CLI: `--zero_shot_generation_mode {smc_rollout,no_rollout_sequential}`
- `no_rollout_sequential` samples the K masked residues sequentially and ranks only terminal completed particles.
- In `no_rollout_sequential`, rollout completions are disabled and resampling is disabled; trace candidates use `stage=terminal_no_rollout`.
- Existing `smc_rollout` behavior remains unchanged for baseline comparison.
- Script: `scripts/run_aav_mixed_rank_ft_no_rollout_n128_20260603.sh`

Active runs:
- `aav_no_rollout_cluster_n128`
  - Output: `outputs/aav_zero_shot_evodiff_ft_rank_mixed_no_rollout_cluster_k4_n128_kl2_trace_batch64_20260603`
  - Log: `outputs/aav_zero_shot_evodiff_ft_rank_mixed_no_rollout_cluster_k4_n128_kl2_trace_batch64_20260603.stdout.log`
  - GPU pin: `CUDA_VISIBLE_DEVICES=GPU-fcb13561-e5da-20e1-2ff7-a9bdbfa68c26` (RTX 2080 Ti)
  - Config: AAV, n_queries=128, mixed_explore_exploit, K=4, reward_mode=rank, KL=2, lr=1e-5, 5 FT epochs, replay=all, batch_size=64, finetune_batch_size=16, seeds 1-5, traces on, `--smc_vocab cluster`, `--zero_shot_generation_mode no_rollout_sequential`.
- `aav_no_rollout_full_n128`
  - Output: `outputs/aav_zero_shot_evodiff_ft_rank_mixed_no_rollout_full_k4_n128_kl2_trace_batch64_20260603`
  - Log: `outputs/aav_zero_shot_evodiff_ft_rank_mixed_no_rollout_full_k4_n128_kl2_trace_batch64_20260603.stdout.log`
  - GPU pin: `CUDA_VISIBLE_DEVICES=GPU-a3d54441-08da-ed66-fa0f-6aaaaf97baea` (RTX 2080 Ti)
  - Config: AAV, n_queries=128, mixed_explore_exploit, K=4, reward_mode=rank, KL=2, lr=1e-5, 5 FT epochs, replay=all, batch_size=64, finetune_batch_size=16, seeds 1-5, traces on, `--smc_vocab full`, `--zero_shot_generation_mode no_rollout_sequential`.

Both started at 2026-06-03T14:33:49+02:00 and reached trace-writing/generation for seed 1.

Restart/update for AAV no-rollout sequential ablations:
- Initial launch at 2026-06-03T14:33:49+02:00 failed after the first terminal batch because `no_rollout_sequential` passed the full 2D token matrix to `tokenizer.untokenize`, which expects one sequence at a time.
- Fixed in `src/prospero/inference.py` by untokenizing each sampled sequence row independently.
- Relaunched both sessions at 2026-06-03T14:35:20+02:00 on the same 2080 Ti UUIDs.
- Health check: both runs cleared the previous crash point, generated the first 64 terminal candidates, and entered generation round 2 for seed 1.

## 2026-06-03 RL vs original CNN mean-max plots

Regenerated the `outputs/rl_vs_og` comparisons using mean max fitness (`Best score`) instead of top-100 mean fitness (`Performance`). Active AAV ablation folders were not included/touched; plots use the existing completed seed_grow KL=2 zero-shot FT outputs and original ProSpero CNN outputs from the prior comparison scripts.

Generated files:
- `outputs/rl_vs_og/rl_vs_original_prospero_cnn_variable_k_mean_max.png`
- `outputs/rl_vs_og/rl_vs_original_prospero_cnn_k128_mean_max_trajectories.png`
- `outputs/rl_vs_og/rl_vs_original_prospero_cnn_k8_mean_max_trajectories.png`

Notes:
- K=128 trajectory and K=8 trajectory plot the mean over available seeds of per-round `Best score`.
- Variable-K plot uses final-round `Best score` per budget.
- Original CNN GFP has only 4 completed seeds at K=128; all other listed comparisons are 5/5 seeds.

## 2026-06-03 RL vs original CNN multi zero-shot config plot

Generated a K=128 mean-max trajectory plot that keeps zero-shot configs separate rather than averaging across them.

Generated file:
- `outputs/rl_vs_og/rl_vs_original_cnn_zero_shot_configs_k128_mean_max_trajectories.png`

Included configs:
- All tasks: original ProSpero CNN and completed FT seed_grow KL=2 where available.
- AAV additionally: no-FT seed_grow, traced FT seed_grow KL=2, FT mixed rank KL=2, FT GRPO seed_grow KL=2, FT GRPO mixed KL=2.
- LGK additionally: FT seed_grow KL=0.1 and pretrain-top16 + FT seed_grow KL=2.

Excluded active AAV ablation roots:
- bottom-quantile negative n128 runs
- GRPO negative-weight n128 runs
- full-vocabulary SMC n128 run
- no-rollout cluster/full n128 runs

Metric:
- Per-round `Best score`, averaged over available completed seeds, with SEM bands.
- Original CNN GFP still has only 4 seeds at K=128; all other plotted series are 5/5 seeds.

## 2026-06-03 AAV ablation status checkpoint 17:40

Active sessions:
- `aav_no_rollout_cluster_n128`
- `aav_no_rollout_full_n128`

Finished sessions since prior checkpoint:
- Four negative-weight n=128 ablations completed 5/5 seeds.
- Full-vocabulary SMC n=128 ablation completed 5/5 seeds.

Final n=128 AAV metrics, mean +/- SEM over completed seeds:
- baseline rank mixed cluster rollout: mean best `0.6662 +/- 0.0104`, mean top100 `0.6161 +/- 0.0084`, 5/5 seeds.
- baseline GRPO mixed cluster rollout: mean best `0.5680 +/- 0.0095`, mean top100 `0.5221 +/- 0.0090`, 5/5 seeds.
- GRPO mixed negative_weight=0.5: mean best `0.5820 +/- 0.0158`, mean top100 `0.5270 +/- 0.0116`, 5/5 seeds.
- GRPO mixed negative_weight=1.0: mean best `0.5562 +/- 0.0086`, mean top100 `0.4830 +/- 0.0142`, 5/5 seeds.
- bottom-quantile mixed negative_weight=0.5: mean best `0.6178 +/- 0.0188`, mean top100 `0.5635 +/- 0.0179`, 5/5 seeds.
- bottom-quantile mixed negative_weight=1.0: mean best `0.5890 +/- 0.0152`, mean top100 `0.5310 +/- 0.0160`, 5/5 seeds.
- rank mixed full-SMC-vocab: mean best `0.5893 +/- 0.0205`, mean top100 `0.5393 +/- 0.0169`, 5/5 seeds.
- rank mixed no-rollout cluster: mean best `0.6608 +/- 0.0087`, mean top100 `0.6282 +/- 0.0087`, 3/5 completed seeds; seed 4 at iteration 3.
- rank mixed no-rollout full: mean best `0.5728 +/- 0.0123`, mean top100 `0.5084 +/- 0.0057`, 3/5 completed seeds; seed 4 at iteration 4.

Interpretation at this checkpoint:
- Bottom-quantile negative with weight 0.5 is the best completed negative-reward variant, but it remains below the rank mixed cluster rollout baseline on mean best.
- Full 20-AA SMC under the rollout path is worse than constrained cluster SMC.
- No-rollout cluster is very competitive on partial 3/5 data and slightly above rollout baseline on top100 mean, but final comparison should wait for 5/5 seeds.
- No-rollout full is clearly weaker than no-rollout cluster so far.

## 2026-06-03 ESM-2 650M AAV smoke benchmark

Tested `esm2_t33_650M_UR50D` as a masked-logit provider on the A6000 for AAV K=4 terminal constrained decoding.

Script:
- `scripts/smoke_esm2_650m_aav_decode.py`

Outputs:
- `outputs/esm2_650m_smoke_aav_20260603/b64_cluster.json`
- `outputs/esm2_650m_smoke_aav_20260603/b128_cluster.json`

Setup:
- GPU: `GPU-025e10e6-263f-d814-6dd5-added86fc8af` (RTX A6000)
- Model: `esm2_t33_650M_UR50D`
- Params: `651,043,254`
- Task: AAV
- Masks: first traced mixed-explore-exploit masks from the best rank-mixed baseline trace; for batch 128, the 64 first-round masks are repeated for throughput measurement.
- Decode: terminal no-rollout K=4, biological `CHARGE` cluster vocabulary, no fine-tuning, no SMC rollout/resampling.

Results:
- Batch 64:
  - First run included model/checkpoint download.
  - Load/download seconds: `57.41`
  - Decode seconds: `2.17`
  - Mean forward seconds per unmask step: `0.513`
  - Seconds per candidate: `0.0340`
  - Peak CUDA allocated: `3.27 GB`
  - Oracle best in one generated batch: `0.4919`
  - Unique sequences: `64/64`
- Batch 128:
  - Cached load seconds: `7.13`
  - Decode seconds: `4.27`
  - Mean forward seconds per unmask step: `1.022`
  - Seconds per candidate: `0.0333`
  - Peak CUDA allocated: `3.84 GB`
  - Oracle best in one generated batch: `0.4850`
  - Unique sequences: `128/128`

Interpretation:
- ESM-2 650M is feasible on the A6000 for AAV-length masked decoding.
- It is not a literal drop-in for current EvoDiff active fine-tuning, but it is close to drop-in for the no-rollout masked-logit generator/scorer interface.
- Full integration needs an ESM tokenizer/logit adapter and a separate fine-tuning/KL implementation if we want active FT.

## 2026-06-03 ESM-2 650M validation zero-shot alignment benchmark

Implemented and ran ESM-2 masked-marginal zero-shot alignment on ProSpero validation splits.

Script:
- `scripts/score_esm2_zero_shot_prospero_tasks.py`

Output:
- `outputs/esm2_650m_zero_shot_alignment_20260603/predictive_scores`
- Log: `outputs/esm2_650m_zero_shot_alignment_20260603/predictive_scores.log`

Run details:
- Model: `esm2_t33_650M_UR50D`
- Params: `651,043,254`
- GPU: A6000 pinned by UUID `GPU-025e10e6-263f-d814-6dd5-added86fc8af`
- Method: ProteinGym-style masked marginals, `sum_i log p(mut_i | WT masked at i) - log p(wt_i | WT masked at i)`.
- Split: collapsed ProSpero validation sequences.
- Model cached load time: `7.40s`.
- A6000 memory during run: about `8.4GB` used by nvidia-smi.

Validation Spearman, ESM-2 650M masked marginal vs EvoDiff masked marginal:
- AAV: ESM-2 `0.0861`, EvoDiff `0.0965`, delta `-0.0104`.
- AMIE: ESM-2 `0.3617`, EvoDiff `0.2461`, delta `+0.1156`.
- E4B: ESM-2 `0.2928`, EvoDiff `0.3204`, delta `-0.0276`.
- GFP: ESM-2 `-0.1297`, EvoDiff `-0.1161`, delta `-0.0136`.
- LGK: ESM-2 `0.3647`, EvoDiff `0.1064`, delta `+0.2583`.
- Pab1: ESM-2 `-0.0523`, EvoDiff `-0.0659`, delta `+0.0137`.
- TEM: ESM-2 `0.7124`, EvoDiff `0.3756`, delta `+0.3368`.
- UBE2I: ESM-2 `0.5022`, EvoDiff `0.0492`, delta `+0.4530`.

Interpretation:
- ESM-2 650M is much better aligned than EvoDiff on LGK, TEM, UBE2I, and AMIE.
- ESM-2 is roughly tied/slightly worse on AAV, E4B, GFP, and only slightly better on Pab1.
- For replacing EvoDiff in the optimization loop, ESM-2 looks especially promising for LGK/TEM/UBE2I/AMIE, but not obviously for AAV based on global validation alignment.

## 2026-06-03 ProSST zero-shot validation alignment benchmark

Implemented and ran a modular ProSST zero-shot alignment path without installing AlphaFold/OpenFold/PyG into the main environment.

Code/assets:
- Upstream repo clone: `external/ProSST`.
- Script: `scripts/score_prosst_zero_shot_prospero_tasks.py`.
- Outputs: `outputs/prosst_zero_shot_alignment_20260603/predictive_scores`.
- Log: `outputs/prosst_zero_shot_alignment_20260603/predictive_scores.log`.
- Precomputed ProSST ProteinGym structure-token bundle: `outputs/prosst_zero_shot_alignment_20260603/proteingym_benchmark`.
- ProteinGym AF2 PDB archive also cached at `outputs/prosst_zero_shot_alignment_20260603/ProteinGym_v1_AlphaFold2_PDB.zip`, but not needed for scoring because precomputed structure tokens were available.

Run details:
- GPU: A6000 pinned by UUID `GPU-025e10e6-263f-d814-6dd5-added86fc8af`.
- Model: `AI4Protein/ProSST-2048`.
- Params: `117,146,880`.
- Method: ProSST/ProteinGym full-WT forward log-odds, `sum_i log p(mut_i | WT, structure) - log p(wt_i | WT, structure)` over mutated positions.
- Split: collapsed ProSpero validation sequences.
- Model cached load time: `4.42s`.
- No AlphaFold/OpenFold/PyG install was needed.

Structure mapping notes:
- AAV and GFP are exact mappings.
- AMIE, TEM, UBE2I, Pab1, E4B use close ProteinGym structure mappings with identity >= 0.951.
- LGK uses `LGK_LIPST_Klesmith_2015` structure tokens for positions 1-439; ProSpero LGK has 8 C-terminal residues without ProteinGym structure tokens, so mutations there are ignored.

Validation Spearman/Pearson for ProSST log-odds:
- AAV: `0.3411 / 0.3592`.
- AMIE: `0.4098 / 0.2622`.
- E4B: `0.3661 / 0.3567`.
- GFP: `0.2887 / 0.2570`.
- LGK: `0.4588 / 0.3478`.
- Pab1: `-0.0559 / 0.2531`.
- TEM: `0.7494 / 0.6989`.
- UBE2I: `0.4177 / 0.5144`.

Comparison against prior validation Spearman:
- AAV: ProSST `0.3411`, ESM-2 650M `0.0861`, EvoDiff `0.0965`.
- AMIE: ProSST `0.4098`, ESM-2 650M `0.3617`, EvoDiff `0.2461`.
- E4B: ProSST `0.3661`, ESM-2 650M `0.2928`, EvoDiff `0.3204`.
- GFP: ProSST `0.2887`, ESM-2 650M `-0.1297`, EvoDiff `-0.1161`.
- LGK: ProSST `0.4588`, ESM-2 650M `0.3647`, EvoDiff `0.1064`.
- Pab1: ProSST `-0.0559`, ESM-2 650M `-0.0523`, EvoDiff `-0.0659`.
- TEM: ProSST `0.7494`, ESM-2 650M `0.7124`, EvoDiff `0.3756`.
- UBE2I: ProSST `0.4177`, ESM-2 650M `0.5022`, EvoDiff `0.0492`.

Interpretation:
- ProSST is the best of the tested zero-shot alignment models for AAV, AMIE, E4B, GFP, LGK, and TEM on this validation benchmark.
- ESM-2 remains best on UBE2I.
- Pab1 is not aligned for any of these zero-shot methods.
- The ProSST path is modular/deletable: remove `external/ProSST`, `scripts/score_prosst_zero_shot_prospero_tasks.py`, and `outputs/prosst_zero_shot_alignment_20260603` to clean it out.

## 2026-06-03 AAV ProSST zero-shot optimizer launch

Launched ProSST as a modular AAV zero-shot generator/scorer.

Script:
- `scripts/run_aav_zero_shot_prosst.py`
- launcher: `scripts/run_aav_zero_shot_prosst_n128_20260603.sh`

Output:
- `outputs/aav_zero_shot_prosst_mixed_k4_n128_20260603`

Run config:
- Task: AAV
- Model: `AI4Protein/ProSST-2048`
- GPU: A6000 pinned by UUID `GPU-025e10e6-263f-d814-6dd5-added86fc8af`
- Seeds: 1-5 sequential in tmux session `aav_prosst_mixed_k4_n128`
- Queries: `n=128`
- Rounds: `10`
- Masking: `mixed_explore_exploit`
- Mask budget: `K=4`
- Main decoding vocabulary: `CHARGE` constrained cluster
- Generation mode: terminal no-rollout sequential masked decoding
- Score: cumulative ProSST logP(sampled residue) - logP(original residue)
- Trace logging: enabled

Smoke test before launch:
- Output: `outputs/aav_zero_shot_prosst_smoke_20260603`
- Config: seed 1, `n=8`, `2` rounds, batch 16
- Best score reached by round 2: `0.4757`

## 2026-06-03 AAV ProSST zero-shot optimizer completed

Output:
- `outputs/aav_zero_shot_prosst_mixed_k4_n128_20260603`

Run status:
- Completed seeds 1-5.
- tmux session `aav_prosst_mixed_k4_n128` exited normally.

Final round 10 metrics, mean +/- SEM over 5 seeds:
- Best score: `0.5750 +/- 0.0102`.
- Top-100 mean fitness / `Performance`: `0.5277 +/- 0.0083`.
- Top-100 median fitness: `0.5246 +/- 0.0081`.

Per-seed final round 10:
- Seed 1: best `0.5614`, top100 `0.5223`, median `0.5189`.
- Seed 2: best `0.5654`, top100 `0.5160`, median `0.5149`.
- Seed 3: best `0.6067`, top100 `0.5522`, median `0.5489`.
- Seed 4: best `0.5902`, top100 `0.5410`, median `0.5370`.
- Seed 5: best `0.5514`, top100 `0.5069`, median `0.5035`.

Config:
- Model: `AI4Protein/ProSST-2048`.
- Structure tokens: exact AAV crop from `CAPSD_AAV2S_Sinai_2021`, offset 450, length 90.
- Generation: terminal no-rollout sequential masked decoding.
- Masking: `mixed_explore_exploit`.
- Mask budget: `K=4`.
- Query budget: `n=128`, 10 rounds.
- SMC vocab: constrained `CHARGE` cluster.
- Score: cumulative ProSST `logP(sampled residue) - logP(original residue)`.
- Traces: enabled under `AAV/debug_traces`.

Interpretation:
- ProSST zero-shot optimizer is substantially stronger than earlier no-FT EvoDiff zero-shot on AAV.
- It is below the best AAV active EvoDiff fine-tuning mixed rank baseline on final best score (`0.5750` vs prior `0.6662`) and top100 mean (`0.5277` vs prior `0.6161`).
- It is competitive with weaker active FT ablations and much better than the GRPO mixed run (`0.5680` best, `0.5221` top100), while requiring no online fine-tuning.

## 2026-06-03 AAV ProSST fine-tuning smoke tests and full launch

Implemented ProSST active fine-tuning in `scripts/run_aav_zero_shot_prosst.py` behind `--finetune_prosst`.

Fine-tuning objective:
- Rank-weighted masked-token NLL on oracle-queried sequences.
- Random fixed-budget training masks with `K=4`.
- KL regularization to a frozen base ProSST on the same random masked inputs.
- Generation/search remains `mixed_explore_exploit`, constrained `CHARGE`, terminal no-rollout sequential ProSST decoding.

Tiny smoke, seed 1, n=8, 2 rounds, 2 FT epochs:
- No-FT earlier smoke round 2: best `0.4757`, top100/performance `0.1383`, median `0.0736`.
- FT lr=1e-5 KL=2: round 2 best `0.5176`, performance `0.1511`, median `0.1001`; final KL `0.0500`, rel update `0.000334`.
- FT lr=3e-5 KL=2: round 2 best `0.5176`, performance `0.1568`, median `0.1674`; final KL `0.0997`, rel update `0.000988`.

Larger smoke, seed 1, n=32, 3 rounds, 3 FT epochs:
- FT lr=1e-5 KL=2: round 3 best `0.5337`, performance `0.2324`, median `0.2332`; final KL `0.1738`, rel update `0.001194`.
- FT lr=3e-5 KL=2: round 3 best `0.5358`, performance `0.2618`, median `0.2671`; final KL `0.2580`, rel update `0.003152`.

Interpretation:
- ProSST FT is numerically stable in smoke tests.
- lr=3e-5 gave better population quality with still conservative KL/update drift, so it was selected for the first full run.

Full run launched:
- tmux session: `aav_prosst_ft_mixed_k4_n128`
- launcher: `scripts/run_aav_zero_shot_prosst_ft_n128_20260603.sh`
- output: `outputs/aav_zero_shot_prosst_ft_mixed_k4_n128_kl2_lr3e5_20260603`
- task: AAV
- seeds: 1-5 sequential
- n_queries: `128`
- n_iters: `10`
- generation: `mixed_explore_exploit`, `K=4`, constrained `CHARGE`, no-rollout sequential
- FT: replay all queried sequences, random K=4 masks, rank weights, `lambda_kl=2`, lr `3e-5`, 5 epochs/round, batch size 16
- traces and FT metrics enabled

## 2026-06-03 AAV ProSST fine-tuning full run completed

Output:
- `outputs/aav_zero_shot_prosst_ft_mixed_k4_n128_kl2_lr3e5_20260603`
- Summary: `outputs/aav_zero_shot_prosst_ft_mixed_k4_n128_kl2_lr3e5_20260603/summary_round10.json`

Status:
- Completed seeds 1-5.
- tmux session exited normally.
- A6000 idle after completion.

Config:
- Task: AAV
- Model: `AI4Protein/ProSST-2048`
- Generation: `mixed_explore_exploit`, `K=4`, constrained `CHARGE`, no-rollout sequential decoding
- Query budget: `n=128`, 10 rounds
- FT: replay all queried sequences, random K=4 masks, rank weights, KL to frozen base, `lambda_kl=2`, lr `3e-5`, 5 epochs/round, batch size 16

Final round 10 metrics, mean +/- SEM over 5 seeds:
- Round 1 best: `0.5213 +/- 0.0088`
- Round 1 top100/performance: `0.2235 +/- 0.0097`
- Round 10 best: `0.6704 +/- 0.0142`
- Round 10 top100/performance: `0.6364 +/- 0.0113`
- Round 10 median: `0.6344 +/- 0.0115`

Comparison at round 10:
- ProSST no-FT: best `0.5750 +/- 0.0102`, top100 `0.5277 +/- 0.0083`
- ProSST FT: best `0.6704 +/- 0.0142`, top100 `0.6364 +/- 0.0113`
- EvoDiff FT mixed rank: best `0.6662 +/- 0.0104`, top100 `0.6161 +/- 0.0084`
- EvoDiff FT no-rollout cluster: best `0.6653 +/- 0.0057`, top100 `0.6291 +/- 0.0048`
- AAV one-hot ridge: best `0.7038 +/- 0.0089`, top100 `0.6681 +/- 0.0081`
- Original CNN: best `0.7296 +/- 0.0137`, top100 `0.6944 +/- 0.0118`

Final ProSST FT diagnostics by seed, last epoch:
- Seed 1: KL `0.2522`, rel update `0.01923`, grad `1.359`, total FT seconds `285.7`
- Seed 2: KL `0.2614`, rel update `0.01887`, grad `1.390`, total FT seconds `275.4`
- Seed 3: KL `0.2509`, rel update `0.01906`, grad `1.270`, total FT seconds `275.6`
- Seed 4: KL `0.2548`, rel update `0.01919`, grad `1.278`, total FT seconds `275.6`
- Seed 5: KL `0.2549`, rel update `0.01894`, grad `1.377`, total FT seconds `280.5`

Interpretation:
- ProSST FT is a strong success on AAV.
- It beats ProSST no-FT by about `+0.095` best and `+0.109` top100.
- It slightly beats the EvoDiff FT mixed/no-rollout baselines on best score and top100 mean.
- It still trails the supervised 1-hot ridge and CNN surrogate pipelines, but the gap is now much smaller.

## 2026-06-03 ProSST FT batch launches: reward/vocab ablations and all landscapes

Assumption:
- User's `K=8` and `K=128` are interpreted as variable query budgets (`n_queries=8/128`), consistent with prior variable-k plotting labels. Mask budget remains `K_mask=4`.

Code update:
- `scripts/run_aav_zero_shot_prosst.py` generalized from AAV-only to all ProSST-mapped tasks: AAV, AMIE, E4B, GFP, LGK, Pab1, TEM, UBE2I.
- D-shift tasks are not included because no matching ProSST structure-token mapping is available.
- Added ProSST reward modes: `rank`, `grpo_advantage`, `standardized_advantage`, `bottom_quantile_negative`.
- Fixed masked-context construction to mask token IDs directly instead of inserting literal `<mask>` into amino-acid strings.

Smoke tests before launch:
- Generic AAV FT smoke: `outputs/_smoke_prosst_generic_aav`, completed.
- Generic LGK FT smoke: `outputs/_smoke_prosst_generic_lgk`, completed.

Launched tmux queues:
- `prosst_aav_ablate_n128` on RTX 2080 Ti UUID `GPU-fcb13561-e5da-20e1-2ff7-a9bdbfa68c26`.
- `prosst_lgk_ablate_n128` on RTX 2080 Ti UUID `GPU-a3d54441-08da-ed66-fa0f-6aaaaf97baea`.
- `prosst_all_n8` on RTX 2080 Ti UUID `GPU-6faf056b-00ab-15fd-e25c-a149c7dcf3d7`.
- `prosst_all_n128` on A6000 UUID `GPU-025e10e6-263f-d814-6dd5-added86fc8af`.

AAV/LGK reward-vocab ablation output roots:
- `outputs/prosst_ft_reward_vocab_ablation_AAV_n128_20260603`
- `outputs/prosst_ft_reward_vocab_ablation_LGK_n128_20260603`

Ablation configs per task:
- `full_vocab_rank`: full 20-AA SMC vocab, rank FT.
- `grpo_cluster`: constrained CHARGE SMC vocab, GRPO-style advantage FT, negative_weight 0.25.
- `standardized_cluster`: constrained CHARGE SMC vocab, moving-starting-sequence standardized advantage FT, negative_weight 0.25.
- `bottomq_neg05_cluster`: constrained CHARGE SMC vocab, rank positives with bottom-quartile negatives, negative_weight 0.5.

All-landscape current-strategy output roots:
- `outputs/prosst_ft_all_landscapes_n8_rank_cluster_20260603`
- `outputs/prosst_ft_all_landscapes_n128_rank_cluster_20260603`

All-landscape config:
- ProSST FT current best strategy: `mixed_explore_exploit`, mask budget 4, constrained CHARGE SMC vocab, rank reward, lambda_KL 2, lr 3e-5, 5 epochs/round, replay all, traces enabled.
- Seeds 1-5 for each task.

## 2026-06-03 ProSST batch launch adjustment

LGK ablation initially failed on an RTX 2080 Ti with CUDA OOM during ProSST entropy scoring at LGK length using batch/chunk settings inherited from AAV.

Fix applied:
- Added `--entropy_chunk_size` to `scripts/run_aav_zero_shot_prosst.py`, defaulting to 16.
- Relaunched LGK ablation with `batch_size=64`, `finetune_batch_size=8`, `entropy_chunk_size=16`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Restarted the all-landscape n=8 queue with the same safer long-sequence settings.

Active adjusted sessions:
- `prosst_lgk_ablate_n128`
- `prosst_all_n8`

The AAV ablation and A6000 all-landscape n=128 queues were left running.

## 2026-06-03 ProSST 2080 long-sequence FT adjustment

LGK still OOMed on RTX 2080 Ti during ProSST fine-tuning with `finetune_batch_size=8` at length 439, even after reducing generation batch size.

Fix applied:
- Relaunched LGK reward/vocab ablation with `finetune_batch_size=1`.
- Relaunched all-landscape n=8 queue with `finetune_batch_size=1` to avoid future long-sequence OOMs on the 2080.
- Generation batch remains 64 and entropy chunk remains 16.

Active robust sessions:
- `prosst_lgk_ablate_n128`
- `prosst_all_n8`

## 2026-06-04 ProSST FT batch status checkpoint

Active sessions:
- `prosst_aav_ablate_n128`
- `prosst_lgk_ablate_n128`
- `prosst_all_n8`
- `prosst_all_n128`

GPU status at checkpoint:
- GPU0 RTX 2080 Ti `GPU-fcb13561-e5da-20e1-2ff7-a9bdbfa68c26`: AAV ablations active, ~5.8GB.
- GPU1 RTX 2080 Ti `GPU-a3d54441-08da-ed66-fa0f-6aaaaf97baea`: LGK ablations active, ~8.2GB.
- GPU2 RTX 2080 Ti `GPU-6faf056b-00ab-15fd-e25c-a149c7dcf3d7`: all-landscape n=8 active, ~8.3GB.
- A6000 `GPU-025e10e6-263f-d814-6dd5-added86fc8af`: all-landscape n=128 active, ~23.2GB.

No current Traceback/OOM errors after the `finetune_batch_size=1` long-sequence fix.

Completed/partial aggregates at this checkpoint:

AAV reward/vocab ablations, n=128:
- `full_vocab_rank`: 5/5 complete, final best `0.6150 +/- 0.0077`, top100 `0.5771 +/- 0.0090`, median `0.5740 +/- 0.0094`.
- `grpo_cluster`: 5/5 complete, final best `0.6665 +/- 0.0082`, top100 `0.6305 +/- 0.0069`, median `0.6286 +/- 0.0072`.
- `standardized_cluster`: 5/5 complete, final best `0.5360 +/- 0.0092`, top100 `0.4538 +/- 0.0053`, median `0.4493 +/- 0.0055`.
- `bottomq_neg05_cluster`: running, seed 1 at iteration 7 in latest pane; early seed 1 looked strong with best `0.6207` and top100 `0.5854` by iteration 7.

Interpretation so far for AAV:
- Full 20-AA vocab is worse than CHARGE-constrained rank FT.
- GRPO is competitive with the previous rank ProSST FT run and slightly below/near it depending on metric; current GRPO aggregate best `0.6665`, top100 `0.6305` vs prior rank ProSST FT `0.6704`, top100 `0.6364`.
- Standardized moving-baseline advantage is bad for AAV in this setup.

LGK reward/vocab ablations, n=128:
- `full_vocab_rank`: seed 1 complete, seed 2 running. Seed 1 final best `0.0425`, top100 `0.0418`.

All-landscape current strategy, n=8:
- AAV: 5/5 complete, best `0.5049 +/- 0.0057`, top100 `0.2871 +/- 0.0055`.
- AMIE: 5/5 complete, best `0.2372 +/- 0.0017`, top100 `0.0037 +/- 0.0302`.
- E4B: 5/5 complete, best `7.7399 +/- 0.0098`, top100 `6.5941 +/- 0.0603`.
- GFP: 5/5 complete, best `3.5841 +/- 0.0018`, top100 `3.5171 +/- 0.0130`.
- LGK: 2/5 complete, best `0.0309 +/- 0.0010`, top100 `0.0247 +/- 0.0010`.

All-landscape current strategy, n=128:
- AAV: 5/5 complete, best `0.6704 +/- 0.0142`, top100 `0.6364 +/- 0.0113`.
- AMIE: 4/5 complete, best `0.2616 +/- 0.0024`, top100 `0.2579 +/- 0.0023`.

## 2026-06-04 AAV ProSST soft full-vocabulary sweep queued

Implemented `--non_cluster_logit_penalty` in `scripts/run_aav_zero_shot_prosst.py`.

Behavior:
- Only applies when `--smc_vocab full`.
- All 20 amino acids remain sampleable.
- Amino acids outside the original residue's CHARGE cluster receive a logit penalty before softmax.
- Penalty 0 is the existing full-vocab behavior; larger penalties approach CHARGE-constrained behavior but retain off-cluster exploration.

Queued tmux session:
- `prosst_aav_soft_full_sweep`
- It waits until `prosst_aav_ablate_n128` exits before using GPU0.

Output root:
- `outputs/prosst_ft_soft_full_vocab_AAV_n128_20260604`

Sweep configs:
- `rank_penalty_0p5`, `rank_penalty_1p0`, `rank_penalty_2p0`
- `grpo_penalty_0p5`, `grpo_penalty_1p0`, `grpo_penalty_2p0`

Common config:
- AAV, n=128, 10 rounds, seeds 1-5
- mixed masking, mask budget 4
- full 20-AA SMC vocab with non-cluster logit penalty
- ProSST FT, lambda_KL 2, lr 3e-5, 5 epochs/round, replay all
- traces enabled

## 2026-06-04 ProSST FT / vocab ablation checkpoint

Status summary:
- Active tmux: `prosst_lgk_ablate_n128` only.
- Idle GPUs at last check: RTX 2080 Ti GPU0, RTX 2080 Ti GPU2, A6000 GPU3.
- Busy GPU: RTX 2080 Ti GPU1 running LGK ProSST reward/vocab ablation.

Completed AAV reward/vocab ablation: `outputs/prosst_ft_reward_vocab_ablation_AAV_n128_20260603`
- `full_vocab_rank`: 5/5, round10 best 0.6150 ± 0.0077, top100/perf 0.5771 ± 0.0090.
- `grpo_cluster`: 5/5, round10 best 0.6665 ± 0.0082, top100/perf 0.6305 ± 0.0069.
- `standardized_cluster`: 5/5, round10 best 0.5360 ± 0.0092, top100/perf 0.4538 ± 0.0053.
- `bottomq_neg05_cluster`: 5/5, round10 best 0.6450 ± 0.0159, top100/perf 0.6069 ± 0.0095.

Completed AAV soft full-vocab sweep: `outputs/prosst_ft_soft_full_vocab_AAV_n128_20260604`
- `rank_penalty_0p5`: 5/5, round10 best 0.6421 ± 0.0078, top100/perf 0.6014 ± 0.0066.
- `rank_penalty_1p0`: 5/5, round10 best 0.6585 ± 0.0066, top100/perf 0.6171 ± 0.0041.
- `rank_penalty_2p0`: 5/5, round10 best 0.6609 ± 0.0179, top100/perf 0.6193 ± 0.0153.
- `grpo_penalty_0p5`: 5/5, round10 best 0.5963 ± 0.0132, top100/perf 0.5529 ± 0.0113.
- `grpo_penalty_1p0`: 5/5, round10 best 0.6187 ± 0.0178, top100/perf 0.5819 ± 0.0173.
- `grpo_penalty_2p0`: 5/5, round10 best 0.6366 ± 0.0105, top100/perf 0.5924 ± 0.0099.

Completed all-landscape current ProSST FT strategy, K=8: `outputs/prosst_ft_all_landscapes_n8_rank_cluster_20260603`
- AAV: best 0.5049 ± 0.0057, perf 0.2871 ± 0.0055.
- AMIE: best 0.2372 ± 0.0017, perf 0.0037 ± 0.0302.
- E4B: best 7.7399 ± 0.0098, perf 6.5941 ± 0.0603.
- GFP: best 3.5841 ± 0.0018, perf 3.5171 ± 0.0130.
- LGK: best 0.0331 ± 0.0010, perf 0.0230 ± 0.0034.
- Pab1: best 0.8879 ± 0.0279, perf 0.6827 ± 0.0224.
- TEM: best 1.2210 ± 0.0067, perf 0.6472 ± 0.0382.
- UBE2I: best 2.9868 ± 0.0011, perf 2.8884 ± 0.0144.

Completed all-landscape current ProSST FT strategy, K=128: `outputs/prosst_ft_all_landscapes_n128_rank_cluster_20260603`
- AAV: best 0.6704 ± 0.0142, perf 0.6364 ± 0.0113.
- AMIE: best 0.2613 ± 0.0018, perf 0.2577 ± 0.0018.
- E4B: best 8.0974 ± 0.0065, perf 8.0348 ± 0.0101.
- GFP: best 3.6107 ± 0.0020, perf 3.6089 ± 0.0021.
- LGK: best 0.0407 ± 0.0006, perf 0.0396 ± 0.0005.
- Pab1: best 1.7314 ± 0.2016, perf 1.6105 ± 0.2089.
- TEM: best 1.2319 ± 0.0005, perf 1.2295 ± 0.0003.
- UBE2I: best 2.9994 ± 0.0003, perf 2.9980 ± 0.0004.

LGK reward/vocab ablation still running: `outputs/prosst_ft_reward_vocab_ablation_LGK_n128_20260603`
- `full_vocab_rank`: 5/5 complete, round10 best 0.0435 ± 0.0004, top100/perf 0.0426 ± 0.0004.
- `grpo_cluster`: in progress at last check; 3/5 complete with aggregate round10 best 0.0425 ± 0.0005, top100/perf 0.0416 ± 0.0006.
- Remaining LGK configs queued after GRPO in the same tmux session.

Interpretation checkpoint:
- For AAV, pure CHARGE-constrained rank FT is still the best ProSST variant so far: best 0.6704, perf 0.6364.
- Full 20-AA vocabulary hurts. Soft full-vocab penalties recover some performance but do not beat hard CHARGE.
- GRPO cluster is close to rank FT but does not clearly improve it on AAV.
- Standardized moving-baseline reward is poor on AAV.

## 2026-06-04 ProSST FT method comparison plots

Generated comparison plots in `outputs/rl_vs_og/prosst_ft_methods` using `scripts/plot_rl_vs_og_prosst_ft_methods.py`.

Trajectory plots:
- `outputs/rl_vs_og/prosst_ft_methods/ft_methods_vs_original_cnn_k8_mean_max_trajectories.png`
- `outputs/rl_vs_og/prosst_ft_methods/ft_methods_vs_original_cnn_k128_mean_max_trajectories.png`

Fitness histogram plots:
- `outputs/rl_vs_og/prosst_ft_methods/fitness_histograms/*_k8_ft_methods_vs_original_fitness_histograms.png`
- `outputs/rl_vs_og/prosst_ft_methods/fitness_histograms/*_k128_ft_methods_vs_original_fitness_histograms.png`

Included methods:
- Original CNN ProSpero baseline.
- EvoDiff FT seed_grow KL=2 where available.
- EvoDiff pretrain top16 + FT for AAV and LGK.
- EvoDiff FT mixed rank for AAV.
- ProSST FT rank cluster from all-landscape K=8/K=128 runs.

Histogram convention:
- Histograms use queried candidate fitness from `Iter scores`.
- Duplicate sequences are collapsed per method/task/budget/round.
- Red rug marks show retained pool scores from `Scores` when available.

Data caveat:
- Original CNN GFP K=128 has 4/5 seeds in the available baseline folder; other trajectory series included here have 5/5 seeds.

## 2026-06-04 launched ProSST GRPO cluster all-landscape comparison

Launched all-landscape ProSST fine-tuning comparison with GRPO reward and hard CHARGE cluster vocabulary.

Script:
- `scripts/run_prosst_ft_all_landscapes_grpo_cluster_20260604.sh`

Configs:
- `outputs/prosst_ft_all_landscapes_n128_grpo_cluster_20260604`: n_queries/K=128, task set AAV AMIE E4B GFP LGK Pab1 TEM UBE2I, seeds 1-5, mask strategy `mixed_explore_exploit`, mask budget 4, `smc_vocab=cluster`, ProSST FT, KL=2, lr=3e-5, 5 epochs/round, replay all, reward_mode `grpo_advantage`, traces enabled. tmux `prosst_grpo_cluster_n128`, GPU UUID `GPU-025e10e6-263f-d814-6dd5-added86fc8af`.
- `outputs/prosst_ft_all_landscapes_n8_grpo_cluster_20260604`: same config with n_queries/K=8. tmux `prosst_grpo_cluster_n8`, GPU UUID `GPU-fcb13561-e5da-20e1-2ff7-a9bdbfa68c26`.

Purpose:
- Compare ProSST GRPO cluster against current ProSST rank cluster and original CNN in the same plotting pipeline.

## 2026-06-04 zero-shot delta PLL alignment plots

Generated publication-style zero-shot alignment plots using local Nunito font and reusable plotting style.

Style/code:
- `assets/fonts/Nunito-VariableFont_wght.ttf`
- `src/prospero/plotting_style.py`
- `scripts/plot_zero_shot_delta_pll_alignment.py`

Outputs:
- `outputs/zero_shot_alignment_plots_20260604/prosst_delta_pll_vs_delta_fitness.png` and `.pdf`
- `outputs/zero_shot_alignment_plots_20260604/esm2_650m_delta_pll_vs_delta_fitness.png` and `.pdf`
- `outputs/zero_shot_alignment_plots_20260604/evodiff_delta_pll_vs_delta_fitness.png` and `.pdf`
- `outputs/zero_shot_alignment_plots_20260604/zero_shot_alignment_spearman_summary.png` and `.pdf`
- `outputs/zero_shot_alignment_plots_20260604/zero_shot_alignment_plot_summary.json`

Plot convention:
- x-axis: model delta PLL/log-odds versus WT from existing zero-shot predictive score CSVs.
- y-axis: oracle delta fitness = observed fitness - WT fitness.
- WT fitness is pulled from ProSpero train/valid splits when present.
- Points are validation sequences; trend is median delta fitness over delta-PLL quantile bins with IQR ribbon.
- Axes use robust per-landscape limits for visual readability.

Key Spearman rho(delta PLL, delta fitness):
- ProSST: AAV 0.341, LGK 0.459, GFP 0.289, Pab1 -0.056, AMIE 0.410, E4B 0.366, TEM 0.749, UBE2I 0.418.
- ESM2 650M: AAV 0.086, LGK 0.365, GFP -0.130, Pab1 -0.052, AMIE 0.362, E4B 0.293, TEM 0.712, UBE2I 0.502.
- EvoDiff: AAV 0.097, LGK 0.106, GFP -0.116, Pab1 -0.066, AMIE 0.246, E4B 0.320, TEM 0.376, UBE2I 0.049.

## 2026-06-04 zero-shot alignment plot style update

Updated `scripts/plot_zero_shot_delta_pll_alignment.py` and `src/prospero/plotting_style.py` per plotting feedback:
- Removed binned fit curves and IQR ribbons from delta PLL alignment panels.
- Increased overall font sizes in the reusable ProSpero plotting style.
- Made scatter points fully opaque.
- Regenerated PNG/PDF outputs in `outputs/zero_shot_alignment_plots_20260604`.

## 2026-06-04 GRPO cluster progress checkpoint

ProSST GRPO all-landscape comparison:
- `outputs/prosst_ft_all_landscapes_n8_grpo_cluster_20260604` completed; tmux session exited.
- `outputs/prosst_ft_all_landscapes_n128_grpo_cluster_20260604` still running on A6000 in tmux `prosst_grpo_cluster_n128`; currently AAV seed 5 at last check.

K=8 GRPO cluster completed round10 aggregate:
- AAV: best 0.4979 ± 0.011, perf 0.2052 ± 0.0095.
- AMIE: best 0.2356 ± 0.00087, perf -0.1394 ± 0.049.
- E4B: best 7.752 ± 0.023, perf 6.227 ± 0.17.
- GFP: best 3.584 ± 0.00086, perf 3.510 ± 0.015.
- LGK: best 0.03428 ± 0.00068, perf 0.02278 ± 0.0033.
- Pab1: best 0.885 ± 0.056, perf 0.6618 ± 0.024.
- TEM: best 1.226 ± 0.00075, perf 0.5942 ± 0.045.
- UBE2I: best 2.984 ± 0.00032, perf 2.842 ± 0.020.

K=128 GRPO cluster progress:
- AAV 4/5 complete so far: best 0.6496 ± 0.0094, perf 0.6017 ± 0.011.
- AAV seed 5 is in progress.

## 2026-06-04 simplified 0shotProt mean-max plots

Generated simplified publication-style mean-max trajectory plots using Nunito formatting and only three curves:
- `ProSpero`
- `0shotProt (w/ ProSST)`
- `0shotProt (w/ EvoDiff)`

Outputs:
- `outputs/rl_vs_og/zero_shotprot_simplified/zero_shotprot_vs_prospero_k8_mean_max.png` and `.pdf`
- `outputs/rl_vs_og/zero_shotprot_simplified/zero_shotprot_vs_prospero_k128_mean_max.png` and `.pdf`
- `outputs/rl_vs_og/zero_shotprot_simplified/plot_summary.txt`

Notes:
- Removed previous caveat text overlays.
- Used ProSST rank-cluster all-landscape runs.
- Used EvoDiff rank/mixed-rank run for AAV and completed EvoDiff FT landscape runs elsewhere.

## 2026-06-04 simplified plot legend/order update

Regenerated simplified mean-max trajectory plots so `ProSpero` appears last in the legend and is drawn last:
- `outputs/rl_vs_og/zero_shotprot_simplified/zero_shotprot_vs_prospero_k8_mean_max.png` and `.pdf`
- `outputs/rl_vs_og/zero_shotprot_simplified/zero_shotprot_vs_prospero_k128_mean_max.png` and `.pdf`

## 2026-06-04 restyled oracle epistasis/additivity plots

Restyled the old oracle epistasis/additivity plots using the current Nunito visual style and placed them under `outputs/rl_vs_og`.

Source data:
- `outputs/0423_epistasis/epistasis_additivity_all_tasks.json`

Outputs:
- `outputs/rl_vs_og/epistasis_additivity/epistasis_additivity_oracle_scatter.png` and `.pdf`
- `outputs/rl_vs_og/epistasis_additivity/epistasis_distributions_oracle_histograms.png` and `.pdf`
- `outputs/rl_vs_og/epistasis_additivity/plot_summary.txt`

Script:
- `scripts/plot_epistasis_additivity_styled.py`

## 2026-06-04 epistasis plot update excluding D-shift

Regenerated styled oracle epistasis/additivity plots excluding D-shift scenarios. Included tasks: AAV, LGK, GFP, Pab1, AMIE, E4B, TEM, UBE2I.

Updated outputs:
- `outputs/rl_vs_og/epistasis_additivity/epistasis_additivity_oracle_scatter.png` and `.pdf`
- `outputs/rl_vs_og/epistasis_additivity/epistasis_distributions_oracle_histograms.png` and `.pdf`

## 2026-06-04 epistasis histogram scale fix

Regenerated styled oracle epistasis histograms with robust per-task x-limits using the central 1-99% epistasis range. This fixes LGK visual compression from rare large outliers while annotating clipped outliers in each panel.

Updated outputs:
- `outputs/rl_vs_og/epistasis_additivity/epistasis_distributions_oracle_histograms.png` and `.pdf`
- scatter outputs were regenerated unchanged by the same plotting script.

## 2026-06-04 ProSST restricted vs unrestricted vocabulary plot

Generated K=128 mean-max trajectory plot comparing rank-based ProSST with restricted charge vocabulary vs unrestricted amino acid vocabulary.

Included tasks with completed unrestricted rank ablations:
- AAV
- LGK

Outputs:
- `outputs/rl_vs_og/prosst_vocab_ablation_k128/prosst_restricted_vs_unrestricted_vocab_k128_mean_max.png` and `.pdf`
- `outputs/rl_vs_og/prosst_vocab_ablation_k128/plot_summary.txt`

Round10 summary:
- AAV restricted: best 0.6704 ± 0.0142; unrestricted: best 0.6150 ± 0.0077.
- LGK restricted: best 0.04071 ± 0.00060; unrestricted: best 0.04354 ± 0.00044.

## 2026-06-04 launched ProSST unrestricted vocabulary rank ablations

Launched missing rank-based ProSST unrestricted amino acid vocabulary ablations for comparison against restricted charge vocabulary.

Script:
- `scripts/run_prosst_full_vocab_rank_missing_20260604.sh`

Runs:
- `outputs/prosst_ft_all_landscapes_n8_full_vocab_rank_20260604`: K=8, tasks AAV AMIE E4B GFP LGK Pab1 TEM UBE2I, seeds 1-5, tmux `prosst_full_vocab_rank_k8`, GPU UUID `GPU-fcb13561-e5da-20e1-2ff7-a9bdbfa68c26`.
- `outputs/prosst_ft_all_landscapes_n128_full_vocab_rank_20260604`: K=128, tasks AMIE E4B GFP Pab1 TEM UBE2I, seeds 1-5, tmux `prosst_full_vocab_rank_k128_missing`, GPU UUID `GPU-6faf056b-00ab-15fd-e25c-a149c7dcf3d7`.

Existing completed unrestricted K=128 runs retained for comparison:
- AAV: `outputs/prosst_ft_reward_vocab_ablation_AAV_n128_20260603/full_vocab_rank/AAV`
- LGK: `outputs/prosst_ft_reward_vocab_ablation_LGK_n128_20260603/full_vocab_rank/LGK`

## 2026-06-04 23:55 CEST - ablation status checkpoint

- Unrestricted amino-acid vocabulary ProSST rank ablation, K=8: completed all 8 landscapes, 5/5 seeds each, traces enabled. Output: `outputs/prosst_ft_all_landscapes_n8_full_vocab_rank_20260604/`.
  - Final mean max fitness: AAV 0.5096 +/- 0.0044, LGK 0.03335 +/- 0.00141, GFP 3.5860 +/- 0.0030, Pab1 0.8871 +/- 0.0162, AMIE 0.2339 +/- 0.0015, E4B 7.7244 +/- 0.0496, TEM 1.2278 +/- 0.0005, UBE2I 2.9854 +/- 0.0010.
- Unrestricted amino-acid vocabulary ProSST rank ablation, K=128: still running in tmux session `prosst_full_vocab_rank_k128_missing` on RTX 2080 Ti UUID `GPU-6faf056b-00ab-15fd-e25c-a149c7dcf3d7`.
  - AMIE completed 5/5 seeds: final mean max 0.2605 +/- 0.0024.
  - E4B seed 1 currently partial through iteration 5; remaining E4B seeds plus GFP/Pab1/TEM/UBE2I still pending in this launcher.
- ProSST GRPO-cluster K=128 comparison: still running in tmux session `prosst_grpo_cluster_n128` on A6000 UUID `GPU-025e10e6-263f-d814-6dd5-added86fc8af`.
  - AAV completed 5/5: final mean max 0.6441 +/- 0.0092.
  - AMIE completed 5/5: final mean max 0.2618 +/- 0.0014.
  - E4B completed 4/5 with seed 5 partial through iteration 2 at checkpoint; completed-seed final mean max 8.1356 +/- 0.0164.

## 2026-06-05 14:20 CEST - active ablation checkpoint

- Active tmux sessions:
  - `prosst_full_vocab_rank_k128_missing`: unrestricted amino-acid vocabulary ProSST rank ablation, K=128, running on RTX 2080 Ti UUID `GPU-6faf056b-00ab-15fd-e25c-a149c7dcf3d7`.
  - `prosst_grpo_cluster_n128`: ProSST GRPO-cluster K=128 comparison, running on A6000 UUID `GPU-025e10e6-263f-d814-6dd5-added86fc8af`.
- GPU state at checkpoint:
  - RTX 2080 Ti UUID `GPU-6faf056b-00ab-15fd-e25c-a149c7dcf3d7`: active, ~4.9 GB, ~73% util.
  - A6000 UUID `GPU-025e10e6-263f-d814-6dd5-added86fc8af`: active, ~3.4 GB, ~33% util.
  - Other two RTX 2080 Ti GPUs idle.
- Unrestricted vocabulary ProSST rank, K=8: completed all 8 landscapes, 5/5 seeds each.
- Unrestricted vocabulary ProSST rank, K=128: AMIE/E4B/GFP/Pab1 completed 5/5 seeds each. TEM seed 1 currently partial through iteration 9; TEM seeds 2-5 and UBE2I still pending.
  - Final mean max so far: AMIE 0.2605 +/- 0.0024, E4B 8.1168 +/- 0.0107, GFP 3.6166 +/- 0.0014, Pab1 1.7074 +/- 0.179.
- GRPO-cluster K=128: AAV/AMIE/E4B/GFP/LGK completed 5/5 seeds each. Pab1 completed 4/5 seeds; seed 5 partial through iteration 8.
  - Final mean max so far: AAV 0.6441 +/- 0.0092, AMIE 0.2618 +/- 0.0014, E4B 8.1430 +/- 0.0147, GFP 3.6175 +/- 0.0014, LGK 0.04258 +/- 0.00049, Pab1 1.887 +/- 0.200 over completed seeds only.

## 2026-06-05 19:55 CEST - parallelized remaining GRPO UBE2I seeds

- The K=128 ProSST GRPO-cluster run was originally running sequentially in tmux session `prosst_grpo_cluster_n128` on A6000 UUID `GPU-025e10e6-263f-d814-6dd5-added86fc8af`.
- Two idle RTX 2080 Ti GPUs were used to parallelize the remaining UBE2I GRPO seeds:
  - `prosst_grpo_ube2i_gpu0`: UBE2I seeds 2 and 4 on UUID `GPU-fcb13561-e5da-20e1-2ff7-a9bdbfa68c26`.
  - `prosst_grpo_ube2i_gpu1`: UBE2I seeds 3 and 5 on UUID `GPU-a3d54441-08da-ed66-fa0f-6aaaaf97baea`.
- The original A6000 launcher is expected to complete TEM seed 5, then run or skip UBE2I seeds depending on whether the parallel worker files already exist. Watch for partial `seed_*.pkl` files because the launcher only checks that the file is non-empty.
- GPU state after launch: all four GPUs active.

## 2026-06-05 20:04 CEST - launched TEM no-finetune ProSST trace ablation

- Launched tmux session `prosst_tem_noft_n128_trace` on A6000 UUID `GPU-025e10e6-263f-d814-6dd5-added86fc8af`.
- Output folder: `outputs/prosst_noft_tem_n128_cluster_trace_20260605/`.
- Config: task TEM, K=128 queries, 5 seeds, 10 rounds, batch size 64, mask budget 4, `mixed_explore_exploit` masking, cluster/restricted vocabulary, no `--finetune_prosst`, `--debug_generation_trace` enabled.
- Seed 1 started and trace files are already being written under `outputs/prosst_noft_tem_n128_cluster_trace_20260605/TEM/debug_traces/`.
- A6000 after launch: previous process plus new no-finetune TEM process, ~7.2 GB total memory, high utilization.

## 2026-06-05 21:00 CEST - run status checkpoint

- Active tmux sessions: `prosst_full_vocab_rank_k128_missing`, `prosst_grpo_cluster_n128`, `prosst_grpo_ube2i_gpu0`, `prosst_grpo_ube2i_gpu1`.
- Completed and exited: `prosst_tem_noft_n128_trace`.
- TEM no-finetune ProSST K=128 trace ablation completed 5/5 seeds in `outputs/prosst_noft_tem_n128_cluster_trace_20260605/` with all debug traces present. Final mean max fitness: 1.23083 +/- 0.00050.
- Full-vocab ProSST rank K=128 status: AMIE/E4B/GFP/Pab1/TEM complete 5/5. UBE2I complete 2/5 with seed 3 partial through iteration 4. Current completed-seed UBE2I final mean max: 2.99795 +/- 0.000955.
- GRPO-cluster K=128 status: AAV/AMIE/E4B/GFP/LGK/Pab1/TEM complete 5/5. UBE2I complete 3/5, seeds 4 and 5 partial through iteration 4. Current completed-seed UBE2I final mean max: 2.99980 +/- 0.00118.
- GPU state: all four GPUs active; A6000 now only has the original GRPO launcher process after TEM no-finetune completed.

## 2026-06-08 - completed run checkpoint and no-surrogate scope note

- All previously active tmux sessions have exited: `prosst_full_vocab_rank_k128_missing`, `prosst_grpo_cluster_n128`, `prosst_grpo_ube2i_gpu0`, `prosst_grpo_ube2i_gpu1` are no longer present.
- All GPUs idle at checkpoint.
- Full-vocab ProSST rank K=128 completed 5/5 seeds for AMIE, E4B, GFP, Pab1, TEM, UBE2I. Final mean max: AMIE 0.26046, E4B 8.1168, GFP 3.6166, Pab1 1.7074, TEM 1.2302, UBE2I 2.9988.
- GRPO-cluster ProSST K=128 completed 5/5 seeds for AAV, AMIE, E4B, GFP, LGK, Pab1, TEM, UBE2I. Final mean max: AAV 0.6441, AMIE 0.2618, E4B 8.1430, GFP 3.6175, LGK 0.04258, Pab1 1.9490, TEM 1.2304, UBE2I 2.9995.
- TEM ProSST no-finetune K=128 trace ablation completed 5/5 seeds with traces. Final mean max 1.23083.
- No-surrogate scope note: old ProSpero/EvoDiff no-surrogate runs exist for AAV and LGK (`outputs/aav_zero_shot_prospero_20260528`, `outputs/lgk_zero_shot_prospero_20260528`) and AAV fixed-mask route sweeps (`outputs/aav_zero_shot_fixed_mask_k4_20260529`). I do not see an equivalent old no-finetune ProSpero/EvoDiff no-surrogate sweep across all 8 landscapes. The all-landscape zero-shot work later was online EvoDiff/ProSST fine-tuning or ProSST rank-cluster style, not the same original removed-surrogate ProSpero ablation.

## 2026-06-08 - ProSST PUCT/MCTS decoder smoke and AAV test

Implemented a modular PUCT/MCTS decoder for ProSST zero-shot generation.

Code changes:
- Added generic PUCT search core under `src/prospero/search/puct.py`.
- Added ProSST-specific PUCT evaluator under `src/prospero/search/prosst_puct.py`.
- Added `--decode_strategy {sample,mcts}`, `--mcts_simulations`, and `--mcts_c_puct` to the ProSST zero-shot runner. Default remains `sample`, so existing configs are unchanged unless MCTS is requested.
- Added unit tests for generic PUCT behavior and the ProSST adapter with fake logits.

Validation:
- `uv run python -m pytest tests/test_puct_search.py tests/test_prosst_puct.py -q`: passed, 5 tests.
- `uv run python -m compileall -q src/prospero/search src/prospero/runners/run_zero_shot_prosst.py`: passed.
- Runner help confirms MCTS flags are exposed.

Smoke run:
- Command used A6000 pinned by UUID `GPU-025e10e6-263f-d814-6dd5-added86fc8af`.
- Output: `outputs/mcts_smoke_20260608/`.
- Config: AAV, seed 1, one round, 2 queries, batch size 4, mask budget 4, mixed masking, restricted/cluster vocabulary, MCTS decode, 8 simulations.
- Completed cleanly. Best oracle fitness: 0.3782479603.

AAV test run:
- tmux session: `mcts_aav_test_20260608` (completed and exited).
- Output: `outputs/mcts_aav_test_20260608/`.
- Log: `outputs/mcts_aav_test_20260608.log`.
- Config: AAV, seed 1, one round, 16 queries, batch size 32, mask budget 4, mixed masking, restricted/cluster vocabulary, MCTS decode, 32 simulations, `c_puct=1.5`.
- Completed cleanly on A6000 pinned by UUID `GPU-025e10e6-263f-d814-6dd5-added86fc8af`.
- Trace counts: 32 `mask_selected`, 32 `mcts_summary`, 32 `candidate`, 128 `smc_step`, 16 `candidate_selected_for_query`, 16 `oracle_query`.
- Selected zero-shot score range: min 3.697951, median 6.955224, max 11.211452.
- Oracle fitness over selected candidates: min 0.0, median 0.0985704, max 0.3782479603.

Engineering note:
- This first MCTS implementation is intentionally auditable and node-cached, but not batched across particles/tree nodes yet. It is appropriate for small tests and debugging; full 128-query, 10-round campaigns will need batching or lower simulation counts before being compute-efficient.

## 2026-06-09 - fair AAV ProSST MCTS n=128 comparison

Launched and completed a fair one-round AAV comparison for the ProSST PUCT/MCTS decoder using the same query budget as the old stochastic unmasking comparison.

Run:
- tmux session: `mcts_aav_n128_20260609` (completed and exited).
- GPU: A6000 pinned by UUID `GPU-025e10e6-263f-d814-6dd5-added86fc8af`.
- Output: `outputs/mcts_aav_n128_20260609/`.
- Log: `outputs/mcts_aav_n128_20260609.log`.
- Config: AAV, seed 1, one round, `n_queries=128`, `batch_size=256`, `mask_budget=4`, `mixed_explore_exploit` masking, restricted/cluster vocabulary, `decode_strategy=mcts`, `mcts_simulations=32`, `mcts_c_puct=1.5`, traces enabled.

Trace counts:
- 256 `mask_selected`, 256 `mcts_summary`, 256 `candidate`, 1024 `smc_step`, 128 `candidate_selected_for_query`, 128 `oracle_query`.

Round-1 comparison against old ProSST stochastic unmasking (`outputs/aav_zero_shot_prosst_mixed_k4_n128_20260603`, seed 1, round 1):
- MCTS all generated candidates: n=256, zero-shot score median 2.9087, mean 3.4689, max 12.3289, positive fraction 99.6%.
- Stochastic all generated candidates: n=256, zero-shot score median 0.6926, mean 0.9388, max 12.3289, positive fraction 62.5%.
- MCTS selected candidates: n=128, zero-shot score median 4.6099, mean 5.1719, max 12.3289, positive fraction 100%.
- Stochastic selected candidates: n=128, zero-shot score median 2.1103, mean 2.7978, max 12.3289, positive fraction 100%.
- MCTS oracle fitness: median 0.2090, mean 0.1925, max 0.4690, positive fraction 84.4%.
- Stochastic oracle fitness: median 0.1852, mean 0.1844, max 0.5272, positive fraction 79.7%.

Interpretation:
- MCTS materially improves the generated and selected zero-shot score distribution under the delta-PLL objective.
- MCTS modestly improves median/mean oracle fitness and positive-fitness fraction at the same query budget.
- MCTS does not improve the best oracle candidate in this one-round seed-1 comparison; old stochastic unmasking still had the higher max oracle fitness.

## 2026-06-09 - PLM full PLL alignment rerun

Question: previous Spearman alignment tables used fast mutated-position PLM estimates, not proper whole-sequence PLL. Added and ran a full-PLL alignment benchmark.

Code:
- Added `src/prospero/runners/run_plm_full_pll_alignment.py`.
- Computes old estimate Spearman from existing alignment CSVs and recomputes proper whole-sequence PLL by masking every residue position.
- PLMs: EvoDiff OA_DM_38M, ESM2 650M (`facebook/esm2_t33_650M_UR50D`), ProSST 2048 (`AI4Protein/ProSST-2048`).
- ProSST uses whole covered-region PLL; for LGK the unstructured C-terminal tail without structure tokens is excluded.

Runs:
- Smoke: `outputs/plm_full_pll_alignment_smoke_20260609/`, AAV n=4, completed.
- Attempted all-landscape n=64: `outputs/plm_full_pll_alignment_n64_20260609/`, stopped because ESM full PLL was too slow interactively. Partial results through AMIE EvoDiff retained.
- Completed all-landscape n=16: `outputs/plm_full_pll_alignment_n16_20260609/`, A6000 pinned by UUID `GPU-025e10e6-263f-d814-6dd5-added86fc8af`.

Key result:
- Previous alignment score was not proper full PLL.
- Proper full PLL is not consistently better. It helps in some cases but hurts in others, especially where mutated-position log-odds is more directly tied to variant effects.
- Full ESM PLL is the runtime bottleneck: examples from n=16 are AMIE 192s, LGK 341s, TEM 135s.

## 2026-06-09 - GRPO main optimization-loop plots

Generated simplified main mean-max optimization-loop plots using ProSST GRPO-cluster runs instead of rank-cluster runs.

Command:
- `uv run python -m prospero.runners.plot_simplified_zero_shotprot_mean_max --output-dir outputs/rl_vs_og/zero_shotprot_simplified_grpo --prosst-k8-root outputs/prosst_ft_all_landscapes_n8_grpo_cluster_20260604 --prosst-k128-root outputs/prosst_ft_all_landscapes_n128_grpo_cluster_20260604 --prosst-label '0shotProt GRPO (w/ ProSST)'`

Outputs:
- `outputs/rl_vs_og/zero_shotprot_simplified_grpo/zero_shotprot_vs_prospero_k8_mean_max.png`
- `outputs/rl_vs_og/zero_shotprot_simplified_grpo/zero_shotprot_vs_prospero_k8_mean_max.pdf`
- `outputs/rl_vs_og/zero_shotprot_simplified_grpo/zero_shotprot_vs_prospero_k8_mean_max.svg`
- `outputs/rl_vs_og/zero_shotprot_simplified_grpo/zero_shotprot_vs_prospero_k128_mean_max.png`
- `outputs/rl_vs_og/zero_shotprot_simplified_grpo/zero_shotprot_vs_prospero_k128_mean_max.pdf`
- `outputs/rl_vs_og/zero_shotprot_simplified_grpo/zero_shotprot_vs_prospero_k128_mean_max.svg`

Summary file:
- `outputs/rl_vs_og/zero_shotprot_simplified_grpo/plot_summary.txt`

## 2026-06-09 - combined K=8/K=128 optimization-loop visualization

Generated a new simplified combined-budget main optimization-loop visualization for GRPO. K=128 uses saturated method colors; K=8 uses pastel versions of the same colors on the same axes.

Code:
- Added `src/prospero/runners/plot_combined_budget_mean_max.py`.

Outputs:
- `outputs/rl_vs_og/zero_shotprot_combined_budgets_grpo/zero_shotprot_grpo_combined_k8_k128_mean_max.png`
- `outputs/rl_vs_og/zero_shotprot_combined_budgets_grpo/zero_shotprot_grpo_combined_k8_k128_mean_max.pdf`
- `outputs/rl_vs_og/zero_shotprot_combined_budgets_grpo/zero_shotprot_grpo_combined_k8_k128_mean_max.svg`
- Summary: `outputs/rl_vs_og/zero_shotprot_combined_budgets_grpo/plot_summary.txt`

## 2026-06-09 - launched GRPO reproduction run

Launched full GRPO-focused reproduction run after committing cleanup and determinism changes.

Commits:
- Parent repo: `9bc0464 Lock reproduction to GRPO workflow`
- ProSpero submodule: `1f84c0c Make ProSST reproduction GRPO-only`

Run:
- tmux session: `grpo_reproduction_20260609`
- Command: `uv run python scripts/reproduce.py --timestamp grpo_reproduction_20260609 --gpu GPU-025e10e6-263f-d814-6dd5-added86fc8af`
- Output root: `outputs/reproduction/grpo_reproduction_20260609/`
- Launch log: `outputs/grpo_reproduction_20260609.launch.log`
- GPU: A6000 UUID `GPU-025e10e6-263f-d814-6dd5-added86fc8af`

Stages in recipe:
- ProSpero CNN variable-K sanity check.
- 0shotProt ProSST GRPO restricted/cluster vocabulary.
- 0shotProt ProSST GRPO unrestricted/full vocabulary ablation.
- Epistasis additivity tests.

Initial status:
- Started stage `prospero_AAV`.
- Created `outputs/reproduction/grpo_reproduction_20260609/config.json` and `logs/0000_prospero_AAV.log`.

---
name: autotune
description: >
  Tune training resource utilization and speed for the factorized chart generator:
  benchmark throughput/memory across batch size, bf16 mixed precision (AMP), and
  length bucketing, and run an Optuna hyperparameter search over the generator's
  knobs. Use when the user asks to optimize batch size, GPU/memory utilization,
  training speed, or to run a hyperparameter sweep for train_factorized.py. Reports
  the fastest config that fits and the best HPs; does NOT modify a running job.
---

# autotune

Find the fastest training config that fits on the GPU, and search hyperparameters —
without perturbing an in-flight run.

## First, read the GPU correctly

`nvidia-smi` shows two numbers people conflate:

- **memory.used** — how much VRAM the activations/weights occupy.
- **utilization.gpu** — how busy the compute cores are.

**Low memory does NOT mean underutilized.** If `utilization.gpu` is already ~90%+,
the card is compute-bound and raising batch size mostly gives *fewer, larger* steps
at similar wall-clock — not a free speedup. Check util first:

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

## The levers (highest value first)

1. **bf16 mixed precision (AMP)** — `train_factorized.py` runs pure fp32; the RTX 3060
   has Tensor Cores idle. `torch.autocast('cuda', dtype=torch.bfloat16)` speeds up the
   matmuls and ~halves activation memory. bf16 needs no GradScaler (unlike fp16). This
   is usually the biggest win.
2. **Length bucketing** — batches pad to the longest chart, and the onset encoder is
   O(T²), so one long sequence inflates the whole batch. Grouping similar lengths cuts
   wasted compute on padding. Free.
3. **Batch size** — once AMP frees memory, larger batches improve GPU efficiency *if*
   there's per-step launch/Python overhead. Benefit is modest when already compute-bound.
4. **Hyperparameters** (Optuna) — optimizes the val metric, not speed. Separate concern.

## Commands

Run only when the GPU is **idle** — a concurrent job makes timings meaningless (the
scripts print a warning if VRAM is already occupied).

```bash
# Throughput/memory sweep -> recommends fastest fitting config
python experiments/autotune/bench_throughput.py --data_dir data/ --audio_dir data/

# Narrow it down
python experiments/autotune/bench_throughput.py --data_dir data/ --audio_dir data/ \
    --batch_sizes 8 16 32 48 --amp both --bucketing both --warmup 5 --measure 20

# Optuna HP search (val_total, MedianPruner, resumable sqlite study)
python experiments/autotune/optuna_search.py --data_dir data/ --audio_dir data/ \
    --n_trials 30 --epochs 6
```

Files live in `experiments/autotune/`:
- `_harness.py` — shared model/data/step logic, reuses `train_factorized.setup/collect`
  (same warm cache) and replicates its batching/loss exactly.
- `bench_throughput.py` — the speed/memory sweep.
- `optuna_search.py` — HP search; study persists in `optuna_factorized.db`, resume by
  reusing `--study_name`.

## Applying results

The benchmark prints the winning `amp` / `bucketing` / `batch_size`. To use them in real
training, `train_factorized.py` needs three small additions (it doesn't take `--amp`/
`--bucketed` yet):

1. add `--amp` and `--bucketed` flags to `parse_args()`;
2. in `batches()`, when bucketed, sort indices by `len(s['chart'])` and shuffle batch
   order (see `_harness.iter_batches`);
3. wrap the forward + loss in `train_factorized.compute_losses` call with
   `torch.autocast('cuda', dtype=torch.bfloat16)` when `--amp` (no GradScaler for bf16).

Per CLAUDE.md (one change at a time, log everything): apply to the **next** run as a new
MLflow run, not to a job already in progress — AMP/batch changes alter training dynamics.

## Caveats

- bf16 changes numerics slightly; expect tiny metric deltas vs fp32. Validate the chosen
  config trains to comparable val_total before trusting the speedup.
- The benchmark builds a fresh model per config (warm-started from the Phase-1 checkpoint),
  so peak-memory numbers include the real encoder, not a toy.
- Optuna optimizes `val_total` (onset BCE + panel_loss_weight·panel CE), the same quantity
  the trainer checkpoints on — not generation F1. Confirm the best HPs on the full eval.

# Stage 2 — Autoregressive Transformer Generator

*Run: 2026-06-17 (seed 42). Phase 1 splits. 20 epochs, audio encoder warm-started from `standard_ordinal_multi` and frozen for 3 epochs. Eval: 64 val songs, max_gen_len=768.*

## Model

`ChartGenerator` (`src/generation/transformer.py`), 1.12M params:
- Reuses the Phase 1 `AudioEncoder` (128-d) as the conditioning encoder, warm-started from the best classifier (12 tensors loaded).
- Transformer decoder (4 layers, 8 heads, d=128): causal self-attention over step tokens + cross-attention to audio memory; difficulty embedding added at every position.
- Teacher-forced training, masked class-weighted CE (inverse-sqrt over the 16 panel-states to fight empty dominance). Checkpoint on val CE.

Training converged cleanly: val_ce 1.623 → **1.523** (vs per-frame MLP 1.770), so it learned more than the history-blind baseline.

## Headline: decoding matters enormously

Greedy argmax **collapses to all-empty** (onset_F1=0) — the dominant empty state wins every per-frame argmax even though the model learned a real distribution. Sampling unlocks it:

| decoding | onset_F1 | onset_P | onset_R | gen_density | crit_adj | crit_mae |
|---|---|---|---|---|---|---|
| greedy | 0.000 | 0.000 | 0.000 | 0.000 | 0.469 | 1.562 |
| temp=1.0 | **0.300** | 0.210 | 0.577 | 0.536 | 0.562 | 1.312 |
| temp=0.9 top_k=4 | 0.288 | 0.209 | 0.496 | 0.466 | 0.656 | 1.172 |
| temp=0.7 top_k=2 | 0.245 | 0.208 | 0.332 | 0.308 | **0.984** | 0.375 |

Real val density: 0.196. **Floor**: per-frame MLP onset_F1 = 0.053; n-gram crit_adj = 0.977.

## Conclusions

1. **Stage 2 clears the floor on the hard axis.** onset_F1 0.053 → **0.30** (5.7×). The transformer learned rhythmic onset alignment to the audio — the thing both Stage 1 baselines failed at. Greedy decoding hid this entirely; report sampled metrics.
2. **Clear temperature tradeoff.** High temp (1.0) maximizes onset recall/F1 but over-places (density 0.536 >> 0.196) and weakens difficulty fidelity. Low temp + top_k (0.7/2) gives the best difficulty conditioning (crit_adj 0.984 ≈ n-gram, MAE 0.375) and closest density, at some onset recall. **Sweet spot ≈ temp 0.7–0.9 top_k 2–4**: beats the floor on onsets while preserving difficulty conditioning.
3. **The design doc's empty-imbalance concern is real.** Class-weighted CE was not enough to make greedy viable; the win came from sampling. Density is still high at usable temperatures (over-placing).

## Next (Stage 3+ ideas)

- **Density calibration**: focal loss or the factorized onset-then-panel objective (design doc) to control over-placement directly.
- **Decoding**: tune temp/top_k/top_p as first-class hyperparameters; nucleus sampling.
- **Capacity/length**: longer training, KV-cache for full-length (1440) generation (current eval caps at 768 for O(T^2) decode cost).
- **End-to-end**: write best samples to `.sm` via the Stage 0 writer and listen.

Code: `src/generation/transformer.py`, `experiments/generation_transformer/{train_transformer.py,eval_decoding.py}`.
Reproduce eval: `python experiments/generation_transformer/eval_decoding.py --data_dir data/ --audio_dir data/`.

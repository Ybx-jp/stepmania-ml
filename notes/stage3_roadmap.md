# Stage 3+ Roadmap

Stage 2 proved the AR transformer learns onset alignment (onset_F1 0.30 vs the
0.053 floor), but with two honest weaknesses: it **over-places notes** at usable
temperatures, and **greedy decoding collapses to empty**. The active Stage 3
task is **writing generated samples to `.sm`** (separate branch `gen/sample-export`)
so outputs are playable. The remaining directions, roughly in priority order:

## 1. Density calibration (highest value)

The model over-places at temps that give good onset recall (gen_density 0.54 vs
real 0.20 at temp=1.0). Options:

- **Focal loss** instead of class-weighted CE — focuses on hard/rare positives
  without the blunt per-class reweighting that inflates note probability everywhere.
- **Factorized onset-then-panel objective** (from the design doc): head 1 predicts
  onset vs no-onset per frame; head 2 predicts the panel-state given an onset. Lets
  us calibrate the onset rate directly and decode panels only where a step exists.
- **Density-matched decoding**: pick a per-song onset threshold / temperature so
  generated density matches the target difficulty's empirical density.

## 2. Decoding as a first-class hyperparameter

Stage 2 showed decoding dominates the result. Treat temp / top_k / top_p / nucleus
as tunable knobs with a proper sweep, and pick by the onset_F1 vs critic-adjacent
tradeoff rather than the single greedy number reported during training. Update
`train_transformer.py`'s in-training eval to use a sampled setting (greedy is
misleading as the headline).

## 3. KV-cache for full-length generation

Current `generate()` re-runs the decoder over the growing prefix each step (O(T^2)),
so eval caps at `max_gen_len=768`. A key/value cache makes decoding O(T) and unlocks
full 1440-frame (2-minute) songs at reasonable cost. Needed before shipping
real-length charts.

## 4. Capacity & training budget

1.12M params / 20 epochs is small. Try more layers / d_model, longer training with a
schedule, and label smoothing. Watch for overfitting on ~3800 train songs.

## 5. Evaluation hardening

- Report onset_F1 with a small timing tolerance (±1 frame) — exact-frame onset F1 is
  harsh given audio-feature jitter.
- Add playability heuristics (no impossible same-foot patterns, jump-density bounds).
- Hold out a few songs for qualitative listening, not just aggregate metrics.

## 6. Scope expansion (Phase 2.5)

Out of current Phase 1 scope but natural extensions: holds/rolls (extend the
tokenizer + writer beyond binary taps), variable BPM, 32nd-note resolution.

# Phase 2 Design — Generative Chart Creation

*Last updated: 2026-06-17*

## Goal

Generate playable StepMania step charts conditioned on **audio features** and a **target difficulty**, reusing the Phase 1 encoder backbone. Input: a song's audio features + desired difficulty (Beginner/Easy/Medium/Hard). Output: a `(T, 4)` step sequence renderable to a `.sm` file.

Decision (2026-06-17): use an **autoregressive transformer**, not diffusion. Charts are discrete, time-ordered symbol sequences perfectly aligned to audio frames — the natural fit for an AR decoder, and it reuses the Phase 1 `AudioEncoder` directly with lower implementation risk than discrete diffusion.

## What we inherit from Phase 1 (frozen data layer)

- **Chart tensor**: `(T, 4)` binary, panels `[Left, Down, Up, Right]`, 16th-note resolution. (`docs/chart_representation.md`)
- **Audio features**: `(T, 23)` = MFCC(13) + onset_env(1) + onset_rate(1) + tempo(1) + spectral_contrast(7), frame-aligned 1:1 with chart timesteps.
- **Dataset** (`src/data/dataset.py`) `__getitem__` returns: `chart (T,4)`, `audio (T,23)`, `mask (T,)`, `difficulty (scalar 0–3)`, `groove_radar (5,)`. Max sequence length 1440 (2 min @ 16th notes).
- **`AudioEncoder`** (`src/models/components/encoders.py`): `(B,L,23) → (B,L,256)` per-timestep features via Conv1D blocks. **Reused as the conditioning encoder.**
- **Best classifier**: `standard_ordinal_multi` (82.9% test acc) — used both for **audio-encoder warm-start** and as a **generation critic** (see Evaluation).

The alignment guarantee (`audio[t]` ↔ `chart[t]`) is the central asset: generation is "predict the step-state at frame `t` given the audio context and the steps so far."

## Task formulation

### Tokenization
Each timestep's `(4,)` binary vector is one of `2^4 = 16` panel-states. The chart becomes a length-`T` sequence over a vocabulary:

```
vocab = 16 panel-states (0b0000 … 0b1111)  +  {PAD, BOS, EOS}   ->  19 tokens
```

State `b3 b2 b1 b0` ↔ panels `[L,D,U,R]` (e.g. `0b1010` = Down+Right jump). `panel_state_to_token` / `token_to_panel_state` are pure bit-twiddling — trivial and lossless for Phase 1 scope (no holds/rolls).

### Modeling target
At frame `t`, predict `token[t]` given:
- all audio frames (non-causal over audio — the whole song is known at generation time),
- previous step tokens `token[0..t-1]` (causal),
- a target-difficulty embedding.

This is an audio-conditioned autoregressive sequence model.

## Architecture

```
audio (B,T,23) ──> AudioEncoder ──> audio_ctx (B,T,256)         [warm-start from Phase 1]
difficulty (B,) ─> Embedding(4, d) ─> diff_emb (B,d)
step tokens ────> Embedding(19, d) + positional ─> tok_emb (B,T,d)

Decoder block × N:
  causal self-attention over tok_emb
  cross-attention: queries=tok_emb, keys/values=audio_ctx   (time-aligned conditioning)
  + diff_emb broadcast/added at each position
  FFN

head: Linear(d, 19) ─> per-frame logits over vocab
```

- **Conditioning on difficulty**: add `diff_emb` to every position (and optionally prepend as the first token after BOS). Lets one model serve all difficulties and enables difficulty-controlled generation.
- **Cross-attention vs. additive fusion**: start with cross-attention to `audio_ctx`. Because audio/steps are 1:1 aligned, a cheaper ablation is direct per-position fusion (concat `audio_ctx[t]` into `tok_emb[t]`); compare later (one change at a time).
- **Sizes (initial)**: `d=256` (matches AudioEncoder), `N=4–6` layers, 4–8 heads. Keep it small first.

## Loss

- Cross-entropy over the 19-token vocab, **masked by `mask`** (ignore padding).
- **Severe class imbalance**: the empty state `0b0000` dominates most frames. Mitigations to try in order: (1) class-weighted CE, (2) focal loss, (3) factorized objective — first predict onset/no-onset per frame, then which panels given an onset. Start with weighted CE.

## Training

Follows the project training loop pattern (`model.train()` / `model.eval()` / `torch.no_grad()` / `optimizer.zero_grad()`, checkpoint on **validation** metric, `set_seed()` first). Teacher forcing during training. MLflow experiment: `stepmania-chart-generator`.

- **Warm-start** the `AudioEncoder` from `checkpoints/ordinal_exp/standard_ordinal_multi/best_val_loss.pt`. Optionally freeze it for the first few epochs, then unfreeze.
- Reuse stratified splits; never fit on val/test.

## Sampling / inference

Autoregressive decode frame-by-frame: temperature + top-k/top-p over the 19-way logits, mask out `PAD/BOS`, stop at `EOS` or song end. Convert tokens → `(T,4)` → `.sm` via the new writer.

## Evaluation (define before optimizing)

Per-frame token accuracy is misleading (empty dominates). Primary metrics:

1. **Difficulty-conditioning fidelity** *(the elegant tie-back)*: run the Phase 1 `standard_ordinal_multi` classifier on generated charts; measure agreement between predicted difficulty and the requested target. This reuses Phase 1 as a learned critic.
2. **Onset/density F1**: did we place a step where one belongs (ignoring exact panel)? Captures rhythmic correctness.
3. **Panel accuracy | onset**: given a step exists, are the right panels chosen?
4. **Groove-radar distribution match**: compare radar stats of generated vs. real charts at the same difficulty.
5. **Playability heuristics** (later): no impossible same-foot patterns, reasonable jump density, etc.

Baselines to beat (Stage 1): unconditional frequency/bigram model and a per-frame MLP on audio features only.

## Staged plan

- **Stage 0 — Infrastructure (no modeling).** Chart writer (`tensor → .sm`) + tokenizer (`panel_state ↔ token`) + generation-eval module (classifier-critic + onset/density F1). Unblocks everything, zero modeling risk. **← starting here.**
- **Stage 1 — Dumb baselines.** Frequency/bigram + per-frame MLP. Establish the floor.
- **Stage 2 — AR transformer.** Architecture above, audio-encoder warm-start, masked weighted-CE.
- **Stage 3 — Sampling & full eval.** Decode → `.sm`, run the eval suite, iterate one change at a time.

## Open questions / deferred

- Holds/rolls and variable BPM are out of Phase 1 scope; revisit when expanding the data layer.
- Whether to predict at full 1440 length or chunk with overlap (memory vs. long-range structure).
- Cross-attention vs. aligned additive fusion — decide empirically in Stage 2.

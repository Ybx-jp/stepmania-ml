# StepMania Chart Generator

Generate playable [StepMania](https://www.stepmania.com/) dance charts from audio: feed it a song
and a target difficulty, get back a chart whose arrows land on the music and match the difficulty
you asked for — with controllable style and a learned sense of *taste*.

> **Live demo:** _Hugging Face Space — coming soon._
> **Sample charts:** generated `.sm` files re-parse cleanly and play in StepMania, with holds.

This is a from-scratch deep-learning project. The model — a multi-modal, autoregressive,
audio-conditioned generator — turned out to be the *easy* part. The hard part was this:

> **Every standard metric said the charts were great while they didn't *feel* musical.
> So I built a way to measure taste.**

That thread — distrusting your own green metrics, then engineering the missing one — is what the
project is really about. See [The interesting part](#the-interesting-part).

---

## What it does

Charts are sequences of arrows on 4 panels (left/down/up/right), placed in time against a song.
The system models two problems:

1. **Difficulty classification** — given a chart + its audio, predict the difficulty class
   (Beginner / Easy / Medium / Hard).
2. **Conditional generation** — given audio + a target difficulty, *write* the chart: where notes
   go, which panels, taps vs. holds — with optional **style/feel control**.

The classifier from (1) is reused three ways in (2): as a **warm-start** for the generator's audio
encoder, as a **difficulty critic** that checks whether a generated chart reads at the requested
level, and as the backbone for a separately-trained **taste critic** (below).

---

## Results

Generation numbers are on a held-out validation set. The headline metric is **onset F1** — did a
note land on the right frame — because difficulty conditioning turned out to be easy and *rhythmic
alignment to the audio* is the hard axis. `crit-adj` is critic-adjacent difficulty fidelity (the
fraction of charts the difficulty critic reads as the requested level ±1).

| Stage | Model | onset F1 | Notes |
|---|---|---|---|
| Floor | difficulty n-gram + audio MLP | 0.053 | audio-blind; rhythm unsolved |
| Phase 1 | ordinal multi-output classifier | — | 82.9% test acc, **0.835 macro F1** |
| Phase 2 | AR transformer (sampled) | 0.300 | 5.7× the floor; greedy collapses to empty |
| Phase 2 | **factorized onset+panel head (focal)** | **0.748** | crit-adj **0.927**; over-placement + collapse solved |
| Phase 2.5 | layered typed head + hold-aware decode | ~0.77 | generates **holds** at the real rate |

And the metric the project lacked until late:

- **Taste critic** (a learned real-vs-corrupted discriminator): the first quantitative musicality
  signal here — ranks generations in the same order humans do, **REAL 0.823 > BASE 0.290 > CHAOS 0.003**
  (AUC 0.964). Every prior metric was blind to this.

Plus the engineering that makes it practical:

- **Controllability:** trained groove-radar conditioning + **classifier-free guidance** + reference-
  chart **style transfer** — steer density, syncopation, jumps, holds, or transfer another chart's feel.
- **Calibration:** per-difficulty Platt scaling brought onset ECE from ~0.17 to ~0.01.
- **KV-cache:** a hand-rolled incremental decoder, **bit-identical** to the reference path, takes a
  full 1440-frame (~2 min) generation from 33.4s to 3.6s (**9.2×**).
- **Playable output:** generated charts write back to `.sm`, re-parse with valid hold spans, and
  play in StepMania alongside the original for A/B.

---

## The interesting part

Most of the depth is in failure modes diagnosed from probes and fixed — and the realization that
the metrics couldn't see the thing that mattered.

- **Greedy decoding collapses to silence.** Most frames have no note, so argmax learns to place
  nothing; sampling unlocks output but over-places. The fix wasn't a temperature knob — it was a
  **factorized head** with an *audio-driven, non-causal onset predictor* so density is a stable,
  honest threshold independent of autoregressive token feedback (no drift / exposure bias).
- **Weighted loss distorts generation calibration.** Class weights that fix training imbalance push
  greedy decode to *over*-pick the rare class — same story for onset `pos_weight` and for hold
  weights. **Focal loss** lifted the whole F1-vs-fidelity frontier instead of trading along it.
- **Decode was usually the bottleneck, not the model.** Always-Left arrows, unnatural jacks,
  hold spans, crossovers, jump-during-hold — over and over, the model already encoded the right
  thing and the *default decode* was hiding it. A lot of "playability" is post-hoc constraint, not
  model capacity.
- **The metrics were blind to musicality.** Adding chroma/HPSS/metric-phase audio features left
  every offline metric flat — but in a blind playtest the charts were "definitely more musical."
  The numbers couldn't see it. (This is why there's a [playtest log](notes/playtest_log.md) with a
  defect taxonomy and standing hypotheses — qualitative evaluation is a first-class artifact here.)
- **So I built the missing metric.** A learned **taste critic** scores real-vs-fake. v1 failed: it
  learned the *generator's fingerprint* instead of taste, and scored generations backwards. The fix
  was to engineer the negatives — **corrupted-real**: perturb real charts at fixed density/timing so
  the *only* remaining cue is arrow-choice taste. That hit AUC 0.964 and finally ranked generations
  the way a human does. **Lesson: a discriminator optimizes the easiest separating cue — so engineer
  the negatives so the only available cue is the concept you actually want to measure.**

Full write-ups, one per experiment, live in [`notes/`](notes/).

---

## Architecture

```
audio (MFCC / chroma / HPSS / onset / spectral / metric-phase)     target difficulty   [+ radar / style]
        │                                                                  │                    │
        ▼                                                                  ▼                    ▼
  AudioEncoder (Conv1D) ───warm-start───► difficulty embedding  +  groove-radar / style latent (CFG)
        │                                                  │
        └──────────────────► cross-attention ◄────────────┘
                                   │
                ┌──────────────────┼───────────────────┐
                ▼                  ▼                     ▼
        onset head          which-panels head      per-panel type head
   (non-causal, audio,     (autoregressive)       (tap / hold-head / tail)
    BCE / focal)
                └──────────────► chart tensor (T × 4 symbols) ──► .sm writer
```

- **Tokenizer** (`src/generation/tokenizer.py`): lossless `(T, 4)` ↔ token.
- **Factorized generator** (`src/generation/factorized.py`): audio-driven onset + AR panel.
- **Typed generator** (`src/generation/typed_model.py`, `typed.py`): tap/hold symbols + the layered
  onset→which-panels→type head; conditioning (radar, style), CFG, and KV-cached decode.
- **`.sm` writer** (`src/generation/sm_writer.py`): tensor → playable chart, inverse of the parser.
- **Critics** (`src/generation/evaluation.py`, `experiments/realism_critic/`): the difficulty critic
  and the learned taste critic, both reusing the Phase-1 classifier backbone.

---

## Run it

```bash
conda env create -f environment.yml
conda activate stepmania-chart-gen
pip install -e .
```

**Phase 1 — difficulty classifier:**
```bash
python scripts/train.py --config config/model_config.yaml --data_dir data/ --audio_dir data/
python scripts/evaluate.py --checkpoint checkpoints/ordinal_exp/standard_ordinal_multi/best_val_loss.pt \
    --config config/model_config.yaml --data_dir data/ --audio_dir data/
```

**Phase 2 — generator:**
```bash
# layered typed generator on musical features (chroma/HPSS/metric-phase)
python experiments/generation_typed/train_stage1.py
# groove-radar conditioning + classifier-free guidance
python experiments/generation_typed/train_radar.py
```

**Generate playable charts:**
```bash
python experiments/generation_typed/export_typed_samples.py \
    --data_dir data/ --audio_dir data/ --radar "chaos=0.9,air=0.85" --guidance 1.4
```

**Taste critic + best-of-N reranking:**
```bash
python experiments/realism_critic/train_critic.py     # corrupted-real negatives
python experiments/realism_critic/eval_taste.py        # REAL > BASE > CHAOS ranking
python experiments/realism_critic/export_reranked.py   # keep highest-taste of N candidates
```

**Tests:**
```bash
pytest tests/
```

---

## Scope & honesty notes

- 4-panel (DDR-style) charts. Difficulty: Beginner / Easy / Medium / Hard.
- **Holds are supported. Rolls are not** — there are zero rolls in the training data (0/675), so
  the model never learns to place them. A data limit, stated rather than hidden.
- **Musicality is improved, not solved.** Onset F1 in the high-0.7s means most notes land on-beat,
  not every note. Open problems: the `chaos` (syncopation) knob still smears off-grid; the model has
  no song-structure/climax awareness (density stays flat where humans build to a finale); the AR
  decoder has an awkward cold-start. These are tracked in [`notes/playtest_log.md`](notes/playtest_log.md).
- The taste critic is a within-project signal validated against the playtest ranking on this dataset
  — not a general-purpose musicality oracle.

---

## Methodology

Baseline-first ML discipline (enforced by a custom Claude Code plugin):

- A dumb baseline established the floor (onset F1 0.053) *before* any transformer, which is how we
  learned that rhythm — not difficulty — was the hard axis.
- One variable per experiment; everything tracked in MLflow; deterministic seeding across
  torch / numpy / random / cudnn.
- Every finding has a write-up in [`notes/`](notes/), and play-feel that the metrics can't capture
  has its own qualitative ledger in [`notes/playtest_log.md`](notes/playtest_log.md).

---

## Repository layout

```
src/
  data/         chart parser, audio features (MFCC/chroma/HPSS/phase), groove radar, dataset
  models/       LateFusionClassifier, baselines, ordinal heads
  generation/   tokenizer, transformer, factorized, typed, sm_writer, evaluation
  losses/       contrastive + ordinal losses
  training/     trainers, callbacks
experiments/    one folder per experiment (baselines → transformer → factorized → typed →
                conditioning → musical-features → realism-critic)
notes/          a findings write-up per experiment + the playtest log
checkpoints/    trained weights
tests/          unit + regression tests (incl. KV-cache bit-identity)
```

# StepMania Chart Generator

Generate playable [StepMania](https://www.stepmania.com/) dance charts from audio: feed it a song
and a target difficulty, get back a chart whose arrows land on the music and match the difficulty
you asked for — with controllable style and a learned sense of *taste*.

> **Status:** research project, pre-1.0 — MIT licensed. No hosted demo yet; you can
> [generate playable charts locally](#run-it) and read the hands-on [playtest log](notes/playtest_log.md).
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
| Phase 2.5 | layered typed head + hold-aware decode | ~0.77 | generates **holds** at near the real rate |

And the metric the project lacked until late:

- **Taste critic** (a learned discriminator, **AUC 0.964** separating real charts from *corrupted-real*
  ones): the first quantitative musicality signal here. Turned on generations, it ranks them in the same
  order humans do — **REAL 0.823 > BASE 0.290 > CHAOS 0.003** (that CHAOS is the deliberate off-manifold
  *flood*). It **re-validates on the current decode stack**, and independently scored the **chaos-conditioning
  redesign** as more musical (see Limitations) — the metric catching a quality change humans felt but no prior
  number could see. Every prior metric was blind to this. And we can say *what* it measures: an interpretability
  probe (perturbation saliency + Conv1D activation maps, [`critic_interpretability.ipynb`](experiments/realism_critic/critic_interpretability.ipynb))
  shows its "fake" evidence is **off-grid flooding** — removing a bad chart's off-grid notes recovers ~half its
  score, while the on-grid backbone is what "real" rests on.

Plus the engineering that makes it practical:

- **Controllability:** groove-radar conditioning + **classifier-free guidance** to steer density,
  syncopation, jumps, and holds. The shipped path is the in-distribution `--style` surface (it fills a
  loose groove request out onto the learned groove **manifold**); raw `--radar` point-conditioning is
  disabled because it steers off-manifold. Reference-chart **style transfer** — transfer another chart's
  holistic *feel* — is built and validated offline but **experimental: not exposed in the CLI**.
- **Decode-time biomechanical governor:** a per-note two-foot **fatigue** model (foots jacks onto a
  human distribution), a per-region **stamina** relief valve (thins density under sustained workload),
  and an energy-following difficulty **arc** — physical plausibility applied at *decode* rather than
  retrained in (the recurring "the bottleneck is the decode, not the model" theme). Playtest-confirmed
  as "a tasteful edit, not a rewrite."
- **Calibration:** per-difficulty Platt scaling brought onset ECE from ~0.17 to ~0.01.
- **KV-cache:** a hand-rolled incremental decoder, **bit-identical** to the reference path, takes a
  full 1440-frame (~2 min) generation from 33.4s to 3.6s (**9.2×**, batch of 4).
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
  onset→which-panels→type head; conditioning (radar, style), CFG, KV-cached decode, and the
  decode-time fatigue/stamina governor.
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

**Generate a chart for your own song** (no dataset needed — just an audio file):
```bash
python scripts/generate.py --audio path/to/song.ogg --difficulty Hard --out MyGenerated
# optional: --bpm 174 (else estimated), --style "chaos=q0.7" (groove feel)
```
`--out` is the **group** folder; the song is nested inside it as `MyGenerated/<title>/{chart.sm, audio}`,
because StepMania expects `Songs/<group>/<song>/<files>` (a bare song folder dropped into `Songs/` becomes
an empty group and won't show). Drop the `MyGenerated` folder into your StepMania Songs directory to play.
Uses the deployed 42-dim model and the shipped groove manifold; BPM is auto-estimated; songs longer than the
model's ~3–4 min context are truncated. Requires the weights (see below).

**Evaluate on held-out songs** (A/B vs the human chart — needs the training dataset on disk):
```bash
# --style is the in-distribution (manifold) conditioning path; --radar is disabled (off-manifold).
python experiments/generation_typed/export_typed_samples.py \
    --data_dir data/ --audio_dir data/ --style "chaos=q0.9" --guidance 1.5
```

**Taste critic + best-of-N reranking** (reranking is built but not yet playtest-validated)**:**
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
- **Song length is capped to the model's trained context (~3–4 min).** The generator's positional
  encoding is 2048 sixteenth-note frames; longer songs are charted up to that point and truncated
  (with a warning), not chunked. A windowed long-song decode is future work.
- **Musicality is improved, not solved.** Onset F1 in the high-0.7s means most notes land on-beat,
  not every note; and the AR decoder has an awkward cold-start.
- **Chaos / syncopation is in-distribution-bounded (a studied, bounded behavior, not an open defect).** The
  `chaos` knob produces *musical*, choreographed intensity while the conditioning stays on the groove
  **manifold** — the deployed radar path fills unset dims via the Gaussian conditional and projects
  onto the covariance ellipsoid (the in-distribution shell). Pushed *past* the manifold it degrades to
  a uniform 16th "wall" that actually reads as *less* demanding, so literal 16th-note share isn't the
  dial (felt chaos peaks mid-range, not at the flood). The lever is staying in-distribution, not
  cranking harder. **The taste critic confirms this independently:** holding the generator fixed and varying
  only *how chaos is requested*, it scores the in-distribution manifold chaos well above the off-manifold
  mean-pin flood (P(real) **0.228 vs 0.028**, n=64, 73% of songs), and the old flood-request still scores
  ~0.03 on the current model — so the gain is the *conditioning redesign*, not the model. A quantitative
  echo of the by-ear judgment that the flood reads as worse. (A *separator* signal, not a playtest; and
  "more musical" here means "stopped flooding off-grid 16ths," not "added spice.") An interpretability probe
  pins down *why* at the input level: the critic's "fake" evidence lives on the off-grid flood itself —
  removing those notes recovers ~half the score gap — while it leaves *sparse* off-grid syncopation alone (those
  are tasteful). The metric and the manifold conditioning agree on the same axis. ([`critic_interpretability.ipynb`](experiments/realism_critic/critic_interpretability.ipynb))
- **Song structure — a strength, with one open frontier.** Playtests consistently read charts as
  *"in character with the song"* — accents land, choreography escalates, and the model picks jacks
  where the music wants stomping; the audio-driven onset backbone follows energy into an intensity
  arc. What's *less* certain is deliberate **global, whole-song phrase planning** (long-range density
  build, clean phrase boundaries), which frame-local features don't directly encode — the optional
  decode-time governor can impose an energy-following difficulty arc on top. These are tracked in [`notes/playtest_log.md`](notes/playtest_log.md).
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

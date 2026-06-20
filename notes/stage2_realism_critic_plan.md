# Stage 2 plan — realism critic for "taste"

*2026-06-20. Motivated by the playtests: the Stage-1 model plays "mostly right" but has "no sense of
taste" (see `playtest_log.md`). Diagnosis converged on the **training objective** — frame-wise CE
rewards matching the reference token, never "is this a coherent, tasteful chart." This plan adds a
learned taste signal and uses it to improve generation — and gives us the **taste metric** we lack.*

## The idea

Train a **realism critic** (discriminator): given the audio + a chart, predict P(real human chart).
Real charts are positives; generated charts are negatives. The critic learns what's *unrealistic* about
our generations — i.e., it learns taste from the data. Then use it two ways, cheap first:

1. **Best-of-N reranking (decode-time, no retrain)** — generate N candidate charts per song, score each
   with the critic, keep the highest P(real). Tests whether the critic's signal improves play-feel
   before any expensive retraining.
2. **Critic-guided fine-tuning (only if reranking helps)** — push the generator toward high-P(real)
   outputs. Prefer **self-distillation on critic-filtered samples** (generate, keep the top-scoring,
   fine-tune the generator on them) over full adversarial GAN training — more stable on the 3060.

Byproduct: **P(real) on generated charts is the taste metric** we've been missing (every offline metric
so far is blind to musicality). Validate it correlates with play-feel — the base set the user liked
should score higher than the chaos sets.

## Architecture (reuse Phase 1)

The Phase 1 `LateFusionClassifier` (`src/models/classifier.py`) already fuses a dual audio/chart encoder
for difficulty classification. The critic is the same backbone with a **binary real/fake head** instead
of the 4-way difficulty head — reuse `AudioEncoder` + `ChartEncoder` + late fusion. Keep it small
(dataset is ~3820 train samples; a big discriminator will overfit). Audio = the 23-dim features (the
critic doesn't need chroma; it judges chart-vs-audio realism). Chart = binary occupancy or typed.

## Negatives (what makes the critic learn taste, not a shortcut)

A discriminator will cheat on trivial cues (density, hold rate) if allowed. Use varied negatives:
- **Generated** charts from `gen_stage1` (recommended decode) — the thing we want to improve.
- **Mismatched** pairs (real chart + *wrong* song's audio) — forces the critic to judge chart-vs-audio
  *alignment*, not just "is this a plausible chart in isolation". This is the audio-grounding term that
  most directly encodes "taste = fits the music."
- (optional) **corrupted-real** (shuffle/jitter a real chart) — teaches local-coherence cues.
Start with generated + mismatched. Audit what the critic keys on (ablate density etc.) so it isn't a
density detector.

## Staging

- **2a — build + train the critic.** Validate it separates real vs generated (ROC-AUC) AND that it's
  audio-grounded (mismatched pairs score low). Sanity: base set scores > chaos set (≈ the playtest
  ranking). Establishes the taste metric. *(Stop here and check before escalating.)*
- **2b — best-of-N reranking** with the critic; playtest the reranked charts. Cheap, decode-time.
- **2c — critic-guided fine-tune** (self-distillation on critic-filtered samples) only if 2b helps.

## Risks / open decisions

- **Critic shortcuts** (density/length detector instead of taste) — mitigate with mismatched negatives
  + an ablation audit; this is the main thing to watch in 2a.
- **Critic staleness** — as the generator improves (2c), the critic must be refreshed (the adversarial
  loop). For reranking (2b) a static critic is fine.
- **Decision (defer to 2c):** fine-tune method — self-distillation (stable, recommended) vs adversarial.
- **Decision (now, 2a):** negatives = generated + mismatched (recommended) vs generated-only.

## Success criteria

2a: critic ROC-AUC clearly > 0.5 separating real/gen, mismatched pairs score low, and base > chaos
(taste metric tracks play-feel). 2b: a reranked set that the user judges more tasteful than the
unranked stage1 base. 2c (if reached): generator's mean P(real) rises and a playtest confirms it.

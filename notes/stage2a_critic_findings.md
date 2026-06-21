# Stage 2a — realism critic: findings (offline)

*2026-06-20. Two attempts. **v1 (generated negatives) FAILED** — learned the generator fingerprint,
taste metric scored backwards. **v2 (corrupted-real negatives) SUCCEEDED** — AUC 0.964 and the taste
metric ranks generations in playtest order (REAL > BASE > CHAOS). See `stage2_realism_critic_plan.md`.*

## v2 (corrupted-real negatives) — SUCCESS ✅

Fix per v1's lesson: negatives are REAL charts perturbed at FIXED density/timing, so the only cue is
taste — **panels** (per note-frame reassign which panels, keep count → kills arrow coherence) and
**shift** (roll chart vs audio → kills alignment). No generator in training. Warm-start from Phase-1.

- **AUC 0.964** (epoch 12) separating real vs corrupted-real. P(real): real ≈ 0.79, panels ≈ 0.03,
  shift ≈ 0.05 → the critic confidently flags both scrambled-arrow and misaligned charts as fake. It
  learned arrow-choice taste AND audio-alignment (the two things v1 missed).
- **Taste metric (`eval_taste.py`, 64 val songs, P(real)): REAL 0.823 > BASE 0.290 > CHAOS 0.003** —
  matches the playtest exactly (base "more musical", chaos "no taste"). **We now have a quantitative
  musicality signal** — the thing every prior metric (onset_F1, crit_adj, phase, structure) couldn't see.

Unblocks **2b (best-of-N reranking)**: a valid taste scorer means we can generate N candidates and keep
the highest P(real). Checkpoint `checkpoints/realism_critic/best_val.pt` (config embedded).

---

## v1 (generated negatives) — FAILED (kept for the record)

*Result: the critic did NOT give a valid taste metric — it learned the generator's fingerprint, not
musical taste. The validation gate the staged plan was designed to catch before escalating.*

## What ran
`train_critic.py`: `LateFusionClassifier` + binary real/fake head, warm-started from the Phase-1
classifier (118 encoder/fusion/backbone params transfer). 1500 train / 300 val songs, 12 epochs.
Three example types per song: positive (audio, real chart), neg-generated (gen_stage1 base-decode),
neg-mismatch (real chart + wrong audio). Critic = 2.7M params, best val AUC (real vs gen+mismatch) 0.723.

## Results
P(real) by category (best epochs): real ≈ 0.39–0.51, **generated ≈ 0.09–0.15**, **mismatch ≈ 0.32–0.48
≈ real**. Taste test (`eval_taste.py`, 64 val songs, critic P(real)):
```
REAL  0.386
BASE  0.123   (lowest)
CHAOS 0.680   (highest)
```
**Ranking is BACKWARDS** vs the playtest (user: base musical, chaos "no taste"). Expected REAL>BASE>CHAOS.

## Diagnosis — three failure modes
1. **Generator-fingerprint shortcut.** The negatives were gen_stage1 *base-style* outputs, so the critic
   learned to flag base generations as fake *by construction* → base scores lowest. It's a
   "human-authored vs our-generator" detector, NOT a "tasteful vs not" detector. (Note: real 0.386 vs
   base 0.123 at the *same* density ~0.2 → it's keying on generation artifacts, not pure density.)
2. **Audio-grounding failed.** mismatch (real chart + wrong audio) ≈ real (0.40 vs 0.39). The mismatch
   negative was too easy to ignore — minimizing loss via the gen-artifact cue was the easier path, so
   the critic never learned "fits-this-music." The audio-grounding term didn't bite.
3. **Out-of-distribution on chaos.** The critic only saw ~0.2-density charts (real + base-gen +
   mismatch all low density). Chaos (~0.6) is extrapolation → its 0.68 is spurious, not a taste signal.

**Conclusion: 2a fails its validation gate.** The critic is not a taste metric and would be actively
misleading for reranking (2b) — it scores base generations (what we'd rerank) uniformly low and rewards
the dense/off-beat chaos charts the playtest rejected. Do NOT proceed to 2b/2c as built.

## Fix options
- **(A, recommended) Isolate taste with corrupted-real negatives.** Build negatives by perturbing REAL
  charts while *preserving density and onset positions* — e.g. shuffle which-panels among note frames,
  or permute pattern assignments. Then the ONLY difference from a positive is arrow-choice coherence =
  taste. This forces the critic off the gen-fingerprint/density shortcuts onto musical judgment. Cheap
  (no generation needed). Could keep a small gen-negative share but make corrupted-real dominant.
- **(B) Force audio-grounding via contrastive pairing** — same audio, real vs perturbed, as a paired
  (not pooled) objective so the critic must use the audio.
- **(C) Step back.** Bank the Stage-1 musical-feature win (PR #25) and pursue structure (H5) or ship;
  a learned taste metric may be hard on this dataset/model size.

## Lesson
A discriminator optimizes the *easiest* separating cue, not the one we want. "Real vs our-generated"
was easier to learn than "tasteful vs not," so the critic learned the former and the taste framing
collapsed. The negatives must be engineered so the *only* available cue is the target concept
(corrupted-real). Same shape as H8/density-shortcut worries — verified here the hard way.

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

---

## Stage 2b — best-of-N at HARD: the critic is trustworthy at Hard; the generator is *tame*

*2026-06-20, follow-on. Driven by the Hard best-of-N playtest (`notes/playtest_log.md`, 2026-06-20
Hard entry).* `export_reranked.py --difficulty Hard` produced a **collapsed critic table**: best-of-8
P(real) tops out at **0.116** (Deja loin), most songs 0.02–0.05, mean lift +0.032 (the mixed-difficulty
set was +0.444). Yet the charts *played well* ("pretty good" / "very good"), with the user noting they
felt **"too tame for an 11" — "expected more chaos."**

Two readings to disambiguate: (1) critic over-rejects dense/Hard charts (blind spot), or (2) the
generator's Hard charts genuinely differ from real Hard (gap is real).

**Decisive diagnostic** — `diag_real_by_difficulty.py` scores 25 *real* charts per difficulty with the
same critic:

| difficulty | n | mean P(real) | median | min | max |
|---|---|---|---|---|---|
| Beginner | 25 | 0.727 | 0.909 | 0.058 | 0.984 |
| Easy | 25 | 0.852 | 0.986 | 0.007 | 0.988 |
| Medium | 25 | 0.852 | 0.985 | 0.035 | 0.988 |
| **Hard** | 25 | **0.818** | **0.983** | 0.037 | 0.987 |

**Real Hard scores 0.82 — as high as every other difficulty. The critic does NOT over-reject Hard.**
So reading (2): the gap is real. Generated Hard ≈ 0.02–0.12 vs real Hard ≈ 0.82 is a genuine
gen-vs-real difference, and the critic is **trustworthy at Hard**.

**What the gap *is*:** not density (tau matches the real chart's density) but **tameness/on-gridness** —
the generator under-syncopates (H4/H7: sits on-beat), so at Hard density it produces a regular on-grid
chart where a real DDR 11 has syncopated intensity. The critic flags this as un-Hard-like; the user
feels it as "expected more chaos." Same defect, two instruments, agreeing.

**Implications for 2b/2c:**
- **2b (best-of-N) has little headroom at Hard** — *all* candidates are tame, so selection can't surface
  intensity that the generator never draws. Best-of-N is a low/mid-difficulty win, not a Hard fix.
- **2c (critic-guided fine-tuning) at Hard is bottlenecked by H4.** Pushing toward higher critic P(real)
  at Hard means pushing toward syncopation/intensity — exactly the chaos-conditioning the model can't
  render (H4 root: can't see which offbeats deserve notes). 2c likely won't fix Hard until chaos is
  fixed at the conditioning/objective level. The critic is *ready*; the generator's capacity is the gate.
- **Secondary (pad-playability):** one B4U sequence required one-footed crossovers+jacks during a hold
  (`--no_jump_during_hold` was on, `--no_crossovers` was not). Worth testing whether `--no_crossovers`
  (or a crossover-under-hold mask) raises P(real) and removes the awkward sequence.

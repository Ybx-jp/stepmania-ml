# Stage 3 — Factorized Onset-then-Panel Generator

*Run: 2026-06-18 (seed 42). 20 epochs, audio encoder warm-started from `standard_ordinal_multi`, frozen 3 epochs. Train/val capped to 1024 frames; eval on 64 val songs, max_gen_len 768.*

## Model

`FactorizedChartGenerator` (`src/generation/factorized.py`), 1.52M params. Splits the single 16-way per-frame softmax of Stage 2 into two heads (the [[factorized head]] design the density probe prescribed):

- **Onset head** — audio-driven, [[non-causal]] transformer encoder over the audio memory (+ difficulty, position) → per-frame P(step). Does **not** read generated step tokens, so it can't collapse under self-generated context (the [[exposure bias|AR drift]] that killed single-head threshold decoding). Loss: [[BCE]] with [[pos_weight]] ≈ 3.7.
- **Panel head** — autoregressive decoder over step tokens → which of the 15 non-empty patterns, given an onset. Loss: cross-entropy on real-onset frames only.

Generation: onset decided in one non-AR pass (stable, density set by threshold or [[Bernoulli sampling]]); only the panel pattern is decoded autoregressively, only where an onset fired.

## Results (test against the whole Phase-2 ladder)

| decode | onset_F1 | precision | recall | density | crit_adj |
|---|---|---|---|---|---|
| **factorized @ τ (Stage 3)** | **0.763** | **0.757** | **0.800** | 0.206 | 0.734 |
| Stage 2 temp 1.0 | 0.300 | 0.210 | 0.577 | 0.536 | 0.562 |
| probe single-head free-run @ τ | 0.000 | — | — | 0.000 | 0.469 |
| Stage 1 per-frame MLP (floor) | 0.053 | — | — | 0.014 | — |

Onset **ROC-AUC 0.950, PR-AUC 0.825** (single-head probe: 0.813 / 0.469). Target density 0.214, achieved 0.206. Global threshold τ=0.849.

## Conclusions

1. **Both diagnosed problems are solved.** Onset alignment: onset_F1 **0.30 → 0.763** (2.5× over Stage 2, 14× the floor) at *correct* density. No collapse — the audio-driven onset head makes threshold decoding stable, exactly as the probe predicted. Precision jumped 0.21 → 0.76: notes now land in the right places, not by carpet-bombing.
2. **The onset head got much sharper** than the single-head's posteriors (ROC-AUC 0.81 → 0.95). Giving onset its own [[non-causal]] encoder over audio — with no token-feedback to drift — is a big win on its own.
3. **Recall 0.80 + precision 0.76 at real density** is the first genuinely usable generation operating point in the project.

## Caveat / next step: difficulty fidelity

`crit_adj` dropped to **0.734** (vs Stage 2 temp-0.7's 0.984 and the n-gram's 0.977). Likely cause: the eval uses **one global onset threshold** for all difficulties, but each difficulty has its own density — a global cutoff over-places on easy charts and under-places on hard ones, smearing the difficulty signal the critic reads. (The critic's fixed-BPM groove-radar approximation adds some noise too.)

Fix: **per-difficulty thresholds** (calibrate τ separately per class to that class's empirical density), or decode onsets by [[Bernoulli sampling]] from the difficulty-conditioned onset head instead of a single global cutoff. Cheap to try on the trained checkpoint — no retraining.

## Reproduce

`python experiments/generation_factorized/train_factorized.py --data_dir data/ --audio_dir data/ --epochs 20 --warmup_freeze 3 --batch_size 8`
(batch 8 + 1024-frame cap fit the RTX 3060; the onset encoder adds O(T²) memory over Stage 2.)

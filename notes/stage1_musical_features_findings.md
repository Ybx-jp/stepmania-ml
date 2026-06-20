# Stage 1 musical-feature retrain — findings (offline)

*2026-06-20. First stage of the feature retrain (see `notes/feature_retrain_plan.md`). Offline/
quantitative results; the subjective play-feel verdict lives in `notes/playtest_log.md`.*

## What ran

Retrained the layered + groove-radar generator on **41-dim features** (original 23 + chroma 12 +
HPSS onsets 2 + metric phase 4), warm-started from `gen_radar` (only the audio-encoder first conv is
fresh; `load_from_factorized` loads the other 121 tensors), 15 epochs, `warmup_freeze=0` so the fresh
conv trains from epoch 0. Best val 1.226 (≈ the 23-dim radar model's 1.229). Cache `cache/samples_v2`.
Checkpoint `checkpoints/gen_stage1/best_val.pt`. Eval: `eval_musicality.py`, base vs stage1.

## Head-to-head (48 val songs, density-matched decode)

| metric | BASE 23-dim | STAGE1 41-dim | REAL |
|---|---|---|---|
| within-beat on-beat frac | 0.93 | 0.952 | 0.829 |
| phase L1 vs real (↓) | **0.201** | 0.245 | 0 |
| structure corr (↑) | +0.796 | +0.792 | — |
| end-decile density | 0.233 | 0.232 | 0.219 |
| arrow↔chroma align (↑) | −0.018 | −0.014 | +0.039 |
| onset_F1 | 0.758 | 0.755 | — |
| crit_adj | 1.00 | 1.00 | — |

**Offline verdict: no improvement; onset-phase slightly REGRESSED (more on-beat).** Structure,
alignment, and guards unchanged.

## Ablation — the model IS using the features

Teacher-forced pattern-logit sensitivity to zeroing each input block (KL divergence):
`chroma 10.34, phase 4.58, hpss 0.29, mfcc(control) 11.88`.

- **Chroma is used as heavily as the MFCCs** — the pattern head genuinely depends on it. Rules out the
  "warm-start anchored it / it never trained" worry: the new info IS being consumed.
- **Metric phase is used (4.58)** but its net effect pushed onsets MORE on-beat (0.93→0.952): it gave
  the model a clean downbeat signal and the model used it to sit ON the beat — the opposite of musical
  syncopation.
- **HPSS nearly ignored (0.29)** — likely redundant with the existing onset_env.

## Chaos still smears (the H4 mechanism is unchanged)

Within-beat phase of the **chaos knob** (chaos=0.9, g=2.0), generated charts:
```
                          on-beat  16th  8th-off  16th
base chaos_only (old)      0.060   0.320  0.304   0.316   uniform smear (was unplayable)
STAGE1 chaos (new)         0.058   0.340  0.265   0.337   STILL a uniform smear
```
Chroma did **not** fix chaos: cranking the chaos radar dim still destroys the downbeat (≈6% on-beat)
and smears notes uniformly across subdivisions. Chroma informs *which panels*, but the chaos dim drives
*onset timing* via the onset head, which still goes uniformly off-grid rather than landing offbeats on
musical events. (Final playtest verdict pending in `playtest_log.md`.)

## Conclusions / hypotheses (cross-cutting hypotheses tracked in the playtest skill)

- **H6** Informative features are *necessary but not sufficient*: chroma is used (KL 10.3) yet offline
  musicality didn't move. If the playtest also disappoints, the next lever is the *objective*
  (musicality-rewarding, not just frame-wise CE) or an event-grounding architecture — not more features.
- **H7** Metric-phase backfires for syncopation (reinforces on-beat); drop or replace in Stage 2.
- **H8** HPSS is near-redundant with onset_env (KL 0.29); not worth its ~4.4s/sample cost as-is.
- The arrow↔chroma offline metric is **too weak** to trust (real itself only +0.039) — redesign or retire.

## What this is NOT

Not a refutation of the feature hypothesis (H1). The necessary condition is met (the model uses
chroma); the offline proxies are partly inconclusive (weak alignment metric; teacher-forced feature-use
≠ play-feel). The decisive test is the playtest of `gen_stage1` (base + chaos sets generated via
`export_typed_samples.py --features stage1`) — recorded in `playtest_log.md`.

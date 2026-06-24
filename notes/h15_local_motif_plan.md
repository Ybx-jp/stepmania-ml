# H15 Local (Per-Section) Motif Conditioning — Plan

**Status:** scoped 2026-06-24. Decisions: (1) **incremental sectional conditioning** (reuse the existing
MotifBasis + motif vector; change only global→per-section in `_cond`); hierarchical "pick-motif-then-realize"
two-stage decode is BANKED as a follow-up if the evidence leads there. (2) **Run two cheap no-train gates
FIRST** (variance, leakage) before any training.

## Why (the confirmed root cause)
Phase 2 + 2b established the motif lever is REAL but PARTIAL (candle/cross steers; jack axes dead) and the
cause is **base-invariant** (Track-A 42-dim test: candle got STRONGER +1.60→+2.57 but jacks stayed dead). The
mechanism is visible in `typed_model.py`: `_cond(...)` returns one `(B,d)` vector that `onset_logits`/`_decode`
**broadcast identically across all T frames** (`.unsqueeze(1).expand(B,T,-1)`). So `motif_proj(motif)` adds the
SAME constant to every position → it can only shift the pattern head's global bias, never sharpen
which-panel-at-frame-t → weak per-frame gradient (train pattern-loss fell only ~1.5% on Track B). See
`notes/h15_motif_findings.md` (Phase 2 + 2b).

## Core idea
Replace the single broadcast motif vector with a **time-varying motif schedule**: one motif vector per SECTION
(musical phrase), piecewise-constant across the chart. The per-frame conditioning then VARIES and correlates
with local figure content → a real per-frame gradient, AND genuine per-section user control ("breakdown=trills,
drop=candle storm").

**Granularity is a bottleneck spectrum; leakage picks the operating point.**
- global (1 vec, current) = max bottleneck → no leak, weak gradient.
- per-frame (1 vec/frame) = min bottleneck → strong gradient but LEAKS the answer at t.
- **section-level = principled middle**: a phrase's aggregate figure distribution over ~tens of onsets varies
  enough to drive a gradient but is a soft DISTRIBUTION, not a next-panel lookup. The 12-d radar-orthogonal
  projection + the L↔R mirror-fold (LRLR≡RLRL) add lossiness that blocks exact copying.

## The central risk — self-conditioning LEAKAGE (label leakage)
Training self-conditions each section on its OWN realized motif vector. If that vector reveals the panel at a
frame inside the section, train loss plummets as an ARTIFACT and the model ignores audio / is uncontrollable at
inference. Per experiment-design Rules 7–9: **do NOT trust a train-loss improvement; success is proven ONLY on
the leak-free inference steerability test** (the user-set target carries no future, so inference is leak-free
by construction).

## Pre-flight gates (NO training — do first; experiment-design Rules 5, 6, 7)
- **Gate A — does local even VARY?** Encode per-section motif vectors over real charts; compare within-chart
  section-to-section variance vs between-chart variance, swept over section size. If a real chart's sections
  barely differ from its global vector, sectional conditioning has nothing to steer → kill/rethink. Output: the
  section size(s) where within-chart variance is meaningful. (Rule 5: read what REAL data does here.)
- **Gate B — leakage calibration.** Measure how well a section's motif vector predicts the EXACT next panel at
  a held-out onset in that section (a simple predictor: AUC / top-1), as a function of section size. Pick the
  SMALLEST section that VARIES (Gate A) yet is NOT leaky (Gate B) = the honest window, chosen BEFORE training.
- **Gate decision rule:** proceed only if there's a window with (A) within-chart variance clearly above the
  per-frame-shuffled control AND (B) next-panel predictability near the global-vector floor (i.e. the section
  vector is not an answer key). If no such window exists, the incremental lever can't work → revisit
  hierarchical.

## Architecture change (minimal, warm-startable, back-compatible)
- `motif` arg accepts `(B,K)` (global, as today) OR `(B,T,K)` (per-frame schedule). `motif_proj` is a Linear on
  the last dim, so it maps either rank; add the per-frame term per-position instead of broadcasting.
- Refactor `_cond` so the motif term can be `(B,T,d)`: compute the non-motif cond `(B,d)`, broadcast to
  `(B,T,d)`, then ADD `motif_proj(motif)` (broadcast if 2-D, per-position if 3-D). CFG null path unchanged
  (`null_motif` (d,) broadcasts). Zero-init `motif_proj` already makes warm-start a step-0 no-op.
- `generate()` accepts a schedule `(B,T,K)` or a single `(B,K)`; CFG dual-path unchanged.
- Target builder: `local_motif_targets(chart, radar, section)` → `(T,K)` (encode each section's window, expand
  to frames). Built in `collect_typed`.

## Training (`train_motif_local.py`)
Warm-start **gen_motif_hr** (Track A, the just-built 42-dim motif model) — exact no-op at step 0. **STYLE OFF**
(Track A's style encoder is untrained; eval generates style=null anyway). 42-dim cache `cache/samples_v3`.
Self-condition each chart on its per-section motif schedule; CFG dropout radar/motif; focal losses; ~12 ep.
Add motif-schedule DROPOUT/noise so the model can't over-rely (extra leakage insurance).

## Fair success test (leak-free)
1. **Constant-schedule steerability** (`eval_motif.py` as-is, schedule = +z knob k everywhere): directly
   comparable to Track-A global (candle +2.57, jack0 −0.16, jack10 +0.23). HEADLINE = do the jack axes MOVE?
2. **Varying-schedule control** (NEW eval): set knob k in section A, knob k' in section B, using a REAL chart's
   own section sequence (in-distribution, Rule 3); measure realized figures track the schedule section-by-section.
3. **Quality guardrails:** onset_F1, density unchanged (one-change attribution; Rule 11).

## GATES RAN & PASSED (2026-06-24) — `diag_local_motif_gate.py`, 600 real charts
```
 size | within-frac  ord/shuf var | sec-acc  glob-acc  excess(leak)  maj
   16 |    0.694         0.73     |  0.592    0.432       0.160      0.372
   32 |    0.772         0.93     |  0.606    0.424       0.182      0.346
   48 |    0.758         1.04     |  0.613    0.398       0.215      0.317
   64 |    0.742         1.08     |  0.630    0.423       0.207      0.334
   96 |    0.702         1.18     |  0.625    0.431       0.193      0.326
  128 |    0.667         1.16     |  0.631    0.445       0.186      0.348
```
- **Gate A (variance):** within-chart fraction high everywhere (~0.67–0.77 → ~70% of motif variation is LOCAL).
  BUT the ord/shuf ratio is the honest column: at S=16/32 it's <1 → that "local variation" is SAMPLING NOISE
  (a ~8-onset window is a noisy histogram; shuffling onsets gives just as much variance). REAL temporal
  structure (ordered var > shuffled) only emerges at **S≥48**, strengthening to S≈96. (This is exactly why we
  gated before training — S=16 would have trained the model to chase noise.)
- **Gate B (leakage):** section vec predicts its dominant figure ~0.63 vs global ~0.43 → excess ~0.16–0.21,
  FLAT across sizes, DECREASING in the valid range. Bounded/benign (the section carrying real local style =
  the control we want), NOT a near-1.0 answer key.
- **CHOSEN WINDOW: S≈64 frames (~4 measures / a phrase)** — smallest size clearing the structure test
  (ord/shuf 1.08) with high within-frac (0.742), moderate leak, ~10–22 steerable sections per chart. **S=96 =
  conservative fallback** (strongest structure, lowest leak, coarser control). Still add schedule-dropout; only
  the leak-free steerability eval counts as proof (Rule 7).

## BUILT + TRAINED + VERDICT (2026-06-24)
- **Architecture** (`src/generation/typed_model.py`): `_cond` now returns (B,1,d) broadcast OR (B,T,d) when
  `motif` is a per-frame schedule (B,T,K); `onset_logits`/`_decode`/`generate` index it per position (generate
  indexes `cond_emb[:, t:t+1]` in the cached loop). Zero-motif global ≡ zero-schedule (back-compat verified);
  31/31 generation tests pass. (4 pre-existing data-layer test failures, unrelated, don't import the model.)
- **Training** (`train_motif_local.py`): warm gen_motif_hr (exact), per-section schedule via
  `local_motif_targets(chart, radar, basis, section=64)`, style OFF, CFG+motif dropout, **patience-3 early
  stopping** (stopped at the 20-epoch cap; best epoch 17). **val_total 1.216→1.091, val pattern-loss
  ~1.10→0.979 (~11% vs the global descriptor's ~1.5%)** — the per-frame pathway is genuinely used.
- **VERDICT — leak-free steerability (eval_motif.py --ckpt gen_motif_local --highres; own-axis Δ).**
  LOCALITY CONFIRMED, NOT leakage (the loss drop cashed out into real control on a user-set target):

  | knob            | B global | A global | LOCAL g=1 | LOCAL g=3 |
  |-----------------|----------|----------|-----------|-----------|
  | 3 step↔candle   | +1.60    | +2.57    | **+2.00** | +3.43     |
  | 10 jack↔trill   | +0.51    | +0.23    | **+0.57** | +1.52     |
  | 0 jack↔sweep    | +0.15    | −0.16    | **+0.23** | +0.10     |

  - **HEADLINE: jack↔trill revived** (−0.12 global g=1 → **+0.57** local g=1; +1.52 at g=3) — a jack-family
    axis finally moves. Candle stronger. From TWO dead axes to ONE.
  - **HOLDOUT: jack↔sweep (knob 0) still barely moves** (+0.10–0.23). Sweeps = long-range monotonic staircases
    needing multi-frame sequential coherence a section vector can't pin → likely the case for the BANKED
    hierarchical pick-then-realize.
  - **CAVEAT (Rule 8): clean win is g=1** (density matched 0.187, onset_F1 0.70–0.76). At g=3 density inflates
    to 0.30–0.33 + onset_F1 falls to ~0.5–0.67 + cross-talk up to 2.4 — CFG now amplifies per-frame motif on
    the ONSET head too (over-places). Don't headline g=3. Usable operating point ~g=1–2; possible fix =
    decouple motif from the onset head, or cap guidance.

## #1 DONE — decouple motif from the ONSET head (gen_motif_local2, 2026-06-24)
`onset_logits` now passes `motif=None` to `_cond` (motif = which-panels only = a pattern-head concern; the
MotifBasis excludes rhythm/density, which is the radar's job). RETRAINED (Rule 2: no train/inference mismatch)
warm gen_motif_hr, section=64, patience-3 → `checkpoints/gen_motif_local2/best_val.pt`, **val_total 1.0924 (=
v1 1.0912; quality untouched).** 31/31 gen tests pass.
- **DENSITY INFLATION FIXED:** g=3 density 0.30–0.33 → **0.190** (real 0.187), onset_F1 symmetric 0.63. g=3 is
  now USABLE.
- **Deconfounded steerability (Δself), global-A → v1 → v2:**
  - candle (3): +2.57 → +2.00/+3.43 → **+1.92 (g1) / +3.49 (g3)** — rock-solid, the robust lever.
  - jack↔trill (10): +0.23 → +0.57/+1.52 → **+0.18 (g1) / +0.97 (g3)** — DROPPED from v1 because v1's gain was
    partly the DENSITY ARTIFACT (more notes → more trill-like figures; Rule 8). v2 = honest; still a real
    sign-flip over global (g1 −0.12→+0.18) so locality genuinely helps trill, just modestly.
  - jack↔sweep (0): −0.16 → +0.23/+0.10 → **+0.17 (g1) / +0.27 (g3)** — still the lone holdout.
- **VERDICT:** local conditioning = a robust candle lever + a modest-but-real trill lever + a usable g=3
  (density matched), with sweep still stuck. v2 is the deconfounded model to carry forward.

## Banked follow-up
Hierarchical pick-motif-then-realize (stage-1 emits a motif label per section, stage-2 realizes panels
conditioned on it) — MOTIVATED by the jack↔sweep holdout (long-range staircase needs multi-frame coherence a
section vector can't pin). Also: (b) PLAYTEST gen_motif_local2 @ g≈1.5–3 (now density-safe) — does the
measurable candle/trill steer change the FEEL? (validated instrument; offline metrics blind to vibe).

## #3 DONE — varying-schedule per-section control CONFIRMED (eval_motif_schedule.py, gen_motif_local2, 12 songs)
Alternate a knob +z/−z per 64-frame section within ONE chart; re-encode per section; local_Δ = mean(+sections)
− mean(−sections), track_r = corr(per-section realized, target sign). The thing GLOBAL conditioning cannot do.

| knob          | g | global_Δ | local_Δ | local/global | track_r | density/F1   |
|---------------|---|----------|---------|--------------|---------|--------------|
| 3 candle      | 1 | +1.36    | **+1.85** | 136%       | **+0.52** | 0.186 / 0.72 |
| 3 candle      | 3 | +2.73    | **+3.21** | 117%       | **+0.70** | 0.195 / 0.68 |
| 10 jack↔trill | 1 | +0.36    | +0.20   | 55%          | +0.05   | 0.186 / 0.72 |
| 10 jack↔trill | 3 | +0.75    | **+1.12** | 148%       | +0.29   | 0.195 / 0.68 |
| 0 jack↔sweep  | 1 | +0.24    | −0.17   | —            | −0.07   | 0.186 / 0.72 |
| 0 jack↔sweep  | 3 | +0.04    | −0.21   | —            | −0.08   | 0.195 / 0.68 |

- **CANDLE = genuine per-section control:** local_Δ ≥ global_Δ (the model is MORE in-domain on varying targets
  — it was trained on them), track_r +0.52→+0.70 (candle-heavy sections land where the schedule says +z,
  step-heavy where −z), quality intact. THE payoff of the whole local-motif effort.
- **jack↔trill:** moderate per-section control at g=3 (local_Δ +1.12, track_r +0.29), weak at g=1.
- **jack↔sweep:** dead per-section too (negative local_Δ, track_r ~0) — the consistent holdout across every test.

## OVERALL H15 LOCAL-MOTIF VERDICT (2026-06-24)
Incremental sectional conditioning DELIVERED a steerable, section-by-section, quality-safe CANDLE/CROSS lever
(+ partial trill), confirming the locality hypothesis end-to-end (gates → loss drop → leak-free constant-knob →
deconfounded onset-decouple → per-section tracking). **jack↔sweep is the one unmoved axis** (long-range
staircase = multi-frame sequential coherence a section vector can't pin) → the standing case for the BANKED
hierarchical pick-then-realize. Carry-forward model = `gen_motif_local2` (motif off the onset head; density-safe
at high guidance). Untested: PLAYTEST (does the measurable candle steer change the FEEL? — user was afk).

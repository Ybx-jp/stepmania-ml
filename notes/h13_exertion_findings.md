# H13 — Exertion / fast-jack findings

**Claim under test (H13, from playtest 2026-06-22):** the generator has no representation of *physical
exertion* — e.g. "a 6-note jack on 1/16s is crazy." A **jack** = consecutive presses on the SAME single
panel; at 16th spacing (adjacent onset frames) that is ONE foot hammering one arrow at 16th speed = brutal.

Probe: `experiments/generation_typed/diag_exertion_h13.py` (reuses the staged quota-free pipeline; persists
the staged onset model to `checkpoints/gen_staged_onset/maskpredict.pt`). Fix: `generate(max_jack_run=...)`
in `src/generation/typed_model.py`. Designed under the `experiment-design` discipline (measure real first;
isolate the variable).

## What real charts do (reference — measured, NOT assumed)
Over **786 val charts**, fast same-panel jacks are essentially absent at every difficulty:

| difficulty | jack-pair-rate | mean max fast-run | charts w/ fast-run ≥3 |
|---|---|---|---|
| Beginner (0) | 0.000 | 1.00 | 0 / 170 |
| Easy (1)     | 0.015 | 1.00 | 0 / 210 |
| Medium (2)   | 0.052 | 1.03 | 1 / 230 |
| Hard (3)     | 0.006 | 1.06 | 2 / 176 |

Human charters **alternate panels** across fast runs (feet alternate) → max same-panel 16th run ≈ 1. A fast
6-note one-panel jack basically does not occur in real charts.

## The result (12 Hard chaotic songs, deployment decode)
`jack-pair-rate` = fraction of 16th-adjacent single-note pairs that land on the SAME panel.

| chart | jack-pair-rate | runs≥4 /1k onsets | max fast-jack-run |
|---|---|---|---|
| REAL | 0.000 | 0.0 | 1.0 |
| **DEPLOYMENT** (staged onset → v4 panels = what was played) | **0.284** | 0.3 | 3.0 (up to **9** on Dancing lovers) |
| CONTROL (REAL onset → v4 panels) | 0.289 | 1.4 | 3.1 |

**Attribution (clean isolation):**
- DEPLOYMENT 0.284 ≫ REAL 0.000 → **H13 CONFIRMED by data**, not just feel. The generator over-produces
  fast same-panel jacks by ~50×.
- CONTROL ≈ DEPLOYMENT → it is the **PATTERN head (which-panel), not onset placement** (mechanism b). Given
  the *exact* notes a human patterned with ZERO jacks, the v4 pattern head still jacks 28% of fast pairs.
- Mechanism (a) RULED OUT: staged onset run-lengths (frac-in-runs≥4 = 0.026, max 5.0) are *shorter* than
  real (0.077, max 6.7). The long runs aren't the problem; the panel assignment within them is.
- **Mechanism:** the pattern head is sampled per-frame (pattern_temperature 0.7) with no notion of foot
  speed; a `next_foot` tracker exists but only feeds `no_crossovers`. Nothing forbids repeating a panel on
  consecutive 16th frames, so sampling lands the same panel and produces a one-foot jack.

## The fix — `max_jack_run` (decode-time, pattern head)
Speed-conditioned anti-jack constraint: track the running FAST (16th-adjacent) same-single-panel run; once
it reaches `max_jack_run`, forbid that panel's single-pattern on the next 16th-adjacent single onset →
forces a different panel (foot alternation, like real). Only 16th-adjacent runs are capped (normal slower
jacks untouched); jumps reset the run. `=1` matches real (strict alternation). Default None (off) in
`generate()`; **default 1 in the export paths** (the exertion fix is part of the new default).

**Validation (same 12 songs, `--max_jack_run 1`):**

| | jack-pair-rate | runs≥4 /1k | max fast-jack-run |
|---|---|---|---|
| DEPLOYMENT, cap off | 0.284 | 0.3 | 3.0 |
| DEPLOYMENT, cap=1 | **0.031** | **0.0** | **1.5** |
| REAL | 0.000 | 0.0 | 1.0 |

9× reduction, brutal 4+ jacks eliminated, now real-equivalent (real Medium itself is 0.05). Onset
run-lengths unchanged (cap only touches panel assignment) → density/structure preserved; no NaN/crash with
full playability constraints active. Test `tests/test_generation.py::test_max_jack_run_caps_fast_jacks`
(pattern-level guarantee = 0 fast same-panel jacks; 29 tests pass).

Residual 0.031 (vs 0 ideal) is from hold-CLOSE frames (a note on a held panel) that the metric reads as a
press but the cap treats differently — real-equivalent, not worth chasing.

## Status / next
- Implemented + offline-validated + **PLAYTEST-CONFIRMED (2026-06-22)**: night in motion "AWESOME! hit all
  the patterns and musicality I was looking for"; get it all "very smooth". H13 CLOSED as solved.
- **PROMOTED to the code-enforced MANDATORY playability set** (`src/generation/playtest_export.py`:
  `MANDATORY_JACK_CAP=1`, `enforce_playability` injects it and refuses None/0 without an override reason) and
  the `playtest` skill constraints table. Default-on `max_jack_run=1` in both export paths; tests
  `test_max_jack_run_caps_fast_jacks` + `test_enforce_playability_jack_cap` (30 pass).
- Strict `=1` (full alternation) confirmed good by ear; `>1` allowed as a looser cap, disabling needs an
  explicit override_reason (e.g. this diagnostic's uncapped baseline).
- NOT a defect, noted from get it all: wanting "more chaotic patterns from the drumline" = the chaos-radar
  INTENSITY dial (validated, `chaos_conditioning_findings.md`), a separate knob from exertion.

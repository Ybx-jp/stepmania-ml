# Jack-heaviness investigation — it's BOTH heads (pattern temperature + onset rhythm)

**Date:** 2026-06-27. **Origin:** the native-mode foot-physics comparison (notes/foot_physics_baseline.md
axis-split) found the learned head's same-panel run-length distribution is jack-heavy vs real (Medium
len3 ~2×, ≥4 ~3–4×). Hypothesis (user): a temperature issue in the onset AND pattern heads. Investigated
under experiment-design "one variable at a time"; Rule 0 surfaced the prior `style_decoding.md` result
(greedy = 88% jacks; pattern_temperature=1.0 → jack-rate 0.20 = real; shipped default 0.7 is lower = jackier).

Harnesses: `probe_jack_temp.py` (Probe 1), `probe_onset_rhythm.py` (Probe 2). 16 rich songs, native decode
(own onset head, radar-conditioned, density matched to REAL), governor OFF to isolate the heads.

## Probe 1 — PATTERN head: temperature IS a strong jack lever (confirmed)
Native, governor off, density held, ONE variable = pattern_temperature, same seed per (song,temp).
**Medium (n=11, reliable):** jackDist falls MONOTONICALLY with temperature, len2/len3/≥4 all → real,
and jump% rises → real too (a bonus, not a cost — Rule 1 side-effect check passed):

| patT | len2% | len3% | ≥4% | jump% | jackDist |
|------|-------|-------|-----|-------|----------|
| 0.70 (shipped) | 72.1 | 18.3 | 9.6 | 23.2 | 0.311 |
| 1.00 | 75.9 | 16.0 | 8.1 | 35.6 | 0.236 |
| 1.50 | 79.3 | 14.8 | 5.9 | 42.9 | 0.167 |
| real | 87.7 | 9.4 | 2.9 | 40.6 | 0 |

So the shipped 0.7 is too greedy → too jacky AND too few jumps; raising temperature fixes both toward real.
**Caveats:** (a) doesn't fully close the gap (residual jack-heaviness even at 1.5); (b) >0.85 leaves the H2
arrow-coherence range — the cost is arrow incoherence, invisible to these metrics, needs by-ear (Rule 8);
(c) **Hard (n=4) RESISTED** — jackDist bounced 0.48→0.39→0.47→0.39→0.44, len2 stuck ~62–67 (real 86). So
Hard's jacks aren't only pattern temperature → led to Probe 2.

## Probe 2 — ONSET head: no temperature lever, but RHYTHM is implicated
**Part A (no-op confirm):** onset_logit_scale (the natural "onset temperature") is `p=sigmoid(scale·ol)`,
monotonic → preserves frame ranking → under quantile thresholding (deployed path) the selected onset set is
IDENTICAL for any scale. Empirically: **0 frames differ** at scale 0.5 or 2.0 across all 16 songs. So there
is NO onset-temperature knob that changes which onsets fire in deployment.

**Part B (rhythm realism, density HELD):** the onset head's contribution to jacks is its SPACING — model
NATIVE onsets vs REAL inter-onset-interval (IOI). The model's rhythm is BLOCKIER than real:

| stratum | src | g1 16th% | g2 8th% | g34 qtr% | gwide% | maxOnsetRun |
|---------|-----|----------|---------|----------|--------|-------------|
| Medium  | real  | 3.6 | 59.5 | 32.8 | 4.2 | 114 |
| Medium  | model | 0.0 | 62.0 | 36.3 | 1.7 | 234 |
| Hard    | real  | 11.0 | 59.4 | 26.7 | 2.9 | 137 |
| Hard    | model | 0.0 | 75.8 | 23.9 | 0.3 | 289 |

The model places **zero 16th-adjacent onsets** (real has 4–11% — the known 16th under-commitment), **over-
weights 8ths** (Hard +16pts), inserts **fewer rests/wide gaps** (gwide ~half real), and strings onsets into
**~2× longer consecutive-onset runs** (gap≤4). A jack run can only live inside an onset run, so the onset head
raises the jack-OPPORTUNITY ceiling (longer, more uniform 8th streams) that the pattern head then fills.

## Conclusion — the user's "both heads" was right, refined
The jack-heaviness is a COMBINATION:
1. **Pattern head — TEMPERATURE.** Shipped 0.7 jacks too readily; ↑temp reduces jacks + adds jumps toward
   real (decode lever, but coherence-capped at ~0.85 by H2 — by-ear needed).
2. **Onset head — RHYTHM (not temperature; the temperature is a no-op).** It lays down 8th-heavy, 16th-absent,
   rest-poor streams ~2× longer than real, raising the jack-opportunity ceiling. A training / onset-calibration
   thread, not a decode knob.
3. **Cross-probe consistency:** Hard's resistance to the pattern-temp fix (Probe 1) is EXPLAINED by Hard's
   worst onset-rhythm blockiness (Probe 2: 8th +16pts, onset runs 2.1× real) — the opportunity is too dominant
   for temperature alone.

## Probe 3 — section-level causality + misallocation (`probe_onset_sections.py`, 16 songs, win=32)
Tests the user's "pattern head = best of a bad situation" causally, and quantifies the playtest
"awkward sections over-noted / empty where active." Native deployed onset head, governor OFF, density
matched. (Metric fix mid-probe: onset-run-length saturates in rich charts — gap≤4 opportunity is
everywhere — so 3A is binned by LOCAL DENSITY, not run-length.)

**3A — the pattern head is the PROXIMATE cause, not just the onset head (refines "best of a bad situation").**
At MATCHED local density the model jacks far more than real, and the SHAPES differ qualitatively (Medium):

| local density | model jackiness | real jackiness |
|---------------|-----------------|----------------|
| [0.2,0.35) | 1.42 | 1.07 |
| [0.35,0.5) | 1.60 | 1.08 |
| [0.5,1.0)  | 1.67 | 1.06 |
| corr(density,jackiness) | **+0.23** | **−0.09** |

REAL jackiness is DENSITY-INVARIANT (~1.07) — humans keep alternating however dense the section gets.
The MODEL's jackiness RISES with density. So the onset head's denser/blockier sections (Probe 2) TRIGGER
it, but the pattern head has a density-coupled jack response humans don't have — same defect as Probe 1
(low pattern_temperature → greedy repeat → worst where there are most consecutive decisions = dense
sections). Probe 1 and 3A are one mechanism. (Governor OFF isolates the head; the fatigue governor partly
counters this in dense sections — the other lever.)

**3B — misallocation confirmed + quantified (Med / Hard):** corr(model_dens, real_dens) 0.48 / 0.49
(only ~half the section variance matches real); corr(model_dens, p_onset) 0.62 / 0.83 vs corr(real_dens,
p_onset) 0.36 / 0.54 — the model tracks raw audio SALIENCE more than humans do → over-notes loud/awkward,
under-notes musically-active-but-quiet (the §8d melodic under-placement). Over/under sections 6%/10% (Med),
9%/11% (Hard) — skews toward UNDER-placement, matching "empty where it should be active."

## Probe 4 — does the FATIGUE GOVERNOR let pattern_temperature rise? (`probe_temp_governor.py`)
H2's "cap pattern_temperature at 0.85 for coherence" predates the governor. Swept pattern_temperature
× governor {OFF, ON λ=2}, native, density matched, bpm passed (§8b). Both arms keep the mandatory
max_jack_run=2 hard cap; the swept knob is the SOFT exponential fatigue_penalty.

**Governor SUPPORTS higher temp on the JACK axis (hypothesis holds there).** Medium maxRun (jack tail):
gov OFF 8/**22**/6/6/6 (erratic, spikes) vs gov ON 8/7/5/7/5 (bounded near real 5) across T=0.7→1.5. And
under the governor, rising temp still improves the distribution: jackDist 0.34→0.22, jump% 19→34 (→ real
41), mJumpStrm up — the governor does NOT kill the jump gain. So the governor catches the jacks the H2 cap
indirectly guarded → you can raise temp for the jack/jump benefit without a jack blowup.

**But it does NOT certify musical COHERENCE, and the metric is ambiguous.** Transition-entropy (scramble
proxy) CLIMBS with temp in BOTH arms (Medium real 2.20; gov ON 2.40→2.81); the governor only slightly
dampens it. KEY caveat (Rule 1): entropy can't separate STRUCTURED variety (good) from RANDOM scramble
(bad) — reducing jacks (low-entropy repetition) NECESSARILY raises entropy. So a rising transEnt is partly
the desired jack-fix, partly possible incoherence; the metric can't split them. The governor bounds
FATIGUE/jacks, not musical structure.

**Verdict:** metrics say "raise it" (governor bounds jacks; temp moves jacks+jumps → real); metrics CANNOT
settle coherence → BY-EAR (Rule 8). Metric sweet spot: gov-ON T≈1.0–1.2 (Medium jackDist 0.26–0.28, maxRun
5–7, jump% 29–31). Exports for the ear test: `outputs/probe_temp_governor/` (gov-ON, T=0.7/0.85/1.0/1.2).
NEXT = playtest A/B those exports; the binding constraint is coherence, which only ears resolve.

## Updated picture of the jack-heaviness (after 4 probes)
- PROXIMATE cause = the PATTERN head's density-coupled over-jacking (low pattern_temperature). Levers:
  pattern_temperature ↑ (coherence-capped) and the fatigue governor (both decode-time, available NOW).
- CONTRIBUTING cause = the ONSET head's blocky/dense rhythm + salience-chasing misallocation (Probe 2/3B) —
  a training/architecture thread (sequence-aware onset, bounded in sequence_aware_onset_plan.md; + the
  melodic-under-placement audio-feature thread, §8d). Not cheaply fixable (06-22 bound).

## Next (each starts with Rule 0)
- Quick decode win: A/B pattern_temperature 0.7 vs ~0.85 BY EAR (the coherence ceiling is the binding
  constraint, and metrics are blind to it). Don't ship >0.85 on metrics alone.
- Deeper: the onset head's blocky rhythm (no 16ths, too few rests, long uniform 8th streams) — connects to the
  known 16th under-commitment + the stamina/breathing (rest) work. Onset-calibration / training thread.
- Stratum caveat: Easy is n=1 (rich set skews Med/Hard) — ignore Easy; Hard n=4 is a lead, not a result.

# H11 transition-responsiveness — re-run on the CURRENT model + canonical defaults

**Date:** 2026-06-28
**Probe:** `experiments/generation_typed/buffered_sectional.py` (canonicalized) → `outputs/h11_rerun/run.log`
**Why:** the original H11 work ran 2026-06-21 on `gen_stage1` (41-dim, pattern_temp 0.7, NO governor). Many
improvements since (the `gen_motif_full_fixed` H19 highres retrain + the full decode-time governor + pattern_temp
→1.0). Re-run with the **canonical PLAYTEST/EXPORTER defaults** (now codified in the `generation-defaults` skill)
to see the delta. The June-21 finding: free-run baseline under-transitions ~4× vs real (the H11 AR-drift gap);
its own follow-up flagged the pooled metric as noisy/song-set-dependent (Rule 11).

## Canonical config used (the whole point of the re-run)
`gen_motif_full_fixed` (42-dim highres, `cache/samples_v3`) + full governor (`fatigue_penalty=2 free=6`,
`stamina_ceiling=50 tau=8 scale=15 breathe=1.2`) + `pattern_temperature=1.0`, `type_temperature=0.4`,
`max_jack_run=2`, playability forced on, `bpm` per song. (The June-21 run had ALL of these wrong/absent.)

## Result — the corrected PAIR (6 rich Hard songs; responsiveness = choreography-change @boundary − @random)

Both arms = FULL canonical config (gen_motif_full_fixed 42-dim highres, pattern_temp 1.0, type 0.4, max_jack_run 2,
**onset_phase_calib (0,1.0) = 16th-unlock**, tau uses the same offset). The ONE variable = the governor.
(NB: this supersedes an earlier single run that used the governor but was MISSING the 16th-unlock; with calib on,
canonical-full baseline rose +0.101 → +0.130.)

POOLED:
| arm | real | **baseline** | sectional |
|---|---|---|---|
| **canonical-full** (governor on) | +0.122 | **+0.130** ≈ real | +0.347 |
| **governor-off** (calib only) | +0.122 | **+0.057** ≈ ½ real | +0.398 |

PER-SONG (the honest readout — signs flip; the pooled mean hides this):
| song | real | base FULL | base GOV-OFF | sectional(full) |
|---|---|---|---|---|
| Dancing lovers | +0.230 | +0.006 | −0.176 | +0.658 |
| First of the Year Equino | +0.130 | −0.095 | +0.042 | +0.293 |
| Deja loin | +0.172 | +0.180 | +0.210 | −0.111 |
| Pound the Alarm | +0.239 | +0.338 | +0.099 | +0.671 |
| Taylor Swift | +0.172 | +0.199 | +0.149 | +0.393 |
| IN BETWEEN | **−0.211** | +0.152 | +0.021 | +0.175 |

## DENSITY-DROPPED cut (the decisive disentangler) — both arms, same charts

To separate "the breathe arc mechanically moves the density dim" from "the choreography genuinely transitions",
recomputed responsiveness on the SAME charts with DENSITY dropped from the descriptor (jump_frac + L/D/U/R only):

| descriptor | real | baseline FULL | baseline GOV-OFF | gov contribution (full − govoff) |
|---|---|---|---|---|
| FULL `[density,jump,L,D,U,R]` | +0.122 | +0.130 | +0.057 | **+0.073** |
| NO-DENSITY `[jump,L,D,U,R]` | +0.105 | +0.091 | +0.019 | **+0.072** |

NO-DENSITY @boundary/@random detail: real 0.676/0.572; FULL 0.709/0.618; GOV-OFF **0.677/0.658**.

## Reading — the density-tautology is REFUTED; the governor adds REAL section-aware choreography

1. **The density-DIMENSION tautology is refuted** — the full−govoff boost is +0.073 with density ≈ +0.072 without,
   so it's not a bookkeeping artifact of the density dim. (Density dim contributes a small, EQUAL ~+0.038 both arms.)
2. **⚠️ BUT the bigger attribution was ALSO WRONG (user-corrected) — the no-density metric does NOT isolate the
   pattern head.** Per the architecture (`conditioning-mechanics` §0/§8): the **ONSET head decides WHEN notes are
   placed (phrasing/density); the PATTERN head only decides WHICH panels, NEVER note count.** `jump_frac` + panel
   mix are computed OVER the onset-selected note-frames — so when the onset/stamina density regime shifts across a
   boundary, the realized panel/jump stats shift too with IDENTICAL pattern-head behavior. The no-density
   responsiveness is therefore STILL largely an onset-side phrasing effect cascading downstream, NOT the pattern
   head authoring transitions. And full−govoff strips BOTH the onset-side stamina/breathe AND the pattern-side
   fatigue, so the +0.072 gap mixes an onset-cascade with a smaller direct fatigue-on-panels effect — it isolates
   neither head. **Retract: "the governor improves transitions via genuine pattern-head choreography."**
3. **Correct attribution: the transition/phrasing gain is ONSET-SIDE.** The governor's section-responsiveness comes
   from the breathe/stamina arc GATING THE ONSET DECISION (§8c) to track section energy — i.e. the onset head (plus
   its governor) authors the phrasing. The bare model under-responds; the onset-gating governor rescues it. The
   pattern head colors panels on whatever frames it's handed; it cannot author phrasing.
4. **What the no-density numbers DO still show (weakly):** the chart's realized panel stats change more at
   boundaries with the governor on — but that's downstream of the onset structure, so it's not evidence about the
   pattern head's own transition behavior either way.
5. **Per-song STILL noisy** (Rule 11): signs flip (Dancing lovers base −0.001 vs real +0.200; First −0.195;
   Pound/Taylor OVER), **IN BETWEEN degenerate** (real NEGATIVE). Pooled = DIRECTIONAL on 6 songs.
6. **Sectional OVERSHOOTS** both arms/descriptors — over-corrects, not shippable.

## Bottom line (twice-corrected)
The decode-time governor brings the PLAYED chart's transition responsiveness to ~real, and that gain is **onset-side
phrasing** (the breathe/stamina arc gating WHICH FRAMES fire to follow section energy), NOT pattern-head
choreography — the no-density metric can't separate the pattern head because panel/jump stats are conditional on the
onset frames. So this re-run does NOT cleanly measure the pattern head's H11 AR-drift at all; it measures the
onset-side phrasing the governor supplies. **To actually isolate the pattern head's transition contribution, hold
the onset frames FIXED** (teacher-force / `onset_override` with the same note positions for both arms, e.g. the real
chart's onsets, governor off) and measure panel responsiveness vs real — only then are jump/panel changes
attributable to the pattern head. Until that's run, the H11/AR-drift claim about the PATTERN head rests on the
June-21 teacher-forced probe + `arc_lag` (cold-start), not on this re-run. NOT a 0.1.0 blocker.

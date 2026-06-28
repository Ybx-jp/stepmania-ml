# Lineage — Onset phrasing & the sparse-harm calibrator (2026-06-27 → 2026-06-28)

**One line:** chased "the chart lags / under-places at musical events" from a mis-attributed PATTERN-head story to
its true ONSET-head home, built the measure-before-build diagnostic, and validated + wired a sparse-harm-in-quiet
phrase calibrator (awaiting ears).

**Status:** ACTIVE. Step 2 done (calibrator wired into `generate()`, A/B installed); the by-ear gate is the open
fork. Branch `gen/full-governor-cond-grid` (commits `26200f2`, `d6bde49`, `fca697f`, `9580b1b`).

**Memory:** [[onset-phrase-calibrator]] (active), roots in [[jack-heaviness]]; corroborates [[fatigue-governor]];
methodology footgun [[dataset-cache-footgun]].

## The hypothesis chain (what we believed → what we learned)
1. **Root (jack-heaviness arc, `notes/jack_heaviness_findings.md`):** the generator's jacks = pattern head
   (proximate) + the onset head's blocky audio-only rhythm (contributing). Spun off the onset 16th-unlock.
2. **16th-unlock readout finding (`notes/onset_alloc_findings.md`):** the onset head's `p_onset` RANKS real 16th
   slots at AUC 0.73, but the global `tau` buries them → a READOUT, not representation, problem. `onset_phase_calib`
   un-buries them globally; open lead = make the un-burial LOCAL/learned. → seed of the calibrator.
3. **Canonical-defaults consolidation (`generation-defaults` skill):** before re-running any onset probe, pinned
   the ONE deployed config (gen_motif_full_fixed 42-dim highres + full governor + pattern_temp 1.0 +
   onset_phase_calib (0,1.0)); `generate()` bare defaults are unplayable/governor-off. Flipped the exporter defaults
   so its bare run = that config. *(Prevented the recurring "stale-subset" probe error — exp-design Rule 2.)*
4. **arc_lag (`notes/arc_lag_findings.md`):** localized the felt HSL cold-start. Breathe arc EXONERATED (a centered
   boxsmooth is zero-phase, architectural); HSL onset tracks audio. By elimination the cold-start = the AR PATTERN
   head. *(Pre-registered decision rule, Rule 9; complaint song, Rules 5/11.)*
5. **H11 re-run (`notes/h11_rerun_findings.md`):** re-measured transition responsiveness on the canonical model +
   the governor-OFF ablation + a density-DROPPED descriptor. **Attribution TWICE-corrected** (the methodology spine
   of this arc): (a) "baseline≈real is mostly the density tautology" → REFUTED by the density-dropped cut
   (boost +0.073 with density ≈ +0.072 without); (b) "the governor improves transitions via pattern-head
   choreography" → WRONG — the no-density metric is conditional on the onset frames, so it never isolated the
   pattern head. Net: the bare AR pattern head under-transitions; the governor's section-responsiveness is
   ONSET-SIDE phrasing (breathe gates the onset decision).
6. **Architecture correction (user; `conditioning-mechanics` §0/§8):** ONSET head decides WHEN notes fire
   (phrasing); PATTERN head decides WHICH panels, NEVER count. So phrasing is the onset head's job — the pattern
   head can't author it. This re-homed the whole investigation onto the onset side.
7. **Phrasing-coherence diagnostic (`notes/phrasing_coherence_findings.md`, `probe_phrasing_coherence.py`):**
   measure-before-build, no-retrain, deployed head, reference = MUSICAL EVENTS not a real chart (objective =
   the model's OWN coherence). Four axes (boundary-snap / burst-in-quiet / clean-tail / perc↔harm fluidity).
   Sharpest gap = axis-2 burst-in-quiet: on HSL the head ANTI-correlates with harmonic onset in quiet (corr_harm
   −0.16) = the melodic under-placement. axis-4 fluidity already EXISTS to amplify.
8. **Step 1 — sparse-harm-in-quiet calibrator (validated, posterior):** hand-crafted onset logit offset
   `gain·quiet_gate·harm`; dynamic range confirmed (Rule 11); at gain~10 corr_harm flips positive on all 3 songs,
   no regression, HELD global density (redistribution into quiet harmonic events).
9. **Step 2 — wired + A/B installed (awaiting ears, Rule 8):** `generate()` got a per-frame `onset_logit_offset`;
   exporter `--harm_calib`; `~/sm-generated/harmcalib_{OFF,ON}`. Fork: reads well → Step 3 (LEARN the offset);
   over-allocates → retune gain/gate/feature.

## Methodology wins to reuse (this arc is a case study for the SKILL.md rules)
- **Rule 0 saved a cycle TWICE:** the planned pattern-head isolation via `onset_override` was a KNOWN-INVALID setup
  (`notes/foot_physics_baseline.md` §post-retraction: forces the pattern head OOD, ≥4-jack share 0.7%→9–14%); and
  the calibrator's existence was already an open lead in `onset_alloc_findings.md`. CHECK THE NOTES FIRST.
- **The density-dropped cut** (Rule 1: does the metric SEE the property) overturned a committed "it's the model"
  read — exactly the failure-mode pattern.
- **Pre-registered decision rules** (Rule 9) on arc_lag/H11 kept attribution honest.
- **The cache footgun ([[dataset-cache-footgun]], `notes/cache_index_bug.md`):** the index-keyed sample cache
  silently fed a `--match` probe stale audio (HSL got kneeso's). Caught by Rule 8 (the identical-rows smell on the
  artifact); blast radius checked (H11 verified clean); fixed with an identity-stamped cache + regression test.

## Cross-arc corroboration
- **[[fatigue-governor]] (governor arc, `notes/governor_release_region.md`, cond-mech §8c):** the breathe arc is
  the ONSET-side density mechanism this arc shows is what carries the governor's transition responsiveness — the
  H11 re-run is the corroborating measurement that the breathe arc "does double duty (playability + phrasing)".
- **[[jack-heaviness]] (`notes/jack_heaviness_findings.md`):** the onset head's blocky audio-only rhythm is the
  upstream cause of the phrasing gap this arc targets; both arcs implicate the onset head's placement.
- **`notes/sequence_aware_onset_plan.md`:** the heavier note-context onset-head program (06-22 verdict: sequence
  signal real but AR explodes / refinement can't bootstrap from audio-only context). The phrase calibrator is the
  cheaper decode-time lever that side-steps that stall.

## Skills in play
`generation-defaults` (the canonical config the probes replicate) · `conditioning-mechanics` §0/§6/§8 (onset vs
pattern responsibilities, `onset_logit_offset`, `onset_phase_calib`) · `experiment-design` (this discipline) ·
`playtest` (the A/B set + by-ear gate).

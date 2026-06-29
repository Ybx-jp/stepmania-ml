# Lineage — Sequence-aware onset head ("WHEN" conditioned on "WHERE") (2026-06-22 → 2026-06-28)

**One line:** the 16th-placement signal lives in the NOTE SEQUENCE (the pattern head's "where"), not the audio —
but the deployed onset head is audio-only/non-causal by design, so it discards it; the AR fix explodes and
refinement couldn't bootstrap from the old audio-only C0. **RE-OPENED 06-28** with a better C0 (the current
deployed pipeline) the 06-22 wall never tested.

**Status:** RE-OPEN RESOLVED (2026-06-28) — **wall STANDS** (controlled negative). Cheap decode-time path closed;
reaching the 0.87 signal needs a RETRAIN, not a decode lever. Probe `probe_seqcontext_c0.py` (controls fired).

**Memory:** roots in [[jack-heaviness]]; corroborates/【depends-on】 [[onset-phrase-calibrator]] (the decode-time
phrase calibrator is the *side-step* this thread's full fix would replace); [[fatigue-governor]] (stamina = the
only existing where→when bridge).

## The hypothesis chain (what we believed → what we learned)
1. **06-22 `diag_seqcontext_probe`:** 16th-localization AUC audio-only **0.649** → causal note-context **0.935** →
   both 0.944. Placement is SEQUENCE-determined (run coherence); audio is nearly secondary once note-context is
   present. The deployed onset head is audio-only + non-causal (a deliberate Stage-2 anti-drift choice) → it
   DISCARDS the 0.93 signal. (`notes/sequence_aware_onset_plan.md`.)
2. **Naive AR note-context onset EXPLODES:** density 0.73–0.83 vs real 0.18 (runaway feedback). Signal real, AR
   formulation wrong-shape.
3. **Iterative refinement couldn't BOOTSTRAP:** the real v4 C0 scored 0.456 (anti-correlated — places 16ths in the
   WRONG spots), so the refiner falls back to ~audio-only (0.666). VERDICT (06-22): *"refinement cannot bootstrap
   good placement from a bad C0, and audio-only is our only C0 — circular."* Parked; ship-state = decode-time
   amount control (calib/conditioning).
4. **Architecture re-confirmation (06-28, `typed_model.py`):** decode is strictly one-way — `p_onset` precomputed
   (audio-only) → stamina thins (ceiling-only) → pattern "where" → fatigue. "Where" NEVER feeds "when". Stamina is
   the ONLY where→when bridge: foot-cost → threshold, suppress-only, biomechanics not music. → the isolation is
   real and structural, exactly the user's "the heads can't articulate it" reframing.
5. **RE-OPEN (06-28):** the 0.456 C0 was the OLD audio-only `gen_highres_v4`. The CURRENT deployed pipeline
   (`gen_motif_full_fixed` + pattern_temp 1.0 + full governor) is a much-better C0 with real run-coherence —
   06-22 could not test it. Probe `probe_seqcontext_c0.py`: TARGET=real onset, CONTEXT swapped audio /
   both_real (ceiling) / **both_C0 (deployed chart)**.
6. **RESULT (06-28) — WALL STANDS (controls fired):** 16th-AUC audio **0.656** (≈floor ✓), both_real **0.871**
   (≈ceiling ✓ POSITIVE CONTROL PASSES), **both_C0 0.667 ≈ audio (5% of gap).** The deployed C0 carries NO
   placement signal beyond audio — because its onsets are audio-only-PLACED, so the "where" just echoes audio.
   CORROBORATION: 06-22's refiner trained-on-v4-C0 = 0.666 ≈ today's eval-on-deployed-C0 0.667 → robust across
   setups/pipelines, not a domain-mismatch artifact. Residual confound (train-on-deployed-C0) not pursued (the
   convergence + root cause make it very unlikely to flip). → cheap decode re-open CLOSED; needs a RETRAIN.
   **Methodology save:** the FIRST run trained on only 20 songs → both_real collapsed to 0.497 (positive control
   DEAD) → caught + re-run at 800 songs before reporting (experiment-design Rule 11: confirm the metric can move).

## Methodology notes (reuse)
- **Rule 5/6 (cheap real reference first):** the 06-28 boundary-snap detour first measured what REAL does (density
  tracks real; figure-character barely snaps) — two cheap probes that REFRAMED "structure" before any build, and
  pointed here. (`phrasing_coherence_findings.md` boundary-snap reframe.)
- **Rule 2 (deployment match):** the 06-22 "circular" verdict is C0-SPECIFIC (it tested v4's C0). Re-testing with
  the DEPLOYED C0 is the fair version; treating the old verdict as model-general would be the error.
- **honest bound:** teacher-forced 0.935 includes "continue-the-run" autocorrelation; gen-time AUC will be lower.
  AR-stability + playtest remain the real tests. A full sequence-aware head is a Phase-2.6 architecture change.

## Cross-arc corroboration
- **[[onset-phrase-calibrator]] (`phrasing_coherence_findings.md`, lineage `onset-phrasing-calibrator-arc.md`):**
  the sparse-harm / `onset_logit_offset` calibrator is the cheap decode-time SIDE-STEP of this thread's full fix —
  it injects a little content→when influence without the AR/refinement machinery. This thread is the "do it
  properly" path; that one is the shippable approximation.
- **[[fatigue-governor]] (`governor_release_region.md`, cond-mech §8c):** stamina is the lone existing where→when
  coupling — proves decode-time placement-gating works, but suppress-only/biomechanical.
- **[[jack-heaviness]]:** the onset head's blocky audio-only rhythm (the upstream cause) is what note-context fixes.

## Skills in play
`conditioning-mechanics` §0/§6/§8 (head decoupling, the one-way decode, stamina gate) · `experiment-design`
(Rules 2/5/6 — fair-version, real-reference-first) · `generation-defaults` (the canonical C0 config the probe
must generate with).

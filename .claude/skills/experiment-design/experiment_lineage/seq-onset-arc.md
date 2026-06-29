# Lineage — Sequence-aware onset head ("WHEN" conditioned on "WHERE") (2026-06-22 → 2026-06-29)

**One line:** the 16th-placement signal lives in the NOTE SEQUENCE (the pattern head's "where"), not the audio —
but the deployed onset head is audio-only/non-causal by design, so it discards it; the AR fix explodes and
refinement couldn't bootstrap from the old audio-only C0. **RE-OPENED 06-28** with a better C0 (the current
deployed pipeline) the 06-22 wall never tested. **06-29: the own-output refiner (the matched train-on-C0 test)
closed 06-28's last confound — NEGATIVE.**

**Status:** CLOSED NEGATIVE — wall AIRTIGHT across FOUR independent directions (every probe's positive control
FIRED). The 0.87 teacher-forced note-context signal is a chart-structural PRIOR, NOT in the audio:
(1) forward audio→16th onset **0.65**; (2) seq refiner MATCHED train-on-deployed-C0 **0.672** (06-29; 06-22 v4-C0
0.666, 06-28 mismatched 0.667 — converge); (3) inverse analysis-by-synthesis critic real-vs-corrupted **0.570**;
(4) inverse critic real-vs-deployed-C0 **0.468** (06-29) — vs the 0.871 note-context ceiling. BOTH the own-output
ITERATIVE-REFINER (user's chosen decode form) and the ANALYSIS-BY-SYNTHESIS critic (user's inverse idea) are DEAD:
the audio likelihood is COARSE/density-compatible only, placement-blind beyond density. In `P(chart|audio) ∝
P(audio|chart)·P(chart)` the likelihood carries no fine placement → it ALL lives in the prior P(chart) = a chart
sequence model → only remaining path = a **causal-AR head retrain** (explodes; serious drift-taming) OR bank the
bound. Probes `probe_seqcontext_c0.py`, `probe_seqcontext_matched.py`, `probe_recon_audio.py`, `probe_recon_critic.py`;
train-C0 generator `gen_train_c0.py`.

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
7. **M0 own-output refiner gate (06-29, `probe_seqcontext_matched.py`) — the MATCHED arm closes 06-28's residual
   confound, NEGATIVE.** 06-28 left one hole: it trained on REAL context and only EVAL'd on C0 (mismatched). The
   own-output iterative refiner (the user's chosen decode form) is the MATCHED test: train a note-context net ON
   deployed-C0 context AND eval ON C0. Required generating deployed-C0 for the train split (800 charts; the "big AR
   run" 06-28 deferred) — done via `gen_train_c0.py` (fresh-extract → 4 GPU shards → merge; ~73 min, GPU-bound).
   RESULT (800 train / 28 eval Hard): audio **0.656** (floor ✓), both_real **0.871** (ceiling, POSITIVE CONTROL
   FIRED ✓), both_c0_mismatched **0.667** (= 06-28 exactly ✓), **both_c0_MATCHED 0.672** (8% of the gap — on top of
   the floor). Training the refiner specifically to read C0 did NOT help → C0 has no exploitable placement structure
   even when matched. CONVERGENCE: 0.666 (v4) ≈ 0.667 (mismatched) ≈ 0.672 (matched) across 3 independent setups.
   **VERDICT: own-output refiner DEAD; the wall is C0-INDEPENDENT** (not specific to v4's anti-correlation — ANY
   audio-only-placed C0 lacks the signal). Root cause unchanged: C0's "where" is a deterministic echo of audio, so
   `(audio, C0)` ⊀ `audio` for placement. **Attribution discipline that held:** positive control fired (0.871≫0.656)
   so the null is REAL not underpowered; the same architecture+budget reaches 0.871 on real context, so the matched
   failure is "no signal in C0," not "net too weak"; mismatched reproduced 0.667 exactly, so the harness is correct.
   Data-cache footgun caught in prep: the stale 800-row `seqctx_train_cache` (split since shrank 800→786 charts of
   3820) → row j ≠ valid_samples[j]; re-extracted FRESH ([[dataset-cache-footgun]]).
8. **ANALYSIS-BY-SYNTHESIS critic (06-29, user idea, `probe_recon_audio.py`→`probe_recon_critic.py`) — the AUDIO
   side of the same wall, DEAD for fine placement.** Idea: a critic D(chart,audio) = the likelihood `P(audio|chart)`
   for a synthesis loop `P(chart|audio) ∝ P(audio|chart)·P(chart)` (taste critic = the prior). Three iterations, each
   flaw caught by a control (the METHODOLOGY is the lesson):
   - **v1 regression `g: chart→audio` — VOID.** recon-all 0.0238 > predict-mean floor 0.0197, mismatch−real +0.0004
     (song-insensitive). A binary chart has no AMPLITUDE → regressing absolute audio energy is ill-posed (+ a
     BatchNorm-over-padding bug). Positive control failed → NOT a finding (experiment-design Rule 11).
   - **v2 contrastive critic, corrupted+mismatch negatives — CONFOUNDED.** AUC(real vs corrupted) 0.764 looked
     viable BUT AUC(real vs mismatch-song) 0.517 (control at CHANCE) exposed it: D took a CHART-ONLY shortcut
     (scores chart-coherence = the taste critic / the PRIOR) and IGNORED the audio. The 0.764 was the prior.
   - **v3/v4 contrastive critic, MISMATCH-song negatives ONLY (force the audio path) — CLEAN, control FIRED.**
     AUC(real vs MISMATCH-song) **0.815** (control ✓ — D uses audio for COARSE density/energy match); AUC(real vs
     DEPLOYED-C0) **0.468** (critic CANNOT prefer real over our generator's placement — mean D c0 0.740 > real 0.732);
     AUC(real vs CORRUPTED-plc, density held) **0.570** (~chance, N=28 ~1.3σ). → audio carries STRONG coarse/density
     compatibility, NO fine-16th-placement beyond density. **Attribution win:** the mismatch-song control is what
     separated "audio-grounded" from "chart-coherence shortcut" — v2's 0.764 would have been a false VIABLE.
   **VERDICT:** the analysis-by-synthesis likelihood is placement-blind → it can correct DENSITY but not fine
   placement; the user's "inverse generator" fallback is bounded by the SAME finding (inverting chart→audio for
   placement needs placement IN the audio). The placement signal is the PRIOR, reachable only by a chart sequence
   model (the AR head). `notes/sequence_aware_onset_plan.md` (the ANALYSIS-BY-SYNTHESIS + 4-way CONVERGENCE sections).

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

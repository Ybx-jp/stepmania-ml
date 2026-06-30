# Lineage — Sequence-aware onset head ("WHEN" conditioned on "WHERE") (2026-06-22 → 2026-06-29)

**One line:** the 16th-placement signal lives in the NOTE SEQUENCE (the pattern head's "where"), not the audio —
but the deployed onset head is audio-only/non-causal by design, so it discards it; the AR fix explodes and
refinement couldn't bootstrap from the old audio-only C0. **RE-OPENED 06-28** with a better C0 (the current
deployed pipeline) the 06-22 wall never tested. **06-29: the own-output refiner (the matched train-on-C0 test)
closed 06-28's last confound — NEGATIVE.**

**Status (2026-06-29): fork (A) is ALIVE but UNDERTUNED — not dead, not fully understood (user's framing, post-playtest).**
History: the wall is CLOSED NEGATIVE (placement is a chart-PRIOR not in audio — 4 ways); the BUILD re-opened cheap
(M1a frozen-`h` conv readout 0.892 ≡ ceiling); M1b-3 broke the DENSITY drift (scheduled sampling, run 1.0 @ real
density). M1b-4/5/6 then found the free-run head, RUN ON THE AUDIO HEAD'S DECODE SURFACE, floods 16ths (62% vs real
4%, backbone collapsed) — and I prematurely committed "placement-hollow / BANKED." **The user overturned that twice
(both correct, experiment-design catches):** (1) M1b-7/8 — the bank measured ONE under-tuned config; the 16th-flood is
the KNOWN chaos-smear failure and a phase down-weight DRAINS it to a real-aligned backbone (precision 0.24→0.62 ≈ the
audio head's 0.61). (2) **THE DECODE SURFACE IS HEAD-SPECIFIC** — the deployed palette was co-evolved with the AUDIO
head and several knobs BREAK / INVERT / are ABSENT for the seq head (tau: global-quantile → must be per-song adaptive;
16th-unlock: polarity FLIPS to a down-weight; rests: the audio head's energy-silences are absent → need an explicit
valve). M1b-9 BUILT that surface (rest valve + self-cal tau + inverted phase lever, `seqonset_decode.py`) → by-ear
"**it's better! still very linear**". So the comparison was never fair (tuned audio head vs an untuned new head). **The
real fork is now STRATEGIC, not technical: is the seq-onset path the right investment for THIS stage of the project?**
— it's viable-but-early, like the audio decode when it first landed (which took many hours of vibe-tuning to blossom).
Open mechanistic leads (next session): the user suspects the head did NOT learn to REST but leans on a HOLD-RELEASE
phantom note to stave off collapse; the per-song density CLIFF (flood↔collapse bimodal, no real-d operating point on
some songs); the "still linear" gap. `notes/onset_placement_findings.md`.

The wall (every probe's positive control FIRED): the 0.87 teacher-forced note-context signal is a chart-structural
PRIOR, NOT in the audio: (1) forward audio→16th onset **0.65**; (2) seq refiner MATCHED train-on-deployed-C0
**0.672** (06-29; 06-22 v4-C0 0.666, 06-28 mismatched 0.667 — converge); (3) inverse analysis-by-synthesis critic
real-vs-corrupted **0.570**; (4) inverse critic real-vs-deployed-C0 **0.468** (06-29) — vs the 0.871 note-context
ceiling. BOTH the own-output ITERATIVE-REFINER (refine a FROZEN audio-only C0) and the ANALYSIS-BY-SYNTHESIS critic
are DEAD: the audio likelihood is COARSE/density-compatible only, placement-blind beyond density. In
`P(chart|audio) ∝ P(audio|chart)·P(chart)` the likelihood carries no fine placement → it ALL lives in the prior
P(chart) = a chart sequence model. **The M1a insight:** that chart sequence model ALREADY EXISTS — it's the
deployed decoder (the pattern head's causal stack); we just never read onset off it. So the "retrain" is an
onset-head ADD on a frozen decoder, not a from-scratch model. Probes `probe_seqcontext_c0.py`,
`probe_seqcontext_matched.py`, `probe_recon_audio.py`, `probe_recon_critic.py`, `probe_seqcontext_frozenh.py` (M1a);
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

9. **M1a — frozen-decoder-`h` representation probe (06-29, `probe_seqcontext_frozenh.py`) — POSITIVE, the cheap
   build greenlit on REPRESENTATION.** The wall says placement = the chart PRIOR (a sequence model). M1a asks the
   build-sizing question: does the DEPLOYED decoder (the pattern head's causal stack) ALREADY encode that prior in
   its per-frame hidden state `h` (`typed_model.py:635`), so the onset retrain is a cheap HEAD-ADD on a FROZEN
   decoder rather than a from-scratch model? Read `h` teacher-forced over REAL typed states (native mode, causal:
   `h[t]` sees states[<t]), train a small onset readout, eval 16th-AUC. 800 train / 98 Hard val:
   | arm | 16th-AUC | read |
   |---|---|---|
   | audio | **0.624** | floor ✓ |
   | both_real (raw-note CNN) | **0.892** | ceiling — POSITIVE CONTROL FIRED ✓ |
   | frozen_h (1×1 readout) | **0.763** | 52% of the gap |
   | frozen_h (conv readout, capacity-matched) | **0.892** | **100% — ≡ ceiling** |
   **VERDICT:** the frozen decoder encodes the ENTIRE placement signal; `frozen_h_conv` ≡ `both_real` (onset-AUC
   0.952 > 0.945 too). M1b can be a small causal-conv onset head on the FROZEN decoder — no unfreeze, no dedicated
   note branch. **Attribution save (Rule 11):** the 1×1 arm's 0.763 looked like "decoder compressed half the signal
   away"; a capacity-matched conv arm OVERTURNED it to 100% — the shortfall was readout temporal-mixing, not lost
   signal. Committing the 1×1 number would have wrongly prescribed an unfreeze. **BOUNDARY (Rule 9):** this settles
   REPRESENTATION, not DRIFT — `h` is teacher-forced on REAL notes (the upper bound a readout could see). At gen
   time the head reads its OWN emitted notes (onset→note→`h`→onset can snowball; 06-22 free-run density 0.73 vs real
   0.18). DRIFT is the lone binding M1b gate (`diag_ar_stability`). `notes/onset_frozenh_findings.md`.

10. **M1b — drift gate (06-29, `probe_seqonset_rollout.py`) — controlled NEGATIVE: the head COLLAPSES free-run.**
    M1a settled representation; M1b is the binding DRIFT test. Train the M1a conv head on teacher-forced `h`, then
    free-run a step-by-step rollout reusing the DEPLOYED decode (`_decoder_step_cached` + `pattern_head`); onset
    decided by the head, tau calibrated on teacher-forced `h` and TRANSFERRED to free-run. 12 Hard val, MEAN:
    | arm | density | read |
    |---|---|---|
    | real | 0.272 | target |
    | TF_rollout (incremental `h` + REAL context) | **0.275** | **CONTROL — ≈ real ⇒ no harness bug (Rule 11)** |
    | FREE-run (incremental `h` + OWN notes) | **0.000** | **COLLAPSE to empty** |
    | seed32_after (32 real frames → free-run) | 0.026 | NOT cold-start — can't sustain from own context |
    **VERDICT:** the teacher-forced-trained head is NOT drift-stable. It's the 06-22 exposure-bias failure as
    COLLAPSE (not explosion): trained where note-context was always real, the head leans on it → at free-run the own
    context is sparse → under-fire → self-fulfilling empty. The audio is in `h` but the head doesn't weight it enough.
    **Attribution discipline:** the TF_rollout control (0.275 ≈ real) cleared the harness FIRST (incremental `h` ≡
    training `h`), so the collapse is a real model property, not a rollout bug; the warm-seed arm then ruled out
    cold-start. **⇒ own-output SCHEDULED SAMPLING is mandatory (M1b-3)** — train the head on `h` from its OWN pass-k
    notes so it learns to fire from audio-in-`h` when context is sparse + sustain runs; re-run this gate after. KILL
    fork (A) only if scheduled sampling also fails. CAVEAT (Rule 9): tap-only/greedy emission is a minor own-context
    simplification (collapse is to ZERO onsets = an onset-head behavior, control fires fine). `notes/onset_seqrollout_findings.md`.
    **⚠️ SEVERITY CONFOUND found in M1b-3 (point 11): the 0.000 was partly TAU-TRANSFER calibration (teacher-forced
    tau applied to free-run logits); the TF-only head's collapse is real but "can't fire AT ALL" overstated it.**
11. **M1b-3 — note-dropout SCHEDULED SAMPLING (06-29, `probe_seqonset_ss.py`) — POSITIVE: the drift wall BREAKS.**
    Fix the collapse cheaply/in-parallel: per batch drop real notes `d∼U(0,1)`, decode `h` from the corrupted
    (sparse/empty) context, train the head to predict the FULL real onsets (`d→1` = empty context → forces firing
    from audio-in-`h`). Decoder FROZEN, grad through head only. Then the drift gate + an ABSOLUTE-threshold SWEEP.
    | tau | free_d | run | (real ≈ 0.27 / 1.05) |
    |---|---|---|---|
    | 0.62 | 0.000 | 0.00 | cliff edge |
    | 0.58 | 0.213 | 0.67 | |
    | ~0.56 | ≈0.27 | ~0.8 | **near-real operating point EXISTS** |
    | 0.50 | 0.39 | 1.00 | stable plateau (run 1.0) |
    | 0.10 | 0.51 | 6.9 | runs begin |
    | 0.02 | 0.89 | 83 | explosion |
    **VERDICT:** the head free-runs COHERENTLY from its OWN context (run 1.0, no collapse/explosion across tau
    0.2–0.55); tau≈0.56 hits real density. **TWO corrections:** (a) M1b's "collapse to 0.000" was CONFOUNDED by
    TAU-TRANSFER (teacher-forced tau, calibrated on dense real-context logits, sat above the sparse free-run logit
    range → buried everything; the §3 "tau from the wrong distribution" bug) — the SWEEP is the fair test that
    overturned the SEVERITY (Rules 7–9); (b) dropout-SS genuinely added the audio-firing the TF-only head lacked.
    **Remaining:** a STEEP calibration cliff (free-run logits concentrated → global tau density-sensitive → use
    self/per-song tau or the density-target/stamina mechanisms); slight OVER-fire at dmax=1.0. **NOT yet shown:**
    placement QUALITY (gen-time 16th-AUC vs real / by-ear) — the gate is density/run-length only. NEXT: 16th-AUC of
    free-run onsets at tau≈0.56, then wire into `generate()` (real types + governors) + by-ear. `notes/onset_ss_findings.md`.
12. **M1b-4 — placement QUALITY (06-29, `probe_seqonset_placement.py`) — NEGATIVE: free-run placement ≤ the audio
    floor.** M1b-3's gate measured density/run-length (STABILITY), blind to WHERE the 16ths land (Rule 1). M1b-4
    measures the free-run head's gen-time 16th-AUC against a bracket on the SAME 12 Hard-val songs: FLOOR = the
    deployed audio onset head (0.751); CEILING = a PURE-TF conv head (0.839, POSITIVE CONTROL >> floor — re-measured
    on this set). FREE-RUN (SS head, own-note `h`, tau plateau 0.45–0.56): **16th-AUC 0.43–0.63, ≤ floor across all
    taus**; realized 16th precision **0.02–0.04** (of the 16ths it fires, ~96–98% miss real 16ths). **Attribution
    save (the FIRST run's control FAILED → harness, not model):** I'd used the SS head's TF pass as "ceiling" (0.736,
    < floor) — but SS training trades ~0.10 of TF accuracy for drift-robustness, so its TF pass is NOT the
    representation ceiling. Adding the pure-TF conv ceiling (0.839) fixed the bracket (Rule 11). Real-note context
    (0.839) beats the audio floor → chart-context carries placement; own-note free-run collapses below it → DRIFT
    destroys it. The 0.839 representation (M1a) is CONTINGENT on real notes in context; the head can't bootstrap the
    16th prior from its own audio-placed notes. `notes/onset_placement_findings.md`.
13. **M1b-5 — taste-critic A/B (06-29, `probe_seqonset_critic.py`) — refutes "the AUC metric is too strict."** User's
    sharp objection: 16th-AUC vs ONE reference penalizes valid ALTERNATIVE phrasing. The fair MUSICALITY gate is the
    realism critic (learned P(real), not exact-match). One-change A/B via the SANCTIONED `onset_override` (NO loop
    surgery — Rule 14) through the DEPLOYED `generate()` + canonical governor config (generation-defaults skill),
    density-matched, radar off, stamina off BOTH arms; baseline = the audio head WITH the 16th-unlock (the STRONGEST
    deployed path, Rule 15). 8 chaotic Hard songs: REAL 0.727 ≫ shuf16 0.270 (control FIRED); AUDIO@d_seq **0.253**
    vs SEQ@d_seq **0.005** (SEQ ≤ AUDIO on EVERY song; never fires "real-like", AUDIO does on 25%). The lenient
    musicality gate ALSO ranks SEQ far below the deployed baseline → "too strict" REFUTED, not just unconfirmed.
14. **M1b-6 — the FAILURE MODE: a self-generated 16th-FLOOD (06-29, by-ear → `probe_seqonset_phase.py`).** User
    played the M1b export (`export_seqonset_ab.py`) and read the seq charts as "bland, only 1/16s" — the chaos-OOD
    smear signature — and asked if chaos conditioning leaked. It did NOT (radar=None verified in the rollout cond, the
    `generate()` call, AND the audio baseline). The phase-share measurement confirms the by-ear read: SEQ free-run =
    **19/19/62%** (quarter/8th/16th) vs real **64/32/4** — a self-generated 16th-flood with the backbone COLLAPSED, no
    chaos knob. Audio arms on the SAME harness give sane backbone-heavy shares (controls clean). This is the MECHANISM
    behind M1b-4's free-run 16th-AUC < 0.5 (ranks 16th frames ABOVE backbone) + precision 0.04: the head INVERTS the
    rhythm — abandons the quarters it can't author, floods the offbeats. **Methodology win (Rule 1 + Rule 8):** AUC and
    the critic compressed this to "worse" without NAMING it; the phase-share metric SEES the property, and the user's
    EAR caught it first — the by-ear gate was load-bearing. (NOTE: I then committed "BANKED/placement-hollow" — that was
    PREMATURE; points 15–17 overturn it. The 16th-flood is a measurement on the WRONG decode surface, not a verdict.)
15. **M1b-7/8 — the user's "did you violate experiment-design?" catch (06-29) — YES; the bank was one under-tuned
    config.** The 16th-flood IS the canonical chaos-smear/backbone-collapse failure the skill's OWN Evidence section is
    built around (overturned there by the coherent-conditioning fair test). I'd committed model-blame WITHOUT the fair
    version (Rule 7) and treated 3 metrics on ONE config as robustness (Rule 11). Two fair tests:
    - **M1b-7 manifold conditioning (`probe_seqonset_cond.py`):** condition the rollout on a backbone-heavy manifold
      groove. NO effect — diagnosed (Rule 11 dynamic-range) as a LIVE-but-WEAK channel: groove moves the seq head's
      logits only ~3% (mean|Δlogit| 0.099 vs scale 2.97), because the seq head reads the decoder's `h` and `h` barely
      encodes groove — **the deployed onset head takes `radar` DIRECTLY**, which is why the user's manifold fix worked
      THERE but doesn't transfer to this head's wiring. Inconclusive (OOD) → faithful fix = a radar-DIRECT head.
    - **M1b-8 phase-rebalance (`probe_seqonset_phasepen.py`):** a penalty on the seq head's 16th-offbeat logits DRAINS
      the flood to a real-aligned backbone — precision **0.24→0.62**, F1 0.27→0.71 ≈ the deployed audio head's 0.70 at
      matched density. So **"placement-hollow / dead" was WRONG; the truth is AUDIO-PARITY backbone** (the sequence-
      context 16th advantage is reachable teacher-forced but does NOT survive free-run; the penalty is binary — the
      cliff jumps 62%→0% at b16≥0.1, no graded 16th control because the head can't RANK which 16ths).
16. **The DECODE SURFACE is HEAD-SPECIFIC (06-29, the user's architectural reframe) — a durable mechanism finding.**
    The deployed palette was co-evolved with the AUDIO head (non-causal, calibrated p_onset, 16th-UNDER-confident,
    naturally silent). The seq head (causal, note-momentum, concentrated logits, 16th-OVER-firing, never silent) BREAKS/
    INVERTS knobs: **tau** (global quantile → per-song ADAPTIVE; the cliff); **onset_phase_calib** (the +1.0 16th-UNLOCK
    polarity FLIPS to a down-weight); **stamina/breathing rests** (ABSENT — the audio head's energy-silences are free;
    the seq head needs an EXPLICIT valve sourced from the audio p_onset). UNAFFECTED: fatigue/jack (per-note "where") —
    but the pattern head is OOD on the seq trajectory ("throwing in jumps"). Folded into `conditioning-mechanics` §8.
17. **M1b-9 — the FAIR decode surface BUILT + playtested (06-29, `seqonset_decode.py`, `probe_seqonset_rest.py`).**
    Rest valve (audio p_onset energy envelope biases the seq logit down in quiet → rests; rests/1k 1.95→3.9 toward real
    5.1) + self-cal tau (BINARY SEARCH best-tracking → all export songs at real density; a quantile-iteration DIVERGED,
    collapsing a song to empty) + inverted phase lever. Regen A/B `~/sm-generated/seqonset_ab_fair`. **By-ear: "it's
    better! still very linear"** — a real improvement (it pauses now), still clearly behind the deployed audio head;
    user's read: NOT a fair test yet (tuning unfinished, like early audio decode). **HOLD-RELEASE HYPOTHESIS (user,
    UNTESTED — next session):** the head may not have learned to REST — it may lean on a hold-release phantom note to
    stave off collapse. Density CLIFF confirmed per-song (flood↔collapse bimodal). **VERDICT: path ALIVE + undertuned;
    the fork is now STRATEGIC (right investment for this stage?), not "is it viable."**

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

# Sequence-aware onset head — the justified program (Phase 2.6)

## Why (the turning point, 06-22)
16th PLACEMENT is SEQUENCE-determined, not audio-ambiguous. `diag_seqcontext_probe.py`: 16th-localization AUC
audio-only **0.649** → causal note-sequence-only **0.935** → both **0.944**. The note sequence (run coherence)
carries the placement signal; audio is nearly secondary once note context is present. Our current onset head
is **audio-only + non-causal** (a deliberate Stage-2 anti-drift choice) → it discards the 0.93 signal → places
isolated audio-salient 16ths instead of coherent runs → the "awkward" play-feel (v7). This supersedes the
"audio-ambiguity ceiling" conclusion ([[chaos_placement_ceiling_SUPERSEDED]], now banner-superseded).
Root insight (user): we measured the wrong objective — global/audio, not local note-sequence coherence.

## The program (3 coordinated pieces; architecture is the de-risked headline)
1. **Architecture — sequence-aware onset head.** Predict onset[t] from `audio[t]` (non-causal, the ANCHOR)
   + `notes[<t]` (causal note context). = the probe's `both` model, integrated into
   `LayeredTypedChartGenerator`'s onset head. **AR-drift mitigation** (Stage-2 onset collapsed to empty):
   (a) AUDIO ANCHOR — audio always signals activity so it can't collapse to empty; (b) scheduled sampling
   (mix teacher-forced + own predictions in training) to close the train/gen gap; (c) KV-cache the causal
   note branch for O(T) decode. GATE: `diag_ar_stability.py` (AR rollout stays stable+coherent?).
2. **Objective — local coherence, not global L1.** Revive the Stage-2 taste critic (corrupted-real negatives;
   REAL>BASE>CHAOS; scores arrow/sequence coherence) as the selection/eval signal. per-song L1 is too coarse
   (v7 matched it yet played awkward). Candidate also: local n-gram / windowed pattern likelihood.
3. **Representation — local chaos + groove manifold.** Chaos as a LOCAL pattern descriptor (sparse-chaos is
   valid; the global scalar conflates chaos with density). Condition on PATTERN properties, not a radar POINT
   — the dims are coupled (jumps→voltage+air CONFIRMED via formulas; stream↔freeze nuance UNVERIFIED — stream
   may count hold occupancy → could be positive not inverse). Radar-point conditioning is WHY chaos went OOD.

## Sequencing (cheap gates before big builds — [[experiment-design]])
- [DONE — YELLOW] AR-stability de-risk (`diag_ar_stability.py`): free-run Bernoulli AR does NOT collapse
  (audio anchor prevents empty) but EXPLODES — density 0.73 (both) / 0.83 (seq) vs real 0.177, mean-run 5.7
  vs real 1.0. Exposure bias the OTHER way: own context says "in a run, continue" → runaway streams. Audio
  anchor moderates (both<seq) but doesn't tame. So: signal real (teacher-forced 0.935), but naive AR
  unstable → **scheduled sampling is MANDATORY, not optional**, with residual risk it doesn't fully tame it.
- [DONE — PARTIAL] Scheduled-sampling de-risk (`--scheduled_sampling`, one-step, eps→0.5): density 0.73→0.66,
  mean-run 5.7→4.0 — HELPED but did NOT tame (still ~3.7× real density, stream-biased). The instability is
  RUNAWAY SELF-FEEDBACK (place note → context says "in a run" → snowball); SS only dampens it. Says: the
  FULLY-AR onset formulation is the wrong shape, not that the signal is bad (still 0.935 teacher-forced).
- [PROPOSED PIVOT] NON-AR note-context / ITERATIVE REFINEMENT: first-pass onset from audio (current stable
  head) → refine with note-context computed over the FIXED first-pass chart (not own streaming output) →
  1-2 passes approximate the real-note context (0.935) with NO runaway loop. De-risk: 2-pass refine probe —
  does it improve placement (run-lengths toward real) while staying stable? Alternatives if it fails:
  density-budgeted AR decode (caps density, may stay stream-biased); full sequential SS with eps→1.
- [ ] Build (only if scheduled-sampling gate passes): causal note-context branch on the onset head
  (warm-start v4 audio branch) + scheduled sampling + KV-cache the note branch. Eval on PLACEMENT (16th-AUC +
  run-length match) + taste critic + playtest. NOT per-song L1.
- [ ] Then objective (taste critic) + representation (local chaos) as follow-on refinements.

## Critic-guided refinement (user idea, 06-22) — the OBJECTIVE layer on top of refinement
generate → critic grades → regenerate-the-weak-parts → keep if better → iterate. Marries the two open
pieces: refinement = the ARCHITECTURE (how to regenerate coherently), taste critic = the OBJECTIVE (what
"better" means). Three flavors, increasing power:
  1. best-of-N + critic (SELECT) — did this (Stage 2b), helped but only picks, never fixes.
  2. critic-guided refinement (this idea, FIX) — critic flags weak bars, regenerate them conditioned on the
     rest (= local inpainting = refinement applied locally), keep if critic improves. Iterate.
  3. critic as a training signal (Stage 2c) — fine-tune toward high-critic outputs.
KEY: (2) NEEDS the refinement mechanism — regenerating a flagged region coherently requires conditioning on
the surrounding FROZEN notes (inpainting); a plain audio-only re-sample just yields another awkward region.
So both the architecture pivot and this idea converge on the SAME foundation (note-context-conditioned
onset); the critic is the guide + stopping rule on top. Caveats: critic grades whole-chart -> need LOCAL
(windowed) scores to target regeneration; iterating toward a critic risks GAMING it (ours is a validated
taste metric REAL>BASE>CHAOS, but cap iterations + keep a playtest in the loop).

## Refinement-mechanism de-risk (DONE — POSITIVE, `diag_refine_probe.py`)
Non-causal frozen-context refiner (denoising: audio + noisy-chart context -> real onset), iterated:
```
  pass        16th-AUC  run-mean  density  Δ      (REAL: density 0.198, run 1.02)
  C0 (input)   0.734     1.12     0.199    —      corrupted rough chart
  refine 1     0.865     1.01     0.198    0.078  big lift, stays real-shaped
  refine 2     0.704     1.01     0.198    0.008  degrades (OOD: own output != training corruption)
  refine 3     0.653     1.01     0.198    0.002
```
- **STABLE — no explosion** (density/run hold at real across passes, Δ->0/converges) — the decisive contrast
  with AR (0.73 density, run 5.7). Breaking the loop into frozen passes works.
- **One pass lifts placement 0.734 -> 0.865** (toward the 0.93 ceiling). Frozen-context refinement recovers
  good placement from a rough chart -> THIS is the architecture.
- Wrinkle: passes 2-3 degrade (refiner only saw synthetic corruption, not its OWN output -> OOD). Fix: train
  on own pass-1 outputs (scheduled-sampling family), or just use ONE pass.
- CAVEAT: the synthetic corruption KEEPS ~half the real 16ths in context, so some of 0.865 reads off
  surviving real notes. The audio-only C0 errs differently (WRONG placement, not drops). -> last cheap gate:
  transfer to a REAL audio-only C0 (train refiner on v4 audio-only outputs -> real) before the full build.

## TRANSFER gate (DONE — SOBERING, `diag_refine_probe.py --real_c0`)
Refiner trained on v4's REAL audio-only C0 (not synthetic corruption), iterated:
```
  pass     16th-AUC  run-mean  density   (REAL: density 0.198, run 1.02)
  C0 (v4)   0.456     1.02     0.198      v4's real rough pass — BELOW chance (its 16ths ANTI-correlate w/ real)
  refine 1  0.666     1.00     0.198      marginal: ~= audio-only ceiling (0.649), NOT the 0.935 ceiling
  refine 2  0.592     ...      ...
```
- STABLE (no explosion, converges) — mechanism robust regardless of C0 source. But placement gain MARGINAL.
- The synthetic test (0.73→0.865) was OPTIMISTIC — it leaked ~half the real 16ths into the context. The REAL
  v4 C0 scores 0.456 (anti-correlated: places 16ths in WRONG spots), so the context is MISLEADING and the
  refiner falls back to ~audio-only (0.666). **Refinement CANNOT bootstrap good placement from a bad C0**, and
  audio-only is our only C0 — circular. The 0.935 ceiling needs good context we can't produce from audio.

## VERDICT (06-22): sequence signal is REAL but NOT cheaply exploitable
AR onset explodes; refinement is stable but can't bootstrap from v4's anti-correlated C0; iterating doesn't
climb. Reaching the 0.935 ceiling requires good context (near-real notes) that audio alone can't produce.
Placement excellence needs a different paradigm (learn the placement DISTRIBUTION from multiple human
chartings; or a much stronger first-pass model), not these levers. **Step back: ship-state = v4 +
amount-control (calib/conditioning); this rigorous bounding is the thesis deliverable.** The critic-guided
loop COULD iteratively improve context but starts from anti-correlated C0 (steep, uncertain) — only if pushing.

## Risks / open questions
- AR-drift survival under generation (the gate above settles direction).
- Cost: causal note branch + scheduled sampling is a real architecture change to the generator (Phase 2.6),
  not a fine-tune. Bigger than anything in the chaos arc.
- The teacher-forced 0.935 includes "continue-the-run" autocorrelation — real and exploitable, but gen-time
  AUC will be lower; the AR-stability + eventual playtest are the honest tests.
- Current best playable model stays **gen_highres_v4** until the new head proves out.

## RE-OPEN (2026-06-28) — architectural framing + a NEW C0 the 06-22 wall never tested
Re-entered from the structure/boundary-snap thread (`phrasing_coherence_findings.md`): after two cheap probes
showed "boundary-snap vs Foote" is not a clean targetable gap (density tracks real; figure-character barely snaps
even in real), the user reframed: **maybe the structure signal IS there, but because "WHEN" (onset head) and
"WHERE" (pattern head) are isolated heads, the model can't ARTICULATE it.** This thread is the quantitative form
of that reframing.

**Architectural framing (confirmed in code, `typed_model.py`):** the decode is STRICTLY one-directional —
`p_onset` is precomputed (audio-only, non-causal, `onset_logits(memory,diff,radar,style)` — no note history, no
fatigue, no stamina, no "where") BEFORE the loop → stamina thins it (CEILING-only, sheds lowest-`p_onset`) →
pattern head decides "where" → fatigue adjusts "where". **The pattern head's "where" NEVER flows back to the onset
"when".** The ONLY where→when coupling that exists is **STAMINA** (8c): it reads the realized FOOT-COST and raises
the onset threshold — i.e. a tiny, hand-crafted, SUPPRESS-ONLY, biomechanics-only slice of the coupling. It proves
the principle (footwork can gate placement at decode time) but carries fatigue, not musical structure; the head
itself stays blind. So the 0.649→0.935 gap (audio-only vs note-context 16th-AUC) is exactly "latent capability the
factored decode discards."

**The NEW angle (why 06-22's "circular" verdict is re-openable):** the wall was *"refinement can't bootstrap from a
bad C0, and audio-only is our only C0."* But the C0 that scored **0.456 (anti-correlated)** was **gen_highres_v4**
— the OLD audio-only generator. The CURRENT deployed pipeline (`gen_motif_full_fixed` + pattern_temp 1.0 + full
governor) is a DIFFERENT, much-improved C0 whose realized charts carry real run-coherence from the pattern head +
governors. **06-22 could not test it because that pipeline didn't exist yet.**

**The decisive no-(generator-)retrain probe (`probe_seqcontext_c0.py`):** reuse `diag_seqcontext_probe`'s
tiny CNN, TARGET = REAL onset, but swap the note CONTEXT source: `audio` vs `both_real` (real context = ceiling)
vs **`both_C0`** (context = the DEPLOYED generator's chart). Train the note-context branch on the full real train
split (PARALLEL extraction via DataLoader workers + own npz cache — the serial path through the stale index-cache
was a 1-hour 1-core hang), eval the deployed-C0 val songs with the context source swapped at TEST time.

### RESULT (2026-06-28) — POSITIVE CONTROL FIRED; the WALL STANDS (a clean, well-controlled negative)
800 real train songs / 28 deployed-C0 eval songs, 16th-localization AUC:
| predictor | 16th-AUC | vs reference |
|---|---|---|
| audio | **0.656** | ≈ 06-22 floor 0.649 ✓ |
| both/real | **0.871** | ≈ 06-22 ceiling 0.935 ✓ **CONTROL PASSES** (signal real + reproduces) |
| both/c0 | **0.667** | ≈ audio → recovers only **5%** of the (real−audio) gap |

- **The deployed C0 carries NO placement signal beyond audio.** both/c0 (0.667) ≈ audio (0.656); the real-context-
  trained seq-branch, fed the deployed chart's notes, responds as if it saw nothing but audio.
- **WHY (root cause):** the C0's onsets were placed by the **audio-only onset head**, so the C0's "where" is a
  downstream ECHO of the audio the audio-branch already has — it contains no INDEPENDENT run-coherence (the thing
  real human charts have, worth +0.22 AUC). The pattern-head + governor improvements don't help because the limit
  is UPSTREAM in onset placement. The improved pipeline did NOT move the number.
- **STRIKING CORROBORATION:** 06-22's refiner *trained on* the old v4 C0 = **0.666**; this eval on the *deployed*
  C0 = **0.667**. Near-identical → the convergence argues the result is NOT an artifact of the train-real/eval-C0
  domain mismatch (the one residual confound). The wall is robust across two very different setups + two pipelines.
- **Residual confound (stated honestly):** fully airtight closure = TRAIN on deployed-C0 context (needs generating
  C0 for hundreds of train songs — a big AR run). NOT pursued: 0.666≈0.667 + the root-cause argument make it very
  unlikely to change the verdict (you can't learn real placement from a context that doesn't contain it — the
  06-22 "can't bootstrap from a bad C0" finding, re-confirmed).

### VERDICT (2026-06-28): the cheap decode-time re-open is CLOSED (negative). Reaching the 0.87 signal needs a
RETRAIN (sequence-aware onset head / learn placement from multiple human chartings), NOT a decode lever — the
deployed C0 is not a good-enough bootstrap context because its onsets are audio-only. The shippable side-step
stays the decode-time phrase calibrator ([[onset-phrase-calibrator]]: harm_calib / onset_phase_calib). This also
ANSWERS the structure question that spawned the re-open: the structure/placement signal IS latent (0.87 TF) but
the factored audio-only-onset decode cannot articulate it — and now we know decode tricks can't reach it either.

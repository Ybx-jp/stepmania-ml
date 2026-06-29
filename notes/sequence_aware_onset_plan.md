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

## SCOPING (2026-06-29) — the retrain program: own-output iterative refiner, onset-head-only
The 06-28 re-open closed the cheap decode path → a RETRAIN is required. This section scopes it. **User decisions
(2026-06-29):** (1) decode formulation = **own-output ITERATIVE REFINER** (not the causal-AR head); (2) scope =
**onset-head-only, minimal** (freeze pattern/type/decoder, warm-start; objective=taste-critic and local-chaos
representation deferred to follow-on). Rationale: of the de-risked options only frozen-context refinement never
EXPLODED (AR did, even with scheduled sampling); its one failure — "can't bootstrap from an audio-only C0" — is
exactly what a proper retrain (train the refiner on its OWN outputs) attacks.

### The Rule-0 collision that gates the program (READ THIS before building anything)
The chosen path is, structurally, the 06-22 **TRANSFER gate** (`diag_refine_probe.py --real_c0`): train a refiner
on `(audio + the generator's own audio-only C0) → real onset`. That returned **refine-1 = 0.666 ≈ the audio floor**
and the lineage recorded "refinement CANNOT bootstrap from a bad C0." Information-theoretically: if C0's onsets are
a deterministic echo of audio, `(audio, C0)` carries no more placement signal than `audio` alone → a single refiner
pass is bounded near audio-only no matter how well trained. So the program MUST open with the fair test that 06-22
did NOT run, distinguishing "genuinely re-openable" from "re-deriving the wall." Two things make it re-openable:
1. **C0 identity.** 06-22 used **v4's** C0 (0.456, *anti-correlated* = actively misleading). The **deployed** C0
   (`gen_motif_full_fixed` + pattern_temp 1.0 + governor) is **neutral** (0.667). A refiner reading a neutral
   context may behave differently than one fighting an anti-correlated one. Untested.
2. **Matched training.** The 06-28 probe is MISMATCHED — trains the note-context net on REAL context, evals on C0
   (→ 0.667). The own-output refiner is MATCHED — train on C0, eval on C0, so it learns to read *this C0's specific
   (incoherent) patterns*. The matched arm is the genuinely-untested thing; it needs train-C0.

### M0 — the go/no-go GATE (cheap; NO generator retrain)
Matched-context probe, 3 arms (extends `probe_seqcontext_c0.py`):
- **audio** — floor (~0.65) · **both_real** — train+eval on REAL context = **POSITIVE CONTROL** (must fire ~0.87,
  else underpowered — 20-song run collapsed it to 0.497, so use the full 800-charts power) · **both_c0_matched** —
  **train on C0, eval on C0** ← THE MEASUREMENT.
- **PASS:** matched climbs meaningfully past 0.667 toward 0.87 → C0 has exploitable structure → own-output refiner
  is alive → build M1. **FAIL:** matched sits at ~0.667 → wall is C0-independent → refiner is dead → pivot
  (causal-AR head, or learn placement from multiple human chartings) BEFORE spending a retrain.
- **Train-C0 generation** (the residual confound 06-28 flagged, now sized): measured ~10 s/chart, ~98 frames/s on
  the RTX 3060. Full 800-chart split ≈ 2.3 h serial → run as **4 process shards** (B=1 AR decode underutilizes the
  GPU; near-linear on 4 cores) ≈ ~35–45 min. **Batched `generate()` is forbidden here** — `onset_threshold`/`bpm`
  are batch-scalars (HANDOFF), batching mis-applies one song's tau/bpm to all → a confounded C0. Tooling:
  `experiments/generation_typed/gen_train_c0.py` (`--extract` fresh-from-current-split → `--shard k --nshards 4`
  unmodified `gen_c0` per song → `--merge` → `cache/seqctx_trainc0_cache.npz`). **Footgun caught 06-29:** the old
  `seqctx_train_cache.npz` (800 rows) is STALE vs the current split (786 songs / 3820 charts) — row j ≠
  valid_samples[j]; the runner re-extracts FRESH ([[dataset-cache-footgun]]) rather than trust it.

### M1 — build (ONLY if M0 passes); onset-head-only, everything else frozen
- A note-context refiner head = the probe's non-causal `both` model (audio memory + causal/windowed note-context
  branch → onset logits), **warm-started** from the deployed audio onset branch; pattern/type/decoder FROZEN.
- **Own-output scheduled sampling** in training (train on its own pass-k charts) = the named fix for the 06-22
  pass-2 OOD degradation; iteration-stable by construction (frozen-context passes never exploded).
- Decode: frozen audio-only first pass → C0 → refiner reads `(audio + frozen C0 notes)` → refined onsets → 1 pass
  (maybe 2 if M0 shows climbing). No KV-cache (non-causal, not AR).
- **Run `/autotune` before this retrain** (batch/AMP/length-bucketing/Optuna) per HANDOFF; M0 needs no autotune.

### M2 — eval on the PROPERTY (not per-song L1, which v7 matched yet played awkward)
16th-AUC at gen time + run-length-distribution match vs real + the taste critic as selection signal + **by-ear
playtest = the binding gate** (experiment-design Rule 8). Deployed model stays `gen_motif_full_fixed` until the new
head proves out by ear.

### M0 RESULT (2026-06-29) — GATE FAILED; own-output refiner DEAD (clean controlled negative)
`probe_seqcontext_matched.py`, 800 train charts / 28 eval Hard songs, 16th-AUC:
| arm | 16th-AUC | read |
|---|---|---|
| audio | **0.656** | floor ✓ |
| both_real | **0.871** | ceiling — POSITIVE CONTROL FIRED ✓ (0.871 ≫ 0.656, run is powered) |
| both_c0_mismatched | **0.667** | reproduces 06-28 exactly ✓ (harness correct) |
| **both_c0_MATCHED** | **0.672** | **THE MEASUREMENT — 8% of the gap, on top of the floor** |

Training the refiner specifically to read deployed-C0 context did NOT climb (0.672 vs mismatched 0.667 vs audio
0.656; ceiling 0.871). This was the pre-registered FAIL condition. The neutral-C0 hypothesis is REFUTED: the wall
is **C0-INDEPENDENT** — across v4 anti-correlated C0 (0.666), deployed mismatched (0.667), and deployed MATCHED
(0.672), all ≈ floor. **Why airtight:** positive control fired (null is real, not underpowered); same arch+budget
reaches 0.871 on real context (so it's "no signal in C0," not "net too weak"). This also CLOSES 06-28's last
residual confound ("train on deployed-C0 — not pursued") — now pursued, confirms the wall.

**⇒ Both cheap paths are now closed:** AR head EXPLODES (06-22), iterative refiner CANNOT BOOTSTRAP (06-29,
airtight). Reaching 0.87 needs a paradigm that produces good context WITHOUT a good first pass: (a) a real
**causal-AR head retrain** taming the drift properly (SS + audio anchor at scale — accept residual risk), or (b)
**learn the placement DISTRIBUTION** from a stronger first-pass / multiple human chartings, or (c) STAY ship-state
(the 06-22 fallback) and invest in the decode-time phrase calibrator ([[onset-phrase-calibrator]]). FORK is open.

### ANALYSIS-BY-SYNTHESIS probe (2026-06-29) — user idea: a critic D(chart, audio) as the P(audio|chart) likelihood
Tested whether the AUDIO can score placement (the inverse of the forward onset head), to drive a synthesis loop
`P(chart|audio) ∝ P(audio|chart)·P(chart)` (taste critic = the prior). Three iterations, each fixing the prior's
flaw (the methodology is the lesson):
- **v1 regression `g: chart→audio` (`probe_recon_audio.py`) — VOID.** Positive control failed: g (recon-all 0.0238)
  was WORSE than the predict-mean floor (0.0197) and song-insensitive (mismatch−real +0.0004). A binary chart has
  no AMPLITUDE info → regressing absolute audio energy is ill-posed (+ a BatchNorm-over-padding bug). Not a finding.
- **v2 contrastive critic, trained on corrupted+mismatch negs (`probe_recon_critic.py`) — CONFOUNDED.** AUC(real vs
  corrupted)=0.764 looked viable BUT AUC(real vs mismatch-song)=0.517 (control at CHANCE) exposed it: D took a
  CHART-ONLY shortcut (scores "does this chart look coherent" = the existing taste critic / P(chart) prior) and
  IGNORED the audio. The 0.764 was the prior, not the audio likelihood.
- **v3/v4 contrastive critic, trained on MISMATCH-song negs ONLY (force the audio path) — CLEAN, CONTROL FIRED.**
  | arm | AUC | read |
  |---|---|---|
  | real vs MISMATCH-song | **0.815** | POSITIVE CONTROL FIRED — D uses audio for COARSE (density/energy) match |
  | real vs DEPLOYED-C0 | **0.468** | deployment: critic CANNOT prefer real over our generator's placement (≤chance; mean D c0 0.740 > real 0.732) |
  | real vs CORRUPTED-plc | **0.570** | fine placement, density held: ~chance (N=28, ~1.3σ — not significant) |

**VERDICT: analysis-by-synthesis via the onset/energy audio likelihood is DEAD for FINE placement** (controlled
negative — control fired). The audio carries STRONG coarse/density compatibility (0.815) but NO fine-16th-placement
signal beyond density (0.57 / 0.47). So in `P(chart|audio) ∝ P(audio|chart)·P(chart)`, the likelihood is
placement-blind → ALL placement must come from the PRIOR P(chart) (a chart sequence model) — which is the AR head
(explodes). The audio cannot substitute for the chart prior. The user's "revisit the inverse GENERATOR if this
fails" is bounded by the SAME finding (inverting chart→audio for placement needs placement IN the audio — it isn't).

### THE CONVERGENCE (4 independent confirmations the placement signal is a chart-PRIOR, not audio)
1. forward audio→16th onset AUC **0.65** (audio weakly informative) · 2. seq refiner MATCHED train-on-C0 **0.672**
(C0 carries no placement beyond audio) · 3. inverse critic real-vs-corrupted **0.570** (audio placement-blind beyond
density) · 4. inverse critic real-vs-C0 **0.468** (audio can't prefer real over our generator). The note-context
ceiling is **0.87** (teacher-forced). ⇒ fine 16th-placement lives in the NOTE SEQUENCE prior, unreachable from audio
by forward / refiner / inverse-likelihood means. **Only remaining path = a chart sequence model (causal-AR head,
with serious drift-taming) — or bank the bound and stay ship-state.**

### M1a (2026-06-29) — the chart sequence model ALREADY EXISTS: the FROZEN decoder (`onset_frozenh_findings.md`)
The "causal-AR head" framing implied a from-scratch sequence model. M1a (`probe_seqcontext_frozenh.py`) shows it's
far cheaper: the deployed DECODER (the pattern head's causal stack) ALREADY encodes the placement prior in its
per-frame hidden state `h` (`typed_model.py:635`). A capacity-matched causal-conv readout on FROZEN `h`
(teacher-forced) hits **0.892 ≡ the 0.87 note-context ceiling (100% of the gap)**; audio floor 0.624, a 1×1 readout
only 0.763 (a readout-capacity confound, NOT lost signal — caught by the matched-capacity arm, exp-design Rule 11).
⇒ M1b = a CHEAP onset-head ADD on a frozen decoder (read onset off `h` in the loop), not a from-scratch retrain.
**BUT this settles REPRESENTATION, not DRIFT** — `h` is teacher-forced on REAL notes (the upper bound a readout
could see). The binding gate stays the AR-stability rollout (the §"Sequencing" de-risk above: free-run density 0.73
vs real 0.18). EXECUTING M1b: wire the conv onset head into `generate()`, decoder frozen, own-output scheduled
sampling, then `diag_ar_stability` + by-ear.

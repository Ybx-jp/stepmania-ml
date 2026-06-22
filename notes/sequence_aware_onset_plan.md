# Sequence-aware onset head — the justified program (Phase 2.6)

## Why (the turning point, 06-22)
16th PLACEMENT is SEQUENCE-determined, not audio-ambiguous. `diag_seqcontext_probe.py`: 16th-localization AUC
audio-only **0.649** → causal note-sequence-only **0.935** → both **0.944**. The note sequence (run coherence)
carries the placement signal; audio is nearly secondary once note context is present. Our current onset head
is **audio-only + non-causal** (a deliberate Stage-2 anti-drift choice) → it discards the 0.93 signal → places
isolated audio-salient 16ths instead of coherent runs → the "awkward" play-feel (v7). This supersedes the
"audio-ambiguity ceiling" conclusion ([[chaos_placement_ceiling_findings]], now banner-superseded).
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
- [in progress] AR-stability de-risk (`diag_ar_stability.py`): does audio-anchored AR onset stay stable
  (density ~real, not →0) and coherent (run-lengths ~real)? both vs seq(no anchor).
  - stable → build the head. collapses → scheduled sampling mandatory first. seq collapses + both stable →
    anchor confirmed.
- [ ] Build: add causal note-context branch to the onset head (warm-start v4 audio branch); scheduled
  sampling; KV-cache the note branch. Train; eval on PLACEMENT (16th-AUC + run-length match) + taste critic
  + playtest. NOT per-song L1.
- [ ] Then objective (taste critic) + representation (local chaos) as follow-on refinements.

## Risks / open questions
- AR-drift survival under generation (the gate above settles direction).
- Cost: causal note branch + scheduled sampling is a real architecture change to the generator (Phase 2.6),
  not a fine-tune. Bigger than anything in the chaos arc.
- The teacher-forced 0.935 includes "continue-the-run" autocorrelation — real and exploitable, but gen-time
  AUC will be lower; the AR-stability + eventual playtest are the honest tests.
- Current best playable model stays **gen_highres_v4** until the new head proves out.

# Scope: the chaos×onset GATE — tie off-beat placement to LOCAL audio, not a global scalar

*2026-07-04. The "harness it completely" ceiling-raiser named by the good-settings thread (`good-settings-
region-arc.md`, memory [[good-settings-region]]). Written to decide **probe vs train** — Phase 0 IS that
decision. Grounded in H4 (`h4_offbeat_signal_findings.md`), the taste_grid referee (`goodregion_findings.md`),
and today's playtest (`playtest_log.md` 2026-07-04). Applies the `experiment-design` + `conditioning-mechanics`
skills. This is a DESIGN/hypothesis doc — nothing here is a measured result yet.*

## Problem (precise, from the FAIR evidence — not a rigged setup)
Under the GOOD-settings peak (`--style chaos=0.9,voltage=0.7,air=0.5,freeze=0.5 --guidance 1.5`, the taste_grid
peak cell), **some songs still overload** — tolerance is song-dependent even at gentle guidance:
- **High School Love:** a long section of GENERIC 1/16 streaming (backbone dissolves into a uniform off-grid flood).
- **Love Vacation:** **1/16s in the SILENT intro, before the music starts** — off-beats placed where the audio
  affords NONE. The cleanest audible instance of the mechanism.
- Contrast **TimeToEye** (same setting): "really fun, solid — what I was looking for."

User's binding takeaway: *"even at a gentler conditioning some songs still get pumped past tolerance; a cheap
'use lower conditioning' isn't sufficient."* → a global guidance/chaos level cannot fix this; the injection
MECHANISM must change.

## Root mechanism (already attributed — do NOT re-derive; `conditioning-mechanics` §1-§3,§6 + H4)
`chaos` is one dim of the 5-dim radar, projected (`radar_proj: 5→d`) and **ADDED** to the onset head's
conditioning vector (§1, additive). So chaos raises a **global additive bias** on the onset logits; CFG
(`ol_guided = ol_u + g·(ol_cond − ol_u)`, §3) amplifies that global shift UNIFORMLY across frames. tau then
lets the raised off-beat logits through everywhere → a uniform 16th flood (backbone → 0). There is **no local
term** telling chaos WHERE off-beats belong, so it can only smear (H4's "degenerate global grid manipulation").

The audio DOES carry a weak local off-beat cue — the H4 high-res grid-pooled onset (**deployed dim 41**, off-beat
AUC ~0.66; perc-onset dim 35 ~0.55) — but the onset head under-uses it, and the chaos scalar is not coupled to it.

## RESOLVED / explicitly OUT of scope (two independent failures — don't repeat them)
- **NOT a feature problem.** Two retrains that ADD the high-res feature failed (H4 §5-§6): `gen_highres`
  (warm-start, dim41 stayed at init norm 0.127 = dead, ablation KL 0.0000) and `gen_highres_v2` (random-init +
  off-beat-weighted onset loss → dim41 engaged, norm 1.04, but effect ~0.017 logits; **chaos STILL smears**,
  on-beat% 4.4 vs real ~85). The off-beat audio signal is weak AND redundant with the coarse onset (dim 13).
  → Do NOT propose "add a better feature / re-weight the onset CE." That lever is spent.
- **NOT a decode QUOTA.** `onset_phase_alloc` (flat per-phase share) SMEARS (experiment-design Rule 13);
  `onset_phase_penalty` gates the downbeat but doesn't rescue chaos (chaos MOVES notes off-beat). The uniform
  `onset_phase_calib` 16th-unlock (b16=+1.0, the shipped default) raises ALL 16th logits — it's the good-song
  lever but it is exactly what floods on the overload songs.
- Placement QUALITY of the 16ths that do land ("awkward 16ths") is a separate ceiling (AUC 0.66 caps precision);
  out of scope here — this thread is about WHERE/WHETHER, keyed to audio.

## The gate: additive-GLOBAL → multiplicative-LOCAL
Replace "chaos adds a uniform off-beat bias" with "chaos SCALES the local off-beat cue." Per off-beat frame `t`:
```
gate[t]     = norm01_over_song( max( highres_onset[t]=dim41 , perc_onset[t]=dim35 ) )   # 0..1 local saliency
offbeat[t]  = 1 at 16th (t%4∈{1,3});  ~0.5 at 8th (t%4==2);  0 on-beat (t%4==0)
Δlogit[t]   = chaos_gain · chaos · offbeat[t] · gate[t]      # raise off-beats ONLY where audio supports them
```
Consequence by construction: a **silent intro** (gate≈0) gets **no** off-beats (kills Love Vacation's phantom
1/16s); a **generic pad section** with no transients stays on-grid (kills HSL's smear); a genuinely busy beat
(gate≈1) still unlocks 16ths (keeps TimeToEye / GC magic). Chaos becomes "how hard to chase local transients,"
not "how far to shift the whole grid."

## Phase 0 — the CHEAP DECODE-TIME PROBE (no retrain; THIS decides probe-vs-train)
The `generate()` **`onset_logit_offset=(B,T)`** hook (typed_model.py:503, the same per-frame path `--harm_calib`
uses) can inject `Δlogit[t]` above at decode time — the tau is recomputed WITH the offset (the exporter already
tau-couples it). So we can test the gate on the DEPLOYED model, no training:
1. **Build** `onset_logit_offset` from the song's own dim41/dim35 per the gate math; **REPLACE** the uniform
   b16 unlock with the gated version (uniform unlock OFF, so we test the gate in isolation) — a probe/exporter
   flag `--chaos_onset_gate gain` (mirror `--harm_calib`; same tau-coupling; `conditioning-mechanics` §6).
2. **Also test a de-smear variant** (keep uniform unlock, SUBTRACT `gain·(1−gate[t])` at off-beats) in case
   raising-only can't claw back the CFG global shift.
3. **Songs:** the overload pair **High School Love + Love Vacation** (must un-smear) vs the good controls
   **TimeToEye + Grand Chariot** (must NOT degrade) — the exact charts with by-ear labels already.
4. **Metric (offline, deployment-matched):** the **anchoring** overload-detector (validated in
   `goodregion_findings.md`: <~0.3 = smear; the 2 overload cells read 0.17 vs 0.89) + backbone quarter-share +
   per-song 16th-share vs the song's real chart. Success = overload songs' anchoring climbs out of the smear
   band and backbone returns, WITHOUT the good songs' anchoring/backbone dropping. **By-ear is the binding gate**
   (Rule 8) — dump the ASCII onset grid FIRST (the thread's cautionary lesson), then playtest.
## Phase 0 — DONE (2026-07-04, `probe_chaos_onset_gate.py` / `cache/chaos_onset_gate_v2.log`): DECODE EXHAUSTED
Ran on the 4 labeled songs at the good peak (chaos0.9/g1.5), THREE arms each changing ONE thing from the canonical
BASE (an earlier run confounded unlock-off WITH the gate — the good-song collapse was the unlock removal; corrected).
Per-arm on_grid/anchor/**s16** (real s16 in []):
- **ADD** (canonical + additive content gate): WORSENED the overload — HSL s16 .95→.79 (still a flood), GC .44→.63
  (more mush). Adding off-beats keyed to a weak cue just adds more smear. Dead.
- **DESMEAR** (canonical + subtract in low-saliency zones): un-smeared the overload cleanly (HSL→pure quarter
  backbone, anchor .08→1.0) **but crushed the good songs IDENTICALLY** (GC s16 .44→**.01**, TimeToEye .53→**.00**,
  grids become `Q···Q···`). Failed the "keep GOOD s16" test.
- **The decisive datum:** DESMEAR crushed GC (.44) and HSL (.95) to the SAME .01. If good 16ths were audio-salient
  and smear 16ths weren't, the subtract would spare GC and hit HSL — it hit both equally → **the loved off-beats sit
  at the SAME low audio-saliency as the smear.** Off-beat placement is NOT audio-determined (charter/groove-driven) —
  H4 proven at the decode surface. No gate keyed on audio saliency can separate expressive from smear off-beats; it
  can only do blanket ops (ADD=more mush, DESMEAR=flatten to on-grid).
- Untested narrow wiggle (Rule 10): a SILENCE-ONLY threshold subtract kills Love Vacation's silent-intro 1/16s but
  can't separate HSL-smear from GC-loved (both non-silent, low-saliency). A band-aid, not the fix. **→ Phase 1.**

**⇒ The audio-keyed gate is the WRONG SIGNAL.** The placement signal lives in NOTE-CONTEXT, not audio — which the
PARKED seq-onset arc already established: the frozen decoder's hidden `h` (causal self-attn over notes[<t]) predicts
real 16th placement at conv-readout AUC **0.892** (`seq-onset-arc.md`, M1a `onset_frozenh_findings.md`), vs the audio
head's 0.66. So the learned gate must key on `h`, NOT dim41/dim35. Phase 1 rewrites accordingly (below).

## Phase 1 — the RETRAIN (SELECTED 2026-07-04): a NOTE-CONTEXT-keyed chaos gate = the seq-onset head with chaos
Phase 0 killed the audio-keyed gate (placement isn't in audio). The placement signal is in NOTE-CONTEXT (`h`, AUC
0.892). So the retrain is NOT "FiLM on dim41/dim35" (the original scope guess — REJECTED); it is a **sequence-aware
onset head that reads the decoder's `h`, with chaos conditioning as the organizing objective.** This MERGES two arcs:
the chaos-gate's need for a placement signal is answered by the parked **seq-onset build** (`seq-onset-arc.md`), which
already has the head (`cache/seqonset_ss_head.pt`), the drift fix (note-dropout SCHEDULED SAMPLING, `onset_ss_
findings.md` — free-run coherent, run-length 1.0), and the head-specific decode surface (`conditioning-mechanics` §8:
adaptive tau, INVERTED 16th lever, explicit rest valve). The chaos angle is the NEW, unifying objective the seq-onset
arc lacked (it was "strategic, not viability" — parked for want of a reason; the tolerance failure IS the reason).

**Architecture:** an onset readout on the FROZEN decoder's per-frame `h` (causal self-attn over notes[<t] + cross-attn
to audio), chaos-CONDITIONED so chaos scales HOW MUCH the note-context off-beat structure is expressed —
`p_onset[t] = σ( readout(h_t, audio_t) + w·chaos·offbeat_term(h_t) )`. Chaos then ADDS *anchored* off-beats (placed by
learned note-context, real-like) instead of a global smear. Warm-start/FREEZE the decoder+pattern+type heads (they
work); train only the onset readout + chaos term (cheap, per M1a).

**Objective (co-arrive — H4 §6: weighting alone is weak):** off-beat-weighted onset loss + a chaos-CONDITIONAL term
that moves mass 8th→16th at high chaos (NOT raise both — `chaos_retrain_scope.md`), trained with note-dropout
scheduled sampling so it's free-run coherent (the seq-onset binding gate = DRIFT). Optional taste-critic as the
musicality signal (frame CE never rewards syncopation).

**Selection metric (deployment-matched — `chaos_retrain_scope.md` val_f16 lesson):** per-song
**|gen_16th_share − real_16th_share|** under COHERENT conditioning; NOT val_total, NOT a global-threshold F1.

**Eval:** the SAME 4 labeled songs (overload {HSL, LoveVac} + good {GC, TimeToEye}) + anchoring + BY-EAR. Success =
tolerance WIDENS: overload songs stop smearing at chaos0.9/g1.5 AND the good songs KEEP their loved 16ths (the exact
thing decode couldn't do — GC/TimeToEye s16 preserved, not crushed to 0).

**STAGE IT — de-risk cheap before the expensive train (Rule 6; H4 burned 2 blind retrains):**
1. **[cheap probe — DONE 2026-07-04 ✅ GREEN, `probe_seqcontext_chaos.py` / `cache/seqcontext_chaos.log`]** frozen-`h`
   16th-AUC STRATIFIED by real off-beat share (=chaos). On the HIGH-chaos Hard stratum (mean real s16 0.132): audio
   **0.618** → frozen_h_conv **0.862** ≈ both_real 0.858 (recovers 102% of the note-context gap; control fired). The
   placement signal is present — even STRONGER than on tame charts (0.771) — exactly where the tolerance failures
   live. Design note: the 1×1 readout is only 0.750; needs a small CAUSAL CONV over `h` (temporal mixing). ⇒ TRAIN.
2. **[train — NEXT]** the onset-readout (causal conv on frozen `h`) + a chaos-conditioned off-beat term, off-beat-
   weighted loss + note-dropout SCHEDULED SAMPLING (drift = the binding gate; reuse `seqonset_ss_head.pt` +
   `probe_seqonset_ss.py`). Frozen decoder/pattern/type heads.
3. **[eval]** the 4-song set (overload {HSL,LoveVac} + good {GC,TimeToEye}) free-run at chaos0.9/g1.5 + anchoring +
   BY-EAR. Success = overload stops smearing AND good KEEPS its 16ths (what decode couldn't do).

## Risks / watch-items (experiment-design)
- **Don't pool** overload vs good songs (Rule 12 — the thread's repeat sin). Report per-song.
- **State "under setting X, observed Y"** until the by-ear gate clears it (Rule 9).
- **DRIFT is the binding gate** (seq-onset arc): the head reads its OWN notes at gen time → onset→note→`h`→onset
  snowball. Scheduled sampling addressed it but left the head UNDERTUNED — a probe replicating the head MUST test it
  FREE-RUNNING, not just teacher-forced (`conditioning-mechanics` §8 fork-A caveat).
- **Head-specific decode surface** (`conditioning-mechanics` §8): the seq head needs adaptive tau + an INVERTED 16th
  lever + an explicit rest valve — do NOT reuse the audio head's palette on it.

## Immediate next step
The Phase-0 decode probe + `--chaos_onset_gate` flag SHIPPED (both single-sourced in `decode_harness.chaos_onset_gate_
offset`; kept for the record/ablation). Next = **Stage-1 de-risk probe** (frozen-`h` + chaos readout, chaos-stratified),
reusing `probe_seqcontext_frozenh.py`. Only train if it clears.
```
Lineage: good-settings-region-arc.md (parent) · h4_offbeat_signal_findings.md (mechanism, failed feature retrains)
         · chaos_retrain_scope.md (retrain discipline) · goodregion_findings.md (anchoring metric + referee)
         · seq-onset-arc.md (the note-context placement signal + the frozen-h head + the drift fix — NOW REVIVED)
```

# HANDOFF — ACTIVE: good-settings region → TOLERANCE-FORMULA derivation. Chaos×onset GATE fork SCOPED + de-risked GREEN, PARKED. seq-onset parked.

**Written 2026-07-04 for the next Claude.** This session investigated the good-settings thread's OPEN FORK (the
chaos×onset gate, the "harness it completely" ceiling-raiser) end-to-end — decode probe → decode EXHAUSTED → retrain
reshaped around NOTE-CONTEXT → Stage-1 de-risk GREEN → **user PARKED the train**. No model change (all diagnostic +
tooling). The user's next-session direction: **go back to the TOLERANCE-FORMULA derivation** (tolerance = f(song
features), the good-settings thread's core goal).

## WHERE WE ARE
Deployed model UNCHANGED = `checkpoints/gen_motif_full_fixed/best_val.pt` (42-dim highres). Canonical decode defaults
UNCHANGED (v0.2.0 `footswitch=False` + `hold_stream_penalty`; block below). This session added TOOLING + a scoped/
parked retrain plan; it did NOT change the deployed generator or the decode defaults.

## THE ACTIVE THREAD (next session, user's call) — TOLERANCE-FORMULA derivation
Goal (unchanged from the thread's origin): **tolerance = f(song features)** — a formula for how far a song can be
cranked before it leaves the real high-chaos envelope. Seed lead: the n=40 sweep found `real_density` ρ≈**−0.37**
(p≈0.02) — DENSER songs = LOWER tolerance — but marginal/uncorrected + collinear with onset-busyness. **Next = SCALE
it:** more songs + PARTIAL correlations to disentangle density from busyness; turn the lead into a predictive formula.
Lineage `experiment_lineage/good-settings-region-arc.md`, memory [[good-settings-region]].
- NEW TOOLING that helps here: `cache/audio_fingerprints_highres.npz` (244 val songs' 42-dim mean+std audio
  fingerprints, identity-keyed) + `cache/song_bpms.npz` + `probe_song_similarity.py` — a fast per-song feature table
  to correlate against tolerance. **⚠️ the fingerprint is tempo-NORMALIZED** (beat-aligned features average tempo
  out) → add BPM explicitly (BPM is the [[quality-feature-attribution]] top quality driver).

## PARKED THIS SESSION — the chaos×onset GATE (retrain ceiling-raiser; plan intact, revivable)
Full plan `notes/chaos_onset_gate_scope.md`. State:
- **Phase-0 decode probe = DECODE EXHAUSTED** (`probe_chaos_onset_gate.py`, `cache/chaos_onset_gate_v2.log`). Two
  ISOLATED arms (each ONE change vs canonical, unlock kept ON): an ADDITIVE audio-keyed gate WORSENED the smear; a
  SUBTRACTIVE de-smear un-smeared the overload (HSL anchor .08→1.0) **but crushed the GOOD songs' loved 16ths
  IDENTICALLY** (GC s16 .44→.01 == HSL .95→.01). ⇒ **off-beat placement is NOT in audio** (H4 at the decode surface):
  a gate keyed on audio saliency can't separate an expressive 16th from a smear 16th.
- **Retrain reshaped:** NOT FiLM-on-audio (rejected). Placement is in NOTE-CONTEXT → **revive the parked seq-onset
  head with chaos as its organizing objective** (merges seq-onset-arc). **Stage-1 de-risk GREEN**
  (`probe_seqcontext_chaos.py`, `cache/seqcontext_chaos.log`): on HIGH-chaos Hard charts the frozen decoder's `h`
  predicts 16th placement at conv-readout AUC **0.862 ≈ ceiling 0.858 ≫ audio 0.618** (control fired). A learned
  note-context gate CAN place high-chaos off-beats. **Binding risk = free-run DRIFT** (Stage-2, scheduled sampling;
  teacher-forced ≠ gen-time). **User parked the multi-hour train** ("not the right direction now").
- To REVIVE: `notes/chaos_onset_gate_scope.md` Phase-1 has the full Stage-2 train design (onset readout = causal conv
  on frozen `h` + chaos-conditioned off-beat term, off-beat-weighted loss, note-dropout scheduled sampling; reuse
  `cache/seqonset_ss_head.pt` + `probe_seqonset_ss.py`; eval the 4 labeled songs free-run + by-ear).

## ⚠️ DISCIPLINE — this thread is a CAUTIONARY case study (read before continuing)
The good-settings thread already had 3 metric misreads + a pooled-n=2 "OOD" claim (all ear-caught). THIS session added
a 5th: **the FIRST gate probe arm turned the canonical 16th-unlock OFF and added the gate = TWO changes at once**
(canonical-defaults + one-change-at-a-time violation, [[experiment-design]] Rule 11) → I misattributed the good-song
16th collapse to "the gate over-corrects" when it was the UNLOCK removal. **User caught it.** Corrected to arms that
each change ONE thing from the canonical BASE. Lesson: keep the canonical decode palette FIXED; change one lever;
`playtest` skill's ASCII grid dump first (Rule 8).

## AWAITING USER
Nothing pending — the gate thread is parked by user decision; next work is the formula derivation (user will start it).

## PARKED — seq-onset fork (unchanged, now CONNECTED to the gate; lineage `seq-onset-arc.md`)
16th placement is a chart-PRIOR not in audio; the cheap frozen-`h` build is ALIVE + UNDERTUNED; DRIFT is the gate.
**This session's Stage-1 de-risk (frozen-`h` 0.862 on high-chaos) is fresh evidence the seq-onset build is the right
substrate for the chaos gate** — the two threads merge if/when the retrain is revived.

Env: conda `stepmania-chart-gen` — call the interpreter DIRECTLY
(`/home/ybx/miniconda3/envs/stepmania-chart-gen/bin/python`); NOT `conda run`. Deployed generation ~10 s/chart; the
954-file val PARSE is ~2–4 min (unavoidable startup). Redirected probe stdout BLOCK-BUFFERS — check GPU util, not the
log, for liveness (or use flush=True prints, as this session's probes do).

**READ-FIRST (in order):** ACTIVE → lineage `good-settings-region-arc.md` → `notes/goodregion_findings.md` (the
referee + the crank-chaos/gentle-guidance rule) → `notes/chaos_onset_gate_scope.md` (the parked gate: Phase-0 verdict +
Stage-1 GREEN + the Stage-2 revive plan) → `notes/h4_offbeat_signal_findings.md` (the parent mechanism + the 2 failed
feature retrains). Load-bearing skills: **experiment-design** (this thread is its cautionary case study — now with the
one-change catch), **conditioning-mechanics** §2/§6/§8 (manifold `--style`, CFG on onset, the `chaos_onset_gate_offset`
lever + the head-specific seq-onset surface), **generation-defaults** §1 (canonical config + the `--hardest`/
`--chaos_onset_gate` flags), **playtest** (the by-ear gate).

## CANONICAL EXPORT DEFAULTS (the deployed config — VALIDATED by `/refresh`)
The bare `export_typed_samples.py` run reproduces what the user plays. These values MUST equal the script's argparse
defaults — `tools/check_export_defaults.py` parses the block below and FAILS the refresh if they drift. Durable mirror
of `generation-defaults` §1; update both (and re-run the validator) on any deliberate change. **This section is
permanent — keep it in every HANDOFF rewrite.**

<!-- CANONICAL-EXPORT-DEFAULTS:START (do NOT hand-edit values; re-run tools/check_export_defaults.py after a change) -->
```
checkpoint = checkpoints/gen_motif_full_fixed/best_val.pt
features = highres
type_temperature = 0.4
pattern_temperature = 1.0
repetition_penalty = 1.0
max_jack_run = 2
jack_penalty = 0.0
fatigue_penalty = 2.0
fatigue_free = 6.0
stamina_ceiling = 50.0
stamina_tau = 8.0
stamina_scale = 15.0
stamina_breathe = 1.2
onset_phase_calib = 0.0,1.0
hold_stream_penalty = 8.0
hold_stream_floor = 0.45
hold_stream_win = 16
footswitch = False
harm_calib = 0.0
harm_quiet_q = 40.0
guidance = 1.0
```
<!-- CANONICAL-EXPORT-DEFAULTS:END -->

New export flags this session (both DEFAULT-OFF → canonical block unchanged): `--hardest` (per-song hardest
difficulty; the non-groove path otherwise picks Beginner) and `--chaos_onset_gate GAIN` (EXPERIMENTAL, Phase-0 showed
the audio-keyed gate fails — kept for ablation only).

## RESOLVED (don't re-derive)
- **The good-settings phenomenon is documented H4/H14** — chaos = a global off-grid shift rendered uniformly; guidance
  FLOODS off-beats past a cliff; the fix is conditioning-mechanism/objective, not decode. EXTEND (per-song cliff), don't re-derive.
- **The backbone flip is NOT the governor** (0%, FULL≡GOV_OFF); CFG-amplified chaos (~70%) + 16th-calib (~30%).
- **`chaos=0.9,g=3.0` is SONG-DEPENDENT, not OOD as a regime** — some songs fantastic, others collapse. Dependence is the subject.
- **Off-beat placement is NOT audio-reachable** (Phase-0 decode gate, this session) — it's note-context (frozen-`h`
  0.862). A decode gate keyed on audio can only do blanket ops (add mush / flatten to on-grid). Don't re-try an audio gate.
- **Prior thread (hold-in-stream) SHIPPED + confirmed** (v0.2.0: `footswitch=False` + `hold_stream_penalty`). Details `hold_in_stream_findings.md`.

## BRANCH / PR STATE  (verify ALL live state via `gh pr view <n>` / `gh pr list` / `git log origin/main`)
- **`--hardest` fix:** commit `d293648` on branch **`feat/export-hardest-difficulty`** → **PR #61** (verify via `gh pr view 61`).
- **THIS refresh + the chaos-gate scope/tooling:** committed onto the same branch (see the refresh commit) → the PR
  (verify `gh pr list`). New: `notes/chaos_onset_gate_scope.md`, `src/generation/decode_harness.chaos_onset_gate_offset`,
  the `--chaos_onset_gate` exporter flag, probes `probe_{chaos_onset_gate,song_similarity,seqcontext_chaos}.py`.
  Gitignored (reproducible): `cache/{audio_fingerprints_highres,song_bpms}.npz`, `cache/*chaos_onset_gate*.log`,
  `cache/seqcontext_chaos.log`, `~/sm-generated/gc_similar_*`.
- **Prior:** v0.2.0 shipped (PR #58). PR #60 (prefetch DataLoader) MERGED. `main` protected by `protect-main`.

## DISCIPLINE
Rule 0 (check notes first). **By-ear is the binding gate** (Rule 8 — it caught all 4 metric misreads + the 2-change
gate arm; DUMP THE ARTIFACT). **One change at a time; keep the canonical palette fixed** (Rule 11 — the fresh catch).
Stratify, don't pool (Rule 12). State findings as "under setup X, observed Y" until the ear clears them (Rule 9).
Match the verb to the evidence ([[claim-precision]]). `playtest_log.md` = subjective only; quantitative → `notes/*_findings.md`.

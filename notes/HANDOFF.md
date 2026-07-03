# HANDOFF — ACTIVE: hold-in-stream fix SHIPPED (2 decode defaults); open forks below. seq-onset fork still parked.

**Written 2026-07-02 for the next Claude.** This session localized the fast-song pattern/type-head defect to
HOLDS-IN-STREAMS and DECODE-FIXED it — shipping TWO new canonical decode defaults. The quality-feature-attribution
thread (which pointed here) stays closed; the seq-onset fork stays parked (§ below).

## WHERE WE ARE
Deployed model UNCHANGED = `checkpoints/gen_motif_full_fixed/best_val.pt` (42-dim highres). Both new levers are
DECODE-time (no retrain). **Two canonical decode defaults shipped this session** (in `decode_defaults.CANONICAL_DECODE`,
both entry points, the canonical block below, generation-defaults §1):
- **`hold_stream_penalty=8, hold_stream_floor=0.45, hold_stream_win=16`** — suppress hold-heads in dense STREAM
  sections. The type head opens holds where a human streams (gen 18% of stream frames vs real ~0%); the pinned foot
  then forces a jack (`no_cross_during_hold` + fatigue). `relu(density−floor)` gate → SPARSE musical holds untouched.
- **`footswitch=False`** — forbid footswitch footing → same-panel runs must be one-foot jacks, so the model
  ALTERNATES instead. Playtest: **"sooooo much better", forbidding footswitch made the model MORE creative.** The
  new footswitch on/off knob also revealed the "brutal 16th voltage" is a FOOTSWITCH STRATEGY, not intrinsic jacks.

Full record: `notes/hold_in_stream_findings.md`; lineage
`.claude/skills/experiment-design/experiment_lineage/hold-in-stream-arc.md`; memory [[hold-in-stream-fix]];
playtests `notes/playtest_log.md` (2026-07-02). Probes (import the shared canonical helpers in
`probe_quality_features.py`): `probe_bpm_hold_decomp.py`, `probe_stream_holdjack.py`, `probe_holdstream_fix.py`.

## THE ACTIVE THREAD — hold-in-stream fix (SHIPPED) + its open forks (lineage `hold-in-stream-arc.md`)
The fix shipped and played as a total success. Open forks, in priority order (all DECODE-time, no retrain):
1. **FREE-FOOT-OVERLOAD gate** — the robust successor to the density-PROXY hold gate. `hold_stream_penalty` gates on
   local onset density, which can't tell a dense EXPRESSIVE hold from a dense jack-forcing one (floor 0.45 works only
   because japa1's pathological hold happens to be the densest). A gate on the predicted free-foot workload /
   forced-jack would generalize. **User's stated next lever.**
2. **16th-jack penalty, tastefully** — reframed by the footswitch finding: the residual intrinsic voltage (runs that
   PERSIST footswitch-off, esp. OH WORLD) is the real target; do NOT blanket-kill (real charts have justified 2-note
   16th jacks — H13 / `foot_fatigue_design.md`).
3. **GRADED footswitch policy** — the hard ban (`footswitch=False`) shipped well, but a graded penalty (allow SOME
   footswitch where musical) may be better; revisit.

## AWAITING USER
Nothing pending — the last playtest (`~/sm-generated/footswitch_ab`, footswitch on/off) came back a "total success"
and the defaults are shipped. The next work is the user's call among the three forks above (they named the free-foot
gate as next). No installed set awaiting a verdict.

## PARKED — seq-onset fork (strategic, unchanged since 2026-06-29; lineage `seq-onset-arc.md`)
16th placement is a chart-PRIOR not in audio (wall CLOSED NEGATIVE 4 ways); the cheap frozen-`h` build is ALIVE but
UNDERTUNED. THE DECODE SURFACE IS HEAD-SPECIFIC (`conditioning-mechanics` §8). Playtested "better, still very linear."
Strategic (right investment now?), not "is it viable." Untested lead: hold-release phantom-rest. Not this session.

Env: conda `stepmania-chart-gen` — call the interpreter DIRECTLY
(`/home/ybx/miniconda3/envs/stepmania-chart-gen/bin/python`); NOT `conda run`. Deployed generation ~10 s/chart; the
954-file val PARSE is ~4 min (unavoidable startup); do NOT `warm_cache()` (eager, ~30 min — use lazy `val_ds[i]`).

**READ-FIRST (in order):** ACTIVE → `notes/hold_in_stream_findings.md` → lineage `hold-in-stream-arc.md` →
`notes/playtest_log.md` (2026-07-02). Load-bearing skills: **experiment-design** (this thread is a clean worked
example — pooled-vs-paired baseline, small-n regression, shared-RNG A/B, by-ear gate), **conditioning-mechanics** §7
(`hold_stream_penalty`) + §8b (`footswitch`), **generation-defaults** §1 (the shipped values).

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

## RESOLVED THIS SESSION (don't re-derive)
- **The fast-song pattern/type-head defect IS decode-fixable** (the quality arc's "not a decode knob" was too
  pessimistic) — the sub-locus is holds-in-streams, removed by `hold_stream_penalty`.
- **POOLED-vs-PAIRED baseline:** a "defect-vs-X" slope must use the song's OWN real as the baseline, not a pooled
  constant (the tail-run-long r=+0.49 was a pooled artifact; paired −0.07). Pooled is only right for distance-to-manifold.
- **Confirm a marginal lead at higher n** — holdrate +0.31@n40 (p=.026) → +0.09@n90. A boundary p at small n is where
  a true effect and an artifact are indistinguishable.
- **A global RATE hides a POSITIONAL defect** — needed a co-occurrence metric aligned to real streams.
- **hold_burst ≠ hold+jack** — hold_burst counts free-foot CROSSES (dist≥1.4); a jack is dist-0, invisible to it.
- **SHARED-RNG A/B** (common random numbers): the exporter restores the RNG before the Edit arm so a decode-knob A/B
  isolates the knob from sampling noise. `--ab_hold_stream` / `--ab_footswitch` use it.
- **The voltage is a FOOTSWITCH strategy** — forbidding footswitch collapses same-panel runs 81–85% (HSL/japa1).

## BRANCH / PR STATE  (verify ALL live state via `gh pr view <n>` / `git log origin/main` — Documentation Discipline)
- **This refresh (hold-in-stream fix):** docs + the 2 shipped decode defaults + the 3 probes — on branch
  **`docs/hold-in-stream-fix`** → PR (see `gh pr list`). Code changed: `typed_model.generate` (hold_stream_*,
  footswitch), `decode_defaults.CANONICAL_DECODE`, `export_typed_samples.py` (flags + shared-RNG A/B),
  `scripts/generate.py`. Gitignored (reproducible): `cache/bpm_hold_*`, `cache/stream_holdjack.csv`,
  `cache/holdstream_fix.csv`, `outputs/holdstream_ab*`, `outputs/footswitch_ab`.
- **Prior (quality-feature attribution):** PR #55 merged; mechanism-narrowing on `docs/bpm-mechanism-decomp` → PR #56.
- **Infra refactor:** PRs #52/#53/#54 merged. **Seq-onset:** PRs #50/#51 merged. `main` protected by `protect-main`.

## DISCIPLINE (this session's worked examples)
- **Rule 0** (check notes first) + **experiment-design pooled-vs-paired / small-n / shared-RNG** — each caught a
  would-be false conclusion here. **By-ear is the binding gate** (Rule 8) — it named the defect AND every fix verdict
  (blunt v1, floor tune, the footswitch success). **Match the verb to the evidence** ([[claim-precision]]).
  One change at a time; `playtest_log.md` = subjective only; quantitative → `notes/*_findings.md`.

# HANDOFF — seq-onset fork (A) is ALIVE but UNDERTUNED; the decode surface is HEAD-SPECIFIC; the fork is now STRATEGIC

**Written 2026-06-29 for the next Claude.** Read this whole header — it corrects a conclusion an earlier commit in
THIS session got wrong. History: the wall is CLOSED NEGATIVE (16th placement is a chart-PRIOR, not in audio — 4 ways);
the BUILD re-opened cheap (M1a: a conv readout on the FROZEN decoder's `h` = 0.892 ≡ the note-context ceiling); M1b-3
broke the DENSITY drift (scheduled sampling). I then measured the free-run head ON THE AUDIO HEAD'S DECODE SURFACE,
saw a 16th-flood, and **prematurely committed "placement-hollow / BANKED" (M1b-4/5/6). The user overturned that TWICE,
both valid experiment-design catches** — do not re-derive the bank.

**What's actually true (M1b-7/8/9):**
- **The bank was one under-tuned config** (Rule 7/11 violation): the 16th-flood is the canonical chaos-smear failure,
  and the deployed decode palette was co-evolved with the AUDIO head — it does NOT transfer to the seq head.
- **THE DECODE SURFACE IS HEAD-SPECIFIC** (the durable finding, now in `conditioning-mechanics` §8). For a causal,
  note-momentum onset head: **tau** (global density quantile → must be per-song ADAPTIVE; the concentrated logits make
  a fixed tau a flood↔collapse cliff), **onset_phase_calib** (the +1.0 16th-UNLOCK polarity FLIPS to a down-weight —
  the seq head 16th-OVER-fires), **rests** (the audio head's energy-silences are FREE; the seq head never pauses →
  needs an EXPLICIT valve sourced from the audio `p_onset` envelope). Per-note fatigue/jack are unaffected, but the
  pattern head is OOD on the seq onset trajectory.
- **A head-appropriate surface was BUILT** (`seqonset_decode.py`: rest valve + binary-search self-cal tau + inverted
  phase lever): it drains the flood to a real-aligned backbone (precision 0.24→0.62 ≈ the audio head's 0.61), the
  chart now PAUSES (rests/1k 1.95→3.9 toward real 5.1) and sits at real density. **By-ear: "it's better! still very
  linear."** A real improvement, still clearly behind the heavily-tuned audio head.

**So the fork is now STRATEGIC, not technical: is the seq-onset path the right investment for THIS stage of the
project?** It is viable-but-EARLY — undertuned and not fully understood, like the AUDIO decode when it first landed
(which took many hours of vibe-tuning to blossom). It is NOT dead. **Sharpest UNTESTED lead (user's hypothesis):** the
head may not have learned to REST — it may lean on a HOLD-RELEASE phantom note to stave off collapse (probe: do the
"rests" coincide with hold tails?). Other open leads: the per-song density CLIFF (flood↔collapse bimodal); the "still
linear" gap; a radar-DIRECT / phase-aware conditioned RETRAIN (the faithful fix — inference-time manifold is only a 3%
echo because the seq head reads `h`, not radar directly).

**Deployed model UNCHANGED; conditioning, fatigue, and stamina confirmed INTACT and untouched** — every probe this arc
was a diagnostic READ on a frozen decoder. The shipped generator is exactly as last played.

Env: conda `stepmania-chart-gen` — call the interpreter DIRECTLY
(`/home/ybx/miniconda3/envs/stepmania-chart-gen/bin/python`); **NOT `conda run`** (it buffers child stdout → empty logs
if killed/polled).

**READ-FIRST (in order):** `notes/onset_placement_findings.md` (M1b-4..9 + the CORRECTED VERDICT at the top of the
correction section) → lineage `.claude/skills/experiment-design/experiment_lineage/seq-onset-arc.md` (the full chain,
status header = current) → `conditioning-mechanics` §8 (the head-specific decode surface) → `onset_ss_findings.md`
(M1b-3 density break). Load-bearing skills: **experiment-design** (Rule 7 fair-version / Rule 11 isolate-the-variable —
this session is the worked example of violating BOTH and the user catching it), **conditioning-mechanics** §8,
**generation-defaults**.

---

## 1. WHERE WE ARE
Deployed model = `checkpoints/gen_motif_full_fixed/best_val.pt` (42-dim highres) + the shipped governor (canonical
block below). All seq-onset work is DIAGNOSTIC (no model change). New tooling this session:
- `seqonset_decode.py` — the head-appropriate surface: `build_rest_env` (audio `p_onset` energy envelope = the rest
  valve), `selfcal_tau` (binary-search best-tracking per-song density calibration; defeats the cliff).
- `probe_seqonset_placement.py` (M1b-4, the AUC bracket + pure-TF ceiling head `cache/seqonset_tfceiling_head.pt`),
  `probe_seqonset_critic.py` (M1b-5, density-matched `onset_override` taste-critic A/B), `probe_seqonset_phase.py`
  (M1b-6, phase shares), `probe_seqonset_cond.py` (M1b-7, manifold conditioning), `probe_seqonset_phasepen.py` (M1b-8,
  phase-rebalance), `probe_seqonset_rest.py` (M1b-9, rest structure), `export_seqonset_ab.py` (the by-ear A/B export).
- `probe_seqonset_rollout.py`'s `rollout()` gained opt-in `collect_logits` / `radar` / `phase_pen` / `rest_env`.
The SS head is saved at `cache/seqonset_ss_head.pt`.

## 2. THE ACTIVE THREAD — the STRATEGIC fork + the technical leads (lineage `seq-onset-arc.md`)
The binding decision is the user's: **is the seq-onset path the right investment now?** If pursued, the technical
leads in priority order:
1. **Test the HOLD-RELEASE hypothesis** — does the seq head genuinely rest, or use hold-release phantom notes to avoid
   collapse? (Cheap diagnostic: correlate the rests with hold tails.) This decides whether the rest valve is real.
2. **A radar-DIRECT / phase-aware conditioned RETRAIN** — the faithful version of the user's manifold fix (feed groove
   to the seq head as a DIRECT input + train phase-aware so it stops over-firing 16ths at the source). `/autotune`
   first. This is the path to the better-than-audio 16th advantage (M1a 0.89) that free-run doesn't yet reach.
3. **Finer rest texture** (the rests are currently too long/clustered vs real) + tame the density cliff.

**Also-open, INDEPENDENT nearest-shippable** (no longer a fork-A fallback — they stand on their own, governor stack
verified intact): (1) perc-gate `harm_calib` re-A/B (gate on `perc_onset` dim-35 absence, not dim-0 energy — fixes the
HSL piano-solo gate-targeting); (2) 1/16-jack OOD — measure japa1 1/16-jack run-length (`calib_foot_fatigue.py`) BEFORE
a `fatigue_penalty` 2→3 A/B.

## 3. AWAITING USER
The fair-surface A/B is installed at `~/sm-generated/seqonset_ab_fair` (Challenge=audio, Edit=seq fair-surface, +
original). The user PLAYED it: "it's better! still very linear" + the hold-release hypothesis. No open by-ear gate;
the pending decision is the STRATEGIC fork (pursue seq-onset now vs the nearest-shippable vs something else).

## CANONICAL EXPORT DEFAULTS (the deployed config — VALIDATED by `/refresh`)
The bare `export_typed_samples.py` run reproduces what the user plays. These values MUST equal the script's argparse
defaults — `tools/check_export_defaults.py` parses the block below and FAILS the refresh if they drift. Durable mirror
of `generation-defaults` §1; update both (and re-run the validator) on any deliberate change. **This section is
permanent — keep it in every HANDOFF rewrite.** (Unchanged this session.)

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
harm_calib = 0.0
harm_quiet_q = 40.0
guidance = 1.0
```
<!-- CANONICAL-EXPORT-DEFAULTS:END -->

## 4. RESOLVED THIS SESSION (don't re-derive)
- **The decode surface is HEAD-SPECIFIC** — never test a new onset head on the deployed (audio-tuned) palette: tau,
  the 16th-unlock, and rests all break/invert (§ header + conditioning-mechanics §8). This is the session's keeper.
- **The "BANKED / placement-hollow" conclusion was PREMATURE** — three metrics on ONE under-tuned config is not
  robustness (Rule 11), and I skipped the fair version (Rule 7). The user caught it. The path is alive + undertuned.
- **The 16th-flood is NOT chaos conditioning** — `radar=None` throughout (verified). It's the head's free-run fixpoint.
- **Inference-time manifold conditioning is a ~3% echo on the seq head** (it reads `h`, not radar directly) — the
  faithful fix is a radar-DIRECT head, not a decode knob.
- **`onset_override` A/B + `enforce_playability`** is the Rule-14-clean way to compare onset sources (skips stamina for
  both arms = controlled). **`install_to_stepmania` rmtrees a same-named group** — build under `outputs/...`, NOT under
  `~/sm-generated`.
- **Self-cal tau must be a BINARY SEARCH best-tracking realized density** — a quantile-of-realized-logits iteration
  DIVERGES on the cliff (collapsed a song to empty).

## 5. BRANCH / PR STATE
- This refresh's docs are on **`docs/seq-onset-placement-banked`** (the branch name predates the correction; PR **#51**
  carries the now-corrected framing). Prior seq-onset work merged via **PR #50** (merge commit `0d30ee2`). **Verify
  live state: `gh pr view <n>` / `git log origin/main`** (CLAUDE.md Documentation Discipline). `main` protected by `protect-main`.
- Gitignored (not committed): `cache/seqonset_ss_head.pt`, `cache/seqonset_tfceiling_head.pt`, `cache/seqctx_frozenh_*`,
  `outputs/seqonset_ab*`.

## 6. INFRA / PERF NOTES
- Deployed generation ~10 s/chart (B=1 AR, RTX 3060); the seq rollout is lighter. The dataset re-parse (4452 files) is
  the slow part of `probe_seqonset_critic`/`export_seqonset_ab` (~min); the cache-based probes skip it.
- `cache/samples_v3` = the deployed highres feature cache; the `seqctx_frozenh_*` caches are a FIXED split (re-extract
  for a fresh subset — [[dataset-cache-footgun]]).
- `/autotune` (never run) — the right tool BEFORE any seq-onset retrain (batch/AMP/bucketing/Optuna).

## 7. DISCIPLINE (this session is the cautionary tale)
- **Rule 7 (run the fair version before blaming the model) + Rule 11 (isolate the variable / confirm dynamic range):**
  I committed "placement-hollow" from ONE under-tuned config; the fair version (head-appropriate decode surface) showed
  the path is alive. The user caught both. When a result indicts a NEW component, suspect YOUR setup first (HARNESS→DATA→MODEL).
- **By-ear is the binding gate** (Rule 8) — it named the flood AND the "still linear"/hold-release leads the metrics missed.
- **Match the verb to the evidence** ([[claim-precision]]): "alive + undertuned" ≠ "viable/shippable"; "audio-parity
  backbone" ≠ "good". One change at a time; `playtest_log.md` = subjective only; quantitative → `notes/*_findings.md`.

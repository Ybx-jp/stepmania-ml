# HANDOFF — per-foot fatigue governor, next build (Stage 2/3)

**Written 2026-06-25 for the next Claude.** Read this, then `notes/foot_fatigue_design.md` (full spec). Branch
`gen/foot-fatigue` (PR #40 → main). Env: conda `stepmania-chart-gen`. Tests: `pytest tests/test_generation.py`
(36/36). Validate generation with the `diag_*` scripts below, NOT by eyeballing code (see "Discipline").

---

## 1. WHERE WE ARE (one paragraph)
We built a biomechanical decode-time governor for jacks/jumps in `typed_model.generate`. The **per-note layer
is done + validated**: a two-foot simulator (positions + per-foot exertion `E`, exp decay; min-exertion foot
assignment; jack-vs-travel costs; graded footswitch) with a **barrier penalty** (a ceiling, not a downward
pull). It governs jacks onto the human distribution (maxJackRun 6.2→4.1, real 3.5) with density held. The
**per-region layer is NEXT and unbuilt**: a *stamina* accumulator that modulates onset *density* coherently, with
its ceiling *breathing with audio energy* = a difficulty **arc**. This resolves three things at once:
holds-blindness, jump-starving, and structural flatness (H5).

## 2. THE MENTAL MODEL (don't lose this)
Two governors, two scopes:
- **Per-note (DONE):** the foot model + barrier penalty shape *which footing* among the notes that exist. Acts
  on `pat_logits` (which panels). Knobs: `fatigue_penalty, fatigue_free, fatigue_cap, jack_weight, travel_weight,
  fatigue_tau, footswitch_pen, bpm`.
- **Per-region (NEXT):** a STAMINA accumulator shapes *how many notes exist where*. Acts on the ONSET decision
  (density). This is the lever the per-note layer fundamentally lacks — placement can only redistribute load,
  never remove it (that's why hold-streams collapse to jacks: see §4).

## 3. THE IMMEDIATE NEXT TASK (Stage 2), and its blocker
**Goal:** when sustained workload is high, thin UPCOMING onset density COHERENTLY (raise tau → the onset head
re-selects its most-salient notes), never veto individual notes.

**ARCHITECTURAL BLOCKER (do this first):** onsets are PRECOMPUTED as a full `(B,T)` mask BEFORE the pattern loop
(`typed_model.py` ~L480, `onset = (p > onset_threshold)`), so a stamina value that evolves *inside* the loop
can't influence it. **Stage 2 requires moving the onset DECISION into the AR loop:**
- Keep `onset_logits` precomputed (audio-driven `p = sigmoid(...)`), and keep the global `tau` (density target).
- Inside the per-frame loop, decide `onset[:,t]` from `p[:,t]` vs an EFFECTIVE threshold `tau + f(stamina)` —
  i.e. a per-frame stamina-driven tau bump (raise the bar when tired). Preserve all existing onset modes
  (`onset_sample`, `onset_phase_alloc` top-k, `onset_override`, CFG-blended logits).
- The two `onset[:,t]` consumers in the layered loop are `active = panel_bits[pat] & onset[:,t]` and `on =
  onset[:,t]` (search them). Replace with the in-loop decision.
- ⚠️ Keep it a CEILING: stamina raises tau only when workload is genuinely high, so overall density isn't dented
  (don't repeat Stage 1's mistake — see §4). Validate density holds except in the genuinely-too-dense spots.

**Stage 2 mechanics:** add `E_slow` per foot (or global), long τ (~several measures, vs the per-note τ~2 beats).
On each note, `E_slow += (the realized footing cost)`; decays slowly. Hold-pinning feeds E_slow (a sustained
one-foot grind during a hold raises stamina fast → thins what follows). When `E_slow > stamina_ceiling`, bump
tau over the next frames.

**Stage 3 (the ARC, after Stage 2 works):** make `stamina_ceiling` BREATHE with audio energy/novelty (user's
chosen shape) — high at the climax (chart allowed to be hard), low in verses (forced rest). Reuse audio features
already in the encoder. Governor owns the ceiling; the breathing gives the arc shape (NO lower bound — would
fight the difficulty/radar conditioning).

## 4. WHAT FAILED AND WHY (don't repeat — these cost real time)
- **Hold-pinning in the PATTERN penalty → non-monotonic jack explosion** (maxJackRun 4→14, *more* penalty → MORE
  jacks). Root cause (user diagnosed): during a hold one foot is pinned; a one-foot WIDE stream costs MORE than a
  jack (`travel_weight·2 > jack_weight`), so minimizing E *picks the jack*. Placement can't fix a fatigue problem
  that's really about note COUNT. ⇒ hold handling belongs on the ONSET side, not pattern. (Reverted.)
- **Stage 1 hard onset veto on `min_aff = decayE+cost ≥ cap` → density CRASH** 0.320→0.145. It vetoed on
  ACCUMULATED fatigue, punching holes through every dense section ("awkward note fallout", user-predicted). ⇒ a
  hard per-note veto is the WRONG tool for stamina; stamina must modulate density COHERENTLY (Stage 2). Genuine
  instantaneous impossibility is rare and mostly handled by `no_jump_during_hold`, so the local hard veto is
  near-vacuous — skip it. (Reverted.)
- **Footswitch loophole / lift bug / free-threshold** — five subtleties in the per-note model (all now fixed +
  documented in `foot_fatigue_design.md`). The pattern: a foot SIMULATOR is mostly bookkeeping; every gap becomes
  a loophole the decoder exploits.

## 5. KEY FILES & ANCHORS
- `src/generation/typed_model.py` → `LayeredTypedChartGenerator.generate` (~L378+). Fatigue state init (~L520),
  penalty block (`if fatigue_on:` in the pat_logits section), foot-update + sp_run tracker (after `pat` chosen).
  Onset precompute ~L480. Onset consumers: `active = ... & onset[:,t]`, `on = onset[:,t]`.
- `notes/foot_fatigue_design.md` — FULL spec, all math, the five subtleties, the two-timescale plan, calibration.
- Diags (run from repo root, conda env): `experiments/generation_typed/diag_foot_fatigue.py` (jump/jack/density
  sweep), `calib_foot_fatigue.py` (vs REAL charts on the egregious rich-Hard set), `diag_jack_exertion.py`.
- `src/generation/playtest_export.py` (mandatory playability incl. MANDATORY_JACK_CAP=2), `experiments/.../
  export_typed_samples.py` (`--jack_penalty`, `--fatigue_penalty`, `--fatigue_free`).
- Skills to consult BEFORE probing/exporting: `conditioning-mechanics` (the decode math), `experiment-design`
  (don't mistake the harness for a model defect — it bit us repeatedly), `playtest`.

## 6. VALIDATED NUMBERS / DEFAULTS (anchors for calibration)
- Per-note governor knobs that work: `fatigue_penalty~2`, `fatigue_free=8` (barrier silent below, governs above),
  `fatigue_cap=30` (hard forbid), `jack_weight=1.0 > travel_weight=0.6`, `fatigue_tau=2` beats, `footswitch_pen=4`.
- REAL rich-Hard targets (from `calib_foot_fatigue.py`): jump% **31**, maxJumpStream **10**, maxJackRun **3.5**,
  jack≥4 **0.8%**, density **0.385**. The model OFF: jump% ~6 (UNDER-jumps), maxJackRun 6, density 0.32. ⇒ the
  model under-jumps + under-densifies these songs (a SEPARATE thread: manifold density / air conditioning).
- Deployed gen model: `checkpoints/gen_motif_full_fixed/best_val.pt` (42-dim highres; the H19 clean-retrain).

## 7. DISCIPLINE (this project's hard-won rules — follow them)
- **ONE change at a time.** We broke this once (barrier + hold-pin together) and the regression was confusing
  until isolated. Isolate with a toggle, re-run the diag, compare.
- **Validate on REAL data via the diags, not by reading code.** Every bug here (H19, zero-overwrite, the fatigue
  loopholes, the density crash) was caught by a number moving wrong on real charts, invisible in the code.
- **Don't optimize a metric that's the wrong target.** The calibration "dist-to-real" was dominated by a jump%
  gap the governor shouldn't close (model under-jumps for other reasons). Match the metric to the actual goal.
- **ml-gloss hook is active** (gloss ML jargon on first use). **Memory** at `~/.claude/projects/.../memory/` —
  read `phase-state.md` for the running state; update it as you go.

## 8. OPEN THREADS (priority order, user's stated plan)
1. **Stage 2 (this handoff)** → if it works, **Stage 3 (the arc)**. This is the user's chosen next build.
2. **Release assessment** — circle back AFTER Stage 2/3, especially if successful. Current release candidate
   (`gen_motif_full_fixed` + jack governor λ=1.5, playtest-validated) is ready independent of fatigue.
3. **SKIPPED by the user:** the `ab_fatigue_*` playtest (jack-penalty vs fatigue). Reason: the fatigue/jack
   trade just shuffles between jacks and jumps without a clear feel win; the user would rather get Stage 2/3
   (the real fix) working than A/B the incomplete version. (Sets still in `~/sm-generated/ab_fatigue_*` if needed.)
- Lower priority / parked: the body-turn rotation discount (per-note model charges full per-foot travel for a
  coordinated body rotation — ranking right, magnitude too high; see foot_fatigue_design.md §math gaps).

## 9. BRANCH/PR STATE
`gen/foot-fatigue` → **PR #40** (base main). #36–#39 all merged to main (H15 motif arc, repr bug fixes, H19
retrain). The fatigue work is opt-in (default off), so merging #40 is safe; it doesn't change default behavior.

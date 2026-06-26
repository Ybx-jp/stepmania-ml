# HANDOFF — fatigue/stamina governor COMPLETE, release prepped (PR #41)

**Written 2026-06-26 for the next Claude.** The biomechanical decode-time governor (Stages 1–3) is built,
validated, playtest-confirmed, and its governor knobs have a vouched-for range table + shipping default. Release is
**prepped but NOT merged**. (NB: the governor table is per-knob ranges the user vouched for — NOT the joint
"region of good settings across all conditioning knobs"; that's an explicit V2 item, see §6.)
Read this, then `notes/governor_release_region.md` (the governor range table) and `notes/foot_fatigue_design.md` (full
spec + every failure). Env: conda `stepmania-chart-gen`. Tests: `pytest tests/test_generation.py` (36/36).
Memory: `~/.claude/projects/.../memory/fatigue-governor.md` is the running state.

---

## 1. WHERE WE ARE (one paragraph)
The generator (`checkpoints/gen_motif_full_fixed/best_val.pt`, 42-dim highres) has a complete decode-time
governor in `typed_model.generate`: **(Stage 1, per-note)** a two-foot fatigue model that foots jacks/jumps onto
the human distribution with density held; **(Stage 2, per-region)** a STAMINA accumulator that thins onset DENSITY
where sustained workload is high (a relief valve — a CEILING, never a global cut), hold-aware; **(Stage 3)** the
stamina ceiling BREATHES with audio energy = a difficulty ARC (climax kept, verses rested). All validated by
`diag_*` scripts AND playtest-confirmed ("a tasteful edit, not a rewrite… DEFINITELY an improvement"). The
**governor knobs have a vouched-for range table + default** (`notes/governor_release_region.md` — per-knob,
playtest-vouched, NOT the joint region) and the exporter default flipped to the
center. Everything is on **PR #41** (`gen/foot-fatigue-stage2` → main, 19 commits), open and awaiting the user.

## 2. IMMEDIATE NEXT (the user's open decisions — do NOT act without them)
1. **Merge PR #41?** It's release-prepped (tests pass, governor ranges vouched, skill+docs updated). Merging is the
   outward/irreversible step = the user's call. Ask squash vs merge-commit. This is a **behavior-changing**
   release (default per-note governor now `fatigue_penalty=2`), unlike the opt-in PR #40.
2. **Re-playtest the floored arc endings + pick the breathe default.** The Stage-3 ending bug was fixed
   (`stamina_breathe_floor`) and the sets regenerated/reinstalled: `~/sm-generated/chaos_stamina_g25_breathe{12,18}`
   (vs `chaos_stamina_g25` flat, `chaos_stamina_OFF`). The user was going to feel whether the endings are fixed and
   pick breathe **1.2 vs 1.8**. Pending entry at top of `notes/playtest_log.md`.

## 3. THE SYSTEM (mental model + the release config)
Two scopes, both in `LayeredTypedChartGenerator.generate`, both need `bpm=` (no bpm → silent). The exact math is in
the **`conditioning-mechanics` skill §8** (read it before any probe/export that touches a governor knob).
- **Per-NOTE foot model** (`fatigue_penalty`): acts on `pat_logits` → which-panels (footwork), never note count.
  `jack_penalty` is the OLD single-foot version, now SUPERSEDED (default 0).
- **Per-REGION stamina** (`stamina_ceiling`): acts on the ONSET decision (now made IN the AR loop, not precomputed)
  → density. A global `E_slow` accumulator (fed the realized per-note footing cost; hold-aware = free-foot grind)
  raises the effective onset threshold when tired. CEILING only; byte-identical to OFF below it; needs `fatigue_on`.
- **Stage-3 ARC** (`stamina_breathe`): the ceiling = `stamina_ceiling·(1+breathe·z_energy[t]).clamp(min=floor·ceiling)`,
  energy = phrase-smoothed z-scored `p_onset`. `stamina_breathe_floor=0.4` stops low-energy outros emptying.

**RELEASE CENTER (shipped defaults):** `fatigue_penalty=2, jack_penalty=0`; stamina + breathe **OFF by default**
(opt-in levers — near-no-op on normal charts; they earn their keep only when conditioning is cranked past the human
density envelope, e.g. `--style chaos=q0.99 --guidance 3.0` → density 0.400). GOOD RANGES (vouched, per-knob — not a joint map): fatigue 1.5–3,
stamina_ceiling 15–50 (<10 = global cut, 200 ≡ off), breathe 1.2–1.8 (floor 0.4). MANDATORY playability (fixed,
code-enforced via `enforce_playability`): hold_aware, no_jump_during_hold, no_cross_during_hold, max_jack_run=2,
pattern_temperature≈0.7.

## 4. WHAT FAILED (don't repeat — these cost real time)
- **Hold-pinning in the PATTERN penalty → non-monotonic jack explosion** (maxJackRun 4→14). A one-foot wide stream
  costs MORE than a jack, so minimizing E picks the jack. Placement can't fix a COUNT problem. ⇒ holds belong on
  the ONSET/density side. (The Stage-2 stamina cost IS now hold-aware; the pattern penalty is still holds-blind —
  fine, the pathology barely exists: pinned frames ~0.14 dense, maxJackRun-in-holds 3 = human.)
- **Stage-1 hard onset veto on accumulated `decayE` → density CRASH 0.320→0.145** (hole-punched every dense
  section). A hard per-note veto is the WRONG tool for stamina; it must modulate density COHERENTLY (the Stage-2
  ceiling). **METHOD failure to learn from:** a prior session invented a "reach/affordability veto," built it on
  accumulated fatigue, refuted IT, and committed "the local layer is near-vacuous" — but the user's ACTUAL design
  (onset hold-aware + per-foot effort) was never tested. Re-read the spec against the build. (experiment-design Rule 16.)
- **Stage-3 abrupt endings:** the breathing ceiling collapsed to ~0 at low-energy outros → empty tail. Fixed by the
  floor. Confirmed on real charts, not the truncated diag (which didn't reach the outro — measure the right artifact).
- **`--radar` mean-pin** is OFF-MANIFOLD (smears); DISABLED (errors → use `--style`). See conditioning-mechanics §2.

## 5. KEY FILES, COMMANDS, ANCHORS
- `src/generation/typed_model.py` → `LayeredTypedChartGenerator.generate`: fatigue block (`if fatigue_on:`), the
  in-loop stamina gate (`if stamina_on:` at loop top), the `ceiling_t` breathing schedule (before the loop), the
  E_slow increment (in the foot-commit block, with the hold-aware override).
- Diags (repo root, conda env): `diag_stamina.py` (paired peak/rest relief), `diag_stamina_holds.py` (pinned vs
  non-pinned frames), `diag_stamina_arc.py` (corr(density,energy)+climax-verse Δ, tail check), `diag_breathe_energy.py`
  (`--match` to probe specific songs), `calib_foot_fatigue.py` (per-note vs REAL Hard charts), `diag_foot_fatigue.py`.
- Exporter: `experiments/generation_typed/export_typed_samples.py` — `--fatigue_penalty` (def 2), `--stamina_ceiling`,
  `--stamina_breathe`, `--style "chaos=q0.99" --guidance 3.0` (the crank). Playtest via the `playtest` skill.
- Skills to consult FIRST: `conditioning-mechanics` (the decode math — §8 now covers stamina/arc), `experiment-design`
  (attribution HARNESS→DATA→MODEL; the two now cross-reference each other), `playtest`.
- `notes/governor_release_region.md` (the vouched governor range table), `notes/foot_fatigue_design.md` (spec+failures), `notes/playtest_log.md`.

## 6. OPEN THREADS (priority order — none block the release)
1. **H-onset-perc-bias** (the user's deeper hypothesis): the onset HEAD under-places on melodic-only sections (the
   HSL piano-solo "feel"). NOT a governor knob — a feature/retrain thread. `diag_breathe_energy.py` REFUTED that the
   breathing energy signal is at fault (p_onset tracks harmonic energy fine); the gap is the onset head itself.
2. **Model UNDER-JUMPS** (6% vs real 31%) — a separate density/air thread. Do NOT tune the governor to close it
   (calib "dist-to-real" is dominated by this — the wrong target).
3. **V2 — MAP THE REAL REGION OF GOOD SETTINGS** (`notes/geometry_feasible_region.md` "V2 — MAP THE REGION"): the
   v1 governor table is per-knob VOUCHED ranges, NOT the joint feasible set. The real region = ALL conditioning
   knobs (radar/motif/figure/CFG/density/phase + governor), coupled + SONG-CONDITIONAL, by ACTUAL experiment +
   measurement (sweep interior, walk boundaries; the easy/hard difficulty corners are the only walks done so far,
   `difficulty_corner_findings`). User explicitly scoped this to v2 (2026-06-26).
4. **GDL/equivariance** (the pad's L↔R / Klein-four symmetry) — a v2 redesign / paper angle (pairs with #3), parked.
5. Parked per-note math gaps: body-turn rotation discount (magnitude too high).

## 7. DISCIPLINE (this project's hard-won rules)
- **ONE change at a time**; isolate with a toggle, re-run the diag, compare.
- **Validate on REAL data via the diags, not by reading code.** Every bug here was caught by a number moving wrong.
- **Measure the property at its resolution.** Stamina relief is REDISTRIBUTION — the density MEAN is blind; use
  paired peak/rest windows (and frame-level for holds). The arc shows in corr(density,energy), not the mean.
- **Don't optimize the wrong metric** (the model under-jumps for non-governor reasons; ignore jump% for the governor).
- **Attribution HARNESS→DATA→MODEL**; write conclusions as conditional until a fair test clears the harness. A
  committed "model defect" that gets overturned means you ran the fair test second instead of first.
- **ml-gloss hook active** (gloss ML jargon on first use). Update `memory/fatigue-governor.md` + `playtest_log.md` as you go.

## 8. BRANCH / PR STATE
`gen/foot-fatigue-stage2` → **PR #41** (base main, OPEN, 19 commits = Stage 2/3 + hold-aware + governor range table + the
`--radar` disable + the conditioning-mechanics §8 update + the skill cross-refs + playtest docs). `origin/main`
already has the per-note governor (PR #40 merged). The 4 full-suite failures (`test_audio_features`/`test_dataset`/
`test_parser`) are PRE-EXISTING on origin/main, unrelated to this PR (it only touches `src/generation/`). 36/36
generation tests pass. **Not merged — the user's call.**

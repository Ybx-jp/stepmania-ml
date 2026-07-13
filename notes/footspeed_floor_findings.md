# Footspeed floor + `--style` density fix (data-layer-v2 decode playability)

**2026-07-05, branch `feat/governor-subdiv-recalib`.** Two v2-grid decode fixes surfaced by the Phase-6 →
governor-recalibration → by-ear loop. Companion to `notes/data_layer_v2_scope.md`, lineage `meter-grid-arc.md`,
memory [[meter-4-4-grid]]; mechanism in `conditioning-mechanics §6/§8`.

## 1. `--style` manifold density — 3× over-placement on the 48th grid (FIXED)
`RadarManifold.target_density` returns `E[density | radar, diff]` as a **fraction of frames with a note**, fit on
the **16th grid** (4 frames/beat). On the 48th grid (12 frames/beat) the SAME notes-per-beat is a 3× smaller
frame-fraction, so applying the 16th-grid fraction placed ~3× too many notes (`compute_tau` fires that fraction of
the 3×-denser frames). Only bites a `--style`/`--match_radar` run (the deployed regime is style-free → density falls
to the v2-parsed source chart, self-consistent). **Fix:** scale `style_density *= 4/subdiv` at the exporter use-site
(the manifold stays 16th-grid-native; convert on consumption). Confirmed: raw 0.400 → 0.133 (×⅓); post-fix
`gen_dens` matched ref (Grand Chariot 0.120 vs 0.111) instead of a ~0.40 wall. `export_typed_samples.py` ~line 568.

## 2. Footspeed floor — sub-16th (48th) flams (FIXED, #1 of the by-ear fork)
**By-ear (2026-07-05):** subdiv-recalibrated Equinox "much better" but "1 or 2 sections kinda unplayable."
**Diagnosis (ascii dump + gap histogram, `scratchpad/footspeed.py`+`ascii_dump.py`):** the offenders were all
**0.33×16th = 34 ms = 1-frame gaps @145bpm = 29 notes/s** — a subdivision only REACHABLE since the 48th grid (on
the 16th grid 1 frame *was* a 16th, the hard floor). The onset head places a duple-16th (t%12==3) next to an
audio-driven triplet (t%12∈{2,4}) one frame apart → an unsteppable flam, often a **max-distance move** (D→U, L→R;
~58 pad-units/s ≈ 3× a comfortable 16th cross). Concentrated at measures 28 (a recurring once-per-beat L→R flam)
and 71–72 (a climax fill, incl. a D→U in 34 ms). Root: the model HEDGES duple-vs-triplet (First of the Year gen
triplet-occupancy 0.14 vs human 0.40, HANDOFF parked-lead c) → places both → they collide.

**Why the recalibrated §8 governors miss it (a real coverage gap):** `max_jack_run` caps only SAME-panel runs
(these are cross-panel); `fatigue` is per-foot (a flam alternating feet halves each foot's rate) and a 2-note burst
never accumulates past `fatigue_free`; `stamina` (tau=8 beats) is orders of magnitude too slow. Nothing enforced a
**minimum inter-note spacing / max footspeed regardless of panel** — the 16th grid never needed it.

**Fix — `min_onset_gap` (a decode-time onset refractory):** the timing-domain, panel-agnostic sibling of
`max_jack_run`. `onset` is precomputed (audio-only, non-causal) BEFORE the AR loop, so with full lookahead we run
**non-maximum suppression**: enforce a min pairwise gap of `min_onset_gap` FRAMES, and in each too-close pair keep
the higher-`p_onset` note (the audio-supported one), drop the weaker hedge note. `typed_model.generate`, right
after `onset` is finalized; skipped under `onset_override`.
- **Default = auto:** `2` on the 48th grid (forbids 1-frame 48ths, PRESERVES 2-frame triplet-16ths — the v2 win),
  `1` on the 16th grid (gap≥1 always → NO-OP). Verified subdiv=4 **byte-identical** to v1; subdiv=12 engages.
- **Validated on the artifact (Equinox, `--features highres_v2`):** the 29 n/s flams **31 → 0**; the genuine 2-frame
  triplet-16ths (14 n/s) **preserved** (9→7); measures 28/71/72 clean; density −5% (0.116→0.110). Exporter
  `--min_onset_gap` (None=auto; raise to 3 to also drop 24ths — but that kills triplet-16ths, so keep 2).

## 3. Triplet phase band — the duple/triplet hedge fix (BUILT, #2 of the fork)
The root cause of §2's flams: the model UNDER-places triplets and hedges them onto adjacent 16ths. The 16th-unlock
(`onset_phase_calib=(b8,b16)`) boosts the 8th + 16th-offbeat bands but the Phase-5 deferral gave TRIPLET positions
NO band. **Fix:** an OPT-IN 3rd calib element `b_trip` applied to the triplet-only frames
(`decode_defaults.triplet_band_positions`: **{2,4,8,10}@subdiv=12**, empty on the 16th grid) — a per-phase logit
offset (same "knee not node", per-song-floating mechanism as the 16th-unlock), so the model COMMITS to triplets
where audio affords them. **Single-sourced:** both the tau side (`apply_phase_calib`) and `generate()` now build the
offset from one helper `decode_defaults.phase_calib_offset` (they can't drift). Exporter: `--onset_phase_calib
"0,1.0,<b_trip>"` (the parser already comma-splits to a 3-tuple). Default `b_trip=0` (off) — a new, by-ear-gated
lever; the canonical palette stays `(0.0, 1.0)`.
- **Artifact-validated (Equinox, `--features highres_v2`):** triplet-occupancy (frac of notes at {2,4,8,10}):
  baseline (floor, no band) **0.107** → **b_trip=0.7 → 0.390**, density held (382→392 notes). Human reference occ
  0.40–0.57 (chart-dependent) → 0.39 lands in the human band (headroom to ~1.0 for more). **The floor (#1) still
  holds with the band ON: 1-frame flams stay 0**, 24ths 7→2 — #1 and #2 compose (band commits triplets, floor
  removes any 1-frame collisions). subdiv=4 byte-identical (triplet band empty + the 2-tuple refactor verified).

**Awaiting user:** by-ear of the triplet band (installed `~/sm-generated/triplet_band_new`, b_trip=0.7) — does the
committed-triplet feel read musical, and is 0.7 the right knee or should it go higher (~1.0)? The footspeed floor
(#1) is the playability SAFETY NET; the triplet band (#2) is the EXPRESSIVENESS fix.

**Awaiting user:** by-ear A/B of the footspeed floor — installed `~/sm-generated/footspeed_new` (floor on) vs
`gov_subdiv_new` (recalibrated governor, floor off).

## 4. No fast-jump cap — the two-foot sibling of `max_jack_run` (BUILT, #3 of the fork)
**By-ear (`triplet_band_new`, Equinox):** with the triplet band ON, the user liked the new **pink notes** (48ths,
{1,5,7,11}) but flagged that "some of them seemed to enable the model to **evade decode playability constraints** …
the fatigue system needs another look" — and explicitly "**don't remove pink notes**." So the fix must KEEP the
onset and fix the FOOTING, not thin the note (that rules out raising `min_onset_gap`).
**Diagnosis (`conditioning-mechanics §8d`, ascii-dumped):** a **JUMP (≥2 fresh presses) at SUB-16th spacing** is the
uncapped hole. The footspeed floor (#2) permits 2-frame gaps (a 24th, ~14.5 n/s); when one of those is a jump
(`D+U→L+R` in ~69 ms) the body can't lift+re-place two feet in time. Nothing else forbids it: the fatigue governor
governs WHICH-panels not WHETHER (it just re-routes, and a 2-note jump splits load across both feet so neither foot's
exertion accumulator trips), and `max_jack_run` caps only SAME-panel runs (`on_jack`).
**Fix — `no_fast_jump` (default ON), a pattern-logit hard cap in `generate()`** (right after the `max_jack_run`
block): when `since_onset < f16` (strictly sub-16th — a 24th/48th gap), forbid every pattern whose fresh-press count
(`(panel_bits & ~held).sum` — the same idiom as `no_jump_during_hold`) is ≥2. Singles have `fresh_cnt ≤ 1` → never
masked, so the fast note is spent as a **playable single and the onset is KEPT**. Pure frame-count gate
(tempo-independent, like `min_onset_gap`); **v1 (`f16=1`) can never fire** (`since_onset ≥ 1` ⇒ `< 1` impossible) →
**byte-identical**.
- **Smoke-verified** (`scratchpad/smoke_nofastjump.py`, synthetic model biased hard toward the L+R jump, onsets every
  2 frames, pure taps): subdiv=4 → all 12 jumps KEPT (branch skipped); subdiv=12 → the phrase-opening jump (gap 99)
  kept, every subsequent 24th-spaced jump forced to a single, onset kept, **0 sub-16th jump violations**. The toggle
  flips cleanly (`no_fast_jump=False` restores all jumps).
- Exporter: `--no_fast_jump/--no-no_fast_jump` (default ON) + `--ab_no_fast_jump` (shared-RNG "Edit" arm = uncapped,
  for the by-ear A/B). `tools/check_export_defaults.py` still ALIGNED (v2-only lever, outside the v1 canonical block).

**✅ BY-EAR PASSED (2026-07-05, `nofastjump_ab`, Equinox `b_trip=0.7` + `--ab_no_fast_jump`, shared RNG):** the capped
(Challenge) and uncapped (Edit) arms read "basically the same" — the cap dulled NOTHING of the pink-note
expressiveness — and the uncapped arm exposed exactly the pathology the cap targets: a **3-jump-jack in sub-16th
space** ("just silly", physically unsteppable). Invisible when not needed, decisive when it is. Default stays ON.
The trailing-note-only mechanism (leader jump survives, every sub-16th note after it → single; rolling backward gap,
NOT f16-cell binning) matched the user's play-feel. Cap = the third and final v2 playability sibling
(`max_jack_run` same-panel / `min_onset_gap` timing-floor / `no_fast_jump` two-foot-jump).

## 5. Hold-stream gate — DEAD on the 48th grid (FIXED); the mirror of the `--style` density bug
**By-ear (`pt_surprise_v2`, Watch Out Pt.2, `--style stream=high,freeze=high` g1.5):** user "**hold-stream gate is
broken**." `freeze=high` floods holds + `stream=high` floods density → holds land IN streams → the pinned foot forces
jacks (the exact defect `hold_stream_penalty` exists to suppress, `hold_in_stream_findings.md`).
**Root cause (same class as §1):** the gate is `relu(dens − hold_stream_floor)` where `dens` = LOCAL ONSET
FRAME-FRACTION (avg_pool over `win=hold_stream_win·f16` frames). A frame-fraction is NOT grid-invariant: a 16th
stream is `dens=1.0` on the 16th grid (a note every frame) but only `~0.33` on the 48th grid (a note every 3 frames,
same music). So `hold_stream_floor=0.45` (16th-calibrated) sits ABOVE every v2 stream density → the gate **never
fires on v2**. The governor pass fixed `win` (a frame COUNT, scales with `f16`) but MISSED `dens` (a FRACTION, does
not). **Confirmed on the artifact** (`scratchpad/holdgate_probe.py`): pre-fix `dens` maxes at **0.271** < 0.45 →
gate fires on **0.0%** of 3360 frames; 6/27 holds sit in dense stream frames.
**Fix (one line, `generate()`):** convert the measured fraction to 16th-native before the floor —
`dens = (dens · subdiv/4).clamp(max=1.0)` — so the floor AND the penalty MAGNITUDE stay v1-calibrated; the `clamp(1.0)`
is the 16th grid's natural frame-fraction ceiling (so a 24th/48th stream doesn't over-penalize past a 16th stream).
subdiv=4 → `·1` + clamp no-op → **BYTE-IDENTICAL** (v1 smoke: Deja loin Hard 18 holds, densities match source). The
DIRECTION mirror of §1 (`--style` scaled a 16th-native value DOWN to v2 frames; here we scale a v2 fraction UP to
16th-native — same principle, keep the calibrated constant native, convert the other operand).
- **Validated (Watch Out Pt.2, `--features highres_v2`):** total holds **28 → 19**; holds in dense/stream frames
  (`dens16>0.45`) **6 → 3** (the deep-stream holds gone; the 3 left sit at the soft gate margin where the penalty is
  ~0.08 — correct graduated behavior); **density HELD 0.110** (gate is type-only, onset-decoupled by construction).
- Uses a HOLD-TYPE-AWARE metric, NOT the presence critic (hold-type-blind — `conditioning-mechanics §7` caveat).
- **⚠️ METRIC CORRECTION (2026-07-06):** the "hold-heads in dense frames 6→3" number above is a PROXY (where a hold
  OPENS) and made the fix look partial. The felt pathology is a **16th-run OVERLAPPING an open hold** (the free foot
  streams while a foot is pinned). On that correct metric the fix is COMPLETE: `scratchpad/dump_holds.py` +
  the run∩hold intersection show pure-16th-runs(len≥4)-in-holds **1→0**, and ZERO gap-3 free-foot pairs inside any
  hold BODY in holdfix (holdbug retains the one at frames 2934–2952 ≈ measure 61). Lesson: match the metric to the
  FELT property (a hold in a dense SECTION ≠ a stream trapped UNDER a hold).
- **⚠️ PARTIAL fix — the defect PERSISTS (2026-07-06, by-ear + corrected analysis):** the A/B favored the fix
  (user: "holdbug set was significantly worse") but the user ALSO confirmed a stream-in-hold REMAINS in holdfix
  ("i played both and knew which was which… i played it"). **My first analysis was WRONG:** I filtered for PURE 16th
  runs (gap-3) and threw away the actual defect — LONG holds (19–24×16th = **5–6 beats!**) with a sustained ONE-FOOT
  stream (**8ths @148bpm ≈ 5 notes/s**) running the whole length. E.g. holdfix `D@2688–2745`: Down held ~5 beats while
  the free foot streams U/R/L 8ths (run of 8). CORRECT metric = free-foot stream ≥4 notes @≤8th UNDER a hold:
  **holdfix 2, holdbug 4** — the subdiv fix HALVED it, didn't solve it.
- **ROOT CAUSE (deeper than the gate):** `hold_stream_penalty` gates only the hold-HEAD on onset density; it can't see
  (a) the hold's DURATION (the automaton runs a hold until the next note on that panel → freeze=high + sparse
  same-panel notes = 5–6 beat monster holds), nor (b) the free-foot stream that develops DURING the hold. **The real
  fix = a FREE-FOOT-OVERLOAD gate:** while a hold is open, if the free foot sustains a stream, FORCE-CLOSE the hold
  (biomechanically: you can't hold one foot AND stream with the other — the human releases the hold to stream). A
  hold-DURATION cap is a simpler complementary guard against the 6-beat monsters. **IN PROGRESS — do NOT mark the
  hold-stream bug fixed.**
- Method note: this is the recurring "match the metric to the FELT property" trap — TWICE here (first the head-density
  proxy, then filtering the stream to pure-16ths). Dump the RAW grid and read it; don't trust an aggregation that
  encodes a too-narrow definition of the defect.

### 5b. ★ NEXT-UP (2026-07-12, was PARKED 2026-07-06) — the free-foot-stream-under-hold fix (defect #3)
**Status flipped to NEXT-UP 2026-07-12:** the user chose to BUILD this BEFORE the taste-critic label matrix —
because #3 is the one open defect that would POLLUTE the labels (it's CANDIDATE-VARYING under `freeze=high` AND
presence-INVISIBLE to the critic, so labels on it can't even be learned; lineage `taste-critic-arc.md` Decisions
2026-07-12b). Design directive below; build it structural-primary per [[structural-over-salience]].

**Two failed levers ruled out (measured, not assumed):**
1. **`hold_stream_floor` tweak** — sweep on Watch Out freeze=high (holds / free-foot-stream-under-hold):
   `floor 0.45 pen 8` → 19 / 2 (canonical) · `0.25/8` → 6 / 0 · `0.20/12` → 2 / 0 · `0.15/16` → 0 / 0.
   It DOES kill the defect but by DELETING HOLDS (19→6→0) — it fights `freeze=high` instead of serving it. The
   head-gate can only remove holds, not thin the stream under them. Dead end for this defect.
2. **Stamina governor (§8c) — ALREADY ON and losing.** ⚠️ CORRECTION: stamina is NOT off by default — the
   `generate()` signature says `stamina_ceiling=None` but `CANONICAL_DECODE["stamina_ceiling"]=50.0` (+
   `stamina_breathe=1.2`), so it ran at 50 in every Watch Out gen (a `ceiling=50` re-run reproduced the canonical
   run exactly: 19 holds / 2 defect; `ceiling=25` didn't help either). WHY it loses: stamina thins the
   **least-salient** onsets (`tired = onset & (p_onset ≤ onset_threshold + bump)`, `bump ≤ stamina_max_bump=0.45`);
   the free-foot stream is REAL AUDIO (`p_onset ≈ 1 > tau+0.45`) so it SURVIVES. Confirms §8d's own "hold-aware
   stamina near-vacuous for holds" caveat — it thins by SALIENCE, the defect needs thinning by POSITION.

**THE FIX (user-approved direction, ~6 lines, NOT built):** extend the EXISTING hold-aware stamina — during an OPEN
hold, lift the salience cap so the accumulated free-foot grind (`E_slow`, already hold-aware §8c) thins the stream
even when it's loud. New param `stamina_hold_bump=None` (default None → skip → **byte-identical**):
```
bump = stamina_max_bump * tanh(excess/scale)                                   # (line ~687) salience-capped 0.45
if stamina_hold_bump is not None:                                              # position-based hold-grind thinning
    hb = stamina_hold_bump * tanh(excess/scale)                                # uncapped up to ~1.0
    bump = where(held_start.any(1), maximum(bump, hb), bump)                   # only WHILE a hold is open
```
`onset_threshold` IS the per-song tau (`onset = p > onset_threshold`, line 535), so `bump→1.0` can drop even a
`p≈1` note. SELF-LIMITING: fewer notes → less grind → `E_slow` falls → bump falls → equilibrium at a sustainable
free-foot rate (thins the stream, does NOT delete the hold). Near-inert where holds aren't grinds (`E_slow` low →
`excess≈0`), so v1 charts ~unaffected. Correct metric = `scratchpad/measure_defect.py` (free-foot stream
≥4 @≤8th under a hold). Skill line FIXED this session: `conditioning-mechanics §8c` no longer says "stamina off by
default" (that error burned this session).

**★ DESIGN DIRECTIVE (2026-07-12, user) — TWO fixes, AUTOMATON PRIMARY, thinning is the RESIDUAL safety-net; they
must be ORDERED or they undermine each other.** The user's principle: **"better hold-aware PATTERNING is a better
principle than less-bad hold STREAMING"** and **"the automaton is a more reliable mechanism than salience thinning"**
(salience points the WRONG way here — the offending notes are the loudest). So the priority INVERTS the paragraph
above: the `stamina_hold_bump` salience-thinning is NOT the primary fix — it is the cleanup for whatever the
structural fix leaves pinned.
- **PRIMARY = a STRUCTURAL hold fix (automaton / pattern head).** Spectrum cheapest→best: (1) automaton FORCE-CLOSE —
  free foot streams under a hold → release the held foot so the section becomes a proper two-foot stream (a decode
  rule on the final symbols; reliable, blunt) and/or a hold-DURATION cap for the 6-beat monsters; (2) pattern/type-head
  LOGIT SHAPING — bias AGAINST opening a long hold when a stream is imminent (make the human-like choice at generation,
  not patch it after); (3) LEARNED (out of decode scope). The user leans 1–2 as the reliable core.
- **SECONDARY = `stamina_hold_bump`** — thins ONLY the residual one-foot grind the automaton chose NOT to release
  (e.g. a hold too short for a release to read well).
- **⚠️ THE INTERACTION (the user's "could come out strangely") = a PIPELINE-ORDERING bug if stacked naively.** The two
  act at DIFFERENT stages: stamina thins at the ONSET GATE (`on_t`, `typed_model.py:760`, EARLY) while the automaton
  decides hold-release at SYMBOL RESOLUTION (`close = held & active`, `:919`, LATE). So stamina thins the free-foot
  stream BEFORE the automaton sees it → the release TRIGGER (the stream) is erased → a monster hold stays OPEN with a
  sparse awkward trickle = exactly the bad outcome. **Resolution:** compute the hold-release decision on the
  PRE-thinning free-foot DEMAND (onset intent), and GATE `stamina_hold_bump` OFF on any hold the automaton is about to
  force-close (thin only the NOT-released residual). Co-tune, by-ear. **NEXT-UP (2026-07-12): build this before the
  critic label matrix.** Correct metric = `scratchpad/measure_defect.py` (free-foot stream ≥4 @≤8th under a hold);
  probe song set = fast + `freeze=high` (e.g. Watch Out Pt.2, the original complaint).

### 5c. ★ BUILT + MEASURED (2026-07-12) — the AUTOMATON FORCE-CLOSE alone solves it; NO thinning residual needed
**The structural PRIMARY (§5b) is built and the salience SECONDARY (`stamina_hold_bump`) turned out to be UNNEEDED.**
New `generate()` knobs `hold_release_run` / `hold_release_gap` (`typed_model.py`, in the `hold_aware` block right
after `free_act`): while a hold pins one foot, count the FREE foot's stream (notes at ≤`hold_release_gap` frames = an
8th, `subdiv//2`); when it reaches `hold_release_run` notes, force-close EVERY open hold that frame (fold into `close`).
`hold_release_run=None` → skip → **BYTE-IDENTICAL** (v1 no-op). Exporter `--hold_release_run`/`--hold_release_gap` +
shared-RNG `--ab_hold_release` (Edit arm flips it).
- **THE USER'S RULE (2026-07-12), implemented exactly:** 3 free-foot notes is an acceptable one-foot flourish; the
  **4th** qualifying note (`run=4`) is the trigger → release AT that frame so the freed foot takes it two-foot. A gap
  wider than an 8th (e.g. a quarter rest) RESETS the run → a 3-then-rest hold is left untouched. Causal AR resolution:
  the 4th note is the "escape" (release frame); it plays two-foot-free, so the metric must NOT count it (require the
  pin to PERSIST past the note). `measure_defect.freefoot_stream_runs(require_persist=True)` — anchor-NEUTRAL
  (holdfix 2 / holdbug 4 preserved).
- **The metric ITSELF was rebuilt this session** (`scratchpad/measure_defect.py`; the prior one was lost with the
  gitignored scratchpad). Validated against the documented anchor EXACTLY: `outputs/watchout_holdfix` = 2,
  `outputs/watchout_holdbug` = 4 (incl. the exact `2700–2745` monster run). Parses an exported `.sm` (48 rows/measure)
  → typed grid → `bipedal_metrics.foot_moves`; run-COUNT of ≥4 free-foot notes @≤8th under a PERSISTING hold.
- **PIPELINE-ORDERING GUARD baked in from line 1:** the release trigger reads PRE-thinning intent (`onset[:,t]`, NOT
  the stamina-gated `on_t`), so a future `stamina_hold_bump` can't erase the trigger (the §5b trap). ✅ never fired
  a regression: the guard is inert today (no bump) and correctly does NOT over-release (See Me Now note-grid IDENTICAL).
- **REFINEMENT (opening-frame consistency):** the first residual (Graceful Anomaly, 3→1 with the naive trigger) was a
  `.H1.` hold that OPENS on the same frame as the free foot's first stream note. The metric counts that note (other
  foot pinned from that instant) but the naive `hold_open=held.any(1)` reads `held` at frame START (False on the open
  frame) → counted one short of 4. FIX: `opening = free_act & head/roll`; `hold_open = held.any(1) | opening.any(1)`;
  and EXCLUDE the opening head from the demand (`& ~opening`) so a lone hold-open doesn't falsely start a run. Watch
  Out's monster opens on a bare `.H..` → this refinement is a NO-OP there (still releases at the same frame); it only
  catches the `.H1.` shape. Now Graceful → 0.
- **MEASURED — Watch Out Pt.2 A/B (shared-RNG, `outputs/holdrelease_byear`):** defect **2 → 0**; NOTES held (390→390,
  onset-decoupled); hold-HEADS preserved (19→20, holds SHORTENED not deleted); frames-under-a-hold 571→506. Artifact
  spot-check: the 9-note `URULULUR` monster under a ~4-beat D-hold releases at the 4th note (`1t..`) → clean two-foot
  stream after. Contrast the DEAD-END `hold_stream_floor` sweep (§5b) which reached 0 only by DELETING holds 19→0.
- **ROBUSTNESS sweep (10 diverse `stream`/`rich` Hard songs, naive trigger, `outputs/holdrelease_sweep` +
  `scratchpad/sweep_measure.py`):** total defect runs **21 → 1** (7/10 songs had it; 6 → 0, Graceful → 1 = the
  opening-frame residual the refinement then closed). **Over-release flags = 0** (the 3 defect-free songs byte-
  identical). Hold-heads 374 → 390 (preserved/shortened, NOT deleted). Density held per-song.
- **★ BY-EAR ROUND 1 (2026-07-12) surfaced 2 MORE escape classes the ≥4@8th metric was BLIND to** (the recurring
  "match the metric to the FELT property" trap — a THIRD time on this defect). User on the Edit track: "a 3-note 16th
  stream broke through… 8th spacing is the fastest allowable note-speed during holds; tight against sub-16th or other
  irregular fast patterns," + a **7.2-beat monster** hold the ≥4-run trigger missed (its under-notes are 8th-runs-of-3
  split by rests). Exact-grid confirmed (every measure = 48 rows → parse is an identity map): `L`-hold then `R`(b84.75)
  →`U`(b85.0) = a 16th on different arrows under the hold. WHY no existing guard caught it: `no_cross_during_hold` only
  forbids true CROSSOVERS (R→U isn't one); the FATIGUE governor is HOLDS-BLIND (§8d — never pins the held foot, prices
  R→U as a cheap two-foot move; pinning it REGRESSED before → the reliable lever is the automaton, per
  [[structural-over-salience]]).
- **★ FIX v2 (2026-07-12, user spec) — SPEED LIMIT + DURATION CAP added to the same release block:**
  (1) **SPEED LIMIT:** an 8th (`hold_release_gap`=subdiv//2) is the fastest allowable note under a hold; ANY free-foot
  note FASTER (gap < that: 16th/24th/48th + irregular 4-5f) whose PRIOR free-foot note was also under a hold
  force-releases NOW (`fast_release`, guarded by `ff_last_uh` so a note at a hold's OPEN frame — fast vs a pre-hold
  note — can't insta-close it). 8ths still accumulate to the `hold_release_run`=4 threshold (the 3-note flourish).
  (2) **DURATION CAP** (`hold_max_beats`, user chose **6**): force-close any hold open > that many beats regardless of
  under-activity (`hold_start` per-panel age). Exporter `--hold_max_beats`; the `--ab_hold_release` Edit arm turns the
  FULL fix on (run=4 + speed limit + 6-beat cap).
- **VERIFIED (invariants 0 across every fix chart — Watch Out/Raise/Giudecca):** fast-under-persisting-hold = 0,
  8th-run≥4 = 0, >6-beat holds = 0. No hold THRASH (min hold 0.75 beats; the fix SHORTENS long holds — ≥4beat 4→2 —
  not fragments them). Density held (390→387). ⚠️ CAVEAT: the fast-under-hold defect is RNG-RARE (1-2/song, specific
  realizations) so a fresh baseline often has 0 to begin with; the fix's guarantee is STRUCTURAL (a fast 2nd note IS
  the release → can never be a 2nd note under a PERSISTING hold). ⚠️ OPEN nuance for by-ear: a note under a hold + the
  RELEASE note a 16th later (the "escape") is playable two-foot but may still READ as a fast-under-hold move; the fix
  makes it two-foot but does not move the transition earlier.
- **STATUS: automaton force-close + speed-limit + duration-cap VERIFIED at the metric + artifact; `stamina_hold_bump`
  residual NOT built (unneeded).** ⏳ **BY-EAR ROUND 2 PENDING** — install-set `~/sm-generated/holdrelease_v2`
  (Challenge=baseline vs Edit=full fix); compare to the round-1 `holdrelease_byear` (old trigger). ⏳ (orig line:) install-set
  `~/sm-generated/holdrelease_byear` (Challenge=baseline monster vs Edit=fix). Ship as canonical (add to
  `decode_defaults.CANONICAL_DECODE` + both CLIs) ONLY after by-ear passes. Harness notes: v2 export re-extracts
  features every run (no cache) so it's slow; `--prefetch_workers` SILENTLY yields nothing when the pool holds broken-
  audio community songs (exit 0, empty) — use `--song_filter` (pre-restricts) or `--prefetch_workers 0`.

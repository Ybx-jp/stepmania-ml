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

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

**Open (fork #2, NEXT):** the triplet PHASE BAND — resolve the duple/triplet hedge at the source (commit to
triplet OR duple instead of placing both), lifting First-of-the-Year triplet occupancy 0.14 → ~0.40. A new,
unvalidated §6 lever (the deliberate no-triplet-band deferral) → needs its own by-ear gate. The footspeed floor is
the playability SAFETY NET; the phase band is the EXPRESSIVENESS fix.

**Awaiting user:** by-ear A/B of the footspeed floor — installed `~/sm-generated/footspeed_new` (floor on) vs
`gov_subdiv_new` (recalibrated governor, floor off).

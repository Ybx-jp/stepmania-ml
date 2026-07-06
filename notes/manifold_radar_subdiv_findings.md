# Manifold / groove-radar subdiv consistency (data-layer-v2)

**2026-07-05, branch `feat/governor-subdiv-recalib`.** Triggered by the user's question "do we need to refit the
manifold — isn't it computed with the corrupted `//4` flooring?" The answer turned out to be more nuanced than
"yes refit," and the investigation caught a coupling that would have made a naive refit a REGRESSION. Companion to
`notes/footspeed_floor_findings.md`; mechanism in `conditioning-mechanics §2` (the manifold).

## The three layers that must agree on the radar convention
1. **groove-radar MEASUREMENT** (`src/data/groove_radar.py`): chaos colors a note by its beat-position quantization.
2. **The MANIFOLD** (`cache/radar_manifold.npz`): a per-difficulty Gaussian over real radar + density.
3. **The MODEL** (`gen_motif_v2_48th_cont`): trained WITH radar as a conditioning input.

For `--style` / `--match_radar` to work, all three must use the SAME radar convention. They currently do (all
tpb=4) — the point of this note is that a refit must NOT break that.

## Two SEPARATE issues (don't conflate — the user's question mixed them)
### (a) Density 3× — a UNIT bug, already fixed, no refit
The manifold density = frac-of-frames-with-a-note, fit on the 16th grid. On the 48th grid the same notes/beat is a
3× smaller frame-fraction. FIXED at the exporter use-site (`style_density *= 4/subdiv`); the manifold's number is
fine, just read in the right units. See `footspeed_floor_findings.md §1`.

### (b) Chaos under-measures triplets — a MEASUREMENT bug, fixed in code, REFIT DEFERRED
- **Traced (not assumed):** chaos uses CONTINUOUS `note_beats` (`stepmania_parser.py:571`, not the floored tensor),
  re-quantized via `round(beat_fraction·tpb)` and looked up in a **color map**. The old map hard-coded the 4-grid
  `{0:0, 1:1, 2:0.5, 3:1}` with **NO triplet-green (1.25)** entry (its own docstring names green for 12th/24th).
  So a triplet defaulted to **1.0 (yellow)** on BOTH grids: v1 `round(0.333·4)=1→1.0`; v2 `round(0.333·12)=4→
  .get(4,1.0)=1.0`. **A refit alone was therefore a NO-OP for chaos** — my first framing ("refit fixes chaos") was
  WRONG; the bottleneck was the color map, not the parse grid. (experiment-design: traced the code before asserting.)
- **THE BINDING COUPLING (why we do NOT refit now):** `dataset.py:104` hard-coded `GrooveRadarCalculator()`
  (tpb=4) for ALL datasets. Verified from the cache: a `samples_v3_48th` sample stores a pre-computed radar
  (chaos=0.189) — the v2 model **trained on tpb=4 radar**. The current v1-fit manifold is ALSO tpb=4. So manifold
  and model AGREE (a shared, internally-consistent convention). **Refitting the manifold with corrected tpb=12
  radar would DE-SYNC it from the model** (the triplet tail shifts up: a triplet contributes 1.25 not 1.0) → a
  regression on `--style`, not a fix. Bounded to the ~7% triplet songs, but wrong-direction.

## The FIX (committed) + what's deferred
- **COMMITTED (correct code, byte-identical on v1, DORMANT for the current model):**
  1. `groove_radar._build_color_values` is now subdiv-aware: color by the note's quantization DENOMINATOR
     (d=1→0, d=2→0.5, d=4→1.0, else→1.25 green). subdiv=4 reproduces the old map EXACTLY; subdiv=12 gives triplets
     {2,4,8,10} and 48ths {1,5,7,11} their 1.25 green.
  2. `dataset.py:104` threads `self.parser.timesteps_per_beat` into the calculator (v1=4 byte-identical; v2=12
     activates the fix).
- **RETRAIN-GATED:** these change v2 radar, so they take effect only on a v2 cache REBUILD + model RETRAIN. The
  existing v2 cache (tpb=4 radar) + current checkpoint are untouched. ⚠️ Do NOT rebuild the v2 cache and reuse the
  current checkpoint (radar mismatch).
- **DEFERRED to that retrain (NOT standalone):** (i) refit the manifold on the v2 parse with the corrected radar;
  (ii) tag the manifold npz with its `subdiv` so the density conversion generalizes to `manifold_subdiv/
  export_subdiv` and the `×4/subdiv` hack drops out cleanly. Until then: **keep the v1 manifold** (consistent with
  the model) + the `×4/subdiv` density fix.

## Stakes / scope
The manifold ONLY affects `--style` / `--match_radar` (groove-knob) exports. The DEPLOYED v2 regime is style-free
(radar=None), so none of this blocks the deploy swap. The density fix (a) is live and correct; the chaos fix (b) is
a latent code improvement that lands with the next retrain.

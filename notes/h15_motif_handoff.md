# H15 handoff — train the pattern head on groove-correlated MOTIFS (the "vibe" lever)

*Written 2026-06-23 for the next session. Begin here.*

## Why this, why now (the breakthrough + what it leaves open)
2026-06-23 playtest produced the project's **FIRST genuinely-good chart by ear**: OH WORLD, glitch style
(`chaos=high,air=low,stream=mod`), manifold-conditioned, CFG guidance **g≈3.5** — "a human would feel good
about composing this… choreographed, a 1/16 storm and sweeps, preserved global arc with a climax, local
symmetry that varied." (`notes/playtest_log.md`, top entry.) So the stack WORKS at the right operating point:
model representation + manifold conditioning + the now-robust decode CAN make an excellent chart.

But two diagnostics bound what GUIDANCE can do:
- **H16 (`notes/h16_harmonic_findings.md`):** guidance is MONOTONIC, not harmonic. It only trades the quarter
  backbone for 16ths (8th line pinned ~60%). The g3.5 sweet spot is a KNEE ("quarter backbone still ~20–25%"),
  not a resonance. Guidance can dial RHYTHM BALANCE but cannot ADD vocabulary.
- **H14/H15 (playtest):** conditioning shifts QUANTITIES (density, 16th-share, holds) coherently, but glitch
  "failed to capture the vibe" at most settings — because **vibe = characteristic pattern MOTIFS** (stompy
  ornamental figures, sweeps), and the pattern head conditions on the radar SCALAR with **no motif vocabulary**.

**The bet (H15):** give the pattern head (which-panels) a representation of recurring, groove-correlated
MOTIFS so it can produce a style's *character*, not just its note counts. This is a MODEL-level lever (vs
guidance/decode), and it's the path from "right quantity" to "right feel."

## The program — cheap gate FIRST (experiment-design discipline)

### Phase 0 — DE-RISK (no training): do motifs even CARRY style?
The whole bet dies cheaply if real charts' motif usage doesn't separate by groove. Before any model work:
1. Define a **motif** = a short window of the which-panels sequence (try 1–2 beats = 4–8 frames; the panel
   pattern id per onset, ignoring empty frames or not — try both). Mine them from real charts
   (`StepManiaParser.convert_to_tensor_typed` → per-frame pattern id via `src/generation/typed.py`
   `panels_to_pattern`).
2. Build a motif vocabulary (count / cluster frequent windows; consider collapsing by symmetry — L/R mirror,
   rotation — since "varied symmetry" matters).
3. **GATE:** does the motif distribution predict the groove bucket? Bin real charts by chaos/stream/freeze
   (use `RadarManifold` / `src/data/song_selection.py`), and measure whether a chart's motif histogram
   separates the buckets (mutual information, or a simple classifier groove→motif-dist, vs a shuffled
   control). **If motifs don't separate by groove → H15 premise is wrong; report and stop.** If they do →
   you have the target representation AND proof the signal exists. (This mirrors the chaos-arc gates that
   saved wasted retrains — `notes/INDEX.md`, the de-risk pattern.)

### Phase 1 — REPRESENTATION (if Phase 0 passes)
- Turn motifs into a conditioning signal: a **motif codebook** (vector-quantize windows) or a motif-histogram
  vector per style, analogous to the 5-dim radar. Decide whether to condition on a *target motif distribution*
  (parallels `radar_proj` + CFG) or to add a hierarchical "pick-motif-then-realize" decode step.
- Keep it COMPOSABLE with the manifold surface: a style (`chaos=high…`) should map to both a radar target AND
  a motif-distribution target (the manifold could carry motif stats per groove bucket, like it now carries
  density).

### Phase 2 — MODEL / TRAINING
- Warm-start from `checkpoints/gen_style/best_val.pt` (the radar+CFG `LayeredTypedChartGenerator`). Add a
  motif-conditioning vector into `_cond` (alongside difficulty + radar + style), with CFG dropout — reuse the
  `train_radar.py` / `train_style.py` machinery in `experiments/generation_typed/`.
- The per-frame pattern CE can't reward motif-level structure → pair with the **taste critic** (REAL>BASE>CHAOS,
  `checkpoints/realism_critic/best_val.pt`, valid per `notes/stage2a_critic_findings.md`) for selection /
  best-of-N / critic-guided fine-tune.

### Phase 3 — EVAL
- **Playtest is the validated instrument** (offline metrics are blind to choreography — proven all arc).
- Quantitative support: does the generated motif distribution match the target? does the critic rise? Build
  the missing **run/sweep-coherence metric** (the H16 repetition proxy floored at ~0 — exact-window match is
  too strict; try edit-distance clustering or symmetry-aware motif matching) so eval isn't ear-only.

## State you inherit (all on branch `gen/radar-manifold`, PR #35; control fixes + diagnostics this session)
- **Model/decode:** `src/generation/typed_model.py::LayeredTypedChartGenerator` — onset → pattern_head (15-way
  which-panels) → type_head. Conditioning assembled in `_cond` (difficulty + `radar_proj` + style). Decode is
  now ROBUST: this session fixed two playability gaps (note-during-2holds → occupancy `held+fresh≤2`;
  hold-close jack leak → cap FRESH single presses). Mandatory controls enforced via
  `src/generation/playtest_export.py::enforce_playability` (`hold_aware, no_jump_during_hold,
  no_cross_during_hold, max_jack_run=1`).
- **Conditioning surface:** `src/generation/radar_manifold.py::RadarManifold` (named-axis steering →
  conditional-fill + project; source-free density via `target_density`). Exporter flag `--style`. Persisted
  `cache/radar_manifold.npz`. Manifold knee insight: pick guidance where quarter backbone ~20–25% (song-dep).
- **Patterns/groove:** `src/generation/typed.py` (NUM_PATTERNS=15, `panels_to_pattern`/`pattern_to_panels`),
  `src/data/groove_radar.py`, `src/data/song_selection.py` (`select_by_groove`).
- **Critic:** `experiments/realism_critic/` (+ `eval_taste.py`). **Diagnostics this session:**
  `diag_guidance_sweep.py` (H14), `diag_harmonic_guidance.py` (H16), `diag_radar_manifold.py` (manifold map).
- **Reference exemplar:** regenerate OH WORLD glitch g3.5 as the "good chart" benchmark to compare against.

## Open side-threads (not blocking H15; pick up opportunistically)
- **Deja loin audio sync (H, suspected):** user gets an unusual cluster of 'great' ratings on Deja loin →
  possible timing offset/BPM error; could confound every Deja-loin play-feel this arc. Verify offset/BPM.
- **H17 song↔style fit:** a style only works where the AUDIO affords it (OH WORLD ornamental sang; Deja loin
  1/4-dominant fought glitch). Consider an audio "ornamentality" measure to pick/warn styles per song.
- **Per-axis guidance codification:** chaos knee ~g3.5, freeze tolerates ~g3+, stream wants higher (target-
  pinned density anyway). Cheap to nail once and bake defaults.
- **g4.5>g4 puzzle:** almost certainly sampling (H16 rhythm is monotonic); only chase if a sweep/run-coherence
  metric shows real non-monotonicity across seeds.

## Discipline (this project's expensive lesson)
HARNESS → DATA → MODEL attribution order. Run the Phase-0 gate (and any fair control) BEFORE training. The
offline metrics CANNOT see choreography/vibe — treat clean numbers with suspicion and confirm by ear. Route
all playtest exports through `enforce_playability`. See the `experiment-design` skill + `notes/INDEX.md`.

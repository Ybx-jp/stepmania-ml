# Good-settings region — the by-ear referee: anchoring is an OVERLOAD DETECTOR, not a quality ranker

**Question:** does the offline real-anchored metric (backbone on-grid / 16th-anchoring, "distance from the real
high-chaos envelope") track the USER'S liking? The `taste_grid` playtest (6 cells chaos×guidance × 2 songs — Grand
Chariot, NIGHT IN MOTION — at the milestone spec `voltage=0.7,air=0.5,freeze=0.5`) is the referee. Metrics measured
on the EXACT rated charts (Rule 8), parsed from `~/sm-generated/ch*_g*/`.

## The data (measured on the rated charts)
| song | chaos | g | on_grid | anchor | RATING |
|---|---|---|---|---|---|
| GC | 0.2 | 1.5 | 1.00 | 1.00 | 4 |
| GC | 0.5 | 1.5 | 0.90 | 0.95 | 5 |
| **GC** | **0.9** | **1.5** | **0.39** | **0.52** | **6.5 ← PEAK** |
| GC | 0.2 | 3.0 | 1.00 | 1.00 | 5 |
| GC | 0.5 | 3.0 | 0.52 | 0.65 | 5 |
| **GC** | **0.9** | **3.0** | **0.03** | **0.05** | **2.5 OVERLOAD** |
| NIM | 0.2 | 1.5 | 0.99 | 1.00 | 3 |
| NIM | 0.5 | 1.5 | 0.77 | 1.00 | 4.5 |
| NIM | 0.9 | 1.5 | 0.51 | 0.96 | 5 |
| NIM | 0.2 | 3.0 | 0.98 | 1.00 | 3 |
| NIM | 0.5 | 3.0 | 0.59 | 0.86 | 4.5 |
| **NIM** | **0.9** | **3.0** | **0.16** | **0.30** | **2 OVERLOAD** |

## Verdict — the metric detects the FAILURE cliff, does NOT rank quality
1. **Validated OVERLOAD DETECTOR (clean).** The two cells the user flagged "conditioning overload, dumped into
   1/16s" are EXACTLY the two lowest-anchoring charts: OVERLOAD anchor **0.17** / on_grid 0.10 vs REST anchor
   **0.89** / on_grid 0.76. Unambiguous separation. "Anchoring below ~0.3" = the ear's "it broke down".
2. **NOT a quality ranker.** Spearman(metric, rating) ≈ **0** (on_grid +0.07 p=.83; anchor +0.05 p=.87). The tell:
   the PEAK (GC 0.9/1.5, rated **6.5**) has anchor **0.52** — LOWER than mediocre GC 0.2/1.5 (anchor 1.00, rated 4).
3. **The relationship is an INVERTED-U:** high anchoring (safe, low-chaos) = mild/mediocre; MEDIUM anchoring (~0.5,
   chaos cranked but coherent) = the peak; COLLAPSED (<0.3) = overload. So the metric's job is **locating the
   overload BOUNDARY**, not scoring quality. Confirms "necessary-but-not-sufficient" with the user's own ratings:
   above the cliff the chart is good; HOW good is then EXPRESSIVENESS, which the metric (correctly) doesn't measure.

**→ Tolerance = the crank where anchoring crashes below ~0.3 (the overload cliff) = the good-region edge.** Within
it, crank chaos for expressiveness. This is the honest, ear-refereed definition (a failure boundary, not a score).

## Actionable good-settings rule: crank CHAOS, keep GUIDANCE gentle
The peak is chaos=0.9 / g=**1.5**; the overload is chaos=0.9 / g=**3.0** — SAME chaos, higher guidance. **Guidance is
the OVERLOAD lever, not the expressiveness lever** — exactly H14's offline finding ("guidance is the wrong lever for
chaos/vibe; it floods off-beats"), now ear-confirmed. For more vibe: raise chaos, keep g≈1.5. (Also re-confirms the
song-dependence: chaos=0.9,g=3.0 was "fantastic" on the milestone songs but OVERLOADS Grand Chariot & NIM — tolerance
is per-song, NOT a regime.)

## Why the overloaded charts break down (user's question) — attributed, not decode-constraints
- **Proximate (measured, `probe_backbone_phase.py`):** CFG-amplified chaos conditioning (~70%) + the 16th-unlock
  calib (~30%), **NOT the governor** (0%, FULL≡GOV_OFF). At g=3.0 CFG triples the chaos delta and the onset mass
  phase-shifts entirely onto the 16th grid (the `_x.x` smear). NOT a decode constraint; NOT the song (same song is
  a 6.5 at g=1.5).
- **Root (H4, `h4_offbeat_signal_findings.md`):** the model has NO local off-beat audio signal (off-beat onset AUC
  ~0.53–0.66). So chaos enters as a GLOBAL scalar; amplified hard it can ONLY smear uniformly — it can't place
  ANCHORED song-specific off-beats the way real charts do. That is the "OOD miscomprehension" the user sensed.
- **The ceiling-raising fix = a CONDITIONING-MECHANISM change (next thread):** a **chaos×onset gate** tying off-beat
  placement to LOCAL audio (perc/onset transients) instead of a global additive bias → high chaos would ADD anchored
  off-beats (real-like, expressive) rather than smear, plausibly making the loved corner land on EVERY song. Retrain/
  architecture, not a decode knob (two decode-only retrains already failed — H4 §6). User: "it can soar higher if
  harnessed completely" = precisely this.

## Feature lead (from the n=40 sweep, `probe_backbone_tolerance.py`) — held pending
`real_density` predicts tolerance across all 3 real-anchored metrics (ρ≈**−0.37, p≈0.02**): DENSER songs collapse to
overload SOONER (low-density mean anchoring-tol 0.59 vs high-density 0.39). Mechanistically sensible (denser = more
mass for the global shift to push off-grid). **CAVEATS:** p uncorrected across ~30 tests (marginal); density +
onset-busyness features are COLLINEAR (one axis, not many); n=40. A LEAD toward the "formula", not the formula.

## Connections
`backbone_phase_findings.md` (the attribution), `real_phase_reference_findings.md` (the Rule-5 real envelope),
`h4_offbeat_signal_findings.md` + `h14_guidance_sweep_findings.md` (parent phenomenon + the guidance-floods lever),
lineage `experiment_lineage/good-settings-region-arc.md`, memory [[good-settings-region]]. Tooling:
`probe_backbone_tolerance.py`; referee measurement inline (parse `ch*_g*/` + `on_grid_share`/`sixteenth_anchoring`).

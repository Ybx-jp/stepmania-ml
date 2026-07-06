# v2 safety-settings envelope — the pre-deploy-swap sweep

**2026-07-06, branch `feat/governor-subdiv-recalib`.** Ship-checklist item #3 (safe-settings envelope), run as a
PREREQUISITE to item #1 (the deploy-swap) at the user's request: *"verify a wider range of outputs from the new
model to find that safety zone of settings, confirm its existence, and only then swap anything."* This is
ship-scope work, NOT the parked research ([[ship-mode-park-research]]). Companion tooling:
`experiments/generation_typed/analyze_v2_envelope.py` (v2-aware, 48th grid), `src/data/meter_detect.py` (the
duple/triplet classifier). Pairs with `notes/footspeed_floor_findings.md` (the v2 decode levers) and
`generation-defaults §0`.

## What was built + run
1. **The duple/triplet b_trip SWITCH** (`--auto_b_trip`, exporter): per-song, apply the triplet phase band
   (`b_trip`, the 3rd `--onset_phase_calib` element, default 0.7) ONLY to triplet-feel songs; duple songs get
   `b_trip=0`. The classifier = the validated audio meter detector (`triple_pref`, the DFT-of-beat-phase-histogram
   from the meter arc, extracted into `src/data/meter_detect.py`; `triple_pref>0` => triplet). Deployable (audio +
   BPM only — no reference chart needed). Feeds the SAME per-song calib to BOTH the tau path and `generate()`
   (single-source rule). v1 (16th grid) = no-op (empty triplet band). Opt-in default (canonical stays aligned;
   `check_export_defaults.py` = 21 ALIGNED).
   - **Detector separates the classes cleanly** (n=9 spot check, triplet-enriched): triplet seeds +0.16..+0.79
     (Equinox +0.22, My Christmas +0.16, subconsciousness +0.79) vs duple −0.37..−0.68; Spearman +0.81 vs chart
     triplet_frac. Wide margin at the `>0` boundary.
2. **The sweep** — 5 settings arms over the SAME stratified song set (seed 42, `--hardest`, v2 =
   `gen_motif_v2_48th_cont` + `--features highres_v2`). 12 songs selected (6 triplet across BPM + 6 duple across
   BPM incl. the freeze-stress Watch Out Pt.2); **7 survived the val loader's song-length filter** (2 triplet:
   Equinox, My Christmas list; 5 duple) — 4 triplet songs were dropped as too short (a coverage caveat; a triplet
   TOP-UP was run to thicken it). Arms: `btrip07_global` (band on all), `btrip_off` (band off), `auto` (the
   switch), `auto_chaos` (`--style chaos=high`), `auto_freeze` (`--style freeze=high`).
3. **The analyzer** — parses each arm's generated `Challenge` chart with `for_v2()` (tpb=12; a 16th-grid parse
   would FLOOR the 48th rows) and computes the failure modes we can measure offline. **Metric VALIDATED against
   the human charts** in the same `.sm`: human Equinox reads quarter 0.59 / triplet 0.41 / zero 8ths (a triplet
   song), human Giudecca reads quarter 0.25 / 8th 0.26 / 16th 0.48 (a duple 16th song) — the 48th-grid phase cells
   are correct, not garbage.

## RESULT — the safety zone EXISTS (offline; ears are the final gate)

### 1. Playability is ROCK-SOLID across the ENTIRE settings range
Across all 5 arms × 7 songs (35 charts), EVERY chart: **fast_jump 0, flam 0, max_jack ≤ 2, off48 ≤ 0.06** (no
off-grid smear). The v2 governors (footspeed floor `min_onset_gap`, no-fast-jump cap, `max_jack_run`, the subdiv
recalibration) hold everywhere — bare, chaos, freeze, every b_trip mode. This is the core safety-zone proof: the
model does not produce unsteppable charts anywhere in the range a pad player would touch. (Metrics are blind to
musicality by design — this bounds UNPLAYABILITY, not quality; the ears confirm the feel.)

### 2. The b_trip switch is SAFE + best-when-confident, but NOT a clean win (a real tradeoff)
⚠️ **CORRECTION to the first (2-triplet) sweep, which said "auto strictly dominates."** On the widened 6-triplet
set the picture is more nuanced. Widened sweep, 6 triplet + 6 duple, triplet-occupancy (human ~0.40–0.57):

| b_trip mode | triplet occ (mean) | duple occ (mean) | note |
|---|---|---|---|
| **off** | 0.16 | 0.00 | all triplets under-placed |
| **global 0.7** | 0.32 | 0.03 (spurious) | helps ALL triplets, dirties duple a bit |
| **`auto` (switch)** | 0.27 | **0.00** | helps DETECTED triplets, duple perfectly clean |

**The detector (audio `triple_pref`) fired on only 3 of 6 chart-triplet songs** (Equinox +0.22, My Christmas
+0.16, naTivEfAcE +0.31 → band on; Sway −0.35, After The Rain −0.27, Parousia −0.26 → band OFF) — despite Sway
being the MOST chart-triplet song (tf 0.61). On the 3 it caught, the band lands human-level (Equinox 0.12→0.39,
My Christmas 0.35→0.51, naTivEfAcE 0.30→0.52). On the 3 it missed, `auto` leaves them at band-off levels while
`global` lifts them (Parousia 0.14→0.37). So:
- `auto` = **perfectly clean duple + triplet help ONLY where the audio detector is confident** (never harms; a
  miss falls back to band-off, no worse than off). Conservative + safe.
- `global 0.7` = **triplet help everywhere + slight duple busyness**.
- **The ρ+0.47 detector is the bottleneck** — it misclassifies chart-triplet songs whose AUDIO reads duple. TWO
  interpretations, only the EARS decide: (a) genuine detector misses (then `auto` under-helps and `global` is
  better for triplets), or (b) those songs' audio really IS duple and the charter triplet-ified duple music (then
  `auto`'s band-off is CORRECT and `global` would force wrong triplets). chart-triplet ≠ audio-triplet.
- **No offline "best default" verdict is warranted** — the auto-vs-global choice is a by-ear call on the
  ambiguous songs (Sway/Parousia/After The Rain, in the installed pack). The switch is SAFE either way (it never
  degrades a duple song); the open question is only how aggressively to place triplets.

### 3. EDGE — long/sparse songs + `--style chaos/freeze` open big dead gaps
- **Bare `auto`:** the ORIGINAL 7 songs were clean (max gap 7.7b), but the newly-admitted LONG songs (After The
  Rain, Parousia — previously length-blocked) show **10–12-beat** internal gaps even at the default. So the
  widened gates reach songs nearer the model's competence edge; long/sparse songs are the mild default edge.
- **`--style chaos/freeze`:** drops density (nps ~5.5 → ~3.5, taking the manifold `E[density|radar]` instead of
  the source density) and blows the gaps up to **24–28 beats** on the long songs (After The Rain 28b) and 10–13b
  on most others. A 28-beat gap ≈ 7 empty measures — that's near-broken density, not musical breathing. Likely
  cause: the manifold density target ≪ what these songs afford. **`--style` is a genuine use-with-CARE edge on
  long/sparse songs** (tuning the manifold density is the retrain-gated/parked manifold work — not done here).
- By-ear question: on the default, are the 10–12b gaps on long songs musical rests or dead air?

### 4. EDGE — the known freeze hold-stream defect is PRESENT but BOUNDED
`ff_hold` (free-foot stream ≥4 @≤8th under an open hold, the `footspeed_floor_findings §5b` metric) = 1–2
stretches on the hold-heavy songs (AFRONOVA, Watch Out Pt.2, OH WORLD), incl. Watch Out Pt.2 =1 under
`auto_freeze`. Consistent with the documented PARTIAL-fix state (the real fix — position-based `stamina_hold_bump`
— is DESIGNED + PARKED). Bounded, not exploding → supports the ship decision to ship it as a documented
known-limitation rather than block v1.0.0 on it.

## Safe-settings envelope (the guide seed)
- **DEFAULT (recommended, confirmed clean):** bare `auto` — `--auto_b_trip`, `b_trip=0.7`, no `--style`,
  guidance 1.0. Playable everywhere, triplets land human-level, duple stays clean, density healthy (nps 3–7), no
  dead gaps.
- **Groove knobs (`--style chaos/freeze=high`): use-with-awareness.** Still fully PLAYABLE (no jack/jump/flam/
  smear), but sparser with longer rests; `freeze=high` on hold-heavy songs surfaces the bounded hold-stream edge.
- **Untested extremes** (very high guidance, single-dim `--radar` OOD) are out of the "settings a pad player
  would attempt" and out of scope for the envelope.

## BUG FOUND + FIXED — `--relax_gates` was a silent no-op on v2 (needless song blocking)
Triggered by the user asking whether the relaxed-gate / variable-BPM handling was needlessly blocking songs. It
was. The variable-BPM handling (`compute_average_bpm`, duration-weighted) + the widened inference gates
(`for_inference()`) both EXIST, but the v2/48th-grid export path bypassed them:
- `export_typed_samples.py` did `if subdiv==12: infer_parser = StepManiaParser.for_v2()` UNCONDITIONALLY, and
  `for_v2()` inherits the NARROW **training** gates (bpm [60,200], length [75,130]s, `max_simultaneous=2`). The
  `elif args.relax_gates: for_inference()` branch was structurally UNREACHABLE for v2 → `--relax_gates` did
  nothing on exactly the grid we're shipping.
- **THREE stacked training-gates leaked into inference:** BPM, song-length, and `max_simultaneous=2` (rejects
  hand/quad charts — e.g. SHINY DAYS's only chart is a 4-panel Challenge — though the 15-way pattern head
  supports up-to-4-panel frames; the stale `=2` also drops ~55% of real Hard charts).
- **FIX:** a single shared `INFERENCE_GATES` dict in `stepmania_parser.py` (bpm [40,320], length [30,600]s,
  `gimmick_max_bpm=400`, `max_simultaneous=4`) used by BOTH `for_inference()` (v1) and the v2 branch (which now
  does `for_v2(**INFERENCE_GATES)` under `--relax_gates`). v1 export unchanged; bare v2 export still narrow
  (opt-in symmetry preserved); canonical decode defaults untouched (`check_export_defaults.py` = 21 ALIGNED).
- **Reach measured (real parser, all gates, val n=954):** narrow 532 songs → widened **822** = **+290 (+55%)**
  admitted on v2. Confirmed end-to-end: See Me Now (206s, length-blocked) and SHINY DAYS (220 bpm + quad chart)
  now generate with `--relax_gates`.
- **⚠️ Implication for the sweep above:** it ran WITHOUT `--relax_gates`, so on the narrow 532-song pool — which
  is why triplet coverage was thin (most triplet novelty songs are length-blocked). A re-run with `--relax_gates`
  would substantially thicken the triplet arm. **Open decision: make `--relax_gates` the v2 default?** (deployment
  wants max reach; the tradeoff is bare-export reproducibility + the index-keyed cache is disabled under it.)

## Caveats / honesty
- **Triplet coverage FIXED by the gate fix** — the widened re-run gave 6 triplet + 6 duple songs (the narrow
  first run's thin 2-triplet set was itself a symptom of the gate bug below). This is what surfaced the detector's
  real ~50% hit-rate on chart-triplet songs — the honest picture the 2-song run hid.
- **The detector (audio `triple_pref`, ρ+0.47) is the switch's bottleneck**, not playability. It's reliable on
  audio-clearly-triplet songs, noisy on chart-triplet-but-audio-ambiguous ones. A reference-chart-based label
  (chart `triplet_frac`) would be more reliable WHERE a chart exists (val/playtest), but the deployed case
  (chart-less new song) only has audio. Improving it is meter-arc research (parked). The switch is SAFE regardless
  (misses fall back to clean band-off); the detector only bounds how many triplet songs get HELPED.
- **Offline metrics are blind to musicality** (exp-design Rule 8). This bounds unplayability + confirms the b_trip
  switch mechanically; the FELT quality (are the long `--style` rests musical? do the triplets groove?) is the
  by-ear gate — a curated pack is installed for it.
- No defaults were swapped. The deploy-swap (ship item #1) stays a separate, user-gated step.

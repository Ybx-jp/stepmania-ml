---
name: playtest
description: >
  Generate playable StepMania sample sets with interesting control-knob toggles (groove
  radar, reference style, decode constraints) for the user to play, and log their hands-on
  playtest feedback to notes/playtest_log.md with commentary, hypotheses, and connections
  to past findings. Use when the user asks for playtest/sample sets, wants to try control
  knobs (chaos/air/stream/voltage/freeze radar, reference-chart style, jumps, crossovers,
  no-jump-during-hold), records how generated charts felt to play, or asks to log/analyze
  playtest feedback. Pairs export_typed_samples.py with a persistent, analyzed playtest log.
---

# playtest

Close the loop between the generator and the user's hands. Two jobs:

1. **Export sample sets** the user can drop into StepMania, varying control knobs so each set
   probes one hypothesis (does this knob *feel* like its name? is this constraint playable?).
2. **Log the feedback as evidence** — never just transcribe. Every entry connects raw feel to a
   hypothesis and to the project's standing threads, so the log compounds into understanding.

## Exporting sets

Driver: `experiments/generation_typed/export_typed_samples.py` (default checkpoint
`checkpoints/gen_style/best_val.pt`, which supports difficulty + radar + reference-style + CFG).
Run from the repo root in the `stepmania-chart-gen` conda env. Writes one playable folder per song
(original audio + `chart.sm` with generated `Challenge` + original chart for A/B) to `--out_dir`.

Key knobs (all decode-time unless noted):

| knob | flag | what it probes |
|---|---|---|
| groove radar | `--radar "chaos=0.9,air=0.85"` (+`--guidance`) | does each radar dim feel like its name? |
| reference style | `--reference PATH --reference_difficulty NAME` (+`--guidance`) | "make it feel like this chart" |
| CFG strength | `--guidance` (1=off, 1.3–1.5 musical, 2–3 strong/forced) | control-vs-musicality tension (H3) |
| arrow coherence | `--pattern_temperature` (0.6–0.8 coherent, 1.0 varied, greedy collapses) | the coherence/variety sweet spot (H2) |
| pad-playable holds | `--no_jump_during_hold` | no jump while a foot is pinned holding |
| no crossovers | `--no_crossovers` | foot-swap legality |
| jumps / panels | `--jump_bias`, `--prefer "U,R"` | pattern preference steering |

Radar dims (0–1, base = dataset mean): `stream, voltage, air, freeze, chaos`
(stream/voltage→density, air→jumps, freeze→holds, chaos→off-grid/varied rhythm).

**Conventions that make sets comparable and fun:**
- **Same `--seed` (default 42) → same songs** across sets, so the user can A/B *the same song* under
  different knobs. Keep `--num_songs` modest (4–8).
- Vary **one idea per set**; name `--out_dir` after it (e.g. `outputs/radar_samples/chaos_air`).
- Default to **playable** settings the user has endorsed: `--pattern_temperature 0.7`,
  `--no_jump_during_hold`. Reserve extreme `--guidance 2+` for a deliberate "see the knob's reach" set,
  and include a gentler companion (g≈1.4) when showing a strong knob.
- Always keep a plain `base`/`base_coherent` set around as the musical reference point.
- After generating, sanity-check a folder re-parses (`StepManiaParser().parse_file`) before handing off.

## Logging feedback

Append to `notes/playtest_log.md` (newest on top). **Each entry must do four things, not one:**

1. **What was played** — which sets/songs, with the knob settings.
2. **Raw feedback** — the user's words, verbatim where vivid ("the Will Smith meme").
3. **Commentary / hypothesis** — *why* might it feel that way? Tie it to mechanism (which head, which
   feature, which decode step). Give the hypothesis a handle (H1, H2, …) and reuse handles across
   sessions so they accumulate evidence (confirm / complicate / refute).
4. **Action / next** — what to change, generate, or test next; leave a checkbox list.

Then look for the **connecting thread** — what does this feedback say together with prior entries?
(Recurring example in this project: the biggest wins keep being *decode-time* fixes, implying the base
model is under-served by its default decode.) Surface those connections explicitly; that's the point of
the log.

When feedback implies a new control knob (like jump-during-hold did), note it as a requested feature
with the mechanism sketch, and — if it's a decode-time constraint — it likely belongs in the
`hold_aware`/pattern-masking section of `generate()`, exposed as an `export_typed_samples.py` flag,
with a test in `tests/test_generation.py`.

## Standing hypotheses (keep current)

- **H1** *(reinforced 06-19 by chaos playtest)* Timing is solved (onset ROC-AUC ~0.9); choreography
  (arrow↔musical-event mapping) is the open axis, bottlenecked by timbre/energy-only features (no
  chroma/pitch). Fix lever: chroma + HPSS + retrain.
- **H2** Pattern-temperature has a coherence/variety sweet spot (~0.7); 1.0 over-randomizes, greedy collapses.
- **H3** Strong CFG guidance trades musicality for control; gentle (≈1.4) keeps the steer and stays musical.
- **H4** *(FULLY confirmed 06-19: stream_voltage & air_only playable, chaos_only/chaos_air/chaos_gentle
  unplayable; phase histogram)* Quantity knobs (density/holds/jumps, all on-grid) steer fine; musicality
  knobs (chaos/syncopation) break. Mechanism: the model renders chaos as a *degenerate global grid
  manipulation* (uniform 8th-offbeat "all blues" at g=1.3, or uniform smear at g=2.0 — only 6% on-beat),
  NOT event-driven syncopation, because it can't see which offbeats deserve a note. Base also
  under-syncopates vs real (0.91 vs 0.80 on-beat).
- **H5** *(new 06-19, measured)* No song-structure/phrase awareness: generated density is structurally
  flat and fades at the end while real charts have an arc (intro→build→climax@~80-90%→outro). The model
  choreographs frame-locally with no global plan. Root: frame-local features + shallow Conv1D receptive
  field. Fix levers (distinct from H1): audio novelty/self-similarity feature, downbeat phase, or a
  wider-context/attention audio encoder. ("Awkward start" is separate — choreographic/sync, not density.)

Defect hierarchy (each layer above "timing" traces to a musically-shallow audio representation; the
metrics onset_F1/crit_adj only score the bottom layer, which is why charts score well yet feel off):
timing (solved) → local choreography (H1) → global structure (H5) → decode polish (jacks, jump-in-hold).

Update these as playtests confirm or break them. The log is the project's qualitative-evidence ledger,
complementing the quantitative metrics (onset_F1, crit_adj) — which by design never capture play-feel.

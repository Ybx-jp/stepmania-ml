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

## ⛔ MANDATORY pad-playability constraints — NON-NEGOTIABLE, code-enforced
EVERY chart the user plays MUST be generated with these decode constraints ON. Each traces to a specific
"unplayable on pad" playtest finding. Shipping a chart without them silently CONFOUNDS the user's subjective
evaluation (the project's primary signal) and ruins the experiment — this has happened; do not repeat it.

| constraint | value | why (finding) |
|---|---|---|
| `hold_aware` | `True` | coherent holds / no orphans (the automaton the others ride on) |
| `no_jump_during_hold` | `True` | pad has ONE free foot while holding → a jump-during-hold is unhittable ("Will Smith meme") |
| `no_cross_during_hold` | `True` | the free foot can't fast-cross/jack while a hold pins the other foot (the B4U finding) |
| `max_jack_run` | `2` (positive cap) | HARD 16th-jack cap. =2 (user-approved 2026-06-25) allows a justified 2-note 16th jack, hard-forbids 3+ at 16th (H13, "6-note 1/16 jack"). |
| `jack_penalty` (+`bpm`) | `~1.5` (soft) | SOFT foot-exertion governor: escalating BPM-aware penalty on extending a same-panel run — gates unnatural jack STREAMS (8th + long), keeps short justified ones, preserves density (re-routes to alternation). `--jack_penalty` in the exporter (default 1.5); needs song BPM. notes/foot_exertion_findings.md |
| `pattern_temperature` | ~0.6–0.85 (def 0.7) | arrow-coherence sweet spot — greedy collapses, >1.0 over-randomizes (H2) |

**RULES:**
1. The mandatory set is **code-enforced**: every export path runs its `generate()` kwargs through
   `src/generation/playtest_export.py::enforce_playability()`, which FORCES the constraints on and RAISES if
   they're explicitly disabled. `export_typed_samples.py` does this by default. **Any NEW export/generation
   script you write for a playtest MUST call `enforce_playability(gen_kwargs)` before `generate()`.**
2. **ANY deviation from this list requires EXPLICIT user approval**, passed as
   `--override_playability "<reason>"` (exporter) / `override_reason=...` (`enforce_playability`). Never
   deviate silently. When in doubt, ASK the user before exporting.
3. If you add a new pad-playability constraint from a playtest finding, add it to `MANDATORY_PLAYABILITY`
   here AND in `playtest_export.py` — the skill and the code stay in sync.

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
- **★ GROOVE-VALIDATE every set (06-21 directive).** A set only tests a hypothesis if its songs actually
  exercise the relevant axis — B4U with 3 holds CANNOT test a hold fix. Use `--groove_select <axis>`
  (`freeze`=holds, `stream`/`voltage`=density, `air`=jumps, `chaos`=syncopation, `rich`=strong across all)
  `[+ --difficulty_select Hard]` so the exporter picks songs that read strongly on that axis AND prints
  their radar profile — confirm the profile before handing off. Pick the axis the knob/fix touches.
  Prefer harder songs (more revealing of decoder/musicality subtleties). Backed by
  `src/data/song_selection.py`. (deja loin is a good general test: strong stream/voltage/freeze/air.)
- **Match the expected profile with `--match_radar` (+ `--guidance ~1.5`).** By default the output groove
  profile drifts from the original (generation isn't radar-conditioned). When you select/expect a specific
  feel, `--match_radar` conditions each song on its OWN source-chart radar so the output tracks it (e.g.
  high-freeze songs actually get more holds). g≈1.5 matches; g≈2+ amplifies (and dents density/difficulty
  since holds are sparse in note-presence). Use it so a groove-validated set's OUTPUT also reads on-axis.
- **Same `--seed` (default 42) → same songs** across sets, so the user can A/B *the same song* under
  different knobs. Keep `--num_songs` modest (4–8).
- Vary **one idea per set**; name `--out_dir` after it (e.g. `outputs/radar_samples/chaos_air`).
- Default to **playable** settings the user has endorsed: `--pattern_temperature 0.7`,
  `--no_jump_during_hold`. Reserve extreme `--guidance 2+` for a deliberate "see the knob's reach" set,
  and include a gentler companion (g≈1.4) when showing a strong knob.
- Always keep a plain `base`/`base_coherent` set around as the musical reference point.
- After generating, sanity-check a folder re-parses (`StepManiaParser().parse_file`) before handing off.

## Logging feedback

**Scope of `notes/playtest_log.md`:** it is the **subjective** ledger — the user's hands-on play-feel
connected to hypotheses. It is NOT where offline/quantitative experiment results go. Those (training
runs, metric head-to-heads, ablations, eval tables) belong in their own `notes/<experiment>_findings.md`
like every other experiment in this repo (e.g. `stage1_musical_features_findings.md`,
`focal_onset_findings.md`). When a playtest entry needs to reference such numbers, *link* to the
findings note rather than pasting the tables in. Keep the log about what the charts *felt* like.

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

- **H1** *(SUPPORTED 06-20 by playtest)* Choreography (arrow↔musical-event) was the open axis,
  bottlenecked by timbre-only features. Stage 1 added chroma (used heavily, ablation KL 10.3) and the
  **plain Stage-1 model played "definitely more musical… mostly right"** — a play-feel win, even though
  every offline metric (onset_F1, phase, structure) stayed flat. The features helped; the metrics just
  can't see it. (Chaos still smears — that's an H4/chaos-knob issue, not the features.)
- **H2** Pattern-temperature has a coherence/variety sweet spot (~0.7); 1.0 over-randomizes, greedy collapses.
- **H3** Strong CFG guidance trades musicality for control; gentle (≈1.4) keeps the steer and stays musical.
- **H4** *(FULLY confirmed 06-19: stream_voltage & air_only playable, chaos_only/chaos_air/chaos_gentle
  unplayable; phase histogram)* Quantity knobs (density/holds/jumps, all on-grid) steer fine; musicality
  knobs (chaos/syncopation) break. Mechanism: the model renders chaos as a *degenerate global grid
  manipulation* (uniform 8th-offbeat "all blues" at g=1.3, or uniform smear at g=2.0 — only 6% on-beat),
  NOT event-driven syncopation, because it can't see which offbeats deserve a note. Base also
  under-syncopates vs real (0.91 vs 0.80 on-beat). **INVESTIGATED + RESOLVED AS NOT-A-FEATURE-PROBLEM
  (06-20/21, `h4_offbeat_signal_findings.md`):** the shipped onset_env is coarse-hop (~93ms) and nearly
  phase-flat (off-beat note-vs-no-note AUC ~0.53≈chance); a high-res onset (hop~128, max-pooled per 16th)
  recovers *some* signal (off-beat AUC →0.66). BUT two retrains adding it (zero-init warm-start, then
  random-init + off-beat-weighted loss) **both FAILED to fix chaos** — even when the feature column was
  retained at full magnitude (v2 norm 1.04), zeroing it shifts onset logits only ~0.017 and chaos still
  smears (~5% on-beat @ CFG2 vs real ~80-90%). The recovered signal is weak and largely redundant with the
  coarse onset; most off-beat placement is groove/pattern (charter style), not audio-onset-determined; and
  chaos enters as a GLOBAL conditioning scalar a local feature can't redirect. **Conclusion: chaos is an
  OBJECTIVE + CONDITIONING-MECHANISM problem, not a feature problem** (Stage1 chroma also didn't fix it).
  Levers: a per-frame chaos×onset *gate* (tie off-beat placement to local feature) and/or critic-guided
  objective (2c). gen_highres/v2 parked. (OLD claim "fix = add high-res feature" — FALSIFIED. was: 2c move
  Hard). The model's ~0.9 onset AUC rests on the metrical prior + density, not audio event detection.**
- **H5** *(new 06-19, measured)* No song-structure/phrase awareness: generated density is structurally
  flat and fades at the end while real charts have an arc (intro→build→climax@~80-90%→outro). The model
  choreographs frame-locally with no global plan. Root: frame-local features + shallow Conv1D receptive
  field. Fix levers (distinct from H1): audio novelty/self-similarity feature, downbeat phase, or a
  wider-context/attention audio encoder. ("Awkward start" is separate — choreographic/sync, not density.)
- **H6** *(revised 06-20)* For *plain* generation the features WERE sufficient (the playtest win, H1) —
  the offline metrics simply couldn't detect it. The "necessary-not-sufficient" failure is specific to
  the **chaos knob**: a decode gate (`onset_phase_penalty`) does NOT rescue it because chaos *moves*
  notes off-beat (suppressing on-beat) rather than layering off-beats on a backbone, so gating →
  near-empty. Fixing chaos needs the conditioning mechanism/objective, not decode or more features.
- **H7** *(new 06-20)* Metric-phase feature backfires for syncopation — it gave the model a clean
  downbeat signal which it used to sit MORE on-beat (0.93→0.952). Drop or rethink in Stage 2.
- **H8** *(new 06-20)* HPSS onsets are near-redundant with the existing onset_env (ablation KL 0.29);
  not worth their ~4.4s/sample extraction cost as-is.

Defect hierarchy (each layer above "timing" traces to a musically-shallow audio representation; the
metrics onset_F1/crit_adj only score the bottom layer, which is why charts score well yet feel off):
timing (solved) → local choreography (H1) → global structure (H5) → decode polish (jacks, jump-in-hold).

Update these as playtests confirm or break them. The log is the project's qualitative-evidence ledger,
complementing the quantitative metrics (onset_F1, crit_adj) — which by design never capture play-feel.

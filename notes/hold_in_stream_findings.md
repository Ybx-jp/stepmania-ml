# Hold-in-stream defect → the `hold_stream_penalty` fix + the footswitch finding — findings

*2026-07-02. Thread: "the pattern/type head is the fast-song quality locus — WHERE inside it?" (spun off the
BPM→quality arc, `quality_feature_attribution_findings.md`). User by-ear observation seeded it: "the model does a
HOLD with a JACK sequence where a human would chart a regular STREAM." Ended with TWO shipped decode defaults.*

## TL;DR
- **Refuted (cheaply):** the fast-song defect is NOT in the hold-SPAN machinery. `probe_bpm_hold_decomp.py` (paired,
  n=90): hold-burst, tail-run-long, and hold-open-rate all sit at the BPM noise floor. The initial tail-run-long
  "signal" (r=+0.49) was a **pooled-reference artifact** — real hold length ALSO rises with BPM; the paired
  (gen − own-real) excess is −0.07 (p=0.67). The holdrate lead (+0.31 @ n=40, p=.026) **regressed to +0.09 @ n=90**
  (a small-n boundary artifact). Method keeper: use the SONG'S OWN real chart as the baseline for a "defect vs X"
  slope, not a pooled constant; confirm a marginal lead at higher n.
- **Confirmed (the user's observation):** `probe_stream_holdjack.py` (n=40) — the type head OPENS HOLDS in dense
  stream sections a human keeps hold-free (**gen 18% of stream frames vs real ~0%**), and an open hold TRIGGERS the
  free-foot jack (jack rate 23%→32% when a hold is open, p=0.008). The jacking is NOT positionally elevated in
  streams (in 0.259 = out 0.259) — it's **holds-for-streams, and holds make jacks** (the `no_cross_during_hold` +
  fatigue chain converts the pinned-foot stream into a jack). NOT the same as `hold_burst` (that counts CROSSES,
  dist≥1.4; a jack is dist-0 → invisible to it).
- **Fixed (no retrain):** `hold_stream_penalty` (typed_model.generate) — suppress hold-heads gated on local onset
  density `relu(density − floor)`, so SPARSE musical holds are untouched by construction. A/B `probe_holdstream_fix.py`
  (n=27 paired): cuts holds in dense frames (−0.086, p<0.001) while low-density holds are preserved (+0.001, n.s.)
  and hold_burst does NOT rise. **Playtest-tuned to floor 0.45 / penalty 8** ("just right" on japa1).
- **BONUS finding (the footswitch diagnostic):** the "brutal 16th-jack voltage" (a pre-existing complaint) is
  **dominantly a FOOTSWITCH strategy, not intrinsic jacks.** New `footswitch` on/off decode knob: forbidding
  footswitch collapses same-panel runs by **81–85%** on HSL/japa1 (all long runs vanish, max→2). Playtest: OFF is
  **"sooooo much better"** — forbidding footswitch forced the model to ALTERNATE (more creative, not less).
- **SHIPPED as canonical defaults (2026-07-02):** `hold_stream_penalty=8, hold_stream_floor=0.45` AND `footswitch=False`.

## The probe chain (each sharper than the last — a worked experiment-design example)
1. **`probe_bpm_hold_decomp.py`** — does the fast-song defect live in the hold-SPAN metrics (burst / tail-length /
   rate) vs BPM, more than the general which-panel (trans_KL)? Denoised K gens, pooled counts. **NULL at n=90**
   (the tail-length +0.49 was pooled-vs-paired confound; holdrate +0.31→+0.09 regression). The choreography-metric
   lens does not localize the BPM defect at all (nothing slopes at n=90) — only the graded critic ever saw it.
2. **`probe_stream_holdjack.py`** — POSITIONAL/co-occurrence, aligned to REAL stream windows (≥6 alternating
   single-tap onsets, gap≤2, no holds). Primary = HOLD-CENTRIC: when a hold pins one foot, does the FREE foot JACK
   (gen 0.46) or alternate (real 0.33)? Secondary = does gen open holds/jack in real-stream frames? **Confirmed the
   root (holds-in-streams 18% vs 0%) + the causal chain (hold → +11pp jack, p=0.008).**
3. **`probe_holdstream_fix.py`** — A/B sweep of `hold_stream_penalty` (paired, guards: `hold_in_lowdens` must
   survive, `hold_burst` must not rise). penalty 3 = the metric sweet spot; the floor `relu(density−floor)` is what
   protects sparse holds (raw density over-cut → HSL 39→1 in v1 playtest).

## The mechanism (code-confirmed)
- **Root:** the TYPE head opens a hold-head (symbol 2) in a dense section a human streams. A GLOBAL hold-rate
  averages this away (why probe 1's rate metric was null) — it's POSITIONAL.
- **Chain:** once the hold pins a foot, `no_cross_during_hold` (typed_model.py) HARD-FORBIDS different-panel singles
  at a 16th gap and ALLOWS the jack; the fatigue governor (foot_fatigue_design.md:170) makes a one-foot WIDE stream
  cost `travel_weight·dist > jack_weight` → prefers the jack. So the hold is the trigger; the jack is downstream.
- **Fix targets the root (the hold), not the jack** — remove the mis-placed hold and the section stays a two-foot
  stream. `hold_stream_penalty` subtracts `penalty · relu(local_density − floor)` from the hold-head logit
  (decoupled from onset/tau → changes tap-vs-hold ONLY, never WHERE/HOW-MANY notes).

## The floor tuning (playtest arc)
- v1 (floor 0.25, penalty 3): "directionally right but TOO BLUNT" — density over-included, cut expressive holds
  (HSL 39→1 vs real 12). Density is a PROXY for "would-be stream a hold converts to a jack"; it over-fires.
- Grounding (density-at-holds distribution, from the installed baseline charts): expressive holds sit ≤0.5;
  japa1's pathological grind reaches 0.69 (only song with holds >0.6). → floor 0.5 spares OH WORLD (max 0.44) and
  ~97% of HSL, catches japa1's tail.
- v2 (floor 0.5, penalty 8, SHARED-RNG A/B): OH WORLD & HSL byte-identical to baseline (fix a no-op below floor),
  japa1 96.75% identical (only the grind edited). Playtest: "improvement, floor slightly too high" → **0.45**.
- **Shared-RNG A/B** (common random numbers): the exporter restores the RNG state before the Edit arm, so the two
  arms are byte-identical until the knob first bites → any felt difference is attributable to the KNOB, not sampling
  noise (a real confound that showed up as OH WORLD Edit 13→25 holds when arms were independent draws).

## The footswitch finding (`footswitch` knob; the diagnostic that redirected the voltage fix)
- User asked for a footswitch on/off knob to disambiguate a same-panel run being a FOOTSWITCH (alternating feet,
  playable) vs EXCESSIVE VOLTAGE (a one-foot jack) — the chart notation is footing-ambiguous.
- Knob: `footswitch=False` sets the governor's footswitch cost `fs_add=∞` → same-panel runs must be one-foot jacks
  (costed by per-foot fatigue + `max_jack_run`), never a footswitch.
- **Diagnostic result (shared-RNG A/B):** forbidding footswitch collapses same-panel runs by **81–85%** on HSL/japa1
  (n≥3 runs → 0, max→2); OH WORLD keeps more (~39% persist, max 4 = it has the most INTRINSIC jacks). So the
  "brutal voltage" is dominantly a footswitch STRATEGY, not intrinsic jacks → the taste lever is the FOOTSWITCH
  POLICY, not (only) the 16th jack penalty.
- **Playtest: OFF is "sooooo much better", "forbidding footswitches forced the model to be more creative, not less."**
  Shipped `footswitch=False` as the default; revisit a GRADED footswitch policy (vs a hard ban) later.

## OPEN / next
- **Free-foot-overload gate** (queued): the hold fix's density gate is a PROXY — it can't tell a dense EXPRESSIVE
  hold from a dense jack-forcing one. A gate on the predicted free-foot workload / forced-jack would be robust across
  songs (floor 0.45 works here only because the pathological hold happens to be the densest). User's stated next lever.
- **16th-jack penalty** (pre-existing, orthogonal): now reframed by the footswitch finding — the residual intrinsic
  voltage (the runs that PERSIST footswitch-off, esp. OH WORLD) is the real target; tune TASTEFULLY (real charts have
  justified 2-note 16th jacks — H13 / foot_fatigue_design.md), not a blanket kill.
- **Graded footswitch policy** — a hard ban shipped well, but a graded penalty (allow SOME footswitch where musical)
  may be better; revisit.

## Artifacts
- Probes: `probe_bpm_hold_decomp.py`, `probe_stream_holdjack.py`, `probe_holdstream_fix.py` (all import the shared
  canonical helpers from `probe_quality_features.py`; CSVs `cache/bpm_hold_decomp*.csv`, `cache/stream_holdjack.csv`,
  `cache/holdstream_fix.csv`).
- Code: `typed_model.generate(hold_stream_penalty, hold_stream_floor, hold_stream_win, footswitch)`;
  `decode_defaults.CANONICAL_DECODE` (the shipped values); `export_typed_samples.py` flags (`--hold_stream_penalty`,
  `--footswitch/--no-footswitch`, `--ab_hold_stream`, `--ab_footswitch`); the shared-RNG A/B path.
- Playtests: `notes/playtest_log.md` (2026-07-02 entries). Cross-refs: `conditioning-mechanics` §7/§8,
  `foot_fatigue_design.md`, `choreography_metrics_findings.md` (hold_burst), `quality_feature_attribution_findings.md`.

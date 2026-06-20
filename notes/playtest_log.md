# Playtest Log

Running record of hands-on playtests of generated charts (dropped into StepMania and played),
plus Claude's commentary, hypotheses, and cross-session connections. Newest entries on top.
Each entry: **what was played → raw feedback → commentary/hypothesis → action.**

Sample sets live under `outputs/` (gitignored). Generation: `export_typed_samples.py`.

---

## 2026-06-20 — Stage 1 model playtest (chaos still smears; decode-time idea)

### What was played
`outputs/stage1_samples/{base, chaos}` — generated from the 41-dim `gen_stage1` model (base =
density-matched, pattern_temp 0.7, no_jump_during_hold; chaos = chaos=0.9, g=2.0). Same 6 songs as the
base-model sets, for A/B. (Offline numbers: `stage1_musical_features_findings.md`.)

### Raw feedback (user)
> "chaos was basically all 1/16s and 1/8s, and i wouldn't really consider it playable on those grounds…
> however, it did [seem] to improve somewhat. still a pretty random opening sequence, so cold-start is
> still an issue. i wonder if some decode-time tuning could help the model express some more musicality,
> with respect to chaos conditioning."

Then, on the plain (non-chaos) `base` set:
> "the base outputs were definitely more musical! it felt mostly right"

### Commentary / hypotheses
- **★ Stage-1 base felt "definitely more musical… mostly right" — the first play-feel WIN for the
  feature retrain.** This is the key result: the offline metrics were flat (`stage1_musical_features_findings.md`),
  but the human verdict on *plain* generation improved. So the chroma features DID help musicality where
  it's a free choice (which-panels following melody/harmony), even though onset_F1/phase didn't move.
  **Reframes H6:** features-not-sufficient was the wrong read for plain generation — they helped; the
  failure is specific to the *chaos knob* (forced off-grid timing), not to the features themselves. The
  metrics just couldn't see the improvement (re: the log's whole reason to exist).
- **Chaos still smears (H4/H6 hold for the chaos knob).** "All 1/16s and 1/8s" matches the offline phase
  histogram (stage1 chaos ≈ 6% on-beat). Chroma is used for *which-panels*; the chaos dim drives onset
  *timing* and still goes uniformly off-grid. "Improved somewhat" = the which-panels/chroma effect.
- **Cold-start persists.** "Random opening" = the AR pattern head starts from BOS with no prior-note
  context, so the first several notes are unanchored. This is the "awkward start" from round 3 (which we
  showed is NOT a density issue) — a decode/AR cold-start problem, plausibly decode-fixable.
- **The user's decode-time idea is sharp and worth testing.** Mechanism candidate — **phase-aware onset
  gating**: the audio-driven onset head localizes well (ROC-AUC ~0.9), so under chaos the off-beat smear
  may be a *thresholding* artifact — cranking chaos lifts ALL off-beat onset logits above a uniform
  threshold. If we require off-beat frames to clear a *higher* bar (or keep only the top off-beats by
  confidence), only musically-salient off-beats survive → *selective* syncopation instead of a smear.
  This was the optimistic read; the diagnostic below tempers it.

### Decode-gate diagnostic + result (this session)
**Diagnostic** — under chaos conditioning (chaos=0.9, g=2.0): corr(off-beat onset prob, audio
onset-strength) = **+0.10** (on-beat −0.07); off-beat onset prob **mean 0.70, std 0.11**. So chaos
floods off-beats to ~0.70 (the smear), with only *weak* audio signal underneath. Built the gate anyway
(`onset_phase_penalty`: on-beat 0, 8th −p, 16th −2p; 25 tests).

**Result — the gate does NOT rescue chaos.** Gated chaos sets came out near-empty / wildly inconsistent
(g=1.5+pen1.0: densities ~0.00–0.02; g=2.0+pen1.0: 0.007 / 0.42 / 0.59 / 0.72). **The key learning:**
chaos doesn't *add* off-beats on top of an on-beat backbone — it *moves* placement off-beat
(suppressing on-beat too). So gating off-beats leaves almost nothing, not "backbone + selective
syncopation." **Decode cannot fix chaos musicality** — confirms the deeper H6 read for the chaos knob:
the fix is the conditioning mechanism / objective, not decode. (The gate is still a useful *general*
metric-anchoring knob; for density-matched use it needs the threshold re-derived AFTER the penalty,
else density collapses — a real interaction bug, currently the gated sets are not density-matched.)

### Action / next
- [x] Diagnostic + gate built & tested → chaos is *moved* off-beat, not layered; decode can't fix it.
- [ ] **Pursue the Stage-1 base WIN**: it's musical — lock it in as the default, and consider Stage 2
      (drop metric-phase H7, drop HPSS H8, add structural feature H5) to push plain musicality further.
- [ ] Chaos knob: needs a conditioning/objective rethink (not decode) — lower priority than the base win.
- [ ] Gate knob: fix the density-matched-threshold interaction (re-derive tau post-penalty) before using
      it for non-chaos metric anchoring.
- [ ] Cold-start: still open (low-temp/primed opening), separate from chaos.

### Connecting thread
The recurring decode lever has a limit here: it unlocked latent quality before (pattern sampling,
hold-aware) because the model *already encoded* the right thing. For chaos it doesn't — the conditioning
genuinely relocates notes off-grid with little audio grounding (+0.10), so there's nothing musical for
decode to extract. The bigger, happier signal: **the plain Stage-1 model is more musical to play** even
though every offline metric said "no change" — vindicating both the feature retrain AND the playtest
log's reason to exist (the numbers can't see musicality; the hands can).

---

## 2026-06-19 (round 4) — chaos/air disambiguation (H4 fully confirmed + mechanism)

### What was played
`outputs/radar_samples/{chaos_only (chaos=0.9,g=2), air_only (air=0.9,g=2), chaos_gentle (chaos=0.9,g=1.3)}`.

### Raw feedback (user)
> "chaos_only was similarly unplayable. air_only was fine. chaos_gentle was just all blues (i think
> that's 1/8 shifted) pretty much."

### Measured corroboration — within-beat note phase (16th grid: 0=on-beat/4th-red, 2=8th-offbeat/blue, 1&3=16th)
```
                       on-beat   16th   8th-off(blue)  16th
air_only   (g=2.0)       0.98    0.00      0.01        0.00   on the beat -> playable
base ORIG  (real)        0.80    0.02      0.16        0.02   human: mostly on-beat + some syncopation
chaos_only (g=2.0)       0.06    0.32      0.30        0.32   downbeat GONE, uniform smear -> unplayable
chaos_gentle (g=1.3)     0.24    0.04      0.66        0.06   "all blues": 66% on the 8th-offbeat
```

### Conclusion — H4 FULLY CONFIRMED, with mechanism
- **air (jumps) is on-grid (98% on-beat) → playable.** Quantity knob, no musical justification needed.
- **chaos destroys the downbeat anchor.** Gentle chaos → ~uniform 8th-offbeat ("all blues", 66% phase-2,
  exactly the user's read); strong chaos → ~uniform smear across all 16th phases (only 6% on-beat).
- **Mechanism (the key insight):** the model renders the chaos dim as a *degenerate GLOBAL grid
  manipulation* — shift everything onto the offbeat, or smear uniformly — NOT *event-driven
  syncopation*. With no melodic/percussive features it has no idea *which* offbeats deserve a note, so
  the only way it can satisfy "more off-grid" is uniformly. A human places an 8th-offbeat note because a
  specific musical hit lives there. This is H1 + H5 made visible in one histogram: the model treats
  rhythm position as a *global statistic to match* (the radar value), not a *response to musical events*.
- **Bonus:** base-generated is *more* on-beat (0.91) than real (0.80) — the model slightly
  **under-syncopates by default** and literally cannot add musical syncopation; the only syncopation it
  can produce is the chaos knob's uniform smear. Strong motivation for the feature retrain.

### Action / next
- [x] Disambiguation done — chaos (off-grid) is the culprit, not air (jumps); g=2.0 overshoot makes it
      worse but even g=1.3 is degenerate (uniform blues). H3 also visible (gentle = less smeared but still
      not musical).
- [ ] **Feature retrain** is now the clearly-indicated move (H1 local + H5 global + this chaos mechanism
      all converge): per-frame chroma/HPSS for event-identity, structural signal for sections. Scoping next.

---

## 2026-06-19 (round 3) — stream_voltage + structural observations (H4 confirmed, H5 born)

### What was played
`outputs/radar_samples/stream_voltage/` (stream=0.9, voltage=0.9, g=2.0). Plus general
observations across the session's sets.

### Raw feedback (user)
> "stream_voltage was a little off but playable. some occasional random notes and unnatural jack
> sequences. broadly speaking, it feels like the model sometimes fails to pair choreography with a
> phase change in the song, and seems to consistently start and end the song awkwardly."

### What this confirms / opens
**H4 CONFIRMED.** stream_voltage (a *quantity* knob — density) is *playable*, while chaos_air (a
*musicality* knob) was *unplayable*. Exactly H4's prediction: quantity knobs steer fine, musicality
knobs break. The "occasional random notes / unnatural jacks" are minor choreography defects (H1/H2
territory — pattern head + decode temperature), not the structural problem below.

**H5 (new) — no song-structure / phrase awareness.** Measured density vs normalized song position
(6 base songs, generated vs their real charts, deciles 0=start..9=end):
```
REAL:  0.05 0.15 0.19 0.17 0.17 0.15 0.16 0.15 0.20 0.14   (intro -> build -> CLIMAX@8 -> outro)
GEN:   0.06 0.17 0.18 0.17 0.18 0.19 0.18 0.17 0.14 0.08   (flat -> FADE)
```
The real chart has a musical *arc* — sparse intro, build, a density **peak at 80–90%** (final
chorus/climax), short outro. The generated chart is **structurally flat and fades at the end** (last
two deciles 0.14/0.08 vs real 0.20/0.14). This is the measured signature of "awkward end" and "fails
to track phase changes": the model has no representation of song sections, so it can't ramp into a
climax or mark a verse→chorus boundary. It choreographs locally, frame by frame, with no global plan.

**"Awkward start" is a separate mechanism.** The data shows generated *start* density (0.06) matches
real (0.05) — the intro is fine densitywise. So the start-awkwardness the user feels is **choreographic
or sync**, not density. Candidates: AR cold-start (decoder begins from BOS with no context), audio
Conv1D edge effects on the first frames, or offset/lead-in sync. Needs its own probe — do NOT fold it
into H5.

### Mechanism notes
- End fade has two possible causes to separate: (a) audio energy genuinely drops at the outro and the
  model faithfully follows it, while the *human charter keeps density up for gameplay climax* (a
  choreographic choice not in the audio) — this is H5/H1; or (b) a *mechanical* decay — the onset head's
  positional encoding / AR context degrades at long positions so onset logits sag regardless of audio.
  **Probe to disambiguate (next):** feed constant-energy synthetic audio and plot onset prob vs
  position — if it sags with no audio reason, it's mechanical (b); if flat, the fade is structural (a).
- H1 (local: arrow↔event) and H5 (global: section structure) are two faces of the same root — the
  feature set is **frame-local timbre/energy**, expressing neither local event-identity (no chroma) nor
  global structure (no novelty/section curve, shallow Conv1D receptive field). Fix levers diverge though:
  H1 → chroma/HPSS per-frame; H5 → a structural signal (audio self-similarity/novelty, downbeat phase,
  or a wider-context / attention audio encoder so the decoder can see beyond the local frame).

### Probe result (done this session) — end-fade is structural, not mechanical
Fed the onset head **constant-energy audio** (fixed feature vector × 1440 frames) so any slope is
purely positional. Onset prob by decile (diff 2): `0.32 0.33 0.35 0.37 0.36 0.35 0.35 0.37 0.36 0.38`
— **flat, slightly rising at the end** (same shape across difficulties 0–3). Conclusions:
- **End-fade = audio-faithful, NOT mechanical.** The model doesn't positionally decay; the real-song
  end-fade is it *following audio energy down at the outro* while humans keep density up for the
  climax/finale. Mechanism (b) ruled OUT; (a) confirmed. **Refines H5:** the model maps audio-energy→
  density too literally; human charting density carries structural/gameplay intent (build to a climax)
  that is *not a pure function of instantaneous audio energy*. Decode/position fixes won't help — needs
  structural signal or a density target that isn't purely audio-energy-driven.
- **Cold-start dip exists but is small:** first decile depressed (0.32 vs ~0.36 steady) with no audio
  reason — a mechanical AR/positional start effect. But real-song start *density* matched real (0.06 vs
  0.05), so "awkward start" is more likely choreographic/sync than density. Still worth a pattern-level
  start probe.

### Action / next
- [ ] **Start probe**: inspect first ~16 frames of generated vs real (pattern + onset + sync offset) to
      characterize the cold-start awkwardness (density ruled out; look at choreography/sync).
- [ ] User still to play the disambiguation sets: chaos_only / air_only / chaos_gentle (H4/H3 split).
- [ ] H5 fix candidates to scope: (1) add an audio *novelty/self-similarity* feature; (2) add downbeat
      phase; (3) widen the audio encoder's receptive field or add self-attention for global context.

### Connecting thread
The defect taxonomy is sharpening into a clean hierarchy:
- **Timing (onset, local)** — solved (ROC-AUC ~0.9).
- **Local choreography (which arrow ↔ which event)** — H1, blocked by no melodic features.
- **Global structure (sections / phrases / climax / start-end)** — H5, blocked by frame-local features +
  shallow receptive field.
- **Decode polish (jacks, random notes, jump-during-hold)** — fixable at decode time (the pattern keeps
  holding: many "problems" are decode/feature gaps, not capacity).
Every layer above "timing" traces back to the audio representation being musically shallow. The
quantitative metrics (onset_F1, crit_adj) score only the bottom layer — which is why they've looked
great while the chart still doesn't *feel* musical. **The playtest log is measuring the layers the
metrics can't see.**

---

## 2026-06-19 (round 2) — radar toggle playtest

### What was played
`outputs/radar_samples/chaos_air/` — radar `chaos=0.9, air=0.85` (others at dataset mean),
`guidance 2.0`, `pattern_temperature 0.7`, `no_jump_during_hold` on.

### Raw feedback (user)
> "chaos_air is totally unplayable. the density wasn't too bad, it was just so unintuitive with
> the music i couldn't establish any kind of rhythm or feel for what should come next."

Crucially the user **ruled out density** as the cause — it's the *rhythmic placement* that's the
problem: no steady pulse to lock to, and no ability to anticipate the next note.

### Commentary & hypothesis
The `chaos` radar dim is *defined* as off-grid-ness / note-quantization variety (see
`groove_radar.py`: chaos scores notes by their position within the beat — on-beat = 0, off-beat
subdivisions score higher). So cranking chaos to 0.9 **and** amplifying at g=2.0 did exactly what the
knob says: it pushed notes off the steady 4th/8th grid onto unpredictable subdivisions. The knob
*works*. But "unpredictable rhythm" is precisely "unplayable" for a human — a player anticipates from
two anchors: (a) the steady grid/pulse and (b) the music itself. Chaos destroys (a), and our weak
choreography never provided (b), so the player is left with nothing to read.

This sharply **reinforces H1 and spawns H4.** A human author's syncopation is *motivated*: the off-grid
note lands on a vocal stab or a snare fill, so it's still readable. Our model has **no melodic/percussive
features** to motivate off-grid placement, so amplified chaos is *rhythmic noise, not musical
syncopation*. Notably the user said `base_coherent` (chaos at its natural ~mean) felt human — so it's
not chaos-the-concept that's broken, it's **chaos amplified beyond what the (musically-blind) model can
justify.** Two confounds to separate: the air=0.85 (jumps) component, and the g=2.0 overshoot (H3).

**H4 (new):** Knobs that demand *musical justification* (chaos / syncopation) expose the feature gap
(H1) far more than *quantity* knobs (stream/voltage→density, freeze→holds). Quantity needs no musical
reason — more on-grid notes is still readable — so those sets should stay playable while chaos breaks.
Prediction to test: stream_voltage, freeze_holds, calm remain playable; chaos is the lone unplayable one.

### Action / next
- [ ] Disambiguate the chaos_air result: generate **chaos-only** (`chaos=0.9`) vs **air-only**
      (`air=0.9`) at g=2.0 — is the unplayability from off-grid (chaos) or jumps (air)? H4 predicts chaos.
- [ ] Generate **chaos at gentle guidance** (`chaos=0.9 --guidance 1.3`) — does moderate chaos become
      readable? Tests whether it's chaos itself or the g=2.0 overshoot (H3).
- [ ] Get the user's read on the other three sets (stream_voltage / freeze_holds / calm) — direct H4 test.
- [ ] If H4 holds, it's the strongest argument yet for the chroma+HPSS feature retrain (H1): chaos can
      only become *musical* if the model can see the musical events it should syncopate to.

### Connecting thread
The knobs are splitting into two families: **quantity knobs (density, holds, jumps)** that steer fine
because they don't need musical reasons, and **musicality knobs (chaos/syncopation, and ultimately
choreographic arrow-mapping)** that fail because the feature set is musically blind. Same root as the
"disconnected from the music" complaint and the choreography gap — every road keeps leading back to H1.

---

## 2026-06-19 — Session: conditioning knobs + first real playtest

### Sets played
- `outputs/style_samples/base/` and `base_coherent/` — no conditioning, density matched per song.
  `base_coherent` = `pattern_temperature 0.7` (less arrow-jumping).
- `outputs/style_samples/{sparse_ref,dense_ref,dense_ref_g1.4}/` — Step 3 reference-style transfer.
- `outputs/radar_samples/{chaos_air,stream_voltage,freeze_holds,calm}/` — groove-radar toggles (g=2.0).

### Raw feedback (user)
1. **`base_coherent` was genuinely fun — "a human author might have reasonably written them."**
   This is the first generated set that played well.
2. **The model places a jump *during* a hold** — fine for a keyboard, but a dance-pad player has only
   one free foot while holding, so a jump-during-hold is unhittable ("the Will Smith meme"). Wants this
   as an on/off decode knob. → **implemented `no_jump_during_hold`** (see commentary).
3. Earlier in session: the heavily style-guided samples (g=2.0 dense reference, ~0.9 density) felt
   "disconnected from the music besides musical time."

### Commentary & hypotheses

**H1 — Timing is solved; choreography (arrow mapping) is the open axis.** The onset head is
audio-driven with ROC-AUC ~0.81–0.95: the model knows *where* notes go in time. What it lacks is a
sense of *which arrow* maps to *which musical event*, because the 23-dim audio features are timbre +
energy only (MFCC, onset, spectral contrast) with **no pitch/melody/harmony** (no chroma, no source
separation). So the pattern head choreographs from learned generic statistics + previous steps, not
from musical content. That `base_coherent` *still* felt human-authored suggests the pattern head's
learned statistics are actually decent — the missing musicality is subtle, not gross. **Predicted
next lever: add chroma + HPSS (harmonic/percussive split) features and retrain** so arrows can track
melody vs drums. (Not yet tested — see roadmap.)

**H2 — Decode randomness was masking the model's quality.** `pattern_temperature 1.0` (added to fix
the old always-Left/jacks greedy collapse) injects arrow-jumping that reads as "unintuitive."
Dropping to 0.7 (`base_coherent`) was the single change that made charts fun. **Hypothesis: there's a
coherence/variety sweet spot around 0.6–0.8; 1.0 over-randomizes, greedy collapses.** Worth a
systematic temperature sweep judged by play-feel, not just panel-balance stats.

**H3 — High CFG guidance trades musicality for control.** g=2.0 forces density/style so hard it
overrides the audio onset head → notes everywhere → disconnected. The *knobs work*, but **strong
guidance and musical alignment are in tension.** Expect the radar sets at g=2.0 to feel more
"forced" than `base_coherent`; a gentler g≈1.3–1.5 likely keeps the steer while staying musical
(cf. `dense_ref_g1.4` was the playable dense one).

**Jump-during-hold knob (implemented):** while a hold is open, a pad player has one free foot, so any
pattern with ≥2 *fresh* presses on non-held panels is forbidden (closing the held panel and single
taps stay legal). Lives in the `hold_aware` automaton in `generate()`; exposed as
`export_typed_samples.py --no_jump_during_hold`. This is a *biomechanical playability* constraint, a
new category alongside the *musical* ones (crossovers) — note both are decode-time automata that need
no retrain, reinforcing that **a lot of "playability" is post-hoc constraint, not model capacity.**

### Connecting thread
Three of the project's biggest wins (always-Left fix, hold-aware decoding, crossover/jump constraints)
were **decode-time fixes, not model changes** — the model keeps being better than its default decode
makes it look. Standing hypothesis: **the base model is under-served by decode; squeeze decode (and a
musicality-aware feature set) before adding capacity.**

### Actions taken
- Implemented + tested `no_jump_during_hold` (24 generation tests pass).
- Added `--radar` and `--no_jump_during_hold` to `export_typed_samples.py`.
- Generated the four radar-toggle sets above for the next playtest round.

### Open / next
- [ ] Playtest the radar sets — does each dim *feel* like its name? (chaos=off-grid?, air=jumps?,
      stream/voltage=density?, freeze=holds?) Record per-dim feel below.
- [ ] Temperature sweep judged by feel (H2).
- [ ] Try radar at gentler guidance (g≈1.3) to test H3.
- [ ] Prototype chroma + HPSS features + retrain to test H1 (the choreography-musicality lever).

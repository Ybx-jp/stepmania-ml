# Playtest Log

Running record of hands-on playtests of generated charts (dropped into StepMania and played),
plus Claude's commentary, hypotheses, and cross-session connections. Newest entries on top.
Each entry: **what was played → raw feedback → commentary/hypothesis → action.**

Sample sets live under `outputs/` (gitignored). Generation: `export_typed_samples.py`.

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

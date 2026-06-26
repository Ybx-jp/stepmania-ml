# Note patterns, motifs, and the motif control surface

The single home for "what are the note patterns / motifs this project talks about, and what is the
motif/figure control surface meant to do." Consolidated so the framing isn't scattered across the
`h15_*` notes, the `conditioning-mechanics` skill, and the playtest log. Cross-refs at the bottom.

---

## 1. What a "note pattern" / "motif" is

A StepMania chart is a sequence of arrows on 4 panels (L/D/U/R). Players and authors don't think in
individual arrows — they think in **patterns**: recurring shapes of footwork that read as a unit.
A **motif**, here, means a recurring note-pattern figure — the *kind* of footwork happening in a
passage (a jack, a stream of crossovers, a candle run), as opposed to the raw note count or timing.

The community vocabulary (see the links below):

- **Jack** — 2+ consecutive notes on the *same* panel (DD, RRR): one foot hammering one arrow.
- **Stream** — a sustained run of single notes (often 1/8 or 1/16), footing alternates.
- **Gallop** — two different arrows a 1/16 apart: a quick "da-dum" grace-note burst, off the main grid.
- **Crossover** — a pattern that forces the left foot onto the right arrow (or vice-versa).
- **Foot-switch** — what looks like a jack but is actually a *foot swap* on the same panel (not a
  same-foot hammer), used to stay alternating.
- **Candle (cross)** — L-D-U / D-R-U shapes where a foot sweeps *through* the center (it'd knock over a
  candle placed there).
- **Trill** — fast alternation between two panels (L-R-L-R).
- **Sweep / staircase** — L-D-U-R style runs that walk across the pad in order.
- **Bracket / hands** — two panels pressed at once with one foot (bracket) or 3+ notes (hands).
- **Lateral** — extended crossovers that *stay* crossed (e.g. the AFRONOVA walk).
- **Freeze-switch** — lift-and-replace on a hold (freeze) to keep alternating feet.

### Reference articles
- **[DDRCommunity — "Basic Patterns That You Need To Know"](https://ddrcommunity.com/basic-patterns-that-you-need-to-know/)**
  — the guide the playtest notes refer to as "the ddrcommunity guide" (jacks, gallops, candles,
  crossovers/foot-switches, etc.).
- **[StepMania Wiki — "Pattern for Dance"](https://stepmania.fandom.com/wiki/Pattern_for_Dance)** — a
  broader pattern glossary.
- **[Zenius-I-Vanisher — "The DDR Dictionary"](https://zenius-i-vanisher.com/v5.2/thread?threadid=1769)**
  — community terms/acronyms (ZIv is also one of the chart-data sources for this project).

---

## 2. The figure vocabulary the model uses

The project collapses passages into a small set of canonical **figure classes** for measurement and
discrete conditioning (`src/generation/motif_codebook.py`, `FIGURE_CLASSES`):

```
["sparse", "jack", "sweep/staircase", "trill", "candle/cross", "jump/bracket", "step"]
```

`figure_token(section)` labels each section by its dominant figure family. (`sparse` = too few onsets
to name.) A discrete family label is used *in addition to* the continuous motif knobs because the
continuous projection entangles some figures — knob-0 "jack↔sweep" is really a jack detector, and
sweep mushes with step/candle (`diag_figure_labels.py`), so a committed per-section figure token
cleanly isolates a figure (especially **sweep**) that the continuous axis cannot.

---

## 3. The motif / figure control surface — intent & status

**Intent (the design goal):** steer the model toward *specific note patterns at musically literate
moments* — i.e. ask for "more candle runs here, a trill on this phrase, less jack on that section,"
section-by-section, so the choreography matches what the music is doing rather than being a uniform
texture. It is the *which-figures* axis of control, orthogonal to the groove-radar *how-much / what
feel* axis and to the biomechanical governor's *is-it-playable* axis.

**Mechanism (two coupled knobs, H15):**
- **Continuous motif knobs** — `MotifBasis` (`MOTIF_DIM = 12`), radar-orthogonal motif-style
  directions; passed to `typed_model.generate(motif=...)`. Shapes *which panels*, decoupled from
  onset/rhythm. Fit artifact: `cache/motif_basis.npz`.
- **Discrete figure schedule** — a per-section `figure` token (the `FIGURE_CLASSES` above), passed to
  `generate(figure=...)`, to commit a section to a figure family.

**Status: PARTIAL / unfinished as a control surface.** Do not present it as done.
- **Candle/cross — works, playtest-confirmed.** A steerable, section-by-section, quality-safe lever;
  `motif_candle_neg` played audibly "more linear" than `motif_candle` on the same song. Operate gently
  (g≈1.4; g2 can break musicality on less-ornamental songs — H3).
- **Trill — steers**, but the offline trill metric is partly confounded by a held-frame/tap artifact
  (H19), so its measured strength is inflated; treat with care.
- **jack↔sweep — the lone dead/weak axis.** The continuous knob can't push *toward* strong sweep; the
  only path to a strong sweep is the **structured-realize / discrete figure** route, not the
  continuous projection. This is the unfinished part of the surface.
- **Dependency note:** `cache/motif_basis.npz` is **not shipped** (unlike `cache/radar_manifold.npz`),
  and `scripts/generate.py` does **not** expose `--motif`/`--figure` (it passes `motif=None`). The
  fatigue/stamina **governor does not depend on the motif basis at all** — governor costs are inline
  constants. So this surface is deliberately parked until it's finished; shipping the basis + exposing
  the knobs is the to-do.

---

## 4. Coverage gap vs. steering gap (H20)

Distinct from "can we steer the figures the model knows" (§3) is "which figures the model knows at
all." The model's repertoire is **over-concentrated on jacks** and **under-covers ornamental
footwork** — gallops, foot-switches, laterals, half-spins, freeze-switches, brackets/hands are barely
produced. That's a **data-coverage / objective** issue (these figures are rare in training; the jack
axis is over-served), **not** a decode knob — no amount of motif conditioning summons a figure the
model never learned. Convenient coincidence: the stuck steering axis (jack↔sweep) and the desired
direction (the user wants jacks *lower*) point the same way, so a working jack-reduction push doubles
as the over-jacked-feel fix.

---

## Cross-references
- Mechanism / decode math: the `conditioning-mechanics` skill (motif = which-panels, radar-orthogonal).
- H15 motif arc detail: `notes/h15_motif_findings.md`, `notes/h15_local_motif_plan.md`,
  `notes/h15_set_characterization.md`, `notes/h15_motif_handoff.md`.
- Play-feel evidence (H15 candle, H18 chaos, H19 trill confound, H20 vocabulary): `notes/playtest_log.md`.
- The other two control axes: groove radar / density (`notes/radar_manifold_findings.md`) and the
  biomechanical governor (`notes/foot_fatigue_design.md`).

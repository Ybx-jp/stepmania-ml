# Experiment-lineage INDEX

One **lineage file per distinct investigative thread** — the chain of hypotheses → probes → findings → pivots →
current state, with cross-links to the source `notes/*_findings.md`, the relevant skills, and the OTHER lineage
files it corroborates or depends on. A lineage file is the "how did we get here, and what's already been ruled
out" map; the `notes/*_findings.md` are the primary results it threads together. See the **directive** in
`../SKILL.md` ("Experiment lineage — maintain these") for when/how to add one.

Status legend: ✅ written · 🟡 stub (notes exist, lineage file pending) · ⬜ not started

| thread | file | status | one-line | primary notes / memory |
|---|---|---|---|---|
| **Quality-feature attribution** | [quality-feature-attribution-arc.md](quality-feature-attribution-arc.md) | ✅ | "which audio features drive per-song generator QUALITY under canonical defaults?" → **BPM/tempo** (faster Hard → worse gens, r=−0.68 p_fw=0.004), mechanism = the **PATTERN/TYPE head at high density** (governor/coverage/onset-timing all ruled out; onset_override A/B confirms). ⚠️ OVERTURNED a committed 3-instrument NULL that was NOISE ATTENUATION (single-gen score ~46% noise, ICC=0.54; 8-gen mean 0.90-reliable). **Method keeper: check target RELIABILITY/ICC before concluding "no signal".** Built 2 non-saturating instruments (choreography distance-to-real; a GRADED critic). | quality_feature_attribution_findings · [[quality-feature-attribution]] · depends-on taste-critic |
| **Onset phrasing & sparse-harm calibrator** | [onset-phrasing-calibrator-arc.md](onset-phrasing-calibrator-arc.md) | ✅ | mis-attributed PATTERN→true ONSET phrasing; diagnostic + sparse-harm calibrator (awaiting ears) | arc_lag / h11_rerun / phrasing_coherence / onset_alloc / cache_index_bug · [[onset-phrase-calibrator]] |
| Biomechanical governor (foot fatigue / stamina / breathe arc) | _governor-arc.md_ | 🟡 | decode-time per-note foot model + per-region stamina + breathing density arc; SHIPPED PR#41 | foot_fatigue_design / governor_release_region · [[fatigue-governor]] |
| Jack-heaviness decomposition | _jack-heaviness-arc.md_ | 🟡 | jacks = pattern head (proximate, temp↑) + onset blocky rhythm (contributing) | jack_heaviness_findings / foot_physics_baseline · [[jack-heaviness]] |
| Chaos / conditioning (manifold vs mean-pin) | _chaos-conditioning-arc.md_ | 🟡 | the attribution-error gold mine: harness OOD, not model defect (manifold conditional-fill) | the failure_modes_postmortem.md examples · conditioning-mechanics §2 |
| Taste-critic transfer / interpretability | _taste-critic-arc.md_ | 🟡 | ranking transfers (REAL>BASE>CHAOS) but near-binary not graded; interp Phase A/B/C | taste_critic_interpretability_plan · [[taste-critic-transfer]] |
| H15 motif / figure steering | _motif-arc.md_ | 🟡 | section-level candle/trill levers; jack↔sweep the lone dead axis (soft-realize ceiling) | h15_*; conditioning-mechanics §4–§5 |
| Sequence-aware onset head | [seq-onset-arc.md](seq-onset-arc.md) | ✅ | "when" isolated from "where"; placement is a chart-PRIOR not in audio (wall CLOSED NEGATIVE 4 ways). BUILD re-opened (M1a: frozen `h` conv readout 0.892 ≡ ceiling) + M1b-3 broke the DENSITY drift (scheduled sampling, run 1.0 @ real density). **fork (A) ALIVE but UNDERTUNED 06-29 (M1b-4..9; I committed "BANKED" and the user overturned it twice — valid catches).** The 16th-flood was measured on the AUDIO head's decode surface; **the decode surface is HEAD-SPECIFIC** (tau→adaptive, 16th-unlock polarity flips, rests need an explicit valve). A head-appropriate surface (`seqonset_decode.py`) drains the flood to a real-aligned backbone that pauses; playtest "better, still very linear". The fork is now STRATEGIC (right investment this stage?), not "is it viable" — viable-but-early like the audio decode when it landed. Lead: hold-release phantom-rest (untested) | onset_frozenh / onset_seqrollout / onset_ss / **onset_placement** · [[onset-phrase-calibrator]] |

When you add a lineage file: flip the row to ✅, link it, and add reciprocal `corroborates`/`depends-on` links in
the related files (e.g. the governor arc and the onset-phrasing arc reference each other on the breathe-arc =
onset-side-density finding).

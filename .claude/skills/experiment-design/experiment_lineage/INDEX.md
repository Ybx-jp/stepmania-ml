# Experiment-lineage INDEX

One **lineage file per distinct investigative thread** — the chain of hypotheses → probes → findings → pivots →
current state, with cross-links to the source `notes/*_findings.md`, the relevant skills, and the OTHER lineage
files it corroborates or depends on. A lineage file is the "how did we get here, and what's already been ruled
out" map; the `notes/*_findings.md` are the primary results it threads together. See the **directive** in
`../SKILL.md` ("Experiment lineage — maintain these") for when/how to add one.

Status legend: ✅ written · 🟡 stub (notes exist, lineage file pending) · ⬜ not started

| thread | file | status | one-line | primary notes / memory |
|---|---|---|---|---|
| **Onset phrasing & sparse-harm calibrator** | [onset-phrasing-calibrator-arc.md](onset-phrasing-calibrator-arc.md) | ✅ | mis-attributed PATTERN→true ONSET phrasing; diagnostic + sparse-harm calibrator (awaiting ears) | arc_lag / h11_rerun / phrasing_coherence / onset_alloc / cache_index_bug · [[onset-phrase-calibrator]] |
| Biomechanical governor (foot fatigue / stamina / breathe arc) | _governor-arc.md_ | 🟡 | decode-time per-note foot model + per-region stamina + breathing density arc; SHIPPED PR#41 | foot_fatigue_design / governor_release_region · [[fatigue-governor]] |
| Jack-heaviness decomposition | _jack-heaviness-arc.md_ | 🟡 | jacks = pattern head (proximate, temp↑) + onset blocky rhythm (contributing) | jack_heaviness_findings / foot_physics_baseline · [[jack-heaviness]] |
| Chaos / conditioning (manifold vs mean-pin) | _chaos-conditioning-arc.md_ | 🟡 | the attribution-error gold mine: harness OOD, not model defect (manifold conditional-fill) | the failure_modes_postmortem.md examples · conditioning-mechanics §2 |
| Taste-critic transfer / interpretability | _taste-critic-arc.md_ | 🟡 | ranking transfers (REAL>BASE>CHAOS) but near-binary not graded; interp Phase A/B/C | taste_critic_interpretability_plan · [[taste-critic-transfer]] |
| H15 motif / figure steering | _motif-arc.md_ | 🟡 | section-level candle/trill levers; jack↔sweep the lone dead axis (soft-realize ceiling) | h15_*; conditioning-mechanics §4–§5 |
| Sequence-aware onset head | [seq-onset-arc.md](seq-onset-arc.md) | ✅ | "when" isolated from "where"; signal real (0.87 TF) but unreachable — **06-28 re-open RESOLVED: wall STANDS** (deployed C0 ≈ audio 0.667; needs RETRAIN not a decode lever) | sequence_aware_onset_plan · [[onset-phrase-calibrator]] |

When you add a lineage file: flip the row to ✅, link it, and add reciprocal `corroborates`/`depends-on` links in
the related files (e.g. the governor arc and the onset-phrasing arc reference each other on the breathe-arc =
onset-side-density finding).

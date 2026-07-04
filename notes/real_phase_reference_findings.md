# Real-data phase reference — how REAL Hard charts get chaotic (the good-region anchor)

**Rule-5 reference (experiment-design skill).** Before mapping how the GENERATED backbone degrades under cranked
chaos/guidance, establish what REAL charts do in the same regime — the reference distribution the generated metric
must be measured *against*, not an arbitrary threshold. Offline, no model: parse real Hard charts, bin by their own
chaos radar, read the phase structure off the real typed grid. `probe_real_phase_reference.py`, real Hard n=176.

## Result — real charts get chaotic by ADDING density on a PRESERVED, better-ANCHORED backbone
| chaos bin | chaos | density | on_grid | qrep(strict) | s16_rate | anchor |
|---|---|---|---|---|---|---|
| Q1 (calm) | 0.04 | 0.280 | 0.99 | 0.80 | 0.01 | 0.41 |
| Q2 | 0.07 | 0.309 | 0.96 | 0.80 | 0.04 | 0.56 |
| Q3 | 0.11 | 0.333 | 0.94 | 0.77 | 0.06 | 0.58 |
| Q4 (chaotic) | 0.25 | 0.356 | **0.85** | **0.68** | **0.15** | **0.73** |

Spearman vs chaos: density **+0.68** (≡ H4's +0.63), on_grid **−0.70**, s16 **+0.70**, qrep −0.32, anchor **+0.23**.
(chaos radar range 0.00–0.54; metrics: `probe_backbone_tolerance.py`. on_grid = onset share on strong beats t%4∈{0,2};
anchor = fraction of 16th-offbeat onsets flanked by a beat carrying a note.)

**Reads:**
- **Chaos ADDS notes** (density +0.68) — it does not merely displace them. Matches H4's chaos↔density +0.63.
- **The backbone is PRESERVED.** Even the most chaotic real Hard keeps **85%** of onsets on strong beats (on_grid
  0.99→0.85) and hits **68%** of active downbeats. Off-beats rise but stay **BOUNDED**: s16 tops out at **~0.15**.
- **The added 16ths get MORE anchored** (0.41→0.73): real chaos = coherent 16th RUNS resolving into beats, not
  scattered off-grid notes.

## The real envelope vs a SONG THAT collapses (Deja loin) — NOT the whole regime
| metric | real chaotic (Q4) | Deja loin `chaos=0.9,g=3.0` |
|---|---|---|
| on_grid | 0.85 | **0.00** |
| s16_rate | 0.15 | **~1.00** (6.7× the real max) |
| anchor | 0.73 | **0.00** |

**`chaos=0.9,g=3.0` is SONG-DEPENDENT, not OOD as a regime.** For Deja loin (n=1, measured) the crank pushes the
chart far outside the real envelope on every metric — the [[H4]] degenerate global-smear, amplified by CFG (H14:
"guidance floods off-beats"). But the user has PLAYED several songs at this exact setting that were "fantastic"
(ear ground truth) — those stay inside the envelope. WHICH songs collapse vs stay real-like is the per-song
**tolerance** the [[goodregion]] sweep maps; do not pool it into "the regime is OOD" (that pooled claim, made
earlier from n=2, broke experiment-design Rules 9+12 — see the lineage file).

## Consequences (the anchor for downstream work)
1. **Tolerance = distance from THIS real high-chaos envelope**, primarily **on-grid** (target ~0.85) and
   **anchoring** (target ~0.73) — both crash to ~0 in the smear. Not `quarter-share`/`quarter-rep` (misleading; see
   `backbone_phase_findings.md` reconciliation). Re-anchors `probe_backbone_tolerance.py`.
2. **Anchoring is the real-vs-degenerate discriminator, and it names the H4 defect:** real chaos = ANCHORED off-beats
   (coherent runs); generated chaos = UNANCHORED off-beats (a global shift). The deeper fix is the known
   conditioning-mechanism problem (a chaos×onset gate), not decode tuning (H4 §6 conclusion).
3. **The manifold-realized chaos (0.44) is at the real edge and is fine** — it is the *guidance* amplification that
   goes OOD. So per-song tolerance ≈ "max guidance staying inside the real envelope."

## Connections
`h4_offbeat_signal_findings.md` (chaos = global off-grid shift; no local off-beat signal; 16th under-commitment),
`h14_guidance_sweep_findings.md` (guidance floods off-beats; critic floored on OOD styles),
`radar_manifold_findings.md` (cranking chaos while pinning density = OOD point real never visits),
`backbone_phase_findings.md` (the generated-side attribution this reference re-anchors), the [[goodregion]] thread.
Tooling: `probe_real_phase_reference.py`.

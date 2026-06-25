# H15 Playtest-Set Characterization (offline, exported charts) — 2026-06-24

`experiments/generation_typed/characterize_sets.py` parses each set's generated `Challenge` chart (the actual
installed `.sm`) and reports playability/rhythm stats + the radar-orthogonal motif KNOBS (the same measure
eval_motif used: candle=k3, trill=k10, jack↔sweep=k0). Purpose: confirm the conditioning LANDED in the files
the user plays (not just in the eval harness), and pre-empt a confounded playtest. 5 songs/set, gen_motif_full.

**METHOD NOTE (exp-design Rule 1):** the first metric (dominant-figure-per-section fraction) was too COARSE —
these Hard songs are trill-saturated, so adding candles rarely flips a section's dominant label; it showed a
null. The radar-orthogonal KNOB (z-score) is the sensitive measure and reconciles with eval_motif. Lesson:
match the characterization metric to the conditioning target.

## Result (candleK/trillK = z; jump/hold/16th = note-fraction)
| set | candleK (Δbase) | trillK (Δbase) | jump | hold | 16th |
|-----|-----------------|----------------|------|------|------|
| base | +0.77 | −0.64 | 0.08 | 0.06 | 0.00 |
| motif_candle (+3 g2) | **+1.42 (+0.65)** | −0.20 | 0.03 | 0.02 | 0.00 |
| candle_gentle (+3 g1.4) | +1.11 (+0.34) | −0.23 | 0.04 | 0.03 | 0.00 |
| candle_neg (−3 g2) | **−1.23 (−2.00)** | −1.45 | 0.05 | 0.03 | 0.00 |
| motif_trill (+3 g2) | +0.88 | **+0.01 (+0.65)** | 0.03 | 0.03 | 0.00 |
| figure_sweep (g1) | +0.74 | −0.61 | 0.07 | 0.05 | 0.00 |
| radar_air | +0.89 | −0.07 | **0.14** | 0.05 | 0.00 |
| radar_freeze | +0.96 | −0.02 | 0.11 | **0.08** | 0.00 |
| radar_stream | +0.86 | −0.49 | 0.11 | 0.05 | 0.00 |
| radar_chaos | +0.84 | −0.30 | 0.10 | 0.05 | 0.00 |
| combo_chaos_candle | +1.41 (+0.64) | +0.26 | 0.05 | 0.02 | 0.00 |
| combo_stream_trill | +0.95 | +0.01 (+0.65) | 0.06 | 0.03 | 0.00 |
| combo_glitch_candle | +0.48 | +0.08 | 0.02 | 0.01 | **0.11** |

## Findings
- **Candle steering LANDED strongly** in the exported charts: candle+ +1.42 vs candle− −1.23 = a 2.65 z swing
  across poles; gentle-guidance correctly weaker (+1.11). So `motif_candle` vs `candle_neg` is a REAL contrast
  for the ear — the playtest is NOT confounded. Survives in combination (combo_chaos_candle +1.41).
- **Trill landed** (trill set ΔtrillK +0.65; broadly trill rises in many sets too).
- **figure=sweep did NOT land** (ΔsweepK −0.07 on knob-0; note knob-0 + pole = JACK, so sweep would push it
  NEGATIVE — it didn't move). Consistent with the proven soft-realize ceiling; imperceptible at export settings.
  If `h15_09_figure_sweep` feels like base by ear, that's EXPECTED.
- **Radar proxies track their names**: air→most jumps (0.14), freeze→most holds (0.08). BUT **chaos barely
  engaged** — 16th≈0.00 almost everywhere (manifold picked a gentle chaos≈0.23); only aggressive combo_glitch
  (g2) made 16ths (0.11). The radar_chaos set may feel mild; a stronger chaos set would need higher target/g.
- **Side effect:** candle conditioning suppresses jumps/holds (0.03/0.02 vs 0.08/0.06) toward pure candle-steps.

## Caveats
Knobs measured against each generated chart's OWN radar (eval_motif used the source-song radar) — directions
match but magnitudes aren't identical to eval_motif's harness. Density is ~flat (0.34) because the manifold
sets density per difficulty and all sets are Hard.

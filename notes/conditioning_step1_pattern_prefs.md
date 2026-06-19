# Conditioning Step 1: decode-time pattern preferences + crossover constraint

*2026-06-18. First conditioning knobs from `docs/conditioning_roadmap.md` — decode-time, no retraining.*

Added to `LayeredTypedChartGenerator.generate`:
- **`pattern_bias`** — a (15,) additive logit bias on the pattern (which-panels) head. Build it with
  `typed.make_pattern_bias(jump=..., panel_prefs=[L,D,U,R])`: `jump>0` favors multi-panel patterns,
  `panel_prefs` favors specific panels.
- **`no_crossovers`** — a per-panel foot automaton (alternate feet on single notes, reset after a
  jump) that masks the crossing single-note pattern (left foot→R, right foot→L) at decode.

Helpers in `typed.py`: `make_pattern_bias`, `count_crossovers` (greedy foot heuristic for eval).

## Result (32 val songs; real: jump 0.11, crossover 0.25)

| setting | jump rate | crossover rate | R panel | onset_F1 | crit_adj |
|---|---|---|---|---|---|
| baseline | 0.16 | 0.25 | 0.29 | 0.784 | 1.000 |
| jump +1.5 | 0.77 | 0.24 | — | 0.784 | 1.000 |
| jump −1.5 | 0.02 | 0.26 | — | 0.784 | 1.000 |
| no_crossovers | 0.29 | 0.00 | — | 0.784 | 1.000 |
| prefer R | 0.35 | 0.41 | 0.68 | 0.784 | 1.000 |

- **Graded jump control** (0.02 ↔ 0.77), **crossovers → 0** on demand, **panel reshaping** (R 0.29→0.68).
- **onset_F1 and difficulty critic unchanged** across all settings — pattern controls only touch which
  panels, not where notes go or the overall difficulty.
- Note: favoring a side panel raises crossovers (more R → more left-foot-on-R); combine with
  `no_crossovers` if needed.

## Usage

`export_typed_samples.py` now takes `--jump_bias`, `--no_crossovers`, `--prefer "U,R"`:
```
python experiments/generation_typed/export_typed_samples.py --data_dir data/ --audio_dir data/ \
    --num_songs 8 --jump_bias 1.0 --no_crossovers
```
`eval_conditioning.py` demonstrates/measures the knobs. Tests: `make_pattern_bias`, `count_crossovers`,
and a `no_crossovers -> 0 crossovers` decode test (20 generation tests pass).

Next conditioning steps (roadmap): trained groove-radar profile (+ CFG), then reference-chart style.

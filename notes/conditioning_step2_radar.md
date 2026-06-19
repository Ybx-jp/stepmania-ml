# Conditioning Step 2: trained groove-radar profile

*2026-06-18. The headline "specific groove-radar profile" knob from `docs/conditioning_roadmap.md`.*

Added a target **groove radar** (5 dims: stream, voltage, air, freeze, chaos) as a trained
conditioning input, generalizing the difficulty embedding into a conditioning vector
`c = difficulty_emb + radar_proj(radar)` added at every decoder position (onset + pattern + type
heads). Trained with **classifier-free-guidance dropout** (15% of batches drop the radar → a learned
null embedding), warm-started from the layered checkpoint. Teacher target = each chart's own
groove radar (free via `GrooveRadarCalculator`); at inference you set any target.

## Controllability (24 val songs; vary one dim 0.1→0.9, others at the dataset mean)

| radar dim ↑ | density | jump rate | hold rate |
|---|---|---|---|
| stream | **0.265 → 0.333** | 0.106→0.093 | 0.041→0.046 |
| voltage | **0.264 → 0.333** | 0.102→0.112 | 0.040→0.038 |
| air | 0.272→0.237 | **0.102 → 0.138** | 0.041→0.044 |
| freeze | 0.272→0.275 | 0.095→0.113 | **0.033 → 0.051** |
| chaos | **0.287 → 0.430** | 0.097→0.100 | 0.045→0.052 |

Each dim moves its expected proxy in the right direction: **stream/voltage → density, air → jumps,
freeze → holds, chaos → density/complexity**. The model learned to obey the radar profile.

## Quality (in-train eval, conditioned on real radar)

onset_F1 0.748, density 0.206 (real 0.217), **crit_adj 0.953** — quality preserved and difficulty
fidelity actually *up* vs the non-radar layered model (~0.77–0.86), since the radar carries
difficulty-correlated information.

## Status / next

**Working knob, modest magnitudes.** Effects are directionally correct at guidance scale 1 (plain
conditioning). To make them *stronger*, apply **classifier-free guidance at inference** — we trained
with the dropout that enables it; CFG extrapolation (`out = uncond + g·(cond − uncond)`, g>1) would
amplify the radar's pull. Not yet implemented (needs a two-pass / dual-cache decode); it's the clear
amplification follow-up.

This also builds the general conditioning-vector that **Step 3 (reference-chart style embedding)**
plugs into: encode a reference chart → latent → concat into `c` alongside difficulty + radar.

Code: `train_radar.py` (CFG dropout), `eval_radar.py` (controllability sweep);
`LayeredTypedChartGenerator` gained `radar_proj`/`null_radar`/`_cond` and a `radar=` arg throughout.
Checkpoint: `checkpoints/gen_radar/best_val.pt`.

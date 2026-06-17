# Stage 1 — Generative Baseline Floor

*Run: 2026-06-17 (seed 42). Phase 1 splits. Val: 786 songs; critic on 128.*

Two deliberately dumb baselines to set the floor before the Stage 2 AR transformer.

| Model | val_loss | onset_F1 | gen_density | crit_exact | crit_adj | crit_mae |
|---|---|---|---|---|---|---|
| n-gram (difficulty bigram) | 0.813 (NLL) | n/a | 0.211 | 0.508 | 0.977 | 0.516 |
| per-frame audio MLP | 1.770 (CE) | 0.053 | 0.014 | 0.219 | 0.484 | 1.484 |

Real val mean density: **0.211**. (val_loss columns are not comparable: n-gram = bigram NLL, MLP = weighted CE.)

## What the floor tells us

1. **Density / difficulty conditioning is EASY.** The audio-blind n-gram reproduces real density exactly (0.211) and the critic reads its difficulty-conditioned charts as the right difficulty remarkably well — 50.8% exact, **97.7% adjacent**. The classifier's difficulty judgment leans heavily on density/groove-radar statistics, which the bigram reproduces by construction. So a high difficulty-fidelity number is cheap; the transformer must clearly *beat* 97.7% adjacent / 50.8% exact to claim conditioning value.

2. **Rhythmic onset alignment is the HARD, unsolved axis.** The n-gram is audio-blind, so its onsets don't align to any specific song (onset_F1 n/a — musically arbitrary). The only audio-conditioned baseline, the per-frame MLP, is terrible at onset_F1 (**0.053**): a single audio frame carries too little signal to localize a step, and even with inverse-sqrt class weighting it collapses toward the empty state (gen_density 0.014 << 0.211). **onset_F1 ≈ 0.05 is the number the Stage 2 transformer must beat** — placing steps in time with the music is the real problem, and it requires temporal context (which both baselines lack).

## Implications for Stage 2

- The transformer's value proposition is **onset alignment via temporal + audio context** — the axis both baselines fail. Track onset_F1 as the headline.
- Difficulty fidelity (critic) is a guardrail, not the goal: must stay near the n-gram's ~0.98 adjacent without sacrificing onsets.
- The per-frame MLP's empty-collapse confirms the design doc's concern about empty-class imbalance — Stage 2 needs class weighting / focal loss or the factorized onset-then-panel objective.

Code: `src/generation/baselines.py`, `experiments/generation_baselines/run_baselines.py`.
Reproduce: `python experiments/generation_baselines/run_baselines.py --data_dir data/ --audio_dir data/`.

# Conditioning to match the source groove — radar beats style

*2026-06-21. `experiments/generation_typed/eval_conditioning_match.py`.* By default the generated groove
profile drifts from the original (generation isn't groove-conditioned). Which conditioning best matches
the source? Compared 3 paths on 6 freeze-selected Hard songs; metric = mean L1 between the generated
chart's groove radar and the source chart's radar (lower = better match).

| approach | stream | volt | air | freeze | chaos | TOTAL |
|---|---|---|---|---|---|---|
| baseline (no conditioning) | 0.06 | 0.09 | 0.21 | 0.34 | 0.12 | **0.81** |
| match_radar (source 5-dim radar, g=1.5) | 0.03 | 0.06 | 0.15 | 0.11 | 0.10 | **0.44** |
| reference_self (source chart via StyleEncoder, g=2.0) | 0.07 | 0.07 | 0.22 | 0.29 | 0.18 | **0.83** |

## Findings
- **`match_radar` wins decisively (0.81→0.44).** Biggest gains exactly where baseline drifts most:
  freeze (holds) 0.34→0.11 and air (jumps) 0.21→0.15. The explicit 5-dim radar directly encodes these
  dims, so CFG steers them.
- **`reference_self` (the "condition on the input chart" path) does NOT help match the profile
  (0.83 ≈ baseline).** The StyleEncoder is a bottleneck (masked mean-pool → one latent) built for STYLE
  TRANSFER of holistic feel; it does not carry the quantitative groove dims (barely moved freeze
  0.34→0.29). Genuine result, not a bug — consistent with `conditioning_step3_style.md` (style transfers
  density only when references differ a lot; here it's self-reference and the radar dims don't transfer).
- **Note:** this measures PROFILE match (radar distance), not FEEL. The style path may still transfer a
  holistic feel the radar can't summarize — a different use case.

## Recommendation
- To **hit a target groove profile** (e.g. groove-validated playtest sets): `--match_radar --guidance 1.5`.
  Now a standing convention in the playtest skill.
- `--reference` / `--reference_self`: reserve for **style/feel transfer**, not profile control.
- g≈1.5 matches; g≈2+ amplifies (and dents density/difficulty since holds are sparse in note-presence).

# Style Fix: Sample the Pattern (playtest-driven)

*2026-06-18. Playtest feedback: generated charts "heavily biased to repeat the previous note, almost always Left."*

## Diagnosis

The pattern (which-panels) head was decoded **greedy** (argmax) — only the *type* was sampled.
Greedy always takes the single most-likely pattern, which is the previous note (a jack) or the
most-frequent panel (Left). Measured (48 val songs) vs real charts:

| decode | L | D | U | R | repeat/jack | entropy | onset_F1 | crit_adj |
|---|---|---|---|---|---|---|---|---|
| greedy (was) | 0.48 | 0.23 | 0.10 | 0.19 | 0.88 | 0.85 | 0.764 | 0.854 |
| **p-sample t1.0** | 0.27 | 0.24 | 0.24 | 0.26 | 0.20 | 2.56 | 0.764 | 1.000 |
| p-sample t1.0 +rep3 | 0.26 | 0.24 | 0.24 | 0.26 | 0.08 | 2.62 | 0.764 | 1.000 |
| **real charts** | 0.28 | 0.23 | 0.23 | 0.26 | 0.20 | 2.37 | — | — |

Greedy = 48% Left, **88% jacks** — exactly the playtest complaint.

## Fix

**Sample the pattern head** (`pattern_sample=True, pattern_temperature=1.0`). Decode-time only,
no retraining — the pattern head was fine, we were just collapsing it with argmax.

- Panel balance → matches real (L/D/U/R ≈ 0.27/0.24/0.24/0.26).
- Jack rate 0.88 → 0.20 = real's 0.20 (no repetition penalty needed; a penalty pushes it *below*
  real).
- Entropy 0.85 → 2.56 ≈ real 2.37 (varied patterns).
- onset_F1 unchanged (0.764 — onset placement is a separate head); crit_adj *improves* (varied
  realistic charts read as the right difficulty better than degenerate Left-jack spam).

Also added `generate(repetition_penalty=...)` to optionally discourage jacks further (not needed
at t=1.0). The earlier greedy pattern was a holdover from maximizing onset_F1 against the
reference; for *style*, sampling is strictly better here at no F1 cost.

## Recommended decode (final, typed generator)

```
generate(onset_threshold=<match density>, type_sample=True, type_temperature=0.4,
         hold_aware=True, pattern_sample=True, pattern_temperature=1.0)
```

`export_typed_samples.py` now uses this by default. Re-exported samples show balanced panels and
real-matching jack rates — stylish, varied, playable.

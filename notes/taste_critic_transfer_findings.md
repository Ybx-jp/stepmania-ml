# Taste-critic re-evaluation against the LATEST decode machinery — findings

*2026-06-26. The PREREQUISITE gate from `geometry_feasible_region.md` §V2 (can the realism critic be the
goodness signal for a best-of-N sweep on TODAY's generations?). Harness:
`experiments/realism_critic/eval_taste_current.py` (mirrors `scripts/generate.py`'s decode path exactly).
Control: the original `experiments/realism_critic/eval_taste.py` (old `gen_stage1` machinery).*

## Verdict
**The critic's taste ranking TRANSFERS to the current decoder — as a SEPARATOR, not a graded scorer.** The
prerequisite is cleared enough to proceed, with one calibration caveat that bears directly on best-of-N.

## The numbers (critic P(real), mean over 64 val songs)
| Rung | OLD `gen_stage1` (control) | NEW `gen_motif_full_fixed` + governor + manifold |
|---|---|---|
| REAL human chart | 0.823 | **0.823** (identical) |
| BASE generation | 0.290 | **0.269** |
| CHAOS generation | 0.003 | **0.228** |

- `REAL > BASE > CHAOS` holds at n=64 on current machinery. Per-song: **REAL > BASE on 86%**, REAL > CHAOS 91%.
- **Control reproduces the 2026-06-20 number EXACTLY** (0.823 / 0.290 / 0.003) → the scoring harness is faithful;
  the NEW numbers are a real measurement, not a harness artifact (experiment-design Rule 7 cleared).
- REAL=0.823 reproduces identically in BOTH runs — expected, the REAL rung is generator-independent (it scores
  the human chart); it's the built-in scoring-path sanity check.

## What's GOOD news for the v2 best-of-N plan
1. **The density-OOD worry is largely allayed.** `corr(P(real), realized_density) = -0.09` across realized density
   0.05–0.44. The score is NOT a density proxy — the specific fear in the v2 note (critic only learned taste in a
   ~0.2 band) does not show up as a density-tracking artifact across this range.
2. **The critic still separates human from generated on current outputs** (REAL 0.82 vs BASE 0.27; 86% per-song).
   It did not go blind on the new decoder's distribution.

## What's the REAL caveat (and it's NOT the one we expected)
**The critic is near-BINARY — a strong separator, a weak grader.** Score distribution (n=64):
| Rung | <0.1 | mid 0.1–0.9 | >0.9 |
|---|---|---|---|
| REAL | 9% | 14% | 77% |
| BASE | 59% | 23% | 17% |
| CHAOS | 58% | 30% | 12% |

Only 14–30% of scores land in the discriminating middle; the rest pile on the 0/1 rails. **For best-of-N
RERANKING this is the binding constraint, not density-OOD:** the N candidates are all "generated" and will mostly
cluster near the LOW rail, so "keep the highest P(real)" often chooses among near-tied ~0.0 values. Best-of-N adds
value only when some candidate jumps to the high rail; otherwise the signal is thin. **Likely needs the low end
spread out (temperature-rescale / recalibrate the critic) before it can rank among generated candidates.**

Secondary: **9/64 generated BASE charts scored ABOVE their own REAL** (false ceiling). Fine for relative ranking,
but tempers "P(real) = quality".

## Harness caveat that shaped the read (conditioning-mechanics checklist item 7)
The CHAOS rung was a **weak manipulation**: manifold `chaos=q0.85` on a song that doesn't afford chaos produces a
chart ≈ BASE (no 16ths to add). So `BASE > CHAOS` per-song is only **47%** (coin-flip); the aggregate gap is
carried by the minority of chaos-affording songs. **The load-bearing contrast in this experiment is REAL vs BASE,
not the chaos rung.** The old CHAOS=0.003 was an OOD mean-pin smear (an easy "obviously fake" target); the new
in-distribution chaos is correctly scored like a normal generation — a feature, not a failure. A proper "bad" rung
for future runs: filter to chaos-affording songs (`--groove_select chaos`) or use a deliberate mean-pin OOD chart.

## Methodology note (for the log)
The n=8 SMOKE looked like a critic collapse (base<chaos, generated>real) — it was small-sample noise; n=64 showed
clean REAL>BASE>CHAOS. Don't read a saturated near-binary metric off 8 samples.

## Follow-up: the CHAOS jump (0.003→0.228) is the CONDITIONING redesign, not the model upgrade
*2026-06-26, `experiments/realism_critic/eval_chaos_mechanism.py`. Prompted by the user's reframe: if P(real)
carries taste signal, the CHAOS rung rising 0.003→0.228 old→new says the NEW chaos conditioning is more tasteful
— matching their play experience. Caveat that motivated the isolation: old→new changed the WHOLE stack (model,
guidance, governor, AND the chaos mechanism), so the delta wasn't yet attributable to the conditioning.*

Isolation: decode BOTH chaos requests with the SAME current model (`gen_motif_full_fixed`), same guidance (1.5),
same governor, scored by the same critic. Only the request varies:
| Arm (model HELD FIXED) | P(real), n=64 | songs >0.1 |
|---|---|---|
| MEANPIN — OLD request (global-mean radar, chaos dim pinned 0.9, OOD) | **0.028** | 3% |
| MANIFOLD — NEW request (`build_target('chaos=q0.85')`, in-dist) | **0.228** | 42% |

- MANIFOLD > MEANPIN: Δ **+0.200**, **73%** per-song.
- **The model upgrade does NOT explain it:** the OLD mean-pin request scores **0.028 on the NEW model** —
  essentially its old-pipeline 0.003. Decoding garbage-request with a better model still gives garbage. The
  tastefulness gain is attributable to the **conditioning redesign** (mean-pin → manifold conditional-fill +
  ellipsoid projection), holding the model fixed.
- **Mechanism of the gain (precise):** the manifold wins largely by REFUSING to smear — realized chaos is only
  ~0.09 (capped to what each song affords; ~0.00 on non-affording songs). "More tasteful" = "stopped flooding
  off-grid 16ths," not "added tasteful chaos spice." Exactly the failure the conditional-fill was built to prevent.
- Caveat: still the critic's SEPARATOR signal, not a playtest — but it now triangulates three ways (design theory,
  the isolated number, the user's hands-on experience). The 17 songs where MEANPIN≥MANIFOLD are both-near-floor ties.

## Next gate before best-of-N ships as the sweep's inner judge
Grounding-in-the-artifact (experiment-design Rule 8) is NOT yet done: confirm that HIGH-P(real) candidates actually
PLAY better than low-P(real) ones *among current-machinery candidates for the same song/setting* (by ear). The
transfer result says the critic ranks REAL>generated; it does NOT yet prove it ranks good-generation >
mediocre-generation, which is what reranking needs. Pair that with the low-end recalibration above.

## Pointers
- Harness: `experiments/realism_critic/eval_taste_current.py` (NEW), `eval_taste.py` (control; patched to
  `strict=False` load so the current `LayeredTypedChartGenerator` class accepts the older `gen_stage1` weights).
- Theory/scope this feeds: [[geometry_feasible_region]] §V2 (region-of-good-settings, the best-of-N sweep plan).
- Critic provenance: `notes/stage2a_critic_findings.md` (v2 corrupted-real critic, AUC 0.964).

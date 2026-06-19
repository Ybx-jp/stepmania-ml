# Conditioning Step 2b: classifier-free guidance at inference

*2026-06-19. Amplifies the trained groove-radar knob from Step 2.*

Step 2's radar conditioning worked but was weak at plain conditioning (guidance scale 1). We trained
with classifier-free-guidance dropout, so we can now extrapolate at inference. Added
`generate(guidance_scale=g)`: at each step run the model twice — conditioned (with radar) and
unconditioned (null radar, a parallel KV-cache) — and blend `out = uncond + g·(cond − uncond)` for
the onset, pattern, and type logits. `g=1` = off (single pass); `g>1` pushes toward the radar target.

## Amplification (24 val songs; vary one radar dim 0.1→0.9, others at mean)

low → high proxy response, g=1 vs g=3:

| dim → proxy | g=1 | g=3 |
|---|---|---|
| stream → density | 0.265 → 0.333 | **0.231 → 0.456** |
| voltage → density | 0.264 → 0.333 | **0.235 → 0.499** |
| air → jump | 0.102 → 0.138 | **0.117 → 0.211** |
| freeze → hold | 0.033 → 0.051 | **0.023 → 0.095** (~4×) |
| chaos → density | 0.287 → 0.430 | **0.295 → 0.736** |

CFG makes every radar dimension a strong knob (~3–4× the plain-conditioning effect). `guidance_scale`
is a tunable strength dial; g=2–3 is a good range, higher overshoots (chaos→0.74 density is extreme).

## Cost / notes

CFG runs a second decoder pass per step (a parallel null-radar cache), so generation is ~2× slower
when `guidance_scale != 1`. Worth it for strong control; default stays g=1 (no overhead). The onset
head guidance is one extra non-AR pass (cheap); the per-step pattern/type guidance is the 2× part.

The radar profile knob is now both **learned** (Step 2) and **strong** (Step 2b). Conditioning
roadmap: Step 1 (pattern prefs) ✓, Step 2 (radar) ✓, Step 2b (CFG) ✓; remaining: Step 3 reference-
chart style embedding.

Code: `generate(guidance_scale=)` + dual-cache in `LayeredTypedChartGenerator`; `eval_radar.py
--guidance`.

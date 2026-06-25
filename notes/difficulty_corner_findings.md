# Difficulty-corner feasibility map — the EASY corner is HEALTHY (release gate, offline)

**Date:** 2026-06-25. **Script:** `experiments/generation_typed/diag_difficulty_corner.py`. **Model:**
`gen_motif_full` (Track-A highres, 42-dim). **Why:** release-readiness — we stress-tested the HARD/chaotic corner
exhaustively but barely the EASY one (the user: "make sure it can still generate easy songs lol"). The
[[geometry_feasible_region]] framing says a release = boundary-walked; this is the low-difficulty boundary.

## Design (controlled — audio held fixed, only difficulty conditioning swept)
6 audios, each generated at TARGET difficulty Beginner→Hard with BASE conditioning (no radar/motif, so the
difficulty axis is isolated), playability ON (max_jack=1, hold-aware, no-jump/cross-in-hold, pattern_temp 0.7).
Density pinned to the manifold's per-difficulty base `E[density|radar=diff-mean]`. Two readouts:
- **nat_dens@0.5** = intrinsic density at a FIXED threshold (mean p_onset>0.5) — does the model itself WANT fewer
  notes when told 'easy' (vs "Hard minus notes", which would keep density flat and only thin at decode)?
- **structure** at the deployed density — quarter/8th/16th-offbeat shares, jump frac, figure-variety entropy.

## Result
```
difficulty  nat_dens@0.5  tgt_dens  gen_dens  quarter   8th  off16  jump  fig_ent  top
Beginner        0.095       0.088     0.087     1.00   0.00   0.00  0.01    1.78   trill
Easy            0.222       0.169     0.165     1.00   0.00   0.00  0.02    1.69   trill
Medium          0.242       0.254     0.251     0.89   0.11   0.00  0.03    1.71   trill
Hard            0.400       0.320     0.313     0.69   0.31   0.00  0.09    1.26   trill(+jump)
```

## Verdict — POSITIVE, the easy corner is in the feasible region
1. **The model KNOWS easy = sparse (not "Hard minus notes").** nat_dens@0.5 rises monotonically
   0.095→0.222→0.242→0.400 with the difficulty token alone (same audio). The difficulty→density relationship is
   internalized in the onset posteriors, not faked at decode. This is the key result.
2. **Backbone SURVIVES and strengthens toward easy.** Quarter share 1.00/1.00/0.89/0.69; 8ths first appear at
   Medium, syncopation 0 throughout (expected at base). Easy/Beginner are 100% on-grid quarter — maximally
   coherent, the opposite of a sparse-scatter failure.
3. **Density monotone & on-target** (gen≈tgt). **Jumps scale with difficulty** (0.01→0.09 — easy has ~none,
   appropriate). **Figure variety holds** (entropy ~1.7 at easy, not collapsed to a single degenerate figure).

## Caveats
- **Low song diversity:** `collect()` took the first 6 val samples = effectively 2 distinct songs (Deja loin ×4
  difficulty-charts + かぐや姫 ×2). The sweep is still valid (each AUDIO swept across all 4 target difficulties),
  but breadth is thin — rerun with `--songs` larger + a diverse pick before treating the numbers as population-level.
- **`top_fig`="trill" everywhere** is partly the figure-detector's residual trill lean on these songs; entropy
  staying ~1.7 is the load-bearing number (variety preserved), not the modal label.
- **Offline only** — this answers STRUCTURE (backbone/density/variety). "Does an easy chart read as *deliberately*
  musically easy" is a FEEL question → needs a playtest of the Beginner/Easy corner (the natural next step).
- Detector uses the H19-correct attacks-only onset tokens (`(chart!=0)&(chart!=3)`), so trill/jack figures here
  are not hold-tail-inflated.

## Next
- [ ] Diverse rerun (`--songs 16+`, varied titles) to confirm population-level.
- [ ] EASY-corner PLAYTEST set (the FEEL gate) — generate Beginner/Easy charts (the beloved songs at Easy needs a
      `--target_difficulty` exporter override; genuine Easy-source songs work without it). Does easy feel deliberate?
- [ ] Optional deeper proxy: taste-critic P(real) of generated easy vs real easy (the critic over-rejects HARD,
      but easy is in-distribution so it should be trustworthy at this corner) — confirms coherence beyond grid stats.

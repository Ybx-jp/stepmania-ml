# Chart-representation integrity audit (pre-retrain) — found + fixed a note-dropping converter bug

**Date:** 2026-06-25. **Why:** before the H19 retrain (which refits `cache/motif_basis.npz` and the motif
training targets, AND rebuilds `cache/samples_v3` — all from the typed chart tensor), sweep for OTHER
symbol-misinterpretation / lossy-construction bugs in the `{0 none,1 tap,2 hold_head,3 tail,4 roll_head}`
pipeline. H19 (figure detector counted hold releases) was one; this looked for siblings + construction loss.
**Tool:** `experiments/generation_typed/diag_repr_integrity.py` (chart-only, parses sampled real .sm, fast).

## ★ HEADLINE BUG (FIXED) — typed converter dropped sub-16th notes via zero-overwrite
`convert_to_tensor_typed` assigned **every** panel of **every** line **unconditionally**:
`arr[ts, p] = TYPED_SYMBOLS.get(line[p], 0)` — including writing **0** for empty panels. On the fixed 16th grid
(`ts = floor(beat*4)`), when two sub-16th rows (24th/32nd/48th) quantize to the same cell, the **later row's
empty panels overwrote the earlier row's notes with zero**. The typed path silently lost those notes; the binary
/radar path (`convert_to_tensor_extended`, which writes *conditionally* — `if char=='1': arr=1.0`) kept them.
- **Magnitude:** typed_attacks − binary_onsets was **mean −42 frames/chart** (and never positive — binary was
  always a superset). Confirmed the lost frames were typed-EMPTY (not tails) and occurred on **hold-free** charts
  → not a hold issue, purely the zero-overwrite. Worst single chart in the debug sample: "the fifth act" lost
  **90** onset frames; "Desert Rose" 24.
- **Fix:** sticky write — `sym = TYPED_SYMBOLS.get(line[p],0); if sym: arr[ts,p] = sym` (matches the binary
  path). Post-fix the gap collapses to **−2.2 frames/chart** (residual = genuine note-vs-note collisions +
  zero-length holds, the real grid floor), tap share 0.192→0.211 (recovered notes). Regression test
  `test_typed_converter_sub16th_collision_keeps_note`. 35/35 gen+parser tests pass.
- **WHY IT MATTERS FOR THE RETRAIN:** the typed tensor is the source of truth for the motif basis, the motif
  training targets, AND the cached training charts (`cache/samples_v3`). So the deployed `gen_motif_full` trained
  on note-dropped charts, and `cache/motif_basis.npz` was fit on them. **The cache rebuild that the retrain
  already requires is exactly what propagates this fix into training.** Bug + cache-recompute are the same event.

## Clean / not-a-bug
- **MIRROR** (`motif_codebook._MIRROR`, figure canonicalization): valid permutation, involution `M(M(x))=x`,
  L↔R correct, U fixed → **PASS**. Figure folding is sound.
- **Groove-radar density NOT tail-inflated:** radar runs on the binary tensor where a hold is a single head cell
  (tail '3' never written), so `stream/voltage = chart_tensor.sum()` counts attacks correctly. (The H19 tail
  bug was specific to the *figure* detector's `onset_tokens`, already fixed.)

## Known limitations / latent (NOT fixed — flagged)
- **Roll-head ('4') dropped from binary/radar:** `convert_to_tensor` and `convert_to_tensor_extended` only handle
  `'1','2','3'` — a roll head is invisible to the radar, while `convert_to_tensor_typed` keeps it (symbol 4).
  **ZERO impact today** (0/233 charts contain a roll; roll_head share 0.0000). Left as-is — fixing needs a cache
  rebuild for no current benefit. If a roll-bearing pack is ever added, add `'4'` handling to `_extended` (and it
  rides the next cache rebuild). `convert_to_tensor` (Phase-1 binary) is the FROZEN classifier path — do not touch.
- **16th-grid quantization (fundamental, not a code bug):** 3.11% of attacks are sub-16th (off-grid); 0.04% are
  lost to true note-vs-note (ts,panel) overwrite; 1375/233≈6 zero-length holds per chart (head+tail in one cell →
  orphan tail → `pair_holds` drops them). These are the cost of the fixed 16th resolution; fully removing them
  needs a finer grid (changes `timesteps_per_beat`, audio hop, every cache) — out of scope, a separate big change.
- **H19-sibling tail-counters that are LOW/NO impact:** `typed.onset_mask` and `count_crossovers` use `!=0`
  (count tails), and `evaluation.onset_density_metrics` binarizes `>0.5` (counts tails) — but these feed the
  *factorized* track / quality readouts, not the gen_motif_full conditioning or training targets. Flagged, not
  fixed; revisit if any becomes load-bearing.

## Pre-existing unrelated failure (not caused here)
`tests/test_parser.py::test_phase1_song_length_rejection` fails with "No BPM events found" (empty `timing_events`
in the test fixture) — fails identically on the pre-fix parser (verified via stash). Separate test-setup issue.

## RETRAIN CHECKLIST (carry-forward — the two fixes that need the cache rebuild)
1. **H19** — refit `cache/motif_basis.npz` with attacks-only `onset_tokens`, retrain `gen_motif_full` on clean
   (non-tail-inflated) motif targets. [[playtest_log]] H19 action.
2. **THIS bug** — rebuild `cache/samples_v3` with the sticky `convert_to_tensor_typed` so training charts regain
   their sub-16th notes; the basis refit + target derivation then run on the corrected charts.
   Both land in one cache-recompute + retrain.

# ⚠️ Dataset sample-cache is INDEX-KEYED — a footgun for subset/`--match` probes (found 2026-06-28)

## The bug
`StepManiaDataset` caches processed samples at `cache_dir/sample_{idx:06d}.pt` (`_get_cache_path`,
`src/data/dataset.py:438`), keyed by INTEGER INDEX with **no song-identity check**. `_load_from_cache(idx)` returns
the file at that index blindly; `warm_cache` SKIPS already-cached indices (`_is_cached`). So if two runs share a
`cache_dir` but build the dataset over a DIFFERENT file set/order, index `i` maps to different songs across runs,
and the second run silently reads the FIRST run's features for index `i`.

## How it bites
Full-set runs (training, exporter, `buffered_sectional` with `vf[:N]` in standard split order) all use the SAME
ordering → index→song stable → cache CORRECT. But any probe that **subsets/reorders** files (`--match`,
`--song_filter`, a hand-filtered `vf`) re-indexes from 0 and collides with the full-set's `sample_00000x.pt`.
Observed: the phrasing probe (`--match "high school love,kneeso,deja loin"`) served **kneeso's audio for HSL**
(identical feature sums; HSL length 1168 instead of its true ~1383).

## Blast radius (this session, checked)
- **`buffered_sectional` / H11 — UNCONTAMINATED** (verified: all 6 selected songs' cached feature-sums == fresh
  extraction; standard `vf[:240]` order matches how the cache was warmed). H11 conclusions stand.
- **`probe_arc_lag` — HSL primary OK** (ran on correct-length HSL audio; "centered boxsmooth is zero-phase" is
  architectural regardless), **kneeso SECONDARY contaminated** (showed T=1168, true 1414) — no conclusion rested
  on it (already flagged "don't over-read the contrast song").
- **`probe_phrasing_coherence` — FIXED** (now `cache_dir=None`).

## The safe workaround (use NOW for any subset probe)
Pass `cache_dir=None` to `create_datasets`/`StepManiaDataset` → fresh on-the-fly extraction, collision-proof
(cheap for a handful of songs). The `generation-defaults` skill probes should follow this for `--match` work.

## The proper fix (NOT yet applied — touches core infra used by training; needs a decision)
Make the cache identity-aware. Lowest-risk: store the song key (`chart_file`+`difficulty_value`) IN the cached
dict and verify it in `_load_from_cache`; on mismatch, recompute (backward-compatible — old entries lacking the
key just recompute once). Alternative: key the path by a hash of the song identity instead of the index. Either
invalidates nothing for correctly-ordered full-set caches but closes the subset-probe footgun.

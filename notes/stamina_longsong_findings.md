# Stamina breathing arc on LONG songs — length-mis-scoped at the CEILING, but doesn't bite the CHART

*2026-07-11, branch `explore/taste-critic-quality-resolution`. Thread: taste-critic-quality arc (lineage
`taste-critic-arc.md`). Probes in scratchpad (`probe_stamina_longsong.py`, `density_by_section.py` + the 9-arm
generation). User hypothesis: the stamina system (tuned on the ≤130s training corpus) neuters choreography on the
longer personal songs.*

## Mechanism (code-confirmed, corrects a misremembering)
There is **no "thirds" split.** The Stage-3 breathing arc (`typed_model.py:687-705`) smooths the onset-energy
envelope (`p_onset`, box `stamina_breathe_win·f16`) then **z-normalizes it over the ENTIRE song's valid frames**
(`z=(env−mean_wholesong)/std_wholesong`); ceiling `= base·(1+breathe·z)`, clamped `≥ floor·base` (0.4). Stage-2
`E_slow` is one global accumulator (decay `exp(−1/(tau·subdiv))`, subdiv-correct).

## Findings
1. **The whole-song z-normalization IS length-mis-scoped** (`probe_stamina_longsong.py`, onset-head only, no AR
   loop): **corr(song duration, global-vs-local ceiling divergence) = +0.827**. Short train songs (~93s) diverge
   0.10 from a local-rolling-z ceiling; long personal songs (~207s) diverge **0.38** (Bye Bye 364s → 0.58). On a
   multi-section song the whole-song mean is a MIXTURE, so a quiet breakdown reads deep-negative z → ceiling floored.
   BUT the "extra sections floored by global" was NOT confirmed (corr +0.21; global floors slightly FEWER on avg) —
   it's a **redistribution / mis-placement** of thinning, not uniform over-thinning.
2. **It does NOT bite the realized chart** (`density_by_section.py`, the fair AR-loop test — 3 arms OFF/GLOBAL/LOCAL
   × Calling/Switch/Bye Bye, Hard, manifold density ~0.107). Density-by-section is near-identical across arms
   (corr→real human chart: Bye Bye 0.68/0.74/0.67 — deployed GLOBAL is the BEST match); mean note counts within ~4%
   (Bye Bye 1342/1295/1328). Stamina only sheds ~4% of onsets at this density, redistributed near-identically
   regardless of normalization → the large ceiling divergence doesn't propagate. **Necessary ≠ sufficient**
   (exp-design Rule 7/9): the cheap ceiling probe looked confirmatory, the fair test overturned its DIRECTION.
3. **BY-EAR (playtest_log 2026-07-11) EXONERATES stamina.** OFF was the WORST arm (Bye Bye bridge "disjointed");
   GLOBAL "tasteful edit" on Switch. Turning stamina off made charts WORSE, the opposite of "neuters." User reframe
   (correct): stamina "wasn't meant to make charts drastically different, just quiet the excess." **GLOBAL stays
   the default.**
4. **`local-z` (the fix) is a MILD PARTIAL win on the QUIET-SECTION axis only** (user correction): it recovered the
   Bye Bye bridge + some quiet-section notes (both LOCAL and GLOBAL beat OFF there; LOCAL the one that recovered it).
   Offline (`harm_offline.py`): local-z +4-5% density in the gated quiet regions. So the user's length instinct was
   **right on the quiet-section axis, wrong on the "stamina neuters density" axis.** Kept as a mild lever for
   defect #2 (quiet under-charge), NOT reverted; NOT the default.

## Code
Non-breaking `stamina_breathe_local_win` (default None = deployed whole-song z) added to `generate()`
(`typed_model.py`) + `scripts/generate.py --stamina_breathe_local_win`. A rolling-window z-normalization; inert
unless opted in.

## Open
Whether local-z's quiet-section benefit is worth productionizing depends on the defect-#2 (harm_calib) by-ear
verdict — the two levers target the same axis. See `taste_critic_v2_findings.md` + `taste-critic-arc.md`.

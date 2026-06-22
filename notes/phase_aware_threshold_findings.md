# Phase-aware onset threshold — unlocking the model's own 16ths (chaos)

Context: chaos radar = 0 because the generator places ~no 16ths. Localized (diag_no16ths) to: the model's
16th onset confidence exists (p_on@16th ~0.4 with the high-res feature) but a single density-quantile
threshold buries it — 16th frames are out-budgeted by the more-confident 8ths (~0.55), so only 0.4–1.3%
of placed notes land on 16ths (real ~4–5%). See [[chaos_mechanism_plan]].

## The early-stop detour (and what it taught us)

User asked the right question: v4 "early stopped at epoch 4" but the BEST checkpoint was epoch 1 — best
at epoch 1 is suspicious when you're trying to learn a new representation.

- **`val_total` is blind to this experiment.** It's ~88% pattern loss; the onset head we change is ~7.5%,
  and even val_onset is dominated by abundant quarter/8th/empty frames, so the rare 16ths we optimize
  barely move it. It was flat (1.264→1.271) across the whole run while 16th behavior changed underneath.
- v5 added a **target-aligned selection metric** (`val_f16`: precision+recall at 16th-phase frames, at the
  val-set operating threshold). On THAT metric, epoch 4 beat epoch 1 (0.253 vs 0.173, +46%). So
  early-stopping on aggregate loss DID stop short of the per-metric optimum.
- **BUT `val_f16` itself was a flawed proxy** — I computed it with a *global* threshold; real generation
  uses a *per-song* threshold. They diverge. Under real decode, the val_f16-selected checkpoint (v5,
  epoch 4) actually placed FEWER 16ths than v4 (0.4% vs 0.8%) with a lower 16th posterior (0.381 vs
  0.415). More training sharpened the abundant 8ths and pushed 16ths FURTHER below the cut.

**Lessons:** (1) when optimizing a rare subpopulation, early-stop/select on a metric aligned with that
subpopulation, not aggregate loss; (2) and make sure that metric's *thresholding matches decode* or it
becomes its own artifact. `val_f16` with a per-song threshold may still be worth it — parked, not killed.
Conclusion that mattered: **training longer is not the lever; the decode threshold is.**

## The fix: phase-stratified allocation (decode-time, no retrain)

Keep the SAME note budget N = (p_on > tau).sum() the single threshold gives, but split N across the three
16th-grid phase bands (quarter t%4==0, 8th t%4==2, 16th t%4 in {1,3}) by a target note distribution
(default real's 0.707/0.252/0.041), picking the top-p_on frames WITHIN each band. Each band gets its own
implicit threshold, so the model's own 16th ranking chooses which 16ths instead of losing globally to 8ths.

Implemented as `generate(onset_phase_alloc=(q,8th,16th))` (src/generation/typed_model.py) — prototyped
first via `onset_override` (no model change) in `experiments/generation_typed/diag_phase_threshold.py`,
then wired in with a unit test (`test_onset_phase_alloc`) and an exporter flag (`--onset_phase_alloc`).

### Result (diag_phase_threshold.py, 20 songs; identical decode)
```
  source / mode               quarter    8th     16th   density  crit_adj
  REAL                          70.2%   24.7%    5.1%
  v4 / global (single tau)      62.8%   35.9%    1.3%    0.270    1.000
  v4 / alloc  (phase-aware)     70.5%   25.4%    4.1%    0.267    1.000
  v5 / global                   70.1%   29.0%    0.8%    0.270    1.000
  v5 / alloc                    70.5%   25.4%    4.1%    0.267    1.000
```
16ths 1.3%→4.1% (real-matched), **same density**, crit_adj holds at 1.0. Export spot-check: Deja loin
(the original "zero 16ths" case) 0.0%→4.2% at identical note count (n=384).

### What the metrics CANNOT settle (→ playtest)
- crit_adj=1.0 only says the redistribution didn't break difficulty — NOT that 16ths land on musically
  right frames.
- `alloc` forces the target share on EVERY song, even ones with little real 16th content. Within a song it
  picks the most-confident 16th frames (good), but the flat per-song budget could smear 16ths into
  sections that shouldn't have them. Only play-feel decides musical-vs-noise.

## The quota WAS smearing — per-song diagnostic (diag_song_chaos.py)

User (correct): a flat per-song quota = smearing. A song that deserves real chaos is capped; a calm song
gets spurious 16ths. The per-song VARIATION is the chaos signal. Tested whether the model even knows which
songs deserve chaos (no generation needed — onsets are decided up front). 60 songs, gen_highres_v4:

```
  within-16th-band discrimination (AUC, p_on vs real-16th-note): 0.742   (0.5 = chance)
  per-song real 16th-rate: mean 4.3%  std 6.6%  range [0.0, 24.0]%

  signal       Spearman(real16,.)   mean16%  std16%  range16%
  raw pon16          0.289
  global             0.459            0.9      1.4   [0.0,  5.2]   best corr, far too timid
  alloc (quota)      0.256            4.2      0.1   [3.8,  4.7]   FLAT = smearing (corr DROPPED)
  calib              0.389            3.6      3.9   [0.0, 24.4]   variable, full real range  <-- WIN
  adaptive(abs bar)  0.344            3.9      4.0   [0.0, 16.9]
```

Findings:
- **The quota (`alloc`) is provably smearing**: std 0.1% (constant), and it DEGRADED corr 0.459→0.256.
- **The model DOES discriminate**: frame-level AUC 0.742 (which 16th frames within a song — the high-res
  feature working; old off-beat AUC was ~0.53). Song-level corr is moderate (0.29–0.46).
- **`calib` (per-phase LOGIT offset b16≈0.19 + per-song threshold) is the win**: variable (std 3.9),
  spans the full real range [0,24%], best volume-matched corr (0.389). It's per-song NORMALIZED.
- **`adaptive` (absolute cross-song bar) is WORSE** than calib (corr 0.344, range caps at 16.9%): an
  absolute bar conflates LOUD songs with CHAOTIC songs (a dense song clears the bar on volume, not 16th
  prominence). calib's per-song threshold normalizes that confound out — which is why it reaches 24%.
- **The ceiling is the MODEL, not decode.** Every method tops out at song-corr ~0.39–0.46. Strong
  frame-local (0.742), weak song-level — the frame-local-feature limitation (= [[h5]] global structure).
  No decode trick exceeds ~0.46; raising it needs a global-structure feature/training change.

Shipped `onset_phase_calib=(b8,b16)` in generate() (+ exporter `--onset_phase_calib`, test). Prefer it
over `onset_phase_alloc` (kept but documented as smearing). Decision (user): BANK calib, then pivot to the
model to raise the song-level signal.

## Playtest handoff (installed) — VARIABLE chaos
A/B, gen_highres_v4, same 6 rich Hard songs: `~/sm-generated/chaos_calib` (--onset_phase_calib 0,0.19) vs
`~/sm-generated/chaos_global` (single threshold). Per-song 16th% (calib): Pound the Alarm 12.0, Dancing
lovers 7.4, First-of-the-Year 7.4, IN BETWEEN 6.5, Taylor Swift 1.0, Deja loin 0.5 (global: 0–2.9 flat,
timid). The variation is real — chaotic songs get chaos, calm songs stay calm.

Question for the hands: does the VARIABLE chaos feel right — chaotic songs musically busier, calm songs
clean — and are the 16ths (where placed) on-the-music (AUC 0.742 says they should be)? Deja loin reads as
calm (0.5%) — does that match the song? If the targeting feels off (wrong songs get chaos), that's the
song-level ceiling (corr 0.4) → the model/feature pivot. See [[playtest_log]].

(Superseded earlier handoffs: phase16_single/phase16_alloc were the flat-quota A/B — quota is smearing.)

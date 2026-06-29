# Onset-head phrasing-coherence diagnostic (2026-06-28)

**Probe:** `experiments/generation_typed/probe_phrasing_coherence.py` → `outputs/phrasing_coherence.png`
**Goal (user):** characterize whether the DEPLOYED onset head derives its OWN coherent choreographic phrasing,
referenced to MUSICAL EVENTS (audio), NOT to a real chart's onsets (fidelity is a faint sanity band only). The
measure-before-build step for a learned onset *phrase calibrator*: per axis, does the signal already exist in
`p_onset` to AMPLIFY, or must the calibrator ADD it? No retrain, no generation; deployed native onset path +
`onset_phase_calib=(0,1.0)`. Songs: HSL + japa1(kneeso) [complaint songs, Rules 5/11] + Deja loin [contrast].

⚠️ **First run was contaminated by a dataset cache bug** (see `notes/cache_index_bug.md` / below). Re-run with
`cache_dir=None` (fresh extraction). Numbers below are the VALID re-run (3 distinct songs).

## Results (VALID run)
| song (bpm) | 1 boundary-snap (width / lag) | 2 burst-in-quiet corr(perc/harm), p_calm→p_burst | 3 tail over-run | 4 perc↔harm range |
|---|---|---|---|---|
| Deja loin (164) | 27f (1.7meas) / **lag +46f** | −0.04 / −0.06, 0.43→0.43 (flat) | 0f | 1.26 (fluid) |
| **HSL (180)** | 19f (1.2meas) / −13f; real **9f** | +0.03 / **−0.16**, 0.43→0.41 | 0f (p_after 0.32) | 1.13 (fluid) |
| kneeso (185) | 25f (1.6meas) / +9f | **+0.35 / +0.31**, 0.30→0.47 | 0f (p_after 0.17) | 1.34 (fluid) |

## Reading (directional — 3 songs, Rule 11; the sharpest signal is on HSL = the right population, Rules 5/11)

- **Axis 2 (burst-in-quiet) = the clearest gap, calibrator must ADD.** On HSL (melodic/quiet song) the head's
  `p_onset` ANTI-correlates with the harmonic onset in quiet phrases (corr_harm **−0.16**, p_burst < p_calm) — it
  does NOT allocate for a sparse melodic/piano event in a quiet passage. This is the H-onset-perc-bias
  (melodic under-placement) showing up exactly where the user predicted ("empty during a vocal fill"). Deja is
  flat (doesn't go empty: p_calm 0.43, doesn't respond). kneeso responds well (perc +0.35, harm +0.31, p_burst
  0.47≫0.30). So quiet-phrase harmonic-burst sensitivity is WEAK/absent on the melodic songs = headroom to ADD.
- **Axis 4 (perc↔harm fluidity) = signal EXISTS, calibrator AMPLIFIES.** All three swing (range >1.1; the plot's
  right panels visibly alternate perc-lean↔harm-lean within each song). The head already rebalances emphasis
  within a song in its posterior (means HSL +0.21 / Deja −0.15 / kneeso +0.34 → not globally perc-biased,
  consistent with `diag_breathe_energy`). The within-song signal to sharpen is already there.
- **Axis 1 (boundary-snap) = MIXED, modest headroom.** Model allocation transitions over ~1.2–1.7 measures
  (19–27f). The breathe envelope is ~19–34f, i.e. ≈ p_onset's OWN width — so the governor's boxsmooth is NOT the
  main smearer; the raw onset allocation is already ~1.2–1.7meas wide. vs real: on HSL real is SNAPPY (9f) while
  model is 19f (~2× wider); Deja's allocation LAGS the boundary +46f (~3 beats late). So boundaries are wider /
  sometimes laggy than real — some headroom to sharpen.
- **Axis 3 (clean-tail) = NOT reproduced at deployed τ here (needs the actual over-run songs).** Realized onsets
  end before audio-end on all 3 (over-run 0f). BUT the POSTERIOR stays elevated past audio-end (HSL p_after 0.32,
  kneeso 0.17) — the head's confidence doesn't cleanly zero at music-end; τ is what's holding back the over-run.
  So the user's by-ear "extra measure" is LATENT in the posterior (would surface at lower τ / other songs) but not
  realized on these three. Caveat: the audio-end detector (energy>10%) may be conservative. Inconclusive — needs
  the songs where the user heard it.

## STEP 1 (2026-06-28) — hand-crafted "sparse-harm-in-quiet" calibrator: MECHANISM VALIDATED (posterior-only)
The cheapest one-change test toward the learned calibrator (experiment-design Rule 6): add a per-frame onset
LOGIT offset `harm_gain · quiet_gate[t] · harm[t]` (un-bury a sparse harmonic onset in a quiet phrase — the
mirror of the head's existing sparse-perc response; `probe_phrasing_coherence.py --harm_gain`). No retrain, no
generation. Dynamic range confirmed (Rule 11) on HSL: corr_harm −0.16 → −0.06 (g5) → **+0.03 (g10)** → +0.19
(g20), p_burst rising above p_calm. At **g10, ALL THREE songs improve with NO regression**, at HELD global
density (it REDISTRIBUTES into quiet harmonic events, quiet-dens up): Deja −0.06→+0.26, HSL −0.16→+0.03,
kneeso +0.31→+0.55. **→ the lever has authority; un-burying harm-in-quiet fixes the melodic under-placement on
the posterior.** CAVEATS: (a) needed gain ~10 is large because the gate (quiet·harm, both <1) shrinks the
effective boost — a learned version would carry larger/better weights; (b) POSTERIOR-only — the real gate is
BY-EAR (Rule 8): does the generated HSL piano solo now get sensible, well-reading notes? = STEP 2.

## STEP 2 (2026-06-28) — wired into generation; A/B installed, AWAITING EARS (the real gate, Rule 8)
Added a per-frame `onset_logit_offset` (B,T) to `generate()` (applied to the onset logits before τ; caller's τ
uses the same offset — same coupling as `onset_phase_calib`). Exporter exposes `--harm_calib <gain>`
(+ `--harm_quiet_q`), computing the SAME sparse-harm offset as the probe (needs `--features highres`). A/B
installed for HSL + japa1: `~/sm-generated/harmcalib_{OFF,ON}` (ON = gain 10). Generation sanity: global density
HELD (HSL 0.388→0.383, kneeso 0.323→0.321 — redistribution not inflation), critic still Hard, holds shifted
(HSL 22→27). 39/39 generation tests pass. **OPEN = by-ear gate:** does HSL's piano solo now get sensible,
well-reading notes, or does it over-allocate? That decides Step 3 (LEARN the offset) vs retune gain/gate/feature.

## STEP 2 VERDICT (2026-06-28) — by-ear gate came back SPLIT; the GATE-FEATURE is the bug (not the gain)
By-ear (playtest_log 06-28): **japa1 PASS** ("fun, expressive, not a smear job, well choreographed" → mechanism
sound). **HSL MEH + the tell:** the 1/16s "came noticeably AFTER the piano solo concluded." HANDOFF §3 pre-
registered this as the "fix the GATE" branch. Probe (`--quiet_feat energy|perc`, `gain=10`, HSL) + a binned
energy/perc/harm + offset-mass dump CONFIRM the mechanism and complicate the naive fix:

- **HSL has NO energy-quiet section until the outro.** Smoothed dim-0 energy sits 0.90–0.96 from frame ~120 to
  ~1320 (then the outro drops to 0.34→0). So the energy gate's "quietest 40%" is just the shallowest energy
  DIPS — and the deepest non-outro dip (bin 1020–1080, energy 0.85) is **percussively LOUD (perc 0.88)**. The
  energy gate dumps **35% of its offset mass there** — a drum-heavy spot, NOT the melodic solo. *That* is the
  "1/16s after the piano solo": the boost piled onto a loud busy section the energy dip happened to mark.
- **The perc gate (dim35-absence) relocates the mass to the lowest-percussion region** (frames ~120–540, ~66% of
  its mass; ~0% at the loud 1020–1080 spot). Directionally the fix — it stops the loud-section mis-fire and
  targets where drums thin out (the plausible solo). **By-ear A/B not yet run** (Rule 8 decides).
- **HONEST metric caveat (experiment-design Rule 1):** the `offset→melodic` scalar I added (harm>p75 & perc<p50)
  said perc was WORSE (0.05 vs 0.11) — but it's a BAD proxy here because HSL's **harm channel (dim36) is weak and
  FLAT (0.22–0.34, never dominant)**, so "harm>p75" is noise, not a solo. The BINNED perc-mass distribution is the
  truthful read; don't cite the scalar. The flat harm channel also means the hand-crafted harm-gate is a BLUNT
  instrument on HSL regardless of gate-feature — there's no sharp melodic spike to chase.
- **WHY the global 16th-unlock (`onset_phase_calib`) DID nail HSL's solo by ear** (unlock16 06-28) while this
  hand-gate struggles: the global lever doesn't LOCATE the solo — it lowers tau on 16th-phase frames everywhere
  and lets the **head's own learned ranking** (AUC 0.73, onset_alloc_findings) place them where afforded. The
  hand-gate instead depends on the weak dim36 feature to FIND the solo. This is **H-onset-perc-bias at the FEATURE
  level**: the melodic content is under-represented in the 42-dim features, so a feature-driven gate is limited;
  a head-ranking-driven lever isn't. → favors LEARNING the offset from the encoder (Step 3) over hand-tuning the
  gate, OR leaning on the global 16th-unlock for the melodic-solo win and scoping harm_calib to songs with a
  genuine low-energy harm-rich phrase.

## AXIS-1 BOUNDARY-SNAP REFRAME (2026-06-28) — the gap was a metric artifact; re-measured on the REALIZED chart it ~matches real
User pivot to attack boundary-snap as the most critical axis (structure = the skeleton the other 3 decorate).
Rule-0 grounding + a readout-vs-representation decomposition (`probe_boundary_snap.py`, 8 Hard songs, no-gen)
**reframed the gap before any build** (experiment-design Rules 2/7/9/11 — the cheap fair test overturned the
motivating number):

| signal (box16 unless noted) | median width | median lag vs Foote |
|---|---|---|
| `p_onset` raw (box4) | 20f | **−1f** (steps AT the boundary) |
| `p_onset` envelope (box16) | 23f | +1f |
| **realized** (p>τ, what's PLAYED) | **12f** | **−40f** |
| **REAL** chart | **19f** | **−30f** |

- **The realized density step is NOT wider than real (12f < 19f) and its timing tracks real (−40 vs −30f).** Both
  model and real LEAD the Foote audio boundary by ~2–3 beats (choreography anticipates the section change — normal,
  and per-song they co-move: Deja −47/−39, HSL −43/−42, japa −39/−33, Dancing −35/−46). So on the DENSITY side the
  onset head's boundary behavior is ~on par with real.
- **The original axis-1 "model 2× wider / Deja lags +46f" was an ARTIFACT:** measured on the smoothed POSTERIOR
  ENVELOPE (`p_alloc`/`p_breathe` box16), NOT the realized chart (Rule 2 deployment mismatch), on 3 songs (Rule 11).
  The raw posterior's biggest gradient sits at Foote (lag ~0); tau-thresholding shifts the realized step to where
  real sits. → don't cite the posterior-envelope width/lag as a model gap.
- **What is STILL untested (where a structure gap could really live):** (a) the PATTERN/FIGURE *character* snap —
  does panel-pattern IDENTITY / figure family change crisply at a boundary? That's the PATTERN head's job, and H11
  hinted the bare AR pattern head UNDER-authors structure (its realized shifts were mostly onset-cascade). This probe
  measured only density. (b) BY-EAR — boundary-snap has never been playtested; all reads are diagnostic numbers.
- **Caveat / not-fully-clean:** a couple songs diverge (Pound the Alarm realized −38 vs real +3) — the model leads
  more than real on some. Minor, not a clean defect. Widen + by-ear before any magnitude claim.

### FIGURE-CHARACTER snap (the pattern-side test) — REAL reference is WEAK/NOISY (`probe_figure_snap.py`, 8 songs, no-gen)
Followed the density reframe by testing the OTHER half — does the FIGURE-FAMILY character (jack/sweep/trill/
candle/jump/step mix, density-ISOLATED fractions over named W=3 windows) change crisply at Foote boundaries?
Cheapest-first = REAL reference only (Rule 5): if real doesn't snap figure-character, the premise is moot.
**Result: real snaps only WEAKLY/inconsistently** — `resp = |Δchar|@boundary − @random` median **+0.10**,
3/8 songs NEGATIVE (Dancing −0.28, IN BETWEEN −0.12, japa −0.05; vs Deja +0.44, Taylor +0.26, HSL +0.18). The
@random baseline is high (0.44–0.80) → figure character varies a LOT everywhere, only a slight boundary bump.
- **Reading:** real Hard charts don't strongly BLOCK-organize figure character at audio-timbre sections — they're
  more continuously varied. So there's no sharp real target the model is missing. Combined with the density
  reframe above, **two cheap probes both say "Foote-boundary-snap" is not a clean, targetable gap.**
- **Power caveat (don't over-conclude — Rule 1/11):** ~4 boundaries/song (Foote TOPK cap), short 64f windows →
  noisy; Foote AUDIO boundaries may not be where the CHARTER's sections fall. "Weak/noisy," NOT "proven absent."
  A better-powered version = within/between-section variance "blockiness" over all frames, or the chart's own
  segmentation — IF we stay quantitative.
- **Recommendation logged:** boundary-snap has NEVER been grounded by ear (all reads are diagnostic numbers, and
  the numbers keep coming back ambiguous). Before more metric work, GROUND IT BY EAR (Rule 8) — generate a set,
  user listens specifically for whether section structure feels crisp/robust. If the ear hears a real gap, THAT
  defines the right operationalization; if not, the structure concern likely lives elsewhere (long-range motif
  recurrence/repetition, not boundary snap).

## Bottom line / next
The learned phrase calibrator has the most headroom on **(2) quiet-phrase harmonic allocation** (ADD — the HSL
melodic under-placement, the sharpest + on the complaint song) and **(1) snappier boundaries** (sharpen). **(4)
the perc↔harm fluidity signal already exists to amplify** (encouraging — the calibrator steers an existing axis,
not a missing one). **(3) tail** is inconclusive on these songs (latent in the posterior; re-probe on the songs
where the over-run is audible). Directional on 3 songs — widen the set before committing magnitudes.

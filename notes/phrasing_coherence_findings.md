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

## Bottom line / next
The learned phrase calibrator has the most headroom on **(2) quiet-phrase harmonic allocation** (ADD — the HSL
melodic under-placement, the sharpest + on the complaint song) and **(1) snappier boundaries** (sharpen). **(4)
the perc↔harm fluidity signal already exists to amplify** (encouraging — the calibrator steers an existing axis,
not a missing one). **(3) tail** is inconclusive on these songs (latent in the posterior; re-probe on the songs
where the over-run is audible). Directional on 3 songs — widen the set before committing magnitudes.

# Taste critic v2 (48th grid) — R1 cleared, R3 still open; the defects it can/can't catch

*2026-07-11, branch `explore/taste-critic-quality-resolution`. Thread: taste-critic-quality arc (lineage
`taste-critic-arc.md`, memory [[taste-critic-transfer]]). Goal: a taste-aligned critic to drive best-of-N
conditioning SELECTION so f48 generation "just works." Probes in scratchpad (`score_personal.py`,
`offgrid_personal.py`, `critic_catches_defects.py`).*

## The diagnostic that framed the arc (both critics on `~/sm-personal`)
- **F1 — the critic ≠ the user's taste.** Hand-made charts score mean P(real) **0.32** vs train-REAL 0.82; 63%
  rail to <0.1. Corpus-graded ≈ "typical of the training set" ≠ the user's charts. Confounds flagged (first-768
  window = intro; varbpm songs mis-gridded) — a CONDITIONAL observation.
- **F2 — the GRID WALL (user-predicted).** Both existing critics are 16th-grid; the generator ships f48/48th. A
  16th critic must FLOOR an f48 chart to score it, deleting the sub-16th placement that IS the f48 signal. The
  user's own charts are ~100% 16th-native (`offgrid_personal.py`, mean 0.2% off-grid) → F1 is NOT a grid artifact,
  but the generator's OUTPUT is invisible to a 16th critic.
- **F3 — the graded objective helps but isn't enough** (16th `realism_critic_graded`): ladder spread ×6 vs binary,
  but still 6/30 monotone and 16th-grid.

## E1.1 — retrain the graded critic on the 48th grid (`train_graded_critic_v2.py` → `checkpoints/realism_critic_graded_v2`)
Ported `train_graded_critic`'s within-song corruption-ladder margin-ranking to the v2 dataset (`for_v2` +
`highres_v2` + `cache/samples_v3_48th`, sliced [:23]; warm-start from the 16th binary critic loads clean;
max_len 2304 = 192 beats @12/beat), **+ a new sub-16th JITTER corruption axis** (displace on-16th notes to pure-48th
cells — the degradation the 16th critic couldn't represent). 100% cache-hit. **R1 CLEARED:** best epoch jitter ladder
`+2.36 → −4.95`, **monotone 0.98** — the critic grades sub-16th placement. (panel 0.30, shift 0.70 — placement is the
strong axis, arrow-choice the weak one, same as the 16th graded critic.)

## E1.2 — does it agree with the ear + catch the playtest defects? (`critic_catches_defects.py`, sliding 2304 windows)
Architectural prediction (held): the presence-based critic CAN see placement (#1 sub-16ths) + coverage (#2 quiet
under-charge) but is BLIND to hold-type (#3 hold foot-speed).
- **(a) ear agreement MIXED:** Switch GLOBAL>OFF ✓; Bye Bye & Calling ~TIED — can't resolve the subtle arm
  differences the ear heard (Bye Bye bridge quality = arrow-choice, presence-invisible).
- **(b) sub-16th correlation −0.93 but CONFOUNDED** (n=3 songs, sub16 ~0% except Bye Bye 0.9%). Lean on (c).
- **(c) TAIL scored worse than BODY by −1.59** (within-song, unconfounded) → the critic catches "worst near the
  end" (defect #1). **Operational R1 confirmation.**
- **(d) R3 STILL OPEN:** critic rates GENERATIONS above the user's HUMAN charts on **2/3 songs** (Switch, Calling;
  only Bye Bye REAL>gen). Confirms F1 on fresh data. E1.1 fixed the grid (R1), NOT taste (R3).

**Verdict:** best-of-N on this critic can select against placement/tail defects (#1) and likely coverage (#2), but
NOT hold foot-speed (#3, presence-blind) or fine choreography quality. Taste alignment (R3 = Phase 2, a preference
reward-model on the user's good/bad labels) is the confirmed remaining crux.

## harm_calib #2 A/B — offline half (`harm_offline.py`, Bye Bye; by-ear PENDING)
Added `--harm_quiet_feat total|perc` to `generate.py` (perc = the cond-mech §6 dim-35 gate). Both gates land and are
orthogonal: **HARM-TOTAL +40% density in total-quiet LULLS**, **HARM-PERC +17% in perc-absent BUSY-HARMONIC
sections** (each fires only its own gate). The two gates target DIFFERENT sections (Jaccard 0.20) — total = silent
lulls (sparse), perc = melodic-lead-over-loud-pad (dense, the case total MISSES). harm_calib is density-PRESERVING
(redistributes, total count ~flat). **AWAITING by-ear: which gate lands the choreography the user missed, gain 10 vs
over-boost.** If neither lands → #2 is an onset-head/retrain matter, not a decode lever → reward-model target.

## The three defects (from the playtest) as the taste-critic's target list
#1 spurious sub-16ths (esp. tail) — critic SEES it (jitter 0.98 / tail−1.59). #2 quiet under-charge — presence, critic
sees; decode leads harm_calib + local-z. #3 hold foot-speed — presence-BLIND; needs the parked free-foot-overload gate.

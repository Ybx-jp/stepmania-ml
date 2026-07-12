# Taste-critic arc — from human/generated separator to a taste-aligned reward model

**Thread:** can the realism/taste critic become a reliable **quality** signal — good-vs-bad among generations,
aligned to the USER's taste — usable to auto-select conditioning (best-of-N) so f48 generation "just works"?

Cross-refs: memory [[taste-critic-transfer]], [[ship-mode-park-research]] (this arc is the 2026-07-11 UN-PARK),
[[good-settings-region]] (the PREDICTION alternative, clean-negative). Skills: `conditioning-mechanics` (critic
input = 23-dim audio + binary note-PRESENCE grid; presence-blind to hold-vs-tap), `generation-defaults` (v2/48th
is the deployed target), `experiment-design`. Depends-on: `meter-grid-arc` (the 16th→48th grid change is the whole
R1 problem). Corroborates: `quality-feature-attribution-arc` (which built the first GRADED critic).

## Hypothesis chain (what we believed → what we learned)
- **Prior state (2026-06-26, [[taste-critic-transfer]]):** the binary critic ranks REAL>BASE>CHAOS on current
  machinery but is **near-BINARY** — a strong separator, weak grader (77% of REAL on the >0.9 rail, generations on
  the <0.1 rail; only 14–30% in the discriminating middle). Flagged as the binding constraint for best-of-N.
- **A graded critic was built (2026-07-01/02, `train_graded_critic.py`, `checkpoints/realism_critic_graded`)** in
  the quality-feature-attribution arc: within-song corruption-ladder margin-ranking + end BCE anchor, warm-started
  from the binary critic. Never carried to a selection loop. 16th-grid.
- **User frame (2026-07-11):** f48 raised quality VARIANCE — "banger when conditioning is right, but I can't
  predict the right knobs per song." Wants the critic to grade subtle good-vs-bad.
- **Framing decision (locked):** pursue **SELECTION** (generate N, critic picks), NOT **PREDICTION** (audio→knobs =
  the [[good-settings-region]] clean-negative, R²~0.09 LOO-CV≈0). Selection sidesteps prediction: you don't need to
  KNOW the right conditioning, only RECOGNIZE a good result.

## Probes + verdicts (2026-07-11 cheap diagnostic, branch `explore/taste-critic-quality-resolution`)
Harness: scratchpad `score_personal.py` (score every `~/sm-personal` hand-made chart with BOTH critics + the
panel-scramble ladder) and `offgrid_personal.py` (48th-grid placement census). Both critics: 23-dim audio (dims
0..22 of the 42-dim highres) + binary presence, first `max_len` frames.
- **HARNESS bug caught FIRST (attribution order):** default `StepManiaParser()` gates rejected 22/26 songs
  ("failed song length requirement") → only 4 Beginner charts scored. Fixed with `for_inference()` (widened gates,
  `max_simultaneous=4`). The 4-chart run would have been a rigged conclusion. (exp-design Rule 0 / HARNESS→…)
- **Finding 1 — the critic is a poor proxy for the USER's taste (n=30):** hand-made charts score mean P(real)
  **0.32** (median 0.024) vs the train-REAL baseline **0.82**; **63% rail to <0.1**, only 27% >0.9. Wild swings on
  charts the user made+likes (Jealous/Lick/Bye Bye ≈0.003 vs Switch/Crazy Maybe/Sure Feels Good ≈0.98). The
  corrupted-corpus objective learns "typical of the TRAINING SET," which disagrees with the user's charts.
  ⚠️ **Confounds flagged (Rule 10, untested):** scored the FIRST 768 frames (intro, maybe unrepresentative) +
  variable-BPM songs (Heroes ≈0.007, Stereo Sayan ≈0.008) are single-hop-16th MIS-GRIDDED (harness artifact, not
  taste). The swing survives both (constant-BPM charts still span 0.003→0.98). NOT yet a committed conclusion.
- **Finding 2 — the GRID WALL (the user predicted it), quantified:** both critics are 16th-grid; the deployed
  generator ships f48/48th. To score an f48 chart the 16th critic must FLOOR it → deletes the triplet/sub-16th
  placement that IS the f48 quality signal. `offgrid_personal.py`: the user's OWN charts are ~100% 16th-native
  (mean 0.2% off-grid) → the low scores in Finding 1 are NOT a grid artifact (clean), BUT the model's f48 OUTPUT is
  invisible to a 16th critic → a best-of-N loop on it is blind to the axis f48 varies on.
- **Finding 3 — the graded objective HELPS but isn't enough:** graded critic ladder spread +1.92 vs binary +0.30
  (margin), early sensitivity +0.88 vs +0.08 — real. But still monotone only 6/30 per-song on this OOD set, and
  still 16th-grid.

## Current state — 3-part arc (each gated; the EAR is the only ground truth)
- **E0.1 (needs ears, no retrain):** does best-of-N even help — is there real quality spread across conditioning at
  fixed song, and is the best candidate reliably good? Independent kill-switch. Seeds preference data. ⬜ PENDING.
- **E1.1 (R1+R2, autonomous): retrain the graded critic on the 48th grid.** `train_graded_critic_v2.py` →
  `checkpoints/realism_critic_graded_v2`. Mirrors `train_motif_figure_v2`'s dataset build (for_v2 + highres_v2 +
  `cache/samples_v3_48th`, 100% cache-hit verified) + `train_graded_critic`'s objective, warm-started from the 16th
  binary critic (loads clean), `max_len=2304` (=192 beats at 12/beat). **NEW: a sub-16th JITTER corruption axis**
  (displace on-16th notes to pure-48th cells) — the degradation the 16th critic couldn't represent; if the v2 critic
  grades it monotonically, **R1 is cleared**. Smoke (40 songs/1 ep): cache-hit 100%, jitter already the most
  responsive ladder (mono 0.55). 🟡 FULL RUN LAUNCHED (1200 songs, 12 ep). GATE: jitter monotone + spread.
- **E1.2:** score E0.1 candidates with the v2 critic; correlate with the ear. Correlates → best-of-N may work as-is.
  Doesn't → **Finding 1 (taste) is binding** → E2. (Running E1.2 with the OLD 16th critic is grid-confounded — this
  is why E1.1 precedes it; validates the user's "start with the retrain" instinct.)
- **E2 (R3, the crux, only if E1.2 shows the gap):** mine `playtest_log.md` (2296 lines) + structured pairs for
  PREFERENCE data; fine-tune the v2 graded critic as a Bradley-Terry REWARD MODEL (warm-start keeps the real-like
  prior, adds the user-taste direction). Guard: the corruption ladder must not collapse (Rule 15 baseline). Biggest
  RISK = label cost; E0.1/E1.1 are designed to green-light or kill BEFORE this spend.
- **E3:** wire `generate.py --best_of N` (candidate spread via the manifold, one varied axis; canonical v2 palette).
  Final gate: "just run it" produces a good chart on a fresh song, no per-song tuning — the user's original goal.

## Results — 2026-07-11 session (probes: `notes/taste_critic_v2_findings.md`, `notes/stamina_longsong_findings.md`)
- **E1.1 DONE — R1 CLEARED.** `train_graded_critic_v2.py` → `checkpoints/realism_critic_graded_v2`. Jitter ladder
  monotone 0.98 (`+2.36→−4.95`) → the critic grades sub-16th placement the 16th critic couldn't see.
- **E1.2 DONE — mixed, R3 open.** `critic_catches_defects.py`: catches the tail/placement defect (tail scored −1.59
  below body, within-song); but ~tied on subtle arm quality (presence-blind), and **rates generations above the
  user's human charts on 2/3 songs** → R3 (taste alignment) is the confirmed remaining crux. Architectural limit
  CONFIRMED: presence-based critic sees placement (#1) + coverage (#2), blind to hold-type (#3).
- **STAMINA long-song sub-investigation (user-prioritized detour) CLOSED — hypothesis REFUTED.** Ceiling IS
  length-mis-scoped (corr(len,divergence)+0.83) but doesn't bite the chart (fair AR-loop density test + by-ear:
  OFF was WORST, GLOBAL best-or-tied). `local-z` fix kept as a MILD partial fix for defect #2 (quiet under-charge),
  NOT the default. Method keeper: the cheap ceiling probe looked confirmatory; the fair test + ear overturned its
  DIRECTION (Rule 7/9, necessary≠sufficient). `notes/stamina_longsong_findings.md`.
- **PLAYTEST (2026-07-11) enumerated the critic's target list — 3 defects:** #1 spurious sub-16ths (esp. tail,
  consistent across songs; critic SEES it), #2 quiet/harmonic under-charge (decode leads: harm_calib + local-z),
  #3 very-high foot-speed during a hold (worst on fast songs; presence-BLIND → the parked free-foot-overload gate).
- **DECODE-FIX track started (user "leaning decode fixes"), defect #2 first.** `--harm_quiet_feat total|perc` added
  to `generate.py` (perc = cond-mech §6 dim-35 gate). Offline (`harm_offline.py`): both gates land + orthogonal
  (TOTAL +40% in lulls, PERC +17% in busy-harmonic; Jaccard 0.20).
- **✅ SESSION 2 (2026-07-11) — decode-fix track advanced 2 defects; probes `probe_{subtail_position,lick_vs_byebye,
  onset_window_sweep,harm_fills_middle}.py`, detail `notes/playtest_log.md`.**
  - **Defect #2 harm_calib gate PASSED by ear** ("did its job"). KEY mechanism nailed: harm_calib is DENSITY-
    PRESERVING (offset baked into tau) → it TRADES, not adds. HARM-TOTAL +40% silent-lull density but −13% out-of-
    gate (percussive) — the user's felt "at the expense of percussive themes." TOTAL+PERC compete for one budget.
  - **Defect #1 (sub-16th tail) → a LENGTH-GATED long-song defect** (jitter 0 in body / short songs; only long
    songs' tails). **harm_calib EXONERATED** (no-harm Lick reproduces the smear; "HARM-PERC worst" = length
    confound + salience). Mechanism = the **onset-head sliding-window TRAILING EDGE** (song-end at the final
    window's under-trained high-local-PE, no later window to heal → quarter-backbone COLLAPSE; user's hypothesis,
    confirmed by the window arithmetic; Lick collapses harder = longer final window, end at local-PE 5351 ≈ ceiling).
    Real-chart control (Rule 5): the user's own Bye Bye chart keeps ~50% quarter through the tail → the gen's
    body→tail collapse is defect-like. **FIX SHIPPED: the HANGOVER PAD** (`onset_logits(tail_hangover=)`;
    `generate.py --onset_tail_hangover auto`; single-sourced tau+decode). **BY-EAR: "definitely better."**
  - **Empty MIDDLES ("long empty sections + scattered notes", both long songs) = a SEPARATE bug: global-tau
    ALLOCATION starves onset-poor regions.** NOT windows (offline sweep: more overlap makes maxgap WORSE via over-
    smoothing) and NOT harm_calib (keys on harmonic ONSETS a sustained hole lacks; +2 even at gain 20). A
    local/windowed tau FIXES it offline (maxgap 371→188, energy-gated respects the arc) but is the Rule-13 global-
    quota anti-pattern → **user SHELVED it** ("too much complexity; master what we have").
  - **USER DECISIONS (end of session):** silence-pad ADOPTED as the hangover default (`hangover_reflect=False`,
    correctness — the true future is silence; ear-validated version used reflection → re-confirm); local-tau
    SHELVED (offline evidence kept, do-not-build without a fresh directive); **OPEN priority = a UNIVERSAL sub-
    train-length window** — short songs (T≤5400) get NO windowing today, so a ~4800-frame song's end sits at the
    under-trained ABS-PE tail → the user believes a ~80%-train-length window applied to ALL songs fixes the broad
    short-song END-degeneration seen even pre-windowing. My `onset_window_sweep` tested smaller-W on Lick's MIDDLES
    (WRONG population, exp-design Rule 5/11) — this claim is UNTESTED.

## Results — 2026-07-12 session (UNIVERSAL sub-train-length window; `notes/universal_window_findings.md`)
The user's OPEN priority, resolved at the onset level (RIGHT population this time — exp-design Rule 5/11 fixed).
- **Premise MEASURED first (Rule 5/6):** v2 train-length dist (`cache/samples_v3_48th/train`, N=4547) median 3120,
  p75 3648, **MAX 5128** — no training song fills the 5400 buffer; abs-PE exposure collapses to 31%/13%/6% by
  position 3500/4000/4320. So any song >~3500 sits its END in the under-trained abs-PE tail, yet `onset_window`
  is pinned at V2_MSL=5400 → windowing NEVER fires for it. The user's mechanism was right.
- **Onset probe (`probe_universal_window.py`, cached VAL, human chart = ground truth, n=60/band):** single-pass
  fires only **30% of real TAIL notes** on the under-trained band (3800–5128) and the tail backbone Herfindahl
  smears **0.610→0.342**. **W3000/W3600 restore** tail recall (~0.60) and Herfindahl to **the human value
  (0.607–0.610 vs 0.600)**. CONTROL band (<3000): NO degeneration + window BYTE-IDENTICAL (no-op) → clean
  specificity (Rule 4/11). **W4320 ~no-op** on the affected bands (fires only T>4320) = the sharpest proof it's an
  abs-PE effect (window must be < the ~3500 degeneration onset to fire). Recommended default **W≈3600** (=p75).
- **Decoded check (`probe_universal_window_decoded.py`):** windowed tail quarter% 33–69 vs single-pass's collapsed
  4–8%, tail jitter 0, dead-tail recovered (DOMINION 227 vs 136 notes) — the onset fix survives the AR decode.
- **✅ BY-EAR PASSED → SHIPPED AS DEFAULT (2026-07-12, `playtest_log.md`):** A/B (Challenge=windowed W3600 / Edit=
  single-pass / human) on the 3 long songs — **windowed won on all three** ("great"/"better"/"fine"; the bland
  choreography = per-song CONDITIONING, user's own read, feeds the best-of-N track). `decode_defaults.UNIVERSAL_
  ONSET_WINDOW=3600`; both CLIs default v2 → 3600 (`check_export_defaults` 27 ✓; v1/short-fit byte-identical).
- **New residual [H-winddown]** (SEPARATE, pre-existing): neither arm winds down into a silence/outro — candidate =
  the window restores tail p_onset peaks so the stamina breathe arc thins the outro less (queued probe).

## Open fork (binding question)
Two live tracks, complementary: (A) **decode-fix the 3 defects** — #2 harm_calib PASSED (density-preserving trade,
documented); #1 tail COLLAPSE fixed (hangover, ear-confirmed) + **short-song END-degeneration FIXED + SHIPPED
(universal window W3600, by-ear PASSED 2026-07-12)**; the empty-MIDDLES half is an open density-allocation problem
(local-tau shelved); NEW **[H-winddown]** outro-taper lead (pre-existing); #3 free-foot gate still parked. (B) **Phase 2 taste-align the critic (R3)** — the
confirmed crux for best-of-N, needs the user's preference labels; the defects ARE its negative targets, so (A) feeds (B).

## Attribution corrections / method keepers (this thread)
- HARNESS→DATA→MODEL caught the gate bug before a 4-chart conclusion (Rule 0). The off-grid census (Rule 8, ground
  in the artifact) DISENTANGLED "critic is OOD on the user's style" (real) from "16th-flooring mangled the charts"
  (not — the charts are 16th-native). The confounds in Finding 1 (intro window, varbpm mis-grid) are stated, not
  yet cleared → Finding 1 is a CONDITIONAL observation, not a committed defect (Rule 9).
- The STAMINA detour is the session's cleanest Rule-7/9 case: a +0.83 ceiling correlation (cheap probe) that
  did NOT survive the fair AR-loop test OR the ear — and the ear even CORRECTED which arm helped (LOCAL, not GLOBAL,
  recovered the bridge), a [[claim-precision]] fix to the durable log.

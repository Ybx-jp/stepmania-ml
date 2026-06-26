# Playtest Log

Running record of hands-on playtests of generated charts (dropped into StepMania and played),
plus Claude's commentary, hypotheses, and cross-session connections. Newest entries on top.
Each entry: **what was played → raw feedback → commentary/hypothesis → action.**

Sample sets live under `outputs/` (gitignored). Generation: `export_typed_samples.py`.

**METHODOLOGY (06-21, user directive): playtest sets must be GROOVE-VALIDATED.** A set only tests a
hypothesis if its songs actually exercise the relevant axis — B4U with 3 holds can't test a hold fix.
Going forward: (1) songs must be hard enough to reveal decoder/musicality subtleties; (2) MORE important,
each set's songs must have strong, relevant groove-radar readings (`select_by_groove`), and the export
must REPORT them so the set is a meaningful test. Deja loin is a good general test — strong stream,
voltage, freeze, AND air. For a hold test, require high **freeze**; for groove, high stream/voltage/chaos.

---

## 2026-06-25 — ⏳ PENDING: Stage-2 STAMINA density governor A/B (does thinning feel like relief?)
### What was generated (not yet played)
Three sets, **same 5 dense Hard songs** (groove-selected on `stream`: High School Love, nightbird lost wing,
AFRONOVA, 突撃ガラスのニーソ姫, IN BETWEEN), `gen_motif_full_fixed` highres, seed 42, identical except `stamina_ceiling`.
Per-note governor identical across all (`fatigue_penalty=2, jack_penalty=0, max_jack_run=2`); only the new Stage-2
STAMINA layer differs. Mandatory pad-playability ON; 3/3 sets re-parse + installed to `~/sm-generated/`.
Guide: `outputs/playtest_stamina/SET_GUIDE.md`. Offline validation in `notes/foot_fatigue_design.md` "STAGE 2".
- `ab_stamina_OFF` (baseline) → `ab_stamina_g25` (validated gentle, ~10–13% thinner, ~Hard) → `ab_stamina_g12`
  (strong, ~33% thinner, →Medium). Per-song densities in the guide.

### What to evaluate (the FEEL the metrics can't see)
- **H-stamina (the core question):** play OFF→g25→g12 on the SAME song. Does each step feel like the chart
  *easing where it was hardest* (relief valve = success), or like notes *missing from a phrase* (dropped notes =
  fail)? The mechanism sheds the LEAST-salient onsets when E_slow is high, so it SHOULD feel like the busy
  stretches breathing, not holes.
- **Selectivity caveat:** these `stream`-selected songs are uniformly dense → E_slow stays high → relief may read
  as global easing rather than "only the hard spots." The per-region selectivity (diag-proven ~20:1) would show
  best on a VARIED-density song (dense climax + sparse verses). If g25 feels good-but-uniform, that's the next set.
- **Calibration:** is gentle (25) enough to feel, or is 12 needed? Gates the shipped default ceiling.
- **NOT a holds test** — diag_stamina_holds.py confirmed stamina is density-general, not holds-aware.

### Action / next
- [ ] Play OFF vs g25 vs g12 same-song (AFRONOVA = cleanest, no holds; High School Love / IN BETWEEN = biggest swing).
- [ ] Log relief-vs-dropped verdict + the ceiling that feels right → sets the exporter default.
- [ ] If promising but too uniform → generate a varied-density set to feel the per-region selectivity.
- [ ] If the feel lands → proceed to Stage 3 (breathing ceiling = the difficulty arc), which builds on this.

## 2026-06-25 — ⏳ PENDING: trill A/B — deployed vs clean-retrained model (the H19 retrain feel gate)
### What was generated (not yet played)
Two sets, `~/sm-generated/ab_trill_A_old` vs `ab_trill_B_fixed`, **identical songs** (the h15 set: Deja loin,
Pound the Alarm, IN BETWEEN, nightbird, japa1) + **identical knob** (trill=+3 g2, matching the old
`h15_08_motif_trill`). Only the MODEL differs: A = deployed `gen_motif_full` (buggy representation), B =
`gen_motif_full_fixed` (retrained after the two repr fixes). Mandatory pad-playability ON both; 5/5 re-parse.
Guide: `outputs/playtest_ab_trill/SET_GUIDE.md`. Offline validation: [[h19_retrain_findings]].

### What to evaluate
- **Trill honesty (the point):** B's trill knob is honestly LOWER offline (+0.32 vs +0.47). Does B's trill feel
  *better-judged* or just *less*? (A/B the SAME song A-vs-B.)
- **HONEST CAVEAT for the "jacks during holds" (your old report):** the retrain did NOT reduce it — measured
  presses-during-holds A 3.1% vs B 5.4% (slightly MORE on B, small/noisy). That phenomenon is GENERATION-side,
  separate from the H19 DETECTOR fix. So this A/B does not promise a cleaner-around-holds feel; watch whether B
  feels busier there.
- **Decision it gates:** swap the default to `gen_motif_full_fixed`? It's offline strictly-better-or-equal
  (candle preserved, trill honest, sweep improved, quality equal), but H15 is a feel thesis → ears decide.

### Action / next
- [ ] Play A vs B same-song; log whether honest-lower-trill is better/wash/worse + the during-holds feel.
- [ ] If B ≥ A by ear → swap default exporter/eval/playtest checkpoint to gen_motif_full_fixed (PR #39).
- [ ] If the during-holds busyness bugs you → open a SEPARATE generation-side thread (not the repr fix).

---

## 2026-06-25 — ⏳ PENDING: jack-penalty vs PER-FOOT-FATIGUE A/B (does fatigue dissolve the jump displacement?)
### What was generated (not yet played)
`~/sm-generated/ab_fatigue_A_jack` (jack_penalty 1.5, the CURRENT default) vs `ab_fatigue_B_foot` (fatigue
penalty 2, jack OFF), SAME rich-Hard songs, `gen_motif_full_fixed`. Guide: `outputs/playtest_ab_fatigue/SET_GUIDE.md`.
Mechanism + calibration: [[foot_fatigue_design]].
### Why — the calibration REFRAMED the problem
The "consecutive jumps" felt earlier were INDUCED by the jack penalty displacing jacks→jumps. On installed charts:
A (jack penalty) has a **59-note jump WALL**, 23.3% jumps; B (fatigue) cuts longest jump stream to **3**, 4.6%
jumps, density identical (0.346). Real rich-Hard charts = 31% jumps / streams to 10 (the model UNDER-jumps), so
the risk now FLIPS: not walls but whether B feels jump-STARVED.
### What to evaluate
- Does B feel cleaner (no jump walls), jacks controlled? Or **too flat / jump-starved** (4.6% vs real 31%)?
- **Earned vs unearned:** does B flatten jump streams that SHOULD be there (musical bursts), or only the ugly ones?
  (The governor penalizes length, can't see musicality — this is the offline-blind question.)
### Action / next
- [ ] Play A vs B same-song; log clean-vs-starved + whether musical jump streams survive.
- [ ] If B better → fatigue REPLACES the jack penalty as default (dissolves displacement). If jump-starved → lower
      λ, OR pursue the SEPARATE model-under-jumps/under-density thread (manifold density/air, not the governor).

---

## 2026-06-25 — ⏳ PENDING: jack-governor A/B — foot-exertion penalty OFF vs ON (the "long jack streams" fix)
### What was generated (not yet played)
`~/sm-generated/ab_jack_OFF` (`--jack_penalty 0`) vs `ab_jack_ON` (`--jack_penalty 1.5`), SAME rich-Hard songs
(seed 42, the IN BETWEEN-type jack-heavy set), same clean model `gen_motif_full_fixed`, same hard cap
(max_jack_run=2). Only the SOFT foot-exertion governor differs. 5/5 reparse. Guide:
`outputs/playtest_ab_jack/SET_GUIDE.md`. Mechanism: [[foot_exertion_findings]].
### Why (audit)
The user's "long jack streams" = 8th-note jacks, which `max_jack_run` (16th-only) was BLIND to. New governor =
escalating BPM-aware penalty on extending a same-panel run. Offline on the INSTALLED charts: longest jack stream
**14→4 notes**, runs≥4 7.0→1.3%, **density identical (0.346)** — re-routes to alternation, doesn't delete.
### What to evaluate
- Does ON feel more natural / less mechanically jacky (esp. IN BETWEEN)? Does it feel like it LOST anything
  (it shouldn't — density preserved)? Is λ=1.5 right / too gentle / too strong?
- **Gates the default:** λ=1.5 is proposed as the standing default for all future playtests.
### Raw feedback (PLAYED 2026-06-25)
> "this definitely solved the unnatural jack problem." — the foot-exertion governor (λ=1.5) WORKS by ear.
Revealed the next issue: **consecutive JUMPS need a similar mechanism.**

### Commentary — the jack fix DISPLACED mass into jumps (measured)
The jack penalty only suppresses the jack-panel SINGLE; that probability redistributes (softmax) into jumps,
AND a jump RESETS the exertion accumulator (escape valve). Measured OFF→ON (same songs): jump rate 15.7→**23.2%**,
jump-laundered-jacks 21→40, **longest consecutive-jump stream 11→59**. So the consecutive-jump problem is partly
INDUCED by our jack fix. ⇒ jump governor needed (next: foot-exertion v2, jumps). H13b handle. Mechanism +
design in [[foot_exertion_findings]].

### Action / next
- [x] Jack governor λ=1.5 VALIDATED by ear ("solved the unnatural jack problem"). Stays the exporter default.
- [ ] DESIGN jump governor (H13b): jumps differ (foot-TRAVEL/span geometry, no single-foot re-route, must not
      fight the AIR radar, shared exertion budget to stop jack↔jump laundering). Designing with the user.

---

## 2026-06-24 — ★ H15 + CHAOS sets PLAYED — candle knob validated by ear, "model might be ready", felt-chaos ≠ proxy

### What was played
The 13 `h15_*` (`gen_motif_full`) + 6 `chaos2_*` sets from the pending entry below. Same `--groove_select`
Hard songs, seed 42, all mandatory pad-playability on, `pattern_temperature 0.7`. Recurring songs the user
named: **japa1** (a Japanese-titled song — extremely hard), **deja loin**, **in between**, **high school love**,
**nightbird lost wing**. (Player self-rates below super-elite, so japa1's hardest charts are read by feel, not cleared.)

### Raw feedback (verbatim where vivid)
- **chaos2_calib (16th≈0.54):** "The quarter backbone dissolved. Local and global structures were preserved,
  and symmetric patterns emerged. Music accents were represented, though it felt like the conditioning tuning
  was off, most songs took a while to get started — like it was trying to balance densities maybe."
- **chaos2_ohworld (16th≈0.61):** "Mostly the same as chaos2_calib, but it felt a little more coherent and
  quite good choreography with accents. **japa1 was awesome!** It was sooooo chaotic but every note felt earned
  and the choreography was very strong and the chart felt in character with the song… really good." → **disputes
  my characterization:** "I would not agree… that this set's 'backbone survives', the songs seemed to largely key
  off 1/16s to hold the groove during verses."
- **chaos2_manifold_q99 (16th≈0.10):** "**japa1 was totally insane** omg haha! Not a large share of 1/16s, but
  it felt insanely chaotic even though the radar didn't think so. Really good chart. high school love had a
  similar feeling, but the radar showed pretty strong chaos for this. **For the first time, I'm thinking this
  model might be ready.** It's cranking out incredibly intense and chaotic charts (as requested) that still feel
  choreographed… meant for the super elite guys who leave the arcade in a deep sweat and a crowd in awe."
- **chaos2_glitch_ood (1.00) + calib_strong (0.94):** "With basically everything being 1/16s these felt really
  streamy and were actually **not hard to play**. The model was definitely still able to comprehend choreography
  and global/local structures, but its expressiveness was limited by the conditioning… the model is finally
  working with enough controls to put something coherent together under a wide range of conditions, though
  performance is likely better in some regions than others."
- **h15_base:** "japa1 still pretty hectic, fun! The model definitely understands what a 'hard' song really is.
  **deja loin used jacks really well** — the model picked up some understanding of when the music especially wants
  some stomping. **in between had soooo many jacks**, really long jack streams too. It seems clear pattern motifs
  were learned. If this is just base, I'd think **jacks are a knob that should pretty much only go lower lol**."
- **radar_stream:** "fun charts! Not majorly different than base, but I could sense a preference for streams."
- **radar_chaos:** "doesn't feel especially chaotic. fun charts though."
- **motif_candle (+3 g2):** "in between felt like it kinda broke from the music, but the candles were there.
  japa1 was fine."
- **motif_candle_neg (−3 g2):** "in between definitely felt more linear than motif_candle. **the knob is working.**"
- **motif_trill (+3 g2):** "japa1 definitely had the trills. It also had a **huuuge jack streams during holds.**
  I wonder if our pattern head confuses a held frame with a tap and scored some trill points that way."
- **combo_chaos_candle:** "yeah I think its working. nightbird lost wing reminded me that there are some classic
  patterns I barely see represented in any of these charts, like **gallops and foot switches**."

### Commentary / hypotheses
- **★ H15 candle knob — SUPPORTED BY EAR (the arc's payoff).** `motif_candle_neg` played "more linear" than
  `motif_candle` on the *same song* (in between), and "the knob is working" is the user's own verdict on the A/B.
  This is the thing every offline metric is blind to: the section-by-section candle lever (Δ+1.7/+3.9 offline)
  audibly changes the footwork. The H15 thesis (a steerable, quality-safe motif vibe lever) has its play-feel
  confirmation. Caveat → next bullet.
- **H3 (guidance×musicality) — reconfirmed on candle.** At g2, `motif_candle` "kinda broke from the music" on
  in between while japa1 stayed fine — same song-dependence as the chaos/glitch arc (H17 fit). The knob works but
  g2 is past the musical operating point on less-ornamental songs; the gentler `_gentle` g1.4 set is the one to
  default to. Pick operating g per song character, don't headline g2.
- **H18 (NEW) — felt-chaos ≠ 16th-share ≠ radar-chaos.** The single most important result here. `manifold_q99`
  (16th≈0.10, low radar) played "insanely chaotic / totally insane"; `calib`/`ohworld` (16th 0.54/0.61) I'd
  characterized as "backbone survives" but the user heard them keying off 16ths; and the *highest* 16th-share
  sets (`glitch_ood`/`calib_strong`, ≈0.94–1.0) played "streamy and NOT hard." So perceived chaos is **non-
  monotonic in 16th-share and decoupled from radar-chaos** — it peaks in the manifold-realized middle and
  *collapses into easy streams* at the OOD flood. This refines H4: the **manifold** chaos path produces musical,
  choreographed intensity (crown jewel confirmed by ear); the smear regime isn't "more chaos," it's a uniform
  16th wall that reads as *less* demanding. The offline knee-bracket numbers measured placement, not the felt
  axis. **Action:** stop treating 16th-share as the chaos dial; the operating sweet spot by ear is `manifold_q99`
  → `ohworld`, and `calib_strong`/`glitch_ood` are past the cliff, not "more."
- **MILESTONE — "for the first time I'm thinking this model might be ready" / "enough controls to put something
  coherent together under a wide range of conditions."** First time the user has called the *system* (not one
  lucky chart, cf. OH WORLD g3.5) plausibly done. Worth treating as a phase gate.
- **H19 (NEW) — possible held-frame/tap confound in the pattern head.** `motif_trill` produced "huge jack streams
  during holds," and the user suspects the pattern/figure detector credits a *held* frame as a tap → spurious
  "trill" score, i.e. the trill axis may be partly measuring hold artifacts rather than alternating taps. This is
  a **measurement/labeling bug suspicion** on the figure head, and it would also inflate the offline trill Δ we've
  been reporting. Connects to the earlier note (memory) that trill's honest gain was already partly a density
  artifact. **Action:** check `motif_codebook` figure/trill labeling — does a hold-active frame get counted toward
  trill/tap mass? If yes, exclude hold-occupied frames from the figure detector and re-measure trill steerability.
- **H20 (NEW) — missing advanced-pattern VOCABULARY (a coverage gap, not a steering gap).** Charts are jack-heavy
  ("jacks should only go lower") and the user notes **gallops and foot-switches are barely represented**, prompted
  by nightbird lost wing. From the ddrcommunity guide, the advanced vocabulary the model isn't producing:
  **gallops** (a quick 1/16 grace-note before a beat — a "da-dum" burst, not on the grid), **foot-switches** (a
  two-note jack that's actually a foot swap, not a same-foot hammer), **laterals** (extended crossovers like the
  AFRONOVA walk that stay crossed), **half-spins**, **freeze-switches** (lift-and-replace on a freeze to keep
  alternating), and **brackets/hands**. This is distinct from H15 (we can *steer* motifs the model already knows);
  here the model's repertoire is **over-concentrated on jacks and under-covers ornamental footwork**. Likely a
  data-coverage / objective issue (these figures are rare in training and the jack axis is over-served), not a
  decode knob. **Note:** the lone stuck H15 axis is jack↔sweep, and the user independently wants jacks *lower* —
  so the holdout axis and the desired direction coincide; a working jack-reduction / sweep-toward push is the
  same lever as fixing the over-jacked feel.
- **chaos2_calib "slow to get started / balancing densities"** → consistent with **H5** (no global density arc;
  awkward starts). The phase-calib recipe may be spending early measures equilibrating density before the groove
  locks. Minor, but it's the same structural-flatness thread.

### Connecting thread
The recurring lesson lands again, sharper: **our offline proxies disagreed with the ear in the direction that
mattered.** The 16th-share "knee" I bracketed put the *best-feeling* chaos (`manifold_q99`) at the bottom and the
*worst* (`glitch_ood`) at the top — perceived intensity is carried by manifold-coherent placement, not raw 16th
count (H18). Meanwhile the candle A/B (H15) and the trill artifact (H19) show the pattern head is both a working
steering lever *and* a possibly-miscalibrated measuring stick. Net: the **system** crossed a felt-quality bar
("might be ready") under manifold conditioning, and the remaining work is (a) trust manifold-chaos over the 16th
proxy, (b) audit the figure head for the hold/tap confound, (c) broaden the pattern vocabulary away from jacks.

### Action / next
- [ ] **H15 candle:** validated — adopt `_gentle` g≈1.4 as the default candle operating point (g2 breaks musicality
      on non-ornamental songs, H3). Mark candle as the shipped, ear-confirmed H15 lever.
- [ ] **H18 chaos:** retire 16th-share as the "chaos dial"; document `manifold_q99`→`ohworld` as the felt sweet
      spot and `calib_strong`/`glitch_ood` as past-the-cliff smear. Consider a perceived-intensity proxy that isn't
      16th-count (manifold-distance? pattern-entropy?) before the next chaos characterization.
- [x] **H19 figure-head audit — CONFIRMED + measurement fixed (2026-06-24).** `motif_codebook.onset_tokens`
      gated on `arr != 0`, admitting symbol 3 (hold/roll RELEASE) as a struck note; since empty hold-body frames
      are dropped, each hold's head+tail became an ADJACENT same-panel pair → phantom jack/trill, one per hold.
      Magnitude scales with hold density (diag_figure_hold_confound.py): trill exports tail-share ~1% / trillK
      bias ~+0.08, but the freeze set ~4% / ~+0.22 and on hold-heavy charts (IN BETWEEN +0.53→−0.12) it FLIPS the
      trill knob's sign. **Measurement-side fix applied** (`active=(arr!=0)&(arr!=3)`, attacks only; generation
      unaffected — it sets the knob directly, never calls onset_tokens). **Paired re-run of trill eval** (k10,
      16 songs, gen_motif_full, ±3z; same seeded charts): g1 Δself +0.58→**+0.52**, g3 +1.28→**+1.15** — ~10% of
      the measured trill gain was the hold-tail phantom on this hold-sparse set (more on hold-heavy content). Trill
      lever REAL, just smaller. Regression test added (test_onset_tokens_excludes_hold_tail), 34/34 pass.
      DEFERRED (the fuller fix): cache/motif_basis.npz was fit tail-inclusive and gen_motif_full trained on
      tail-contaminated targets — refit basis + retrain to fully clean. SEPARATE thread: the user's PLAYED "huge
      jack streams during holds" on japa1 (tail-share only ~1% there) is mostly a GENERATION behavior, not this
      detector miscount — its own follow-up.
- [ ] **H20 vocabulary:** scope a coverage analysis — measure training-set frequency of gallop / foot-switch /
      lateral / bracket figures vs how often the model emits them; the jack over-concentration is the flip side of
      the stuck jack↔sweep axis and the user's "jacks only go lower" wish. This is likely data/objective, not decode.
- [ ] **Milestone:** log "model might be ready" as a phase gate; decide whether the next arc is *polish/coverage*
      (H19/H20) rather than another conditioning capability.

---

## 2026-06-24 — H15 conditioning-knob sets PREPARED (radar + pattern-motif + combined) — ✅ PLAYED (see entry above)
### What was generated (not yet played — user afk)
13 sets from **`gen_motif_full`** (the consolidated H15 deliverable: radar + continuous per-section motif +
discrete figure token), all on the SAME `--groove_select rich --difficulty_select Hard` songs (seed 42) for
same-song A/B; all mandatory pad-playability on; `pattern_temperature 0.7`. Installed as `~/sm-generated/h15_*`.
Guide: `outputs/playtest_h15/SET_GUIDE.md`. Old 44 sets archived → `~/sm-generated-archive/2026-06-24/` (INDEX.md).
- **base:** `h15_00_base` (no conditioning — the reference).
- **groove radar (manifold --style):** `h15_01_radar_stream` g1.5, `h15_02_radar_chaos` g1.8,
  `h15_03_radar_freeze` g1.6, `h15_04_radar_air` g1.6.
- **pattern motif (H15):** `h15_05_motif_candle` (+3 g2), `h15_06_motif_candle_gentle` (+3 g1.4),
  `h15_07_motif_candle_neg` (−3 g2), `h15_08_motif_trill` (+3 g2), `h15_09_figure_sweep` (figure=sweep g1).
- **combined:** `h15_10_combo_chaos_candle`, `h15_11_combo_stream_trill`, `h15_12_combo_glitch_candle`.
- **CHAOS knee bracket** (`chaos2_*`, on `--groove_select chaos` Hard songs; chaos = crown jewel) — by realized
  16th share: `chaos2_manifold_q90` (0.00, too gentle), `chaos2_manifold_q99` (0.10, mild), `chaos2_calib`
  (**0.54**, strong — the phase-calib 16th lever), `chaos2_ohworld` (**0.61**, the OH WORLD recipe),
  `chaos2_calib_strong` (0.94, over), `chaos2_glitch_ood` (1.00, smear). Numbers + recipes in
  `notes/h15_set_characterization.md`. **Key Q for ears: where does musical syncopation become smear?** (calib
  / ohworld are the candidates; expect calib_strong/glitch_ood to feel like a wall of 16ths.)

### Raw feedback
_(pending — fill in after playing)_

### What to evaluate (hypotheses these sets probe)
- **H15 (the thesis):** does the **candle** knob audibly change the *vibe*? Offline it steers strongly
  (Δ+1.7/+3.9, section-by-section) but every offline metric is blind to feel — this is the only way to know.
  A/B `h15_05_motif_candle` vs `h15_07_motif_candle_neg` (opposite poles) and vs `h15_00_base`.
- **H3 (guidance×musicality):** does candle still steer at gentle g1.4 (`_gentle`) while staying musical?
- **Figure sweep:** modest by metric (frequency nudge, can't enforce a staircase) — does it read as more
  runs/staircases at all, or is it imperceptible? (sets the bar for whether option-2 structured realize matters.)
- **H4 (chaos):** does radar `chaos=high` still smear, or feel musical on these Hard songs?
- **Composition:** do the **combined** sets stay coherent (radar + motif together), esp. `combo_glitch_candle`
  (aiming at the OH WORLD g3.5 vibe + candle ornaments)?

### Action / next
- [ ] Play the sets; I'll log feel against H15/H3/H4 and the standing threads.
- [ ] If candle audibly changes vibe → H15 lever VALIDATED by ear (the arc's payoff); pick a default operating g.
- [ ] If figure=sweep is imperceptible → confirms the soft-realize ceiling; option-2 structured realize is the
      only path to a real sweep lever.

---

## 2026-06-23 — ★ BEST CHART YET (OH WORLD glitch g3.5) + "harmonic guidance" hypothesis (H16) + song-style fit (H17)

### What was played
glitch conditioning (chaos=high,air=low,stream=mod) at g=3.5/4/4.5, Deja loin + OH WORLD. Controls verified
clean (max_jack=1, no note-during-2holds, no hands) on all.

### Raw feedback (user)
- **OH WORLD g3.5** — > "OH MY GOD!! THIS WAS ACTUALLY AN INCREDIBLE CHART! A human would feel good about
  having composed this one. it was choreographed, there was a 1/16 storm and some sweeps, global structure
  was preserved with a satisfying climax. local structures with symmetry emerged and were varied from each
  other. a genuinely good chart. wow." (off-beat-16th 0.16)
- **OH WORLD g4** — > "felt like a DEGRADED version of g35. still had local and global structure, but it
  became 1/8 biased and the choreography and symmetry were ruined." (off16 0.33)
- **OH WORLD g4.5** — > "BETTER than g4 while still being 1/8 biased! how interesting! perhaps this guidance
  metric is HARMONIC — like vibrating a string at the right frequencies produces still points with a unique
  signature and intermediate frequencies are garbage." (off16 0.42) — i.e. **non-monotonic** in g.
- **Deja loin g3.5** — struggled; suspects the AUDIO may be slightly misaligned (unusual cluster of 'great'
  ratings vs the normal 'flawless'/'perfect'). **g4.5** — "actually decent. I realized I struggled with g35
  because this song is so 1/4-DOMINANT and not very ornamental — unnatural to chart this way; the model was
  desperate for musical events to latch onto." (off16 forced to 0.31→0.59, high for a 1/4 song)

### Commentary / hypotheses
- **★ THE BREAKTHROUGH.** OH WORLD g3.5 is the FIRST chart the user has called genuinely good/human-quality —
  choreography, local symmetry that varies, preserved global arc with a climax, a tasteful 16th storm. This
  validates the whole stack by ear: the model's representation + manifold conditioning + the now-robust
  decode CAN produce a musically excellent chart at the right operating point. The thesis (controllable,
  human-quality generation) has its first existence proof.
- **H16 (NEW) — guidance may be NON-MONOTONIC / "harmonic."** g3.5 great → g4 degraded (1/8-biased,
  choreography ruined) → g4.5 partially recovers (still 1/8-biased). The user's resonance metaphor: discrete g
  values are "nodes" of coherent output; intermediates degrade. **MUST be falsified against sampling noise
  FIRST** (generation is stochastic; one good g3.5 draw vs one bad g4 draw could be luck, not a node) —
  [[experiment-design]] rule 11. Plausible MECHANISM if real: at fixed manifold density, tau is recomputed
  per g, so CFG shifts WHICH frames clear threshold; certain g may lock the selected onsets onto the
  metrical grid (quarter/8th/16th emphasis) coherently, others land 1/8-biased. Test: fine g sweep × multiple
  SEEDS on OH WORLD, measure the rhythm distribution (catch the 1/8-bias node) + seed variance.
- **H17 (NEW) — song↔style FIT gates the result.** Same guidance, opposite outcomes: OH WORLD (ornamental →
  off16 0.16 natural) sang; Deja loin (1/4-dominant → off16 forced to 0.31) fought the glitch style ("model
  desperate for events to latch onto"). The off16 the audio yields at a given g is the tell — a style only
  works where the SONG affords it. Implies: pick the style to the song (or warn when a style is off-fit for
  the audio), don't force ornamental conditioning onto a 1/4 song. Connects to the manifold realizability
  idea but at the AUDIO level, not just the radar level.
- **Deja loin AUDIO-MISALIGNMENT suspicion** (separate, check before trusting Deja loin play-feel): the
  'great'-rating cluster suggests a timing offset on this song's audio/chart sync. Could confound every Deja
  loin evaluation this whole arc. Worth verifying the offset/BPM.

### Connecting thread
The same per-axis-guidance tuning (H14) that looked like a simple "push harder" knob turns out to have
STRUCTURE (H16, if real) and to interact with song character (H17). And the breakthrough chart needed BOTH
the right operating point AND a song that affords the style — plus the now-fixed decode floor. The offline
metrics still can't see any of "choreography/symmetry/climax"; only the ear found g3.5. The harmonic test is
the rare case where a quantitative proxy (rhythm distribution) might finally track a felt-quality cliff.

### Action / next
- [ ] **H16 falsification (cheap, first):** OH WORLD, fine g sweep (3.0..5.0 step 0.25) × 3 seeds; measure
  quarter/8th/16th shares + a repetition/symmetry proxy; is the non-monotonic 1/8-bias REPRODUCIBLE beyond
  seed noise (→ harmonic) or within it (→ a lucky draw)? Don't commit "harmonic" until this clears.
- [ ] **H17:** characterize style-fit by the off16 the audio yields per g; flag off-fit (forcing ornamental
  on a 1/4 song). Maybe an audio-ornamentality measure to pick/warn styles.
- [ ] Deja loin audio sync check (offset/BPM) — rule out the misalignment confound.
- [ ] Save OH WORLD g3.5 as the project's reference "good chart" exemplar.

---

## 2026-06-23 — Guidance is a per-axis knob (H14) + TWO mandatory-control GAPS caught by ear (now fixed)

### What was played
The g=3 vs g=5 A/B (same song Deja loin), 3 styles × 2 guidances, manifold-conditioned.

### Raw feedback (user)
- **glitch g3** — "pretty bad. felt awkward, never got into rhythm." **g5** — "actually… better than g3
  surprisingly. maybe this is a very SENSITIVE parameter where there's some magic range that just works. g3
  felt uncertain. g5 was actually musical, but 1/16s were the primary line so it was too unintuitive to
  groove to."
- **hold g3** — "pretty fun! HOWEVER, notes were put down when both feet were occupied by hold arrows. aren't
  we controlling for that? I also noticed glitch g3 had **illegal 1/16 jack sequences**." **hold g5** — "also
  quite fun! no illegal sequences."
- **stream g3/g5** — "fine / basically the same but some awkward sequences more filled in. maybe even higher
  conditioning would be better."
- Takeaway: "**guidance effect is a parameter worth tuning PER AXIS.** I think the model has a strong enough
  representation with a robust enough decode that tuning guidance will be worthwhile."

### Commentary / hypotheses
- **H14 refined — per-axis guidance, and chaos has a narrow "magic range."** The sweep + ear agree: density
  steering is guidance-independent (target-pinned); freeze tolerates guidance gracefully; CHAOS is sensitive
  — g=3 under-commits (awkward/uncertain, off16~0.11), g=5 floods 16ths as the primary line (off16 0.71, the
  H4 smear edge). The good band is BETWEEN (→ this session's g=3.5/4/4.5 test). So conditioning IS a usable
  control surface per the user's read; it just needs per-axis tuning. NOT a global guidance default.
- **★ TWO MANDATORY-CONTROL GAPS, caught by ear, CONFIRMED by audit, now FIXED.** The user's instinct ("ensure
  controls are truly in place; if they were but still broken, that's useful info") was exactly right — the
  controls WERE applied (enforce_playability), but conditioning exposed logic gaps:
  1. **note-during-2holds** (hold_g3: 7 frames). `no_jump_during_hold` only forbade ≥2 FRESH presses while a
     hold was open — a SINGLE fresh press while BOTH feet are hold-pinned (no free foot) slipped through, and
     3–4-panel "hands" were allowed entirely when no hold was open. FIX: occupancy-based — forbid any pattern
     where `held + fresh > 2` (the pad has 2 feet). 
  2. **fast jack leak** (glitch_g3: max-run 3 despite max_jack_run=1). A `{tap, hold-close}` jump reads as a
     SINGLE in the chart, but the cap tracked the PATTERN (a 2-panel jump) and reset the jack counter, so
     consecutive same-panel singles leaked across hold-close frames. FIX: track + cap on FRESH single presses
     (what a foot actually re-hits), not the pattern. Manifests only at full song length / higher density
     (768-frame regen got lucky; the 1440 export didn't) — a why-test-at-deployment-scale lesson.
  Both verified gone at g=3, full length, on Deja loin + IN BETWEEN; tests added. **Useful info confirmed:
  decode constraints written against the PATTERN can be bypassed by the hold automaton's pattern→symbol
  remap; constraints must be expressed on the FINAL playable symbols (fresh presses / occupancy).**

### Connecting thread
The "robust decode" the user is counting on for guidance tuning had two holes that only a hard conditioning
push (more holds + denser 16ths) surfaced — consistent with the standing lesson that each new regime
(here: strong CFG) re-stresses the decode. With the holes closed, the per-axis guidance program (H14) rests
on a now-actually-enforced playability floor. Chaos's "magic range" is the immediate test; the deeper vibe
gap is still H15 (motifs).

### Action / next
- [ ] **This session:** glitch at g=3.5/4/4.5 on Deja loin + IN BETWEEN (controls verified in place) — find
  the chaos magic range between under-commit (g3) and 16th-flood (g5).
- [ ] Per-axis guidance: stream maybe wants HIGHER conditioning (user: "even higher would be better"); freeze
  ~g3; chaos in the 3.5–4.5 band. Codify per-axis guidance once the bands are confirmed by ear.
- [ ] H15 motif work still the real chaos/vibe lever (guidance alone tops out at the smear).

---

## 2026-06-22 — Manifold conditioning is COHERENT (augments the representation) but SUB-THRESHOLD (H14) + the "vibe" gap (H15)

### What was played
The 3 manifold-aware style sets (`style_glitch_tech` / `style_stream_machine` / `style_hold_ballad`), same
4 groove-rich Hard songs (Heart Attack, nightbird lost wing, Deja loin, IN BETWEEN), guidance 1.5,
pattern_temperature 0.7, density manifold-derived (source-chart-free). A/B = the SAME song across styles.

### Raw feedback (user)
- **glitch_tech** — "basically worked." Idea: "train the pattern head on motifs correlated with the groove
  profile. glitch tech is a stompy ornamental vibe that works really well for crunchy edm/dubstep tracks…
  these charts weren't bad, but they failed to capture the glitch tech vibe properly."
- **stream_machine: Heart Attack** — "basically the same chart as its glitch tech version, but a noticeable
  notch streamier! if nothing else, this shows that **conditioning is actually augmenting the output from the
  representation, not some incoherent decode bull shit**." nightbird lost wing similar.
- **hold_ballad: Heart Attack** — "very similar… the conditioning was directionally correct, but **too weak
  to get the model to diverge from its belief of what the chart should be. I wonder if there is some decision
  boundary we need to cross** for the conditioning to have a major effect."

### Commentary / hypotheses
- **H14 (NEW) — manifold conditioning is COHERENT but SUB-THRESHOLD.** The key positive: the same song
  (Heart Attack) is *noticeably streamier* under stream_machine than glitch_tech, in a coherent direction →
  the radar conditioning genuinely augments the learned representation (it's not decode noise, and not OOD
  breakage — the manifold kept it in-distribution). BUT cross-style divergence is small (hold_ballad ≈
  glitch_tech on Heart Attack): the audio prior dominates and gentle (g=1.5, 0.85-quantile) on-manifold
  targets don't move the chart far. The user's "decision boundary" intuition = the conditioning may be
  roughly LINEAR-but-shallow; it needs more push to visibly diverge. **Cheap levers (no retrain):** (a) CFG
  guidance 1.5→2→3, (b) more extreme on-manifold targets (vhigh/max quantiles vs 0.85), swept on one song.
  Mirror image of the chaos-OOD lesson: there, pushing a point too FAR off-manifold broke musicality; here,
  on-manifold is coherent but too SOFT — so there's a usable middle band to find.
- **H15 (NEW) — "vibe" = groove-correlated MOTIFS the pattern head lacks (the user's training idea).** glitch
  tech "failed to capture the vibe" because vibe is characteristic PATTERN MOTIFS (stompy ornamental figures
  for crunchy edm/dubstep), not density/distribution. The pattern head conditions on the radar SCALAR but has
  no vocabulary of style-specific motifs, so steering shifts *quantities* (streamier, more holds) but not the
  *character*. Idea: mine recurring pattern motifs from real charts, correlate them with the groove profile,
  and condition/train the pattern head on motif identity. Model-level lever (vs H14's decode-level push) —
  the path from "right numbers" to "right character." Same "quantity solved, character is the frontier"
  pattern as choreography (H1/H4) and the taste critic.

### Connecting thread
Two layers, cleanly separated by this A/B. (1) DECODE/strength (H14): conditioning works and is coherent;
it's just gentle — a guidance/extremity sweep should find where styles visibly diverge without going OOD.
(2) REPRESENTATION/character (H15): even at full strength, the model can only move quantities it already
knows; capturing a *vibe* needs groove-correlated motif vocabulary it was never trained on. The manifold
made the ASK coherent and proved conditioning is real — the remaining gap is exactly the project's standing
theme (quantity is handled; musical character isn't, and the offline metrics can't see it).

### Action / next
- [ ] **H14 cheap sweep (first):** one song, all 3 styles, guidance ∈ {1.5, 2, 3} × target extremity ∈
  {high(0.85), vhigh(0.95), max(0.98)} → find the "decision boundary" where styles visibly diverge; watch
  musicality so we stop before OOD. No retrain.
- [ ] **H15 scope (the model bet):** plan note for mining groove-correlated pattern motifs + conditioning the
  pattern head on them. The "vibe" lever; pairs with the taste critic.
- [ ] Integrate manifold steering into the staged (new-default) generator (its onset isn't radar-conditioned
  yet — panels only) so density/stream steering is fully live there too.

---

## 2026-06-22 — H13 exertion cap CONFIRMED by ear ("AWESOME") → promoted to MANDATORY

### What was played
`~/sm-generated/h13_jackcap` — the new-default staged quota-free generator WITH the H13 exertion cap
(`max_jack_run=1`: forbid extending a same-single-panel run on 16th-adjacent frames → forces foot
alternation like real charts). Chaotic Hard songs; **night in motion** (where the 6-note 1/16 jack was
flagged) and **get it all**.

### Raw feedback (user)
- **night in motion** — > "AWESOME! hit all the patterns and musicality I was looking for!"
- **get it all** — > "very smooth. I would want to see more chaotic patterns derived from the drumline on a
  greater intensity. it may have been fine for the settings, not necessarily a flaw in the model."

### Commentary / hypotheses
- **H13 CONFIRMED BY EAR — the fix lands.** The exact song that exposed the brutal 6-note jack last session
  is now "AWESOME." The offline probe (`notes/h13_exertion_findings.md`) said the cap drops fast-jack-pair
  rate 0.284→0.031 (real ~0) and the exported night-in-motion .sm measures jack-rate 0.000 / max-run 1; the
  ear agrees. This closes H13 as a SOLVED decode-time issue. **→ PROMOTED to the code-enforced MANDATORY
  playability set** (`enforce_playability` now injects `max_jack_run` and refuses None/0 without override;
  skill table updated) alongside no_jump/no_cross_during_hold. Another **decode-time fix win** — continues
  the project's standing thread (the base model is under-served by its default decode; squeeze decode first).
- **"get it all" = a CHAOS-INTENSITY knob observation, NOT a defect (user agrees).** The export conditions
  each song on its OWN groove radar at guidance=1, so it renders that song at its baseline intensity. Wanting
  "more chaotic patterns derived from the drumline" = dial the chaos radar UP (validated: chaos conditioning
  moves realized 16th share 0.3%→26% monotonically, `notes/chaos_conditioning_findings.md`). The lever exists
  and is shippable; this is a settings/intent choice, not a model failure. Connects to the post-exertion
  frontier: now that AMOUNT, COHERENCE, and EXERTION are handled, the live dial is intensity/chaos *intent*.

### Connecting thread
The defect hierarchy keeps getting peeled from the bottom by DECODE-time fixes: timing (solved) → amount
(H12 quota-free) → exertion (H13 cap, now solved) — none needed a retrain. What remains is upper-layer
*intent/musicality*: choreographic phrasing (H1/H4: dancing lovers' backbone+accents) and chaos-INTENSITY
steering (get it all's drumline) — the latter already has a working knob (chaos conditioning), the former is
the open critic/objective problem. The offline metrics still can't see any of this; the playtest log is
where each layer gets confirmed peeled.

### Action / next
- [x] Promote `max_jack_run=1` to MANDATORY (`enforce_playability` + skill table + test). Default-on in both
  export paths.
- [ ] **get it all (optional):** export a higher-chaos variant (`--radar "chaos=0.7"` or `--match_radar
  --guidance ~1.5`) so the drumline drives more 16ths — confirm the intensity dial feels right by ear.
- [ ] Still queued: funky summer beach BPM de-confound (88 vs ~176 + trim silent tail); dancing lovers
  catalogued as the H1/H4 phrasing case for the critic/objective work.

---

## 2026-06-22 — QUOTA-FREE staged generation SUCCEEDED — "WE HAVE EVOLVED" (H12 vindicated; new H13 exertion)

### What was played
`quotafree` staged generation (per-phase CONFIDENCE THRESHOLD, no global count quota; difficulty-bucketed
calibration; mandatory playability set ON) on three songs: **night in motion, dancing lovers, funky summer
beach**. Also tried `quotafree_v4` (v4 panels under the same quota-free decode).

### Raw feedback (user)
- **night in motion** — > "TOTALLY SUCCEEDED! global structure preserved. coherent patterns. tasteful
  application of 1/16s." Revealed a cross-session pattern: > "the model doesn't have any representation of
  exertion intensity, like doing a 6-note jack on 1/16s is crazy haha — that was really the only significant
  issue."
- **dancing lovers** — > "the 1/16s were fine respective to the rest of the chart, but the whole chart
  didn't represent the music very well. I would have expected a chaotic pattern more like '1, 2, 3-4, 5-6,
  1, 2, 3-4, 5-6' — that's what the song really wanted as a backbone with some chaotic accents to shake up
  the pattern as the music invites."
- **funky summer beach** — > "pretty good! global structure is there, but the model doesn't seem to feel the
  drum fills or synth leads quite right in a choreographic sense." (Likely confounded: BPM probably misread
  88 vs true ~176, and the audio has a long untrimmed silent ending.)
- > "I tried quotafree_v4, it's not better. **WE HAVE EVOLVED.**"

### Commentary / hypotheses
- **H12 VINDICATED BY EAR — the win is real.** Replacing the oracle global per-phase budget with a per-phase
  *confidence threshold* (amount emerges locally per-frame given context) preserved global structure AND
  produced coherent patterns AND tasteful 1/16s on night in motion — exactly the failure mode (smeared
  16ths, quota-driven incoherence) that global counts caused. Confirms the standing thread across the WHOLE
  project: **global quotas/counts damage coherence; the amount must emerge locally.** This is the first
  staged-generation set to land a clean "totally succeeded."
- **quotafree_v4 ≠ better** → the win is the QUOTA-FREE DECODE, not the v4 panel set. The staged
  mask-predict placement (Phase-3) under local thresholds is the live direction. ("WE HAVE EVOLVED" = the
  staged + quota-free combo is now our best-feeling generator, past v4.)
- **H13 (NEW) — the model has no representation of EXERTION / physical intensity.** A 6-note jack on 1/16s
  is musically placeable but humanly brutal — the model optimizes note↔audio fit with zero notion of how
  hard the resulting pattern is on the legs/feet. This is distinct from H2 (arrow coherence) and the
  playability constraints (which are *legality*, not *effort*). It's a recurring cross-session smell, now
  named. Mechanism: nothing in the objective or decode scores cumulative physical load (jack length, stamina
  density, foot-speed). Lever candidates: a decode-time exertion penalty (cap jack run length / foot-speed
  on fine grids), or an effort-aware term in selection. Cheapest first probe: a 16th-grid jack-run-length cap.
- **dancing lovers complicates H1/H4 (choreography), NOT the quota fix.** The 16ths were locally fine but the
  chart didn't capture the song's wanted *backbone-with-accents* structure ('1,2,3-4,5-6…'). This is the
  same choreography/syncopation gap (right notes, wrong WHERE / no structured accenting) — the model places
  on a metrical+density prior, not event-driven phrasing. Quota-free fixed AMOUNT/coherence; it does not add
  musical phrasing. Consistent with H4-resolved-as-conditioning/objective problem.
- **funky summer beach — drum fills / synth leads not felt choreographically (H1 thread), but CONFOUNDED.**
  Before trusting this as a model signal, the BPM (88 vs likely 176) and untrimmed silent tail must be ruled
  out — a halved BPM would scramble the grid alignment that choreography rides on. Treat as suggestive, not
  evidence, until re-run with corrected BPM + trimmed audio.

### Connecting thread
Three sessions of staged work converge: **(1)** global quotas hurt (H12) → confirmed by ear here; the fix
was local thresholds, not a better quota. **(2)** Once AMOUNT and COHERENCE are handled locally, the
remaining gaps are the *upper* layers of the defect hierarchy — choreography/phrasing (H1/H4: dancing
lovers' backbone, funky's drum fills) and a brand-new one, **exertion/effort (H13)** — never quantity or
legality. The decode-time wins keep coming (H12, playability); the open frontier is musical *meaning* and
human *playability-of-effort*, neither of which the offline metrics see.

### Action / next
- [x] Lock in quota-free staged decode as the new default (supersedes v4). Staged onset model persisted to
  `checkpoints/gen_staged_onset/maskpredict.pt` (train-or-load); per-phase thresholds calibrated on the Hard
  bucket (q≈0.82/8th≈0.64/16th≈0.23). Export = `diag_maskpredict_staged.py --export_dir`.
- [x] **H13 probe + fix DONE (offline) — `notes/h13_exertion_findings.md`.** CONFIRMED by data: deployment
  fast-jack-pair-rate 0.284 vs REAL 0.000 (786-chart reference); ISOLATED to the PATTERN head (control on
  real onsets jacks identically → not onset placement). Fix = `generate(max_jack_run=1)` speed-conditioned
  anti-jack cap (forces foot alternation like real). Drops 0.284→0.031, kills all 4+ jacks (max run 3→1.5).
  Default-on in both export paths; test added. **PLAYTEST-CONFIRMED 2026-06-22** (night in motion "AWESOME")
  → PROMOTED to the code-enforced MANDATORY playability set. See the newer entry above.
- [ ] **funky summer beach re-test (de-confound):** fix BPM (try 176) + trim the silent tail, re-export
  before concluding anything about drum-fill/synth-lead choreography.
- [ ] dancing lovers: catalog as a clean H1/H4 phrasing case (wanted backbone+accents, got flat-ish) for the
  choreography/critic work — quota-free is not the lever here.

---

## 2026-06-22 — Phase-3 staged mask-predict > v4 (BUT infra failure: dropped no_cross_during_hold)

### What was played
`~/sm-generated/phase3_staged` (staged mask-predict onset, oracle per-phase budget, v4 panels) vs
`phase3_staged_v4` (v4 baseline), chaotic Hard songs. **INFRA FAILURE: both used an ad-hoc export
(`_gen_v4_panels`) that DROPPED `no_cross_during_hold`** → streams-during-hold (unplayable on pad).

### Raw feedback (user)
> "it has come to my attention that you have not been carrying forward critical pieces of infra... i noticed
> that this gen had streams during a hold, which i've already surfaced as unplayable on pad... [run an audit
> + assert the control knobs in the skill + add code assertions]. phase3_staged seems to be an improvement
> over phase3_staged_v4. i suspect that the oracle budget is interfering with the model's sense of global
> structure. global quotas have consistently damaged pattern coherence in all tests we've run."

### Commentary / hypotheses
- **★ PROCESS FAILURE (mine), now fixed.** A custom export bypassed the canonical exporter's playability
  flags and dropped `no_cross_during_hold` → unplayable streams-during-hold in BOTH sets. Tainted the
  absolute feel (the relative staged-vs-v4 comparison is still ~fair since both had the same handicap). FIX:
  `src/generation/playtest_export.py::enforce_playability()` is now CODE-ENFORCED in every export path
  (forces hold_aware + no_jump/no_cross_during_hold on; raises if disabled without `--override_playability`);
  the playtest skill now leads with a NON-NEGOTIABLE constraints table. **Audit'd mandatory set:** hold_aware,
  no_jump_during_hold, no_cross_during_hold, pattern_temperature ~0.7.
- **Phase-3 direction is PROMISING:** staged mask-predict placement > v4 EVEN confounded by unplayability →
  the joint generative paradigm's placement is on the right track (the unconfounded re-export should confirm).
- **H12 (new, user: confirmed across ALL tests): GLOBAL QUOTAS damage pattern coherence.** The oracle
  per-phase budget (a global count quota) is suspected to interfere with the model's global structure — same
  pattern as `onset_phase_alloc` (smearing), fixed-density chaos, and forced per-phase shares. The amount
  should emerge LOCALLY (per-frame confidence given context), not be imposed as a global count. The quota was
  a crutch to escape the 0%-16th starvation; the real fix is a per-phase CONFIDENCE THRESHOLD in the staged
  16th pass so 16ths are placed where the model is confident (local, coherent), not to hit a quota.

### Action / next
- [x] Audit playtest feedback for mandatory control knobs; assert in skill + code (`enforce_playability`).
- [ ] Re-export `phase3_staged` playability-fixed (no_cross_during_hold on) for a clean re-play.
- [ ] QUOTA-FREE staged generation (H12): replace the oracle top-K per-phase count with a per-phase
  confidence threshold (amount emerges locally). Re-feel coherence.

---

## 2026-06-22 — v7 additive retrain: right 16th AMOUNT, WRONG placement + cold-start regression (approach exhausted)

### What was played
`~/sm-generated/v7_chaos` (gen_highres_v7, reweighted BCE w8=1/w16=10, --match_radar) vs `v7_baseline` (v4),
chaotic Hard songs. Dancing lovers, Star Trail.

### Raw feedback (user)
> "dancing lovers chaos was very awkward, didn't align with musical themes at all. star trail chaos was the
> same. it seems that the chaos set really suffered from cold-start as well, both songs had very long empty
> sequences to start the chart, and felt reluctant to really start charting notes. perhaps our adjustment
> destroyed the model's global structure awareness. i'm not convinced that tuning would save this approach."

### Commentary / hypotheses
- **PLACEMENT ceiling confirmed BY EAR.** Dancing lovers v7 matched real's rhythm DISTRIBUTION exactly
  (34/40/26 vs 33/40/27) yet "didn't align with musical themes." The AUC-0.67 16th-localization ceiling →
  right AMOUNT, wrong WHERE. The feature probe already said features can't fix this; ear confirms it.
- **COLD-START is a v7 REGRESSION (confirmed by data, not just feel).** v7 starts later + sparser than v4
  on every song (Dancing lovers 1st-note 158 vs v4 82 vs real 32; intro density 0.000 vs 0.094 vs 0.172).
  Mechanism: down-weighting 8ths (w8 3→1) pulled onset confidence off the simple 8th-driven intros, and the
  shift toward mid-song 16ths let the GLOBAL density threshold spend budget there → starved the intro. So
  the reweighting + global-threshold decode redistributed notes AWAY from where structure needs them. The
  user's "destroyed global structure awareness" is right in effect.
- **VERDICT: v7 is NOT a win** — traded coherent structure + musical alignment for rhythm-distribution
  numbers. **v4 is the better current model** (coherent intros, sane 16ths); v7 should not ship.
- **The approach (reweighted BCE + global-threshold decode) is EXHAUSTED** (agree w/ user). The constraints
  are ARCHITECTURAL: per-frame onset head (placement ceiling, can't model a 16th RUN) + global-threshold
  decode (no structural/density plan → intro starvation). Tuning trades one failure for another.

### Action / next
- [ ] Keep v4 as the current best playable model (do NOT ship v7).
- [ ] The justified bet (user's fork): rethink onset MODELING — a sequence/context-aware onset head that can
  model 16th RUNS (placement) and density ARCS (structure) together. HONEST caveat: real bet, uncertain it
  breaks the 0.67 ceiling (could be partly audio ambiguity), and a phase-change build, not a fine-tune.
- [ ] Decision for the user: scope the architecture bet, or step back. NOT more reweighting/decode.

---

## 2026-06-22 — Moderate chaos conditioning (Hard) BROKE musicality: chaos = quarter→8th grid-fining smear

### What was played
`~/sm-generated/chaos_mod` (gen_highres_v4, --radar chaos=0.3, density-matched) vs `chaos_tame` (chaos=0.2).
6 rich Hard songs. Dancing lovers, First of the Year. (NOTE flaw: BOTH sets were chaos≥0.2, already past the
quarter→8th flip — no quarter-backbone reference was provided. Bad A/B design.)

### Raw feedback (user)
> "chaos_mod: Dancing lovers was just awful. the entire musicality broke. tried first of the year too.
> horrible. **the bias for 1/8 instead of 1/4 notes is totally unjustified, both defaulted to 1/8 as the
> main line.** the 16ths were awkward. i don't think we can math our way into this at decode time. we need
> to look at the model."

### Commentary / hypotheses
- **H4/H6 CONFIRMED by ear, NOT resolved.** I mis-framed the conditioning sweep (quarter share ↓, 8th/16th
  ↑) as "specific, not a smear." It IS the smear: chaos collapses the QUARTER BACKBONE into an 8th main line
  (sweep: quarter 78.7%→32% by chaos 0.25; 8th becomes 63%). The model learned chaos = uniform global
  grid-fining (quarters→8ths→16ths), exactly the H4/H6 degenerate mechanism — the high-res feature let it
  PLACE 16ths but did NOT teach it structured syncopation. Posterior stats were blind to the backbone loss.
- **The quarter→8th flip is hypersensitive**: even chaos=0.2 ("tame") is already 8th-dominant. Plain v4
  (no chaos cond) is 69% quarters (correct backbone) — so the chaos CONDITIONING specifically destroys it.
- **Decode/conditioning is exhausted as a lever for chaos (user directive).** calib/radar move the posterior
  but can't impose musical structure (preserve backbone + syncopate tastefully). → pivot to the MODEL.
- Suspected model root causes to probe: (1) radar "chaos" dim conflated with density/8th-ness in TRAINING
  data → model learned "chaos = finer grid" but that's not what we want; (2) frame-wise onset objective has
  no backbone-preservation term; (3) the taste critic (REAL>BASE>CHAOS, valid metric) already KNOWS these
  are bad → Stage 2c critic-guided fine-tune is the model-level lever.

### Action / next
- [x] Diagnostic done (`diag_real_chaos_rhythm.py`, 264 real Medium+ charts): **real chaos ADDS density on
  top of a preserved backbone.** density rises 0.226→0.341 with chaos (corr +0.63); quarter NOTE-RATE
  ~flat 0.195→0.168 (corr only −0.37); 16th share 0.2%→12.3% (corr +0.61). Quarter SHARE falls (87%→49%)
  only because the denominator grows. **The radar chaos is DEFINED as an off-beat sum (quarter 0 / 8th 0.5
  / 16th 1.0), so "raise chaos" = "more off-beats" — and I held DENSITY FIXED while raising it, forcing the
  model to DELETE quarters to fit off-beats = the backbone collapse the user heard. Self-inflicted setup
  artifact, not (purely) a model defect.**
- [x] FAIR re-test DONE (`--target_density 0.34`, chaos=0.3): **FAILED to rescue the backbone — it's the
  MODEL.** Given the exact density real high-chaos charts use (0.34, ample budget for backbone + off-beats),
  the model STILL guts quarters: q/frame 0.047 vs real 0.17 (notes on ~19% of quarter positions vs real
  ~68%), 64% 8ths. My fixed-density setup made it worse but was NOT the cause. **DIAGNOSIS: the model has
  no protected metric backbone** — rhythm is a per-frame probability soup; chaos conditioning SUPPRESSES
  quarter p_on (0.625→0.564) while inflating off-beats, so off-beats crowd quarters out under any threshold.
  Real charts treat the quarter pulse as a near-inviolable invariant and layer chaos ON TOP. Model never
  learned that. (Did NOT hand off chaos_fair for play — numbers directly measure the "8th main line"
  complaint and confirm it persists; would only reconfirm.)
- [x] COHERENT re-test DONE (`--match_radar` on real high-chaos Hard songs, real density — the truly-fair
  in-distribution test). **OVERTURNS the "model guts backbone" conclusion.** Backbone is PRESERVED: GEN
  quarter/frame 0.129 vs REAL 0.154 (survival 0.83), quarter share 35–51% ≈ real 33–50%. The catastrophic
  collapse (q/frame 0.047) was ENTIRELY my OOD conditioning (chaos high + density at mean — a combo real
  data never has; chaos & density corr +0.63). I drew a model-defect conclusion from a rigged test — wrong,
  and the prior commit ("no protected backbone") is SUPERSEDED.
  **The REAL, narrower defect: 16th UNDER-COMMITMENT.** GEN runs 49–65% 8th / 0–19% 16th; REAL runs 35–43%
  8th / 7–28% 16th. The model substitutes 8ths where real charts commit to 16ths — defaults to 8ths as its
  "busy" rhythm even under coherent high-chaos conditioning. THIS is the "8th bias." Plus placement quality
  ("awkward 16ths", AUC-0.742) is a separate model issue.
- [ ] PLAYTEST `~/sm-generated/chaos_coherent` (GEN vs REAL per song, same high-chaos songs): does the
  backbone-preserved version feel musical (vs the "awful" rigged set)? Is the 8th-over/16th-under gap the
  audible deficiency? METHODOLOGY LESSON: stop testing the model with incoherent conditioning then blaming
  it — condition in-distribution.
- [ ] MODEL WORK (narrowed target): make the model COMMIT to 16ths where real charts do (not 8th-substitute)
  + better 16th placement. Levers: Stage 2c critic-guided fine-tune; a 16th-recall-weighted objective at the
  pattern/type level (v4 did it for onset; the substitution is downstream); revisit the high-res feature's
  reach. NOT backbone-preservation (that works).

---

## 2026-06-21 — Chaos periodic-groove (MUTE test): model understands structure; generates NO 16ths (chaos=0)

### What was played
`~/sm-generated/chaos_groove_groove` (imposed audio-grounded periodic off-beat groove per section,
3 accents/measure, on-beat backbone from p_on). Deja loin, **on mute** (isolates chart structure from
music alignment).

### Raw feedback (user)
> "deja loin: played on mute and discovered something VERY INTERESTING! a very clear timing pattern
> '1,2,3,4,5,6,_,8' with variations of note patterns that progressed in intensity. global song structure
> is very well understood. intensity gradient for pattern escalation. wonderful variety of patterns.
> HOWEVER it never chose to produce any 1/16ths." + "the groove radar shows ALL generated charts have no
> chaos factor."

### Commentary / hypotheses
- **My "mechanical" prediction was WRONG (instructive).** Offline ac16=0.97 read as a rigid loop, but that
  was only ONSET periodicity; the PATTERN head layered varied, escalating choreography on the periodic
  skeleton → structured-and-musical, not robotic. Periodic onset skeleton + pattern variety is a PROMISING
  combo. Metric missed the play-feel again.
- **Model understands structure/intensity better than H5 credited.** The audio-driven p_on backbone tracks
  energy → an intensity arc; H5's flat-density/end-fade was likely a decode artifact (global threshold).
- **★★ THE finding: generated chaos ≈ 0 — the model NEVER produces 16ths.** The chaos problem from the
  OUTPUT side: rhythmic vocabulary caps at quarters+8ths under ANY conditioning/decode → the chaos knob has
  nothing to amplify. Connects to H4: the onset feature (~93ms ≈ one 8th) is 8th-resolved → p_on can't
  resolve adjacent 16ths → thresholding never places a 16th. The high-res feature was the right lever but
  H4-v2 didn't engage it (KL≈0).

### Action / next
- [ ] **Localization probe (next): WHY no 16ths?** note-fraction by metric phase (quarter/8th/16th) for
  real vs gen_stage1 vs gen_highres; and p_on distribution by phase. p_on never high at 16ths → onset can't
  represent them (H4/resolution); p_on high but decode skips → decode. Localizes the chaos fix.
- [ ] Keep the periodic-groove skeleton (it worked structurally); the gap to fill is 16ths.

---

## 2026-06-21 — Hold-fix on a freeze-validated Hard set (Dead Heat = WIN; KIM = data timing bug)

### What was played
`~/sm-generated/holdfix2_fixed` — freeze-validated Hard songs (freeze 0.60–0.87, 5–25 holds), the first
*real* hold test (vs B4U's 3 holds). `no_cross_during_hold` on.

### Raw feedback (user)
> "kim possible was such a weird song to play, almost like the audio was misaligned? seemed like 1/4 and
> 1/8 noteskins were swapped."
> "dead heat was a fun one! it had really interesting complexity, very intense jumps and jacks. it
> definitely felt like a proper hard/expert. good note patterns during holds."

### Commentary / hypotheses
- **★ Dead Heat = the hold fix CONFIRMED on a valid test, and the model's ceiling is higher than the
  easy tests showed.** "Proper hard/expert," "intense jumps and jacks," and crucially **"good note
  patterns during holds"** — exactly what `no_cross_during_hold` targets. So on a freeze-validated,
  properly-hard song the model produces a genuinely good expert chart. Two lessons: (a) the hold-cross
  fix works where it matters; (b) **groove-validated selection reveals real quality** the under-grooved
  easy songs (B4U) hid — vindicates the 06-21 methodology directive.
  **Follow-up direct A/B (same song): "both fixed and baseline were really fun, fixed was better on the
  basis of the hold decode changes."** Clean same-song confirmation — `no_cross_during_hold` is the cause
  of the improvement. Fix conclusively validated; PR'd.
- **KIM POSSIBLE = a DATA timing bug (audio↔OFFSET mismatch), NOT the model.** Diagnosed: BPM is CORRECT
  (stored 122 ≈ librosa 123, ratio 0.99) — so NOT the 2×-BPM I first guessed. And the GENERATED chart is
  properly on-grid (phase% 73 on the quarter / 27 on the 8th, even *more* on-beat than the original's 61).
  So the chart is internally correct; the grid is just mis-aligned to *this audio file* — i.e. the audio
  was re-encoded with different leading silence than the chart's `#OFFSET` expects, shifting everything
  (original AND generated) against the music → "misaligned, colors swapped." A per-song data issue, common
  in community packs. Implication beyond one song: audio↔offset mismatches are **label noise in the
  audio↔chart alignment** — the model trains on mis-timed pairs and generates on a wrong grid for them. A
  dataset-wide onset-vs-grid alignment audit could be a data-quality lever (the chaos/groove metrics all
  assume correct beat alignment).

### Action / next
- [ ] Diagnose KIM POSSIBLE's parsed bpm/offset vs the audio's actual tempo (below). If 2× → confirms the
  data-timing-bug class.
- [ ] Consider a dataset-wide bpm-vs-librosa-tempo audit (how many charts are mis-timed = training noise).
- [ ] Dead Heat win → the hold-cross decode fix is validated; ready to PR + fold into the default decode.
- [ ] H11 (transitions) still the standing next-investigation.

### What was played
`~/sm-generated/holdfix_{baseline, fixed}` (gen_stage1, 6 songs, pattern_temp 0.7). baseline =
`no_jump_during_hold`; fixed = + `no_cross_during_hold` (free foot can't fast-cross while a hold pins the
other foot — the B4U one-foot-jacks-during-hold awkwardness). Offline: hold_burst 8.7%→4.7% (real 4.0%),
density unchanged. See `hold_cross_decode.md`.

### Raw feedback (user)
> "B4U is too easy to tell anything — baseline only has 3 holds in all."
> "deja loin (fixed) was better than baseline. both had awkward sequences in the beginning of the song
> and during the bridge/breakdown. i sense the model does well continuing a sequence but struggles to
> adapt to transitions."
> "the other songs were not well suited to this test."

### Commentary / hypotheses
- **Hold-cross fix CONFIRMED (partial, on the one valid song):** deja loin fixed > baseline. The metric
  (hold_burst) predicted the B4U complaint AND a fix that lands it on real felt better — the metric↔feel
  link holds both ways. But only deja loin had enough holds to test; B4U (3 holds) and the rest were
  un-validated for freeze → the methodology lesson above.
- **★ H11 (new): the model handles STEADY-STATE well but fails at TRANSITIONS.** Awkwardness clustered at
  the *start* and the *bridge/breakdown* — exactly the song-section boundaries where the music changes
  character. "Does well continuing a sequence, struggles to adapt to transitions." This unifies the
  long-standing "cold-start / awkward opening" thread with mid-song breakdowns: both are TRANSITIONS. The
  AR pattern head continues a motif fine but has no signal for "the section just changed, re-choreograph."
  Connects to H5 (no global/phrase structure) — H11 is the *local* symptom of it (transition points)
  vs H5's *global* symptom (density arc). Likely root: frame-local features + no phrase/boundary signal
  (audio novelty / self-similarity boundaries would mark transitions).

### Action / next
- [ ] **Build groove-validated song selection into the exporter** (the methodology directive) — select +
  REPORT songs by groove radar; freeze for hold tests, rich/multi-dim for general. Re-export holdfix on
  high-freeze songs for a real test. [in progress]
- [ ] **Transition hypothesis (H11):** does awkwardness localize to audio section-boundaries? Measure
  generated-error vs audio novelty/self-similarity boundaries. Fix levers: a boundary/novelty audio
  feature, or phrase-position conditioning. Pairs with H5.
- [ ] PR the hold-cross decode fix once confirmed on a freeze-validated set.

### What was played
`~/sm-generated/chaos_gate_{gated, smear}` (from `chaos_gate.py`). **gated** = decode-time selective gate:
on-beat backbone + top-K off-beat accents keyed on audio saliency (high-res onset dim41), `chaos_frac=0.3`,
density preserved. **smear** = the trained chaos knob (radar chaos=0.9, CFG g=2) for contrast. Offline the
gate looked good: accents land on ~2.5× louder-than-average audio events (vs smear's 1.0× uniform).

### Raw feedback (user)
> gated — "reach the sky without you: didn't feel very musical. deja loin: also didn't feel musical. any
> local sequences that felt good i think are just random chance."
> smear — "was just unplayable completely, a wall of notes."

### Commentary / hypotheses
- **The gate fixed PLAYABILITY but not MUSICALITY — a real but partial result.** smear = unplayable wall
  (the trained chaos knob is broken, as known); gated = playable and controlled but the off-beats feel
  arbitrary ("random chance"). So the selective gate is a strictly better chaos knob (playable, density-
  matched), yet it does not produce *musical* syncopation.
- **★ This resolves the H4/chaos branch: syncopation is NOT audio-placeable.** The gate placed accents on
  genuine audio events (2.5× selectivity) and it STILL felt arbitrary — exactly the pre-registered branch
  ("if it feels off despite landing on audio events → syncopation is groove/pattern, not audio-placeable").
  Confirmed. Landing on an audio onset ≠ a musical accent.
- **H10 (new 06-21): musical syncopation is a RHYTHMIC-PATTERN / periodicity property, not per-frame
  placement.** Real charters make syncopation from *repeated rhythmic figures* (a groove: an off-beat
  motif held across a phrase, 3-3-2 clave, a consistent backbeat), i.e. temporal STRUCTURE. The gate is
  *pointillist* — each off-beat chosen independently by local audio saliency, with no notion of repetition
  or periodicity — so even "correct" accents don't cohere into a groove → feels like scatter. This unifies
  every chaos failure: decode gate, chroma, high-res ×2, and now audio-selective placement all operate
  per-frame; none model rhythmic pattern. The missing ingredient is sequence/structure, not features or
  placement.
- **Connects to H5 (no global structure) and the AR pattern head.** Both are "the model choreographs
  frame-locally with no plan." Syncopation-as-groove is the rhythmic-axis version of the same gap.

### Action / next
- [x] Chaos gate playtested → audio-selective placement is NOT the lever. gate kept as a *playable* chaos
  knob (beats the smear wall) but not a musicality fix. `chaos_mechanism_plan.md` A1 updated.
- [ ] **Strategic: chaos/syncopation is now a hard rhythmic-structure problem** (features ✗×2, decode gate
  ✗, audio-selective gate ✗). Either (a) deprioritize chaos and bank the cheaper high-confidence wins
  (mirror aug, hands-filter retrain, 2c critic fine-tune for general taste), or (b) commit to rhythmic-
  pattern modeling: periodicity-aware placement (repeat an off-beat figure across beats), groove-template
  conditioning (condition on a rhythm pattern, not a scalar), or learned rhythmic motifs in the AR head.
- [ ] If pursuing (b): cheapest probe = periodicity-aware gate (place an off-beat figure, then REPEAT it
  at the same beat-phase in neighboring beats) — tests "does rhythmic repetition feel more musical than
  scatter?" before any retrain.

## 2026-06-20 — Stage 2b best-of-N at HARD difficulty (gens feel good but critic scores ~0; the gap is *tameness*)

### What was played
`outputs/reranked_hard/best` (installed as `~/sm-generated/reranked_hard_best`) — best-of-8 reranked,
all songs forced to **Hard** (conditioning + density target + A/B original all Hard). Offline the whole
critic table had *collapsed* vs the mixed-difficulty set: best tops out at **0.116** (Deja loin), most
0.02–0.05, mean lift +0.032 (was +0.444). The open question going in: does Hard *feel* worse (critic
right, generator degrades at density) or feel fine (critic over-rejects density)? Two songs so far.

### Raw feedback (user)
> "B4U (best) was pretty good! I wonder if the critic scored it poorly because of 1 quite unnatural
> sequence during a hold arrow — it would have required doing crossovers and jacks with one foot.
> otherwise i didn't see any problem with it."
> "Our Soul (best) was very good! The model clearly was able to detect the 'drop' in the music after the
> build-up. For a song rated '11' and that music profile I would have expected some more chaos."

### Commentary / hypotheses
- **★ Hard gens FEEL good ("pretty good" / "very good") despite near-zero critic P(real). So the Hard
  collapse is NOT "generator degrades at density" — it's the *other* reading, but with a twist.** The
  twist is in the user's own words: "expected more chaos for an 11." The charts are density-matched (tau
  set from the real Hard chart) but rhythmically **tame / on-grid** (H4/H7: the model sits on-beat,
  under-syncopates). A real DDR 11 has syncopated streams/intensity; a tame on-grid chart at the same
  density reads as *un-Hard-like*. **This unifies the offline + felt evidence:** the critic correctly
  flags "this doesn't look like a real Hard chart" (it's too tame), the user feels the same thing as
  "expected more chaos," AND it still plays pleasantly because tame ≠ bad. The critic is doing real work
  at Hard; the scores are just compressed near zero because the gens genuinely differ from real Hard.
- **H9 (revised)** Best-of-N earns its keep at low/mid difficulty (big lift, top draw ≈ human). At Hard
  the *level* of every candidate drops (all tame), so there's little good to select and the lift
  compresses — but the survivors still play well. The headroom at Hard is **intensity/syncopation**
  (the H4 chaos defect), not gross correctness. Consistent with "near-human on some draws, just
  inconsistent" + "the model can't render syncopation."
- **The B4U defect is pad-playability, and the user's critic-score hypothesis is plausible.** "Crossovers
  and jacks with one foot during a hold" = a foot pinned by a hold while the free foot is asked to cover
  a crossover + repeated taps — physically gross. `--no_jump_during_hold` was ON, but `--no_crossovers`
  was NOT, and the hold-aware mask doesn't forbid crossovers-under-hold. The critic sees only the binary
  note grid, but that specific pattern (dense taps clustered on one side during a sustained column) may
  read as unrealistic — so the awkward sequence *could* be part of why this candidate scored low. Cheap
  to test: re-export B4U with `--no_crossovers` and see if critic P(real) rises.
- **"Detected the drop" is local audio responsiveness, NOT a refutation of H5.** A drop is a *local*
  energy change (frame-local features see it fine); H5 is about the *global* arc/phrasing plan over the
  whole song. Tracking a drop = onset density following audio energy locally, which the model already
  does well. So this is a nice play-feel moment but doesn't touch the no-global-structure finding.

### Action / next
- [x] **Decisive diagnostic — DONE, gap is real.** `diag_real_by_difficulty.py`: real Hard charts score
  **0.82 mean / 0.98 median** (as high as every other difficulty), vs generated Hard ~0.02–0.12. The
  critic does NOT over-reject Hard — it's trustworthy there; the generator's Hard charts are genuinely
  un-Hard-like (tame/on-grid, the H4 syncopation defect). Numbers in `stage2a_critic_findings.md`
  (Stage 2b section). Consequence: best-of-N has little headroom at Hard (all candidates tame), and 2c
  at Hard is bottlenecked by H4 — the critic is ready, the generator's capacity is the gate.
- [ ] Re-export B4U (and the set) with `--no_crossovers` added; check whether critic P(real) rises and
  whether the awkward-during-hold sequence disappears. If yes, consider a `--no_crossovers` default for
  Hard, or a crossover-under-hold mask in `generate()`.
- [ ] Play the remaining 4 Hard songs (Deja loin, ヤマト…, INSERTiON, KIM POSSIBLE) to confirm the
  "tame but pleasant" pattern holds across the set, not just these two.
- [ ] Connect to 2c: if the diagnostic says the gap is real-and-about-intensity, then critic-guided
  fine-tuning at Hard should push toward *more syncopation/intensity* — which circles back to the
  unsolved H4 chaos-conditioning problem. 2c may not fix Hard until chaos is fixed.

---

## 2026-06-20 — Stage 2b best-of-N reranking playtest (the critic picks the good draw)

### What was played
`outputs/reranked_samples/{best, first}` — same 6 songs, two versions each. `best/` = highest-taste
candidate out of N=8 (selected by the Stage-2a corrupted-real taste critic); `first/` = the first/
unranked candidate (the normal single-sample baseline). A/B the *same* song across the two folders.
(Offline numbers — critic lift, per-song scores — in the Stage-2 findings, not here.)

### Raw feedback (user)
> "deja loin (best): VERY GOOD! it was clearly musical. only a few odd sequences. model generated some
> really interesting patterns, felt creative."
> "deja loin (first): still decent. less musically aligned but not incoherent."
> "I played one of the japanese ones in best, seemed appropriately musical for the difficulty level. it
> would be interesting to see how it feels on a high difficulty."
> "in general, the best group i think suffered less from the cold-start problem… I think it is probable
> that the critic is doing a good job actually picking the best chart."

### Commentary / hypotheses
- **★ Best-of-N is a play-feel WIN, and the taste critic's selection corresponds to felt quality.** The
  song with the biggest offline critic lift (deja loin) was the standout by hand ("VERY GOOD! clearly
  musical… felt creative"), and its `first/` counterpart was merely "decent." That the *ranking* and the
  *hands* agree on the same song is the validation Stage 2a was built for: the critic isn't scoring a
  shortcut the user can't feel — it's tracking taste. **New handle H9.**
- **H9** *(new 06-20, supported)* The base generator already produces near-human-quality charts on *some*
  draws; it's just inconsistent. A taste critic reliably picks the good draw, so best-of-N is a cheap,
  no-retrain quality boost dialable by N. Implication: the headroom is in *consistency*, not peak
  capability — which is exactly the premise of 2c (critic-guided fine-tuning to make every draw that good).
- **Cold-start interacts with selection (connects to the H1-session "cold-start persists" thread).** User:
  "best group suffered less from the cold-start problem." Plausible mechanism: a bad opening sequence
  drags a candidate's taste score down, so the critic *implicitly* deprioritizes cold-start-damaged draws.
  Best-of-N may be partially laundering the AR cold-start problem by selection rather than fixing it.
  Caveat (user's own): "perhaps that's just random chance" — N=8 over 6 songs is thin; don't overclaim.
- **Creativity, not just correctness.** "really interesting patterns, felt creative" matters: the critic
  is rewarding good *structure*, not penalizing the generator toward a bland safe-mean. Best-of-N keeps
  the generator's variety and skims the top — the opposite failure mode from greedy collapse (H2).

### Action / next
- [ ] Export a **high-difficulty** best-of-N set (user explicitly curious how it feels on Challenge/hard).
  Same songs, same N, just a higher `--difficulty` — tests whether the critic's taste tracking holds when
  patterns get dense.
- [ ] Widen the sample base before trusting H9 / the cold-start claim: more songs (and/or larger N) so
  "best suffered less from cold-start" isn't a 6-song coincidence.
- [ ] If high-difficulty also feels better in `best/`, that's the green light for **2c** (critic-guided
  fine-tuning) — the goal being to move the *whole* draw distribution up to where best-of-N's top draw is.
- [ ] Consider an ablation: does the critic specifically down-rank cold-start openings? (correlate
  per-candidate taste score with an opening-sequence-quality proxy) — would confirm the laundering mechanism.

---

## 2026-06-20 — Stage 1 model playtest (chaos still smears; decode-time idea)

### What was played
`outputs/stage1_samples/{base, chaos}` — generated from the 41-dim `gen_stage1` model (base =
density-matched, pattern_temp 0.7, no_jump_during_hold; chaos = chaos=0.9, g=2.0). Same 6 songs as the
base-model sets, for A/B. (Offline numbers: `stage1_musical_features_findings.md`.)

### Raw feedback (user)
> "chaos was basically all 1/16s and 1/8s, and i wouldn't really consider it playable on those grounds…
> however, it did [seem] to improve somewhat. still a pretty random opening sequence, so cold-start is
> still an issue. i wonder if some decode-time tuning could help the model express some more musicality,
> with respect to chaos conditioning."

Then, on the plain (non-chaos) `base` set:
> "the base outputs were definitely more musical! it felt mostly right"

Then, on `chaos_gated` (chaos=0.9, g=2.0, `onset_phase_penalty=1.0`):
> "it did feel a bit more musical than chaos, but it doesn't have any sense of taste. The decode-time
> gate may have helped, but I think you're right that it won't be enough to fix it."

### Commentary / hypotheses
- **★ Stage-1 base felt "definitely more musical… mostly right" — the first play-feel WIN for the
  feature retrain.** This is the key result: the offline metrics were flat (`stage1_musical_features_findings.md`),
  but the human verdict on *plain* generation improved. So the chroma features DID help musicality where
  it's a free choice (which-panels following melody/harmony), even though onset_F1/phase didn't move.
  **Reframes H6:** features-not-sufficient was the wrong read for plain generation — they helped; the
  failure is specific to the *chaos knob* (forced off-grid timing), not to the features themselves. The
  metrics just couldn't see the improvement (re: the log's whole reason to exist).
- **Chaos still smears (H4/H6 hold for the chaos knob).** "All 1/16s and 1/8s" matches the offline phase
  histogram (stage1 chaos ≈ 6% on-beat). Chroma is used for *which-panels*; the chaos dim drives onset
  *timing* and still goes uniformly off-grid. "Improved somewhat" = the which-panels/chroma effect.
- **Cold-start persists.** "Random opening" = the AR pattern head starts from BOS with no prior-note
  context, so the first several notes are unanchored. This is the "awkward start" from round 3 (which we
  showed is NOT a density issue) — a decode/AR cold-start problem, plausibly decode-fixable.
- **The user's decode-time idea is sharp and worth testing.** Mechanism candidate — **phase-aware onset
  gating**: the audio-driven onset head localizes well (ROC-AUC ~0.9), so under chaos the off-beat smear
  may be a *thresholding* artifact — cranking chaos lifts ALL off-beat onset logits above a uniform
  threshold. If we require off-beat frames to clear a *higher* bar (or keep only the top off-beats by
  confidence), only musically-salient off-beats survive → *selective* syncopation instead of a smear.
  This was the optimistic read; the diagnostic below tempers it.

### Decode-gate diagnostic + result (this session)
**Diagnostic** — under chaos conditioning (chaos=0.9, g=2.0): corr(off-beat onset prob, audio
onset-strength) = **+0.10** (on-beat −0.07); off-beat onset prob **mean 0.70, std 0.11**. So chaos
floods off-beats to ~0.70 (the smear), with only *weak* audio signal underneath. Built the gate anyway
(`onset_phase_penalty`: on-beat 0, 8th −p, 16th −2p; 25 tests).

**Result — the gate helps marginally but does NOT rescue chaos.** Gated chaos sets came out near-empty /
inconsistent (g=1.5+pen1.0: densities ~0.00–0.02; g=2.0+pen1.0: 0.007 / 0.42 / 0.59 / 0.72). Played, the
g=2.0 gated set "felt a bit more musical than chaos, but… doesn't have any sense of taste." **The key
learning:** chaos doesn't *add* off-beats on top of an on-beat backbone — it *moves* placement off-beat
(suppressing on-beat too). So gating off-beats leaves almost nothing, not "backbone + selective
syncopation." **Decode cannot fix chaos musicality** — confirms the deeper H6 read for the chaos knob:
the fix is the conditioning mechanism / objective, not decode. (The gate is still a useful *general*
metric-anchoring knob; for density-matched use it needs the threshold re-derived AFTER the penalty,
else density collapses — a real interaction bug, currently the gated sets are not density-matched.)

**"No sense of taste"** is the phrase to carry forward: the model places notes that are individually
plausible but lack musical *judgment* (which off-beats are worth hitting, when to vary). That's not a
feature gap (chroma is in and helped plain play) nor a decode gap (the gate proved decode's ceiling) —
it points squarely at the **training objective**: frame-wise CE rewards matching the reference token,
never "is this a tasteful choreography." This is the strongest signal yet toward an objective-level
Stage 2 (see H6).

### Action / next
- [x] Diagnostic + gate built & tested → chaos is *moved* off-beat, not layered; decode can't fix it.
- [ ] **Pursue the Stage-1 base WIN**: it's musical — lock it in as the default, and consider Stage 2
      (drop metric-phase H7, drop HPSS H8, add structural feature H5) to push plain musicality further.
- [ ] Chaos knob: needs a conditioning/objective rethink (not decode) — lower priority than the base win.
- [ ] Gate knob: fix the density-matched-threshold interaction (re-derive tau post-penalty) before using
      it for non-chaos metric anchoring.
- [ ] Cold-start: still open (low-temp/primed opening), separate from chaos.

### Connecting thread
The recurring decode lever has a limit here: it unlocked latent quality before (pattern sampling,
hold-aware) because the model *already encoded* the right thing. For chaos it doesn't — the conditioning
genuinely relocates notes off-grid with little audio grounding (+0.10), so there's nothing musical for
decode to extract. The bigger, happier signal: **the plain Stage-1 model is more musical to play** even
though every offline metric said "no change" — vindicating both the feature retrain AND the playtest
log's reason to exist (the numbers can't see musicality; the hands can).

---

## 2026-06-19 (round 4) — chaos/air disambiguation (H4 fully confirmed + mechanism)

### What was played
`outputs/radar_samples/{chaos_only (chaos=0.9,g=2), air_only (air=0.9,g=2), chaos_gentle (chaos=0.9,g=1.3)}`.

### Raw feedback (user)
> "chaos_only was similarly unplayable. air_only was fine. chaos_gentle was just all blues (i think
> that's 1/8 shifted) pretty much."

### Measured corroboration — within-beat note phase (16th grid: 0=on-beat/4th-red, 2=8th-offbeat/blue, 1&3=16th)
```
                       on-beat   16th   8th-off(blue)  16th
air_only   (g=2.0)       0.98    0.00      0.01        0.00   on the beat -> playable
base ORIG  (real)        0.80    0.02      0.16        0.02   human: mostly on-beat + some syncopation
chaos_only (g=2.0)       0.06    0.32      0.30        0.32   downbeat GONE, uniform smear -> unplayable
chaos_gentle (g=1.3)     0.24    0.04      0.66        0.06   "all blues": 66% on the 8th-offbeat
```

### Conclusion — H4 FULLY CONFIRMED, with mechanism
- **air (jumps) is on-grid (98% on-beat) → playable.** Quantity knob, no musical justification needed.
- **chaos destroys the downbeat anchor.** Gentle chaos → ~uniform 8th-offbeat ("all blues", 66% phase-2,
  exactly the user's read); strong chaos → ~uniform smear across all 16th phases (only 6% on-beat).
- **Mechanism (the key insight):** the model renders the chaos dim as a *degenerate GLOBAL grid
  manipulation* — shift everything onto the offbeat, or smear uniformly — NOT *event-driven
  syncopation*. With no melodic/percussive features it has no idea *which* offbeats deserve a note, so
  the only way it can satisfy "more off-grid" is uniformly. A human places an 8th-offbeat note because a
  specific musical hit lives there. This is H1 + H5 made visible in one histogram: the model treats
  rhythm position as a *global statistic to match* (the radar value), not a *response to musical events*.
- **Bonus:** base-generated is *more* on-beat (0.91) than real (0.80) — the model slightly
  **under-syncopates by default** and literally cannot add musical syncopation; the only syncopation it
  can produce is the chaos knob's uniform smear. Strong motivation for the feature retrain.

### Action / next
- [x] Disambiguation done — chaos (off-grid) is the culprit, not air (jumps); g=2.0 overshoot makes it
      worse but even g=1.3 is degenerate (uniform blues). H3 also visible (gentle = less smeared but still
      not musical).
- [ ] **Feature retrain** is now the clearly-indicated move (H1 local + H5 global + this chaos mechanism
      all converge): per-frame chroma/HPSS for event-identity, structural signal for sections. Scoping next.

---

## 2026-06-19 (round 3) — stream_voltage + structural observations (H4 confirmed, H5 born)

### What was played
`outputs/radar_samples/stream_voltage/` (stream=0.9, voltage=0.9, g=2.0). Plus general
observations across the session's sets.

### Raw feedback (user)
> "stream_voltage was a little off but playable. some occasional random notes and unnatural jack
> sequences. broadly speaking, it feels like the model sometimes fails to pair choreography with a
> phase change in the song, and seems to consistently start and end the song awkwardly."

### What this confirms / opens
**H4 CONFIRMED.** stream_voltage (a *quantity* knob — density) is *playable*, while chaos_air (a
*musicality* knob) was *unplayable*. Exactly H4's prediction: quantity knobs steer fine, musicality
knobs break. The "occasional random notes / unnatural jacks" are minor choreography defects (H1/H2
territory — pattern head + decode temperature), not the structural problem below.

**H5 (new) — no song-structure / phrase awareness.** Measured density vs normalized song position
(6 base songs, generated vs their real charts, deciles 0=start..9=end):
```
REAL:  0.05 0.15 0.19 0.17 0.17 0.15 0.16 0.15 0.20 0.14   (intro -> build -> CLIMAX@8 -> outro)
GEN:   0.06 0.17 0.18 0.17 0.18 0.19 0.18 0.17 0.14 0.08   (flat -> FADE)
```
The real chart has a musical *arc* — sparse intro, build, a density **peak at 80–90%** (final
chorus/climax), short outro. The generated chart is **structurally flat and fades at the end** (last
two deciles 0.14/0.08 vs real 0.20/0.14). This is the measured signature of "awkward end" and "fails
to track phase changes": the model has no representation of song sections, so it can't ramp into a
climax or mark a verse→chorus boundary. It choreographs locally, frame by frame, with no global plan.

**"Awkward start" is a separate mechanism.** The data shows generated *start* density (0.06) matches
real (0.05) — the intro is fine densitywise. So the start-awkwardness the user feels is **choreographic
or sync**, not density. Candidates: AR cold-start (decoder begins from BOS with no context), audio
Conv1D edge effects on the first frames, or offset/lead-in sync. Needs its own probe — do NOT fold it
into H5.

### Mechanism notes
- End fade has two possible causes to separate: (a) audio energy genuinely drops at the outro and the
  model faithfully follows it, while the *human charter keeps density up for gameplay climax* (a
  choreographic choice not in the audio) — this is H5/H1; or (b) a *mechanical* decay — the onset head's
  positional encoding / AR context degrades at long positions so onset logits sag regardless of audio.
  **Probe to disambiguate (next):** feed constant-energy synthetic audio and plot onset prob vs
  position — if it sags with no audio reason, it's mechanical (b); if flat, the fade is structural (a).
- H1 (local: arrow↔event) and H5 (global: section structure) are two faces of the same root — the
  feature set is **frame-local timbre/energy**, expressing neither local event-identity (no chroma) nor
  global structure (no novelty/section curve, shallow Conv1D receptive field). Fix levers diverge though:
  H1 → chroma/HPSS per-frame; H5 → a structural signal (audio self-similarity/novelty, downbeat phase,
  or a wider-context / attention audio encoder so the decoder can see beyond the local frame).

### Probe result (done this session) — end-fade is structural, not mechanical
Fed the onset head **constant-energy audio** (fixed feature vector × 1440 frames) so any slope is
purely positional. Onset prob by decile (diff 2): `0.32 0.33 0.35 0.37 0.36 0.35 0.35 0.37 0.36 0.38`
— **flat, slightly rising at the end** (same shape across difficulties 0–3). Conclusions:
- **End-fade = audio-faithful, NOT mechanical.** The model doesn't positionally decay; the real-song
  end-fade is it *following audio energy down at the outro* while humans keep density up for the
  climax/finale. Mechanism (b) ruled OUT; (a) confirmed. **Refines H5:** the model maps audio-energy→
  density too literally; human charting density carries structural/gameplay intent (build to a climax)
  that is *not a pure function of instantaneous audio energy*. Decode/position fixes won't help — needs
  structural signal or a density target that isn't purely audio-energy-driven.
- **Cold-start dip exists but is small:** first decile depressed (0.32 vs ~0.36 steady) with no audio
  reason — a mechanical AR/positional start effect. But real-song start *density* matched real (0.06 vs
  0.05), so "awkward start" is more likely choreographic/sync than density. Still worth a pattern-level
  start probe.

### Action / next
- [ ] **Start probe**: inspect first ~16 frames of generated vs real (pattern + onset + sync offset) to
      characterize the cold-start awkwardness (density ruled out; look at choreography/sync).
- [ ] User still to play the disambiguation sets: chaos_only / air_only / chaos_gentle (H4/H3 split).
- [ ] H5 fix candidates to scope: (1) add an audio *novelty/self-similarity* feature; (2) add downbeat
      phase; (3) widen the audio encoder's receptive field or add self-attention for global context.

### Connecting thread
The defect taxonomy is sharpening into a clean hierarchy:
- **Timing (onset, local)** — solved (ROC-AUC ~0.9).
- **Local choreography (which arrow ↔ which event)** — H1, blocked by no melodic features.
- **Global structure (sections / phrases / climax / start-end)** — H5, blocked by frame-local features +
  shallow receptive field.
- **Decode polish (jacks, random notes, jump-during-hold)** — fixable at decode time (the pattern keeps
  holding: many "problems" are decode/feature gaps, not capacity).
Every layer above "timing" traces back to the audio representation being musically shallow. The
quantitative metrics (onset_F1, crit_adj) score only the bottom layer — which is why they've looked
great while the chart still doesn't *feel* musical. **The playtest log is measuring the layers the
metrics can't see.**

---

## 2026-06-19 (round 2) — radar toggle playtest

### What was played
`outputs/radar_samples/chaos_air/` — radar `chaos=0.9, air=0.85` (others at dataset mean),
`guidance 2.0`, `pattern_temperature 0.7`, `no_jump_during_hold` on.

### Raw feedback (user)
> "chaos_air is totally unplayable. the density wasn't too bad, it was just so unintuitive with
> the music i couldn't establish any kind of rhythm or feel for what should come next."

Crucially the user **ruled out density** as the cause — it's the *rhythmic placement* that's the
problem: no steady pulse to lock to, and no ability to anticipate the next note.

### Commentary & hypothesis
The `chaos` radar dim is *defined* as off-grid-ness / note-quantization variety (see
`groove_radar.py`: chaos scores notes by their position within the beat — on-beat = 0, off-beat
subdivisions score higher). So cranking chaos to 0.9 **and** amplifying at g=2.0 did exactly what the
knob says: it pushed notes off the steady 4th/8th grid onto unpredictable subdivisions. The knob
*works*. But "unpredictable rhythm" is precisely "unplayable" for a human — a player anticipates from
two anchors: (a) the steady grid/pulse and (b) the music itself. Chaos destroys (a), and our weak
choreography never provided (b), so the player is left with nothing to read.

This sharply **reinforces H1 and spawns H4.** A human author's syncopation is *motivated*: the off-grid
note lands on a vocal stab or a snare fill, so it's still readable. Our model has **no melodic/percussive
features** to motivate off-grid placement, so amplified chaos is *rhythmic noise, not musical
syncopation*. Notably the user said `base_coherent` (chaos at its natural ~mean) felt human — so it's
not chaos-the-concept that's broken, it's **chaos amplified beyond what the (musically-blind) model can
justify.** Two confounds to separate: the air=0.85 (jumps) component, and the g=2.0 overshoot (H3).

**H4 (new):** Knobs that demand *musical justification* (chaos / syncopation) expose the feature gap
(H1) far more than *quantity* knobs (stream/voltage→density, freeze→holds). Quantity needs no musical
reason — more on-grid notes is still readable — so those sets should stay playable while chaos breaks.
Prediction to test: stream_voltage, freeze_holds, calm remain playable; chaos is the lone unplayable one.

### Action / next
- [ ] Disambiguate the chaos_air result: generate **chaos-only** (`chaos=0.9`) vs **air-only**
      (`air=0.9`) at g=2.0 — is the unplayability from off-grid (chaos) or jumps (air)? H4 predicts chaos.
- [ ] Generate **chaos at gentle guidance** (`chaos=0.9 --guidance 1.3`) — does moderate chaos become
      readable? Tests whether it's chaos itself or the g=2.0 overshoot (H3).
- [ ] Get the user's read on the other three sets (stream_voltage / freeze_holds / calm) — direct H4 test.
- [ ] If H4 holds, it's the strongest argument yet for the chroma+HPSS feature retrain (H1): chaos can
      only become *musical* if the model can see the musical events it should syncopate to.

### Connecting thread
The knobs are splitting into two families: **quantity knobs (density, holds, jumps)** that steer fine
because they don't need musical reasons, and **musicality knobs (chaos/syncopation, and ultimately
choreographic arrow-mapping)** that fail because the feature set is musically blind. Same root as the
"disconnected from the music" complaint and the choreography gap — every road keeps leading back to H1.

---

## 2026-06-19 — Session: conditioning knobs + first real playtest

### Sets played
- `outputs/style_samples/base/` and `base_coherent/` — no conditioning, density matched per song.
  `base_coherent` = `pattern_temperature 0.7` (less arrow-jumping).
- `outputs/style_samples/{sparse_ref,dense_ref,dense_ref_g1.4}/` — Step 3 reference-style transfer.
- `outputs/radar_samples/{chaos_air,stream_voltage,freeze_holds,calm}/` — groove-radar toggles (g=2.0).

### Raw feedback (user)
1. **`base_coherent` was genuinely fun — "a human author might have reasonably written them."**
   This is the first generated set that played well.
2. **The model places a jump *during* a hold** — fine for a keyboard, but a dance-pad player has only
   one free foot while holding, so a jump-during-hold is unhittable ("the Will Smith meme"). Wants this
   as an on/off decode knob. → **implemented `no_jump_during_hold`** (see commentary).
3. Earlier in session: the heavily style-guided samples (g=2.0 dense reference, ~0.9 density) felt
   "disconnected from the music besides musical time."

### Commentary & hypotheses

**H1 — Timing is solved; choreography (arrow mapping) is the open axis.** The onset head is
audio-driven with ROC-AUC ~0.81–0.95: the model knows *where* notes go in time. What it lacks is a
sense of *which arrow* maps to *which musical event*, because the 23-dim audio features are timbre +
energy only (MFCC, onset, spectral contrast) with **no pitch/melody/harmony** (no chroma, no source
separation). So the pattern head choreographs from learned generic statistics + previous steps, not
from musical content. That `base_coherent` *still* felt human-authored suggests the pattern head's
learned statistics are actually decent — the missing musicality is subtle, not gross. **Predicted
next lever: add chroma + HPSS (harmonic/percussive split) features and retrain** so arrows can track
melody vs drums. (Not yet tested — see roadmap.)

**H2 — Decode randomness was masking the model's quality.** `pattern_temperature 1.0` (added to fix
the old always-Left/jacks greedy collapse) injects arrow-jumping that reads as "unintuitive."
Dropping to 0.7 (`base_coherent`) was the single change that made charts fun. **Hypothesis: there's a
coherence/variety sweet spot around 0.6–0.8; 1.0 over-randomizes, greedy collapses.** Worth a
systematic temperature sweep judged by play-feel, not just panel-balance stats.

**H3 — High CFG guidance trades musicality for control.** g=2.0 forces density/style so hard it
overrides the audio onset head → notes everywhere → disconnected. The *knobs work*, but **strong
guidance and musical alignment are in tension.** Expect the radar sets at g=2.0 to feel more
"forced" than `base_coherent`; a gentler g≈1.3–1.5 likely keeps the steer while staying musical
(cf. `dense_ref_g1.4` was the playable dense one).

**Jump-during-hold knob (implemented):** while a hold is open, a pad player has one free foot, so any
pattern with ≥2 *fresh* presses on non-held panels is forbidden (closing the held panel and single
taps stay legal). Lives in the `hold_aware` automaton in `generate()`; exposed as
`export_typed_samples.py --no_jump_during_hold`. This is a *biomechanical playability* constraint, a
new category alongside the *musical* ones (crossovers) — note both are decode-time automata that need
no retrain, reinforcing that **a lot of "playability" is post-hoc constraint, not model capacity.**

### Connecting thread
Three of the project's biggest wins (always-Left fix, hold-aware decoding, crossover/jump constraints)
were **decode-time fixes, not model changes** — the model keeps being better than its default decode
makes it look. Standing hypothesis: **the base model is under-served by decode; squeeze decode (and a
musicality-aware feature set) before adding capacity.**

### Actions taken
- Implemented + tested `no_jump_during_hold` (24 generation tests pass).
- Added `--radar` and `--no_jump_during_hold` to `export_typed_samples.py`.
- Generated the four radar-toggle sets above for the next playtest round.

### Open / next
- [ ] Playtest the radar sets — does each dim *feel* like its name? (chaos=off-grid?, air=jumps?,
      stream/voltage=density?, freeze=holds?) Record per-dim feel below.
- [ ] Temperature sweep judged by feel (H2).
- [ ] Try radar at gentler guidance (g≈1.3) to test H3.
- [ ] Prototype chroma + HPSS features + retrain to test H1 (the choreography-musicality lever).

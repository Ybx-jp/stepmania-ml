# HANDOFF — ★ Active = taste-critic arc, critic-v3 REBUILD phase. Next = E1.2 ear-correlation on the E0.1-v2 set.

**Updated 2026-07-13.** This session (a) caught + fixed a STRUCTURAL flaw in the graded_v2 critic (it hard-truncated
at 2304 frames → tail-blind on ~70% of songs) via a **full critic-v3 rebuild**, and (b) rebuilt the **E0.1 best-of-N
spread set** (the first attempt was trash). The deployed GENERATOR is unchanged; this is critic + eval infrastructure.
Ship v1.0.0 remains a PARALLEL scheduled track (memory [[project-end-state-plan]]) — not touched this session.

## WHERE WE ARE
- **Deployed generator UNCHANGED:** v2 48th-grid `gen_motif_v2_48th_cont` + `--features highres_v2`; both CLIs default
  to it; canonical decode palette below is intact (no decode-behavior change this session → GOLDEN gate untouched).
- **★ critic-v3 BUILT + TRAINED** (`checkpoints/realism_critic_v3/best_val.pt`, gitignored). Closes 3 of the 4 captured
  critic gaps in one from-scratch model. Code: `experiments/realism_critic/{windowed_critic.py, train_critic_v3.py,
  eval_critic_v3_gates.py}`. Findings: `notes/critic_v3_findings.md`. **Two WINS measured** (length-fix + locality),
  **two SOFT SPOTS** (panel/shift) — see the active-thread section.
- **★ E0.1-v2 spread set GENERATING** (verify: bg job `b5r8l2irf` + `outputs/probe_results/e01v2_plain.log` +
  `outputs/e01v2_spread/logs/`). 27 songs (BPM×length factorial) × 5 candidates (plain + moderate-`--style` at
  g=1.0/1.5/2.0/3.0). When the 4 style runs finish: run `python scratchpad/e01v2_merge.py` (→ `outputs/e01v2_merged/`
  6→5-slot folders + `E01v2_RANKING.md`), then install (`src.utils.sm_install.install_to_stepmania('outputs/e01v2_merged')`).

## ★ ACTIVE THREAD — taste-critic arc (lineage `.claude/skills/experiment-design/experiment_lineage/taste-critic-arc.md`)
Goal: a taste-aligned critic that auto-SELECTS the best of N conditioning variants (best-of-N) so f48 generation "just
works" per-song. SELECTION, not PREDICTION. The EAR is the only ground truth for every offline metric.
- **critic-v3 (the rebuild):** WindowedLocalCritic = **soft-min over multi-scale overlapping windows** (2304/1152,
  1152/576, 576/288; tail always covered; soft-min = worst-region-dominates → non-gameable for the E4 optimization
  ladder) + **42-dim audio** (was `audio[:,:23]`) + **TYPED per-panel chart** (hold-type visible; was binarized).
  Objective = graded corruption ladder (jitter/panel/shift, rank+anchor) + a **LOCALITY term** (a tail-only corruption
  must drop TAIL windows only). From scratch (warm-start broken by the input change). WINS: R1 jitter mono 0.98;
  **length-fix** tail-response uniform short/mid/**long>3600** (the old critic truncated 180/200 val songs); **locality**
  clean (first/mid/last-third defect → only overlapping windows drop, 0.00 leakage). SOFT SPOTS: **panel mono 0.56 /
  shift mono collapsed 0.88→0.01** — root = within-window MEAN-pool is ORDER-destroying (v3.1 lever = mean+MAX / order-
  aware within-window pool). Preference objective (E2) rides on top LATER, once E0.1 labels exist.

## ★ NEXT ACTION / OPEN FORK (binding)
1. **Finish E0.1-v2:** when `b5r8l2irf` completes, merge + install, hand the user `E01v2_RANKING.md`.
2. **E1.2-redux (the arbiter):** correlate critic-v3's ranking of the 27×5 candidates with the user's TASTE ranking.
   - Correlates → best-of-N viable; the panel/shift soft spots don't bite selection → proceed to E2 (preference labels).
   - Diverges specifically on configuration quality → the order-aware within-window pool (v3.1) is the first fix.
   E1.2 is BOTH the critic payoff gate AND the arbiter of whether the soft spots matter. **Do NOT "fix" panel-blindness
   before the ear says so** (arc discipline: ear is ground truth).
3. Pending offline gate (cheap, optional, no user): **42-vs-23 audio ablation** (retrain critic-v3 with audio[:,:23],
   compare discrimination — does the full audio earn its keep).

## ⏳ AWAITING USER — binding questions
- **Rank the E0.1-v2 set** (once installed) in `outputs/e01v2_merged/E01v2_RANKING.md`: per song, `Rank` (best→worst),
  `Spread_real` (y/n), `Top_is_banger` (y/n). THE ORACLE IS THE USER'S TASTE — not vs plain, not vs a human chart
  (human-ref slot was DROPPED; the user found val human charts to be maxed gibberish). These rankings ARE E1.2's labels.
- Older still-open by-ear item (from a prior session, unrelated): silence-pad hangover re-confirm on a long-song play.

## THE E0.1 DESIGN (why the first set was trashed — don't repeat)
- **First set (TRASHED):** `--groove_select rich --hardest` picked axis-MAXED reference charts + `--match_radar`
  amplified their maxed radar → the model was forced into a crazy conditioning corner. User: "trash / maxed gibberish".
- **E0.1-v2 (correct):** MODERATE-radar songs (no maxed dim), **NO `--match_radar`**, guidance swept over a **FIXED
  MODERATE `--style`** (`stream=mod,voltage=mod,air=mod,freeze=mod`). VERIFIED mechanism: CFG guidance is a NO-OP
  without a conditioning target (`typed_model.py:539` — `do_cfg` needs radar/style/motif/figure), so the fixed style is
  what guidance amplifies. Songs picked by a custom binned selector (`scratchpad/e01v2_select.py`), driven into the
  exporter via **PATH-based `--song_filter`** (short titles over-match → 105 files; full `.sm` paths are unique; exclude
  comma-paths — they break the comma-split). BPM×length factorial tests whether the guidance knee correlates with
  cheap-to-bin audio attributes (a best-of-N predictor).

## CANONICAL EXPORT DEFAULTS (VALIDATED by `tools/check_export_defaults.py`)
The bare `export_typed_samples.py` run reproduces what the user plays; these MUST equal its argparse defaults.
**Permanent section — keep in every rewrite.** (UNCHANGED this session.)
<!-- CANONICAL-EXPORT-DEFAULTS:START (do NOT hand-edit values; re-run tools/check_export_defaults.py after a change) -->
```
checkpoint = checkpoints/gen_motif_v2_48th_cont/best_val.pt
features = highres_v2
type_temperature = 0.4
pattern_temperature = 1.0
repetition_penalty = 1.0
max_jack_run = 2
jack_penalty = 0.0
fatigue_penalty = 2.0
fatigue_free = 6.0
stamina_ceiling = 50.0
stamina_tau = 8.0
stamina_scale = 15.0
stamina_breathe = 1.2
onset_phase_calib = 0.0,1.0
auto_b_trip = True
triple_pref_thresh = 0.0
grid_snap = auto
grid_snap_keep_triplets = True
onset_window = 3600
onset_tail_hangover = auto
hold_stream_penalty = 8.0
hold_stream_floor = 0.45
hold_stream_win = 16
footswitch = False
hold_release_run = 4
hold_release_gap = None
hold_max_beats = 6.0
harm_calib = 0.0
harm_quiet_q = 40.0
guidance = 1.0
```
<!-- CANONICAL-EXPORT-DEFAULTS:END -->

## BRANCH / PR STATE (verify ALL live state via `gh pr view` / `git log origin/main`)
- This session's work is on **`explore/e01-bestofn-spread`** (branched off `main`). The refresh docs + critic-v3 code
  land via a PR from this branch — verify its number/state via `gh pr list`.
- `checkpoints/realism_critic_v3/`, `outputs/`, `transcripts/` are GITIGNORED (not committed).
- Untracked & NOT mine (leave): `.gitignore` mod adding `.claude/commands` (pre-existing, not this session's).

## READ-FIRST (in order)
Memory [[taste-critic-transfer]] (active thread) → this HANDOFF → lineage `taste-critic-arc.md` (Results 2026-07-13 =
the critic-v3 rebuild + the E4 3-gap capture) → `notes/critic_v3_findings.md` (the wins + soft spots + the mean-pool
order trade-off) → `notes/INDEX.md`. Load-bearing skills: **experiment-design** (the ear is ground truth; HARNESS→DATA→
MODEL — the flaw was caught by the user, verified in code before rebuilding), **generation-defaults** (§0 the bare run
now loads the RIGHT v2 model — the old "legacy gen_style default" TRAP note was corrected this session).

## DISCIPLINE
**The EAR is the deciding vote** — E1.2 (does critic-v3 rank like the user) arbitrates whether the panel/shift soft
spots matter; do NOT fix them offline first. **Verify volatile state in code, not from docs** — the critic flaw, the
CFG-needs-a-target mechanism, and the exporter checkpoint default were all confirmed by reading the code this session
(and the last two overturned stale claims). **One change at a time. Match the verb to the evidence** ([[claim-precision]]:
critic-v3 has MEASURED wins + soft spots, not a clean sweep). **Decode changes still face the GOLDEN gate** (none this session).

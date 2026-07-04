# HANDOFF — ACTIVE: good-settings TOLERANCE-FORMULA. Derived + EAR-CONFIRMED. Optimizing the fit (label-denoise run in flight).

**Written 2026-07-04 for the next Claude.** This session picked up the tolerance-formula derivation (tolerance =
f(song features)) and took it to an EAR-CONFIRMED result, then chased the second factor. All diagnostic + tooling —
NO model or decode-default change. An expanded label run is in flight (see AWAITING USER).

## WHERE WE ARE
Deployed model UNCHANGED = `checkpoints/gen_motif_full_fixed/best_val.pt` (42-dim highres). Canonical decode defaults
UNCHANGED (v0.2.0 block below). This session added probes + notes + a formula; it did NOT touch the generator or the
decode palette.

## THE ACTIVE THREAD — tolerance = f(song features): THE FORMULA (derived + ear-confirmed)
Goal: predict how far a song can be cranked before its 1/4 backbone flips to a 1/16 smear. **RESULT (durable):**
- **`env_strongbeat_frac` (SB)** = fraction of the audio onset-envelope MASS (highres dim 13) on strong-beat 16th
  frames (`t%4∈{0,2}`) — a pure audio×phase-grid feature, no reference chart, no model forward.
- **SB predicts per-song tolerance** ρ≈+0.63 / R²≈0.33 and **SUBSUMES** the old `real_density` lead (density needed a
  reference chart; partial density|SB → n.s.). BPM is null for tolerance (≠ the quality target).
- **SB predicts the per-song FLIP GUIDANCE** `g₀ ≈ 0.77 + 1.62·SB` to **±0.28 guidance-units** (ρ+0.72, R²0.44),
  via a per-song LOGISTIC-CLIFF fit on a dense guidance×k4 sweep (`probe_flip_point.py`). Cliff SHARPNESS `w` is an
  INDEPENDENT axis (SB predicts WHERE the cliff is, not HOW sharp).
- **PROSPECTIVELY EAR-CONFIRMED 3/3** (`notes/playtest_log.md`, 2026-07-04): 3 songs spanning SB × {below-g₀,
  above-g₀} — every SAFE chart coherent, every OVERLOAD degraded; the g=2.0 same-guidance-opposite-verdict
  cross-check landed; high-SB Take It gracefully degraded (shallow cliff). **The FIRST ear result on this thread to
  AGREE with the offline metric** (the prior 5 were ear-overturned misreads). BONUS: g=1.0 > g=2.0 expressiveness →
  g₀ is a per-song SAFETY CEILING; the sweet spot sits GENTLE-side (crank chaos, keep guidance gentle).
- **SECOND-FACTOR HUNT (Rule-6/CV-disciplined) = LABEL-NOISE-limited, NOT feature-limited:** density (the a-priori
  favorite) OVERFITS in LOO-CV (raises in-sample R², LOWERS CV); bpm/onset_rate likewise; `d22_std` (a harmonic
  channel) helps ongrid but HURTS anch (metric-inconsistent, untrusted); NO better SB variant (coarse onset_env beats
  the highres-onset dim34 — the smoothing IS the feature: chaos smears sustained WEIGHT not transients). SB sits at
  the CV ceiling (~0.26) for k=2/n=40 labels. The proven lever is DENOISE (k↑) + more songs — the flip-point k4
  target already fit at 0.44 vs the k2 scalar's 0.25.
Lineage `experiment_lineage/good-settings-region-arc.md`; findings `notes/tolerance_formula_findings.md`; memory
[[good-settings-region]].

## AWAITING USER / IN FLIGHT
- **Expanded flip-point run LAUNCHED this session** (background task `byhqlmhpx`; **verify it finished via the task
  output / `ls -la cache/flip_point_v2.csv`** — do NOT assume): 32 songs spanning SB **0.07–0.84** × dense 8-pt
  guidance × k4 → `cache/flip_point_v2.csv` (+ `.log`). **When it lands:** (1) re-fit `g₀ ~ SB` on the expanded CLEAN
  labels (tighter band?), (2) RE-HUNT the second factor on the clean labels + full 84-dim fingerprint
  (`cache/audio_fingerprints_highres.npz`, identity-keyed) — a real second factor could NOT surface under k=2 noise;
  now it gets a fair shot. Reuse the LOO-CV / nested-CV discipline from `tolerance_formula_findings.md`.
- **Then, user's open menu** (their call): OPERATIONALIZE (ship a per-song guidance recommendation from SB — cap at
  g₀, default gentle-side — as an `export_typed_samples.py` flag); or derive the RECOMMENDED guidance (the
  expressiveness PEAK below g₀, needs ears/a new proxy since expressiveness isn't offline-measurable); or scale
  further. The flip_test playtest is DONE (confirmed, logged).

## PARKED — the chaos×onset GATE (retrain ceiling-raiser; plan intact, revivable)
Full plan `notes/chaos_onset_gate_scope.md`. Phase-0 decode EXHAUSTED (off-beat placement is NOT audio-reachable —
`probe_chaos_onset_gate.py`); the retrain reshaped to the seq-onset head revived with chaos, Stage-1 de-risk GREEN
(frozen-`h` 0.862 on high-chaos ≫ audio 0.618, `probe_seqcontext_chaos.py`). User parked the train. Revive via the
scope doc's Phase-1. The tolerance formula does NOT depend on the gate (the gate would RAISE the ceiling for
low-tolerance songs; the formula PREDICTS tolerance under the current model).

## ⚠️ DISCIPLINE — this thread is a CAUTIONARY case study (read before continuing)
The good-settings thread had 5 ear/user catches (3 metric misreads + a pooled-n=2 OOD claim + a 2-change gate arm).
This session ADDED the counter-example: the tolerance metric, refereed correctly as an OVERLOAD DETECTOR, PASSED the
ear prospectively (3/3). Method keepers proven this session: **predict the target DIRECTLY (don't proxy density — a
mechanism-faithful audio feature BEAT+SUBSUMED the reference-chart lead); fit a LOGISTIC CLIFF (pools all points) not
a threshold crossing; judge added features by LOO-CV (in-sample R² just fits the k=2 label noise — density looked
great in-sample and FAILED CV); faithful playtest = re-measure the exported chart's own anchoring vs the k-mean, and
do NOT re-roll milder draws (biases the test).**

## CANONICAL EXPORT DEFAULTS (the deployed config — VALIDATED by `/refresh`)
The bare `export_typed_samples.py` run reproduces what the user plays. These values MUST equal the script's argparse
defaults — `tools/check_export_defaults.py` parses the block below and FAILS the refresh if they drift. Durable mirror
of `generation-defaults` §1; update both (and re-run the validator) on any deliberate change. **This section is
permanent — keep it in every HANDOFF rewrite.**

<!-- CANONICAL-EXPORT-DEFAULTS:START (do NOT hand-edit values; re-run tools/check_export_defaults.py after a change) -->
```
checkpoint = checkpoints/gen_motif_full_fixed/best_val.pt
features = highres
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
hold_stream_penalty = 8.0
hold_stream_floor = 0.45
hold_stream_win = 16
footswitch = False
harm_calib = 0.0
harm_quiet_q = 40.0
guidance = 1.0
```
<!-- CANONICAL-EXPORT-DEFAULTS:END -->

For the tolerance/flip probes: the milestone HIGH-chaos spec is `--style "chaos=0.9,voltage=0.7,air=0.5,freeze=0.5"`
(manifold snap → realized radar chaos≈0.44) swept over `--guidance`; everything else = the canonical block.

## RESOLVED (don't re-derive)
- **The tolerance FORMULA's predictor is `env_strongbeat_frac`** (audio strong-beat mass), ρ≈0.63, ear-confirmed. It
  predicts the flip guidance `g₀≈0.77+1.62·SB`. Density is SUBSUMED (and OVERFITS as a 2nd factor). See above.
- **The good-settings phenomenon is documented H4/H14** (chaos = global off-grid shift; guidance FLOODS off-beats past
  a per-song cliff). The backbone flip is CFG-amplified chaos (~70%) + 16th-calib (~30%), **NOT the governor** (0%).
- **Off-beat placement is NOT audio-reachable** (Phase-0 decode gate) — a decode gate keyed on audio can only do
  blanket ops. The real fix is a NOTE-CONTEXT (`h`) learned gate = the parked seq-onset retrain.
- **The tolerance metric (16th-anchoring cliff) is an ear-validated OVERLOAD DETECTOR** — it PASSED the ear
  prospectively (3/3). It locates the failure BOUNDARY; expressiveness (unmeasured) ranks the good ones.

## BRANCH / PR STATE  (verify ALL live state via `gh pr view <n>` / `gh pr list` / `git log origin/main`)
- **This session's work is on branch `feat/tolerance-formula`** (off the prior `docs/chaos-onset-gate-refresh` HEAD).
  Commit `20dabea` (the formula + probes + findings + playtest + lineage) + this refresh commit. **Verify pushed
  state / any PR via `gh pr list` / `git log origin/main..HEAD`.** The refresh opens a PR to `main`.
- New this session: `probe_{tolerance_audio_density,flip_point,sb_variants}.py`, `notes/tolerance_formula_findings.md`.
  Gitignored (reproducible): `cache/{tolerance_audio_density,tolerance_audio_density_n60,flip_point,flip_point_v2}.csv`
  + `cache/flip_point*.log`; the playtest set `~/sm-generated/flip_test/` (outside the repo).
- **Prior:** v0.2.0 shipped (PR #58). `--hardest` (PR #61) MERGED. `main` protected by `protect-main`.

## READ-FIRST (in order)
`notes/tolerance_formula_findings.md` (the formula + confirmatory CV + ear validation + the second-factor verdict) →
lineage `good-settings-region-arc.md` → `notes/goodregion_findings.md` (the referee: anchoring = overload detector) →
`notes/playtest_log.md` (the 3/3 ear confirmation) → `notes/chaos_onset_gate_scope.md` (the parked gate). Load-bearing
skills: **experiment-design** (LOO-CV discipline; this thread is its cautionary case AND now its counter-example),
**conditioning-mechanics** §2/§6 (manifold `--style`, CFG on onset, phase metric), **generation-defaults** §1
(canonical config the probes replicate), **playtest** (the by-ear gate; faithful-export re-measure).

## DISCIPLINE
Rule 0 (check notes first). **By-ear is the binding gate** (Rule 8 — it caught 5 misreads AND validated this formula).
**One change at a time; keep the canonical palette fixed** (Rule 11). **Judge added features by LOO-CV, not in-sample
R²** (the k=2 labels overfit — density looked great in-sample, FAILED CV). Stratify, don't pool (Rule 12). Match the
verb to the evidence ([[claim-precision]]). `playtest_log.md` = subjective only; quantitative → `notes/*_findings.md`.

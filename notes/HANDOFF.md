# HANDOFF — seq-onset wall CLOSED NEGATIVE, but the BUILD RE-OPENED CHEAP (M1a); EXECUTING M1b drift gate

**Written 2026-06-29 for the next Claude.** Prior sessions bounded the seq-aware-onset wall DEAD 4 ways (placement =
a chart-PRIOR, not in audio). THIS work asked the build-sizing question and got a favorable answer: **M1a
(`probe_seqcontext_frozenh.py`) shows the FROZEN deployed decoder's hidden state `h` ALREADY encodes the full
placement signal** — a capacity-matched conv readout on `h` hits **0.892 ≡ the note-context ceiling (100%)** (audio
floor 0.624; 1×1 readout 0.763 = a capacity confound caught by Rule 11). So fork (A) is NOT a from-scratch retrain —
it's a CHEAP onset-head ADD on a frozen decoder. M1a settles REPRESENTATION; **DRIFT is the lone binding gate, now
being built as M1b.** Env: conda `stepmania-chart-gen` — call the interpreter DIRECTLY
(`/home/ybx/miniconda3/envs/stepmania-chart-gen/bin/python`); **NOT `conda run`** (it buffers all child stdout until
exit → empty logs if the process is killed/polled).

**READ-FIRST (in order):** `notes/onset_frozenh_findings.md` (M1a — the freshest result: frozen `h` ≡ ceiling, the
cheap-build greenlight + the DRIFT boundary) → `notes/sequence_aware_onset_plan.md` (the M1a addendum + the SCOPING +
M0 + ANALYSIS-BY-SYNTHESIS + 4-WAY CONVERGENCE — the whole story) → lineage
`.claude/skills/experiment-design/experiment_lineage/seq-onset-arc.md` (hypothesis chain incl. M1a + the methodology
wins) → `onset-phrasing-calibrator-arc.md` (the parent active arc, with the nearest-shippable pending items). Load-bearing skills: **experiment-design** (Rule 11 positive-control — it's the
hero of this session), **conditioning-mechanics** §8 (the WHEN↔WHERE isolation note, now "CLOSED NEGATIVE 4 ways"),
**generation-defaults** (the canonical decode/export config).

---

## 1. WHERE WE ARE
Deployed model UNCHANGED: `checkpoints/gen_motif_full_fixed/best_val.pt` (42-dim highres) + the shipped governor
(canonical block below intact). M1a was a DIAGNOSTIC probe (no model change); M1b (in progress) WILL add an onset
head + a `generate()` branch — once it proves out by ear, not before. New tooling: `probe_seqcontext_frozenh.py`
(M1a — the frozen-`h` representation probe, the audio/both_real controls + the frozen_h/frozen_h_conv arms). New
findings: `notes/onset_frozenh_findings.md`. New caches (gitignored): `cache/seqctx_frozenh_{train,val}.npz` (800
train all-diff + 98 Hard val, typed states; both present → the probe skips the 4452-file dataset re-parse).

## 2. THE ACTIVE THREAD — seq-aware onset, CLOSED NEGATIVE (lineage `seq-onset-arc.md`)
The 0.87 teacher-forced note-context placement signal is a chart PRIOR, NOT recoverable from audio — confirmed FOUR
independent ways, every positive control FIRED:
| direction | result | vs |
|---|---|---|
| forward audio→16th onset | 0.65 | — |
| M0 seq refiner MATCHED (train-on-deployed-C0) | **0.672** | ceiling both_real 0.871 (fired) |
| analysis-by-synthesis critic, real vs CORRUPTED-placement (density held) | **0.570** | ~chance (N=28) |
| analysis-by-synthesis critic, real vs DEPLOYED-C0 | **0.468** | control real-vs-mismatch-song 0.815 (fired) |

⇒ the cheap own-output REFINER is DEAD (C0 carries no placement beyond audio even when trained to read it); the
ANALYSIS-BY-SYNTHESIS likelihood is DEAD for fine placement (audio is coarse/density-compatible only). In
`P(chart|audio) ∝ P(audio|chart)·P(chart)` all placement is the PRIOR → only a chart sequence model can author it.

**M1a (06-29) — that chart sequence model ALREADY EXISTS: the FROZEN decoder** (`probe_seqcontext_frozenh.py` →
`notes/onset_frozenh_findings.md`). 16th-AUC, 800 train / 98 Hard val:

| arm | 16th-AUC | read |
|---|---|---|
| audio | 0.624 | floor |
| both_real (raw-note CNN) | 0.892 | ceiling — POSITIVE CONTROL FIRED |
| frozen_h (1×1 readout) | 0.763 | 52% (a readout-capacity confound) |
| **frozen_h (conv readout)** | **0.892** | **100% — ≡ ceiling** |

The deployed decoder's `h` (`typed_model.py:635`) ALREADY encodes the full placement signal; a small causal-conv
onset head on the FROZEN decoder reaches the ceiling — no unfreeze, no dedicated note branch. **Attribution save
(Rule 11):** the 1×1's 0.763 looked like "decoder lost half the signal"; the capacity-matched conv arm overturned it
to 100%. **BOUNDARY: settles REPRESENTATION, NOT DRIFT** (`h` teacher-forced on REAL notes = the upper bound; gen-time
the head reads its OWN notes → snowball risk). DRIFT is the lone binding gate.

## 3. EXECUTING M1b (the binding DRIFT gate) — fork resolved toward (A)
M1a greenlit the cheap frozen-head build on REPRESENTATION → the strategic fork resolved toward **(A)**, now being
built as **M1b**. Plan (onset-head-only, decoder FROZEN):
- Add a causal-conv onset head reading the decoder's per-frame `h`; WIRE it INTO `generate()`'s loop — decide
  onset[t] from `h[t]` instead of the precomputed audio-only onset (today: `typed_model.py:481` precompute → the
  loop reads `h` at :635 for pattern/type only). Train the head (BCE on real onsets) with **own-output scheduled
  sampling** (the named fix for AR drift); decoder/pattern/type FROZEN, warm-start the audio onset branch.
- **DRIFT GATE = `diag_ar_stability.py`** (free-run density + run-length-distribution vs real — 06-22 free-run hit
  density 0.73 vs real 0.18; scheduled sampling only dampened to 0.66). Drift-taming stack: audio anchor +
  scheduled sampling + the STAMINA ceiling as a hard density backstop (cond-mech §8c). **KILL if it explodes.**
- `/autotune` BEFORE the retrain (batch/AMP/length-bucketing/Optuna). Deployed model stays `gen_motif_full_fixed`
  until M1b proves out **by ear** (Rule 8). If M1b's drift gate FAILS → fall back to (B) BANK + the nearest-shippable.
- **(Nearest-shippable FALLBACK, from the parent onset-phrasing arc):** (1) **perc-gate harm_calib re-A/B** — wire
  `--quiet_feat perc` into `export_typed_samples.py`'s `_sparse_harm_offset`, regen `~/sm-generated/harmcalib_ON`
  for HSL, user plays: do the 1/16s land IN the piano solo? (2) **1/16-jack OOD** — measure japa1 1/16-jack
  run-length vs real (`calib_foot_fatigue.py`) BEFORE a `fatigue_penalty` 2→3 A/B (by ear). Play-feel →
  `notes/playtest_log.md`; quantitative → a `notes/*_findings.md`.

## 4. RESOLVED THIS SESSION (don't re-derive)
- **M1a — the frozen decoder's `h` already encodes the FULL placement signal** (conv readout 0.892 ≡ ceiling). So
  fork (A) is a CHEAP onset-head ADD on a frozen decoder, NOT a from-scratch retrain. The 1×1 readout's 0.763 (52%)
  is a READOUT-CAPACITY confound, NOT "decoder lost the signal" — the capacity-matched conv arm overturned it to
  100% (Rule 11). Don't re-run the 1×1-only test and conclude the decoder is lossy.
- **`conda run` BUFFERS child stdout until exit** → empty logs when a run is killed/polled (cost two confusing
  cycles this session). Use the env interpreter directly: `/home/ybx/miniconda3/envs/stepmania-chart-gen/bin/python`.
- **Own-output iterative refiner = the 06-22 transfer gate in disguise** — looked re-openable (deployed C0 is
  neutral 0.667 vs v4's anti-correlated 0.456), but the MATCHED train-on-C0 test (M0) lands at 0.672 → the wall is
  C0-INDEPENDENT (any audio-only-placed C0 lacks the signal). This CLOSED 06-28's last residual confound.
- **Analysis-by-synthesis: three probe iterations, two false starts caught by controls** — v1 regression VOID
  (binary chart has no amplitude → worse than predict-mean); v2 critic CONFOUNDED (learned chart-coherence shortcut,
  control at chance 0.517); v3/v4 clean (mismatch-only training forced the audio path; control 0.815, measurement
  0.47–0.57). The mismatch-song positive control was the hero — it separated "audio-grounded" from "chart-prior."

## CANONICAL EXPORT DEFAULTS (the deployed config — VALIDATED by `/refresh`)
The bare `export_typed_samples.py` run reproduces what the user plays. These values MUST equal the script's
argparse defaults — `tools/check_export_defaults.py` parses the block below and FAILS the refresh if they drift.
This is the durable mirror of the `generation-defaults` skill §1; update both (and re-run the validator) on any
deliberate change. **This section is permanent — keep it in every HANDOFF rewrite.** (Unchanged this session.)

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
onset_phase_calib = 0,1.0
harm_calib = 0.0
harm_quiet_q = 40.0
guidance = 1.0
```
<!-- CANONICAL-EXPORT-DEFAULTS:END -->

## 5. BRANCH / PR STATE
- This refresh's docs are on branch **`docs/seq-onset-frozenh-m1a`** (off `main`), with PR **(this refresh)**.
  Prior refresh: **PR #49** (`docs/seq-onset-closed-negative` → `main`); earlier **PR #47**, **#48**. **Do NOT trust
  any merge/open state written here — verify live: `gh pr view <n>` / `git log origin/main`** (CLAUDE.md
  "Documentation Discipline"). `main` is protected by `protect-main`.
- New cached artifacts are gitignored (not committed): `cache/seqctx_frozenh_{train,val}.npz`.

## 6. INFRA / PERF NOTES (cost real time — know these)
- **Deployed-C0 generation throughput:** ~10 s/chart, ~98 frames/s (B=1 AR) on the RTX 3060. 4 PROCESS shards ≈
  1.8× (GPU saturates at 99% with 4 streams — compute-bound, not launch-bound; more shards won't help). **Batched
  `generate()` is FORBIDDEN for C0** — `onset_threshold`/`bpm` are batch-scalars; batching mis-applies one song's
  tau/bpm to all → a confounded C0. Use `gen_train_c0.py`'s shard pattern.
- **Stale-cache footgun (re-bit):** `cache/seqctx_train_cache.npz` (800 rows) is STALE vs the current split (786
  songs / 3820 charts) — row j ≠ valid_samples[j]. Always re-extract FRESH from the current split ([[dataset-cache-footgun]]).
- **autotune skill** (never run): the right tool BEFORE the AR-head retrain (batch/AMP/length-bucketing/Optuna).

## 7. DISCIPLINE (load-bearing — this session proved it twice)
- **experiment-design Rule 11 (confirm the metric can MOVE / the positive control fires):** the analysis-by-synthesis
  v1 (predict-mean floor) and v2 (mismatch-song at chance) NULLs were both CAUGHT as VOID/CONFOUNDED by their
  controls — without them I'd have reported false findings. ALWAYS pair a null with a fired positive control.
- **Rule 0** (grep notes + lineage + skills BEFORE a probe — the refiner was a re-disguised 06-22 gate);
  **HARNESS→DATA→MODEL**; **by-ear is the binding gate** (Rule 8) for the shippable items.
- **One change at a time;** `playtest_log.md` = subjective only; quantitative → `notes/*_findings.md`; arc → lineage.

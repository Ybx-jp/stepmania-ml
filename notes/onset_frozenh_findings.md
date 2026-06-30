# M1a — frozen-decoder-h onset readout: representation sufficiency (2026-06-29)

**Thread:** seq-aware onset, fork (A) build-sizing (lineage `seq-onset-arc.md`). **Probe:**
`experiments/generation_typed/probe_seqcontext_frozenh.py`. **Status:** POSITIVE on the REPRESENTATION axis;
DRIFT untested (the binding M1b gate).

## Question
The seq-onset wall is CLOSED NEGATIVE 4 ways → the only path to the 0.87 placement signal is a causal-AR onset
retrain. Before that retrain, size its CHEAPEST variant: a small onset head bolted onto the EXISTING decoder's
per-frame hidden state `h` (`typed_model.py:635`, `h = self._decoder_step_cached(...)`). `h[t]` already fuses
audio (cross-attn) + the CAUSAL note history (self-attn over states[<t]) — the probe's `both` info set, but routed
through the FROZEN deployed decoder (trained for the PATTERN head). **Does the frozen `h` preserve the placement
signal the raw-note CNN extracts (the 0.87 `both_real` ceiling)?**

## Setup (mirrors the seqcontext harness; controls comparable to the arc)
- 800 REAL train songs (all difficulties), eval on 98 Hard val songs. 16th-localization AUC. TARGET = real onset.
- `h` extracted teacher-forced over REAL typed states, NATIVE mode (radar/style/motif/figure=None, matches `gen_c0`),
  causal: `h[t]` sees states[<t] only (`_decoder_input` shift + causal mask) — same strictly-past discipline as `both`.
- Arms: `audio` (Probe, floor) · `both_real` (Probe, audio+raw causal notes — the **positive control**, must hit ~0.87)
  · `frozen_h` (1×1 per-frame readout on `h`) · `frozen_h_conv` (4-layer dilated causal-conv readout on `h`,
  **capacity-matched** to `both_real`'s note branch).

## Result
| predictor | onset-AUC | 16th-AUC | gap recovered |
|---|---|---|---|
| audio | 0.888 | **0.624** | floor |
| both_real | 0.945 | **0.892** | ceiling — POSITIVE CONTROL FIRED (0.892 ≫ 0.624) |
| frozen_h (1×1) | 0.858 | **0.763** | 52% |
| **frozen_h (conv)** | 0.952 | **0.892** | **100%** |

## Verdict (REPRESENTATION): the frozen decoder ALREADY encodes the full placement signal
- `frozen_h_conv` = 0.892 ≡ `both_real` (and onset-AUC 0.952 > 0.945). The frozen deployed decoder did **not**
  compress placement away — a small causal-conv readout on `h` surfaces 100% of what the raw-note CNN extracts.
- The 1×1 arm's shortfall (0.763, 52%) was **readout capacity** (per-frame, no temporal mixing), not lost signal —
  this confound was caught and killed by the capacity-matched arm (experiment-design Rule 11). The signal lives in
  `h` but is spread across neighboring frames; a ~4-layer causal conv reads it.
- **BUILD CONSEQUENCE:** the cheapest M1b is viable on representation — a small causal-conv onset head reading the
  **FROZEN** decoder's `h`; no unfreeze, no dedicated note branch. The deployed decoder is a sufficient sequence
  encoder for placement; today's decode just never reads onset off it (`h` → pattern/type heads only).

## BOUNDARY — what this does NOT settle (experiment-design Rule 9/10)
This is REPRESENTATION sufficiency, NOT DRIFT. `h` here is teacher-forced on **REAL** notes = the upper bound a
frozen-head readout could ever see. At generation time the head reads the model's OWN generated notes, and the
onset→note→`h`→onset feedback can snowball (the 06-22 `diag_ar_stability`: free-run density 0.73–0.83 vs real 0.18;
scheduled sampling only dampened to 0.66). **A high number here does NOT prove the causal-AR head works at gen
time.** The frozen-head finding only sharpens the drift question: since the readout is on the same `h` the pattern
head already produces, gen-time "own context" = the `h` chain itself — does placing onsets from `h`, fed back into
the next `h`, stay near real density?

## Next (M1b drift gate — the binding test)
Wire a causal-conv onset head reading `h` INTO `generate()`'s loop (decide onset[t] from `h[t]` instead of the
precomputed audio-only onset), keep the decoder frozen, train the head (BCE on real onsets) with own-output
scheduled sampling, then run `diag_ar_stability.py`: free-run density + run-length-distribution vs real. KILL if it
explodes after the drift-taming stack (audio anchor + scheduled sampling + the stamina ceiling as a density
backstop, cond-mechanics §8c). Deployed model stays `gen_motif_full_fixed` until it proves out by ear (Rule 8).

## Repro
`/home/ybx/miniconda3/envs/stepmania-chart-gen/bin/python experiments/generation_typed/probe_seqcontext_frozenh.py
--max_train 800` (NOT `conda run` — it buffers output until exit; use the env python directly). Caches:
`cache/seqctx_frozenh_{train,val}.npz` (gitignored); both present → dataset re-parse skipped. d_model=128.

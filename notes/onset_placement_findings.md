# M1b-4 — PLACEMENT QUALITY: the cheap frozen-decoder SS build is PLACEMENT-HOLLOW (2026-06-29)

**Thread:** seq-aware onset, fork (A) (lineage `seq-onset-arc.md`). **Probe:**
`experiments/generation_typed/probe_seqonset_placement.py` (reuses the saved M1b-3 head `cache/seqonset_ss_head.pt`;
trains + caches a pure-TF ceiling head `cache/seqonset_tfceiling_head.pt`). **Status:** NEGATIVE for the metric that
matters — M1b-3 broke the DENSITY/run-length drift, but the free-run head does NOT place 16ths on the right frames.
The cheap build (frozen decoder + dropout-SS head) is placement-hollow. This OVERTURNS the optimistic M1b-3 read.

## The question (the binding NEXT from `onset_ss_findings.md`)
M1b-3 showed the dropout-SS head free-runs at real density (0.27) + real run-length (1.0). But density/run is
STABILITY, not musicality. This probe asks: at the operating point, does the free-run head RANK real 16th-onset
frames above non-onset 16th frames — toward the M1a teacher-forced ceiling (16th-AUC 0.892) or only the audio floor?

## Setup — a bracketed AUC made fair by an embedded positive control (exp-design Rule 11)
16th-AUC = among 16th-OFFBEAT frames (t%4∈{1,3}), AUC of predicted onset prob vs real onset (the EXACT M1a metric,
`diag_seqcontext_probe.auc`, pooled over val frames). Three arms, SAME Hard-val songs, predictions pooled:
- **FLOOR** = the DEPLOYED audio-only onset head (`model.onset_logits`, native, NON-causal) — the head this would replace.
- **CEILING** = a PURE teacher-forced conv head (HReadConv on real-note `h`, NO scheduled-sampling dropout) = the M1a
  0.892 representation upper bound, re-measured on THIS set. **POSITIVE CONTROL.**
- **FREE-RUN** = the SS head free-running (own-note `h`, the incremental `_decoder_step_cached` decode = `generate()`).
- (secondary) **SS head TF** = the SS head on real-note `h` — quantifies how much TF accuracy SS training sacrificed.

**FIRST RUN'S CONTROL FAILED — and the failure was the harness, not the model (HARNESS→DATA→MODEL).** I had used the
SS head's TF pass as the "ceiling" (0.736), which did NOT clear the floor (0.751). Cause: the SS head is trained with
heavy note-dropout (dmax 0.9) to be *free-run-robust*; its TF pass is **not** the representation ceiling — SS trades
~0.10 of TF placement accuracy for drift-robustness. Fix = the pure-TF conv ceiling arm (0.839 >> floor → control fires).

## Result (12 Hard val, cap 512; pooled 16th-AUC)
| arm | onset-AUC | 16th-AUC | read |
|---|---|---|---|
| FLOOR (deployed audio head, non-causal) | 0.916 | **0.751** | the head we'd replace |
| CEILING (pure-TF conv, real `h`) | 0.949 | **0.839** | POSITIVE CONTROL fired (>> floor; cf M1a 0.892) |
| SS head, TF `h` | 0.931 | 0.736 | SS sacrificed −0.103 of TF placement |
| **FREE-RUN (SS head, own `h`)** | 0.41–0.60 | **0.43–0.63** | **≤ floor across the plateau — MEASUREMENT** |

Free-run 16th-AUC by tau (the calibration cliff makes point estimates noisy; DIRECTION is robust):
`tau 0.45 → 0.457 · 0.50 → 0.575 · 0.55 → 0.625 · 0.56 → 0.426`. Best (0.625) is still **0.13 below the audio floor
(0.751)** and **0.21 below the ceiling (0.839)**. Realized 16th-placement at the operating tau: **precision 0.02–0.04**
(of the 16ths it fires, ~96–98% are NOT where real 16ths are; recall 0.20–0.67 is saturation from over-firing, not aim).

## Why this is the decisive metric (exp-design Rule 1 — the metric must SEE the property)
M1b-3's "wall broken" rested on DENSITY (0.27) + RUN-LENGTH (1.0) matching real. **You can match both by firing the
right NUMBER of isolated onsets on the WRONG frames** — and that is exactly what happens (precision 0.04). The
density/run metrics are BLIND to placement; they moved the right way while placement broke. So M1b-3 broke the
density-drift wall, NOT the placement wall.

## Attribution (clean, via the bracket)
- Real-note causal context (CEILING 0.839) genuinely beats the non-causal audio floor (0.751) → chart-context DOES
  carry 16th placement, and causality is not the killer (a causal head with real notes wins).
- Own-note free-run collapses to ≤ floor → **drift destroys the chart-context placement advantage.** The 0.839
  representation (M1a) is CONTINGENT on the real notes being in context; the head cannot bootstrap the 16th placement
  from its OWN audio-density-placed notes, because that placement is precisely the chart-PRIOR the audio lacks — the
  4-way-closed wall (`seq-onset-arc.md`) reasserting itself one level up (M1a exposed the signal GIVEN real notes; it
  is not free-run-authorable from audio).
- The halves split was sign-unstable across taus (+0.092 / −0.149) → NOT a clean late-rollout snowball; the placement
  is weak THROUGHOUT, consistent with "the head never reconstructs 16th placement from own context," not "drift accrues."

## Verdict — the CHEAP frozen-decoder build is placement-hollow (re-closes what M1a/M1b-3 re-opened, for QUALITY)
M1a (representation in `h`) + M1b-3 (density drift fixable by dropout-SS) were both correct on their own terms, but the
metric that ships — does it place 16ths musically — fails. Bolting this head into `generate()` would fire at the right
density on the WRONG frames (likely WORSE by ear than the current deployed audio onset head; precision 0.04).

## Boundary / what would change the conclusion (Rule 9/10)
- RANKING under drift, governors OFF, greedy tap-only — NOT a playtest (by-ear stays the binding gate, untested here).
- 12-song cap-512 set: the floor-ceiling gap is COMPRESSED (0.088 vs M1a's 0.27 on ~28 songs) → recovered-% is a noisy
  ratio; the DIRECTION (free ≤ floor ≪ ceiling, precision 0.04) is robust across 4 taus + 2 runs, but the magnitude is not.
- **Untested escalation (the open door):** true own-output rollout SS / DAgger, or an AUDIO-ANCHOR (blend the audio
  onset logits with the `h`-readout so placement stays audio-grounded), or a joint unfreeze-retrain. The CHEAP build is
  dead for placement; an escalation is the EXPENSIVE retrain the arc was trying to avoid and is UNPROVEN — do not assume
  it rescues placement (the wall says the 16th prior is not in the audio in either direction).

## NEXT (decision)
- **BANK fork (A)** as: representation present (M1a), density-drift fixable (M1b-3), **placement-quality NOT recovered by
  the cheap dropout-SS build (this)**. Fall to the nearest-shippable on the verified-intact governor/conditioning stack
  (HANDOFF §3 fallback): (1) perc-gate `harm_calib` re-A/B (HSL piano solo), (2) 1/16-jack OOD `fatigue_penalty` 2→3
  measurement. OR
- **Escalate** to audio-anchor / true rollout-SS only with eyes open that it's the expensive path and the prior wall
  predicts low odds for fine placement.

## M1b-5 — the TASTE-CRITIC gate confirms it (the "AUC too strict" hypothesis is REFUTED, not just unconfirmed)
**Probe:** `experiments/generation_typed/probe_seqonset_critic.py`. The user's sharp objection to M1b-4: 16th-AUC is
measured vs ONE reference chart, so it penalizes musically-VALID-but-DIFFERENT placement (a false negative). The fair
gate for musicality is the realism/taste critic (a learned P(real), NOT exact-match) — which by design rewards valid
alternative phrasing. So this A/B routes the seq head's onsets through the DEPLOYED `generate()` (via the sanctioned
`onset_override`, NO loop surgery — exp-design Rule 14) with the canonical governor/playability config
(generation-defaults skill) and scores with the critic. ONE change (onset trajectory), density-matched to the seq
head (d_seq), radar off, stamina off for BOTH arms (override skips it), per-note fatigue + playability on.

Result (8 chaotic Hard songs, d_seq 0.338 vs real 0.371):
| arm | critic P(real) | read |
|---|---|---|
| REAL | 0.727 | CONTROL high |
| shuf16 | 0.270 | CONTROL low (broken) — control FIRED (REAL ≫ shuf16) |
| AUDIO@real_d | 0.010 | deployed baseline at REAL density (critic is density-brittle: 0.37 is dense) |
| **AUDIO@d_seq** | **0.253** | the FAIR one-change baseline (= the known ~0.27 *generated* score) |
| **SEQ@d_seq** | **0.005** | MEASUREMENT — far below the baseline at the SAME density |

The critic is NEAR-BINARY (taste-critic-transfer: REAL ~0.82 / generated ~0.27), so read it per-song: on the 2 songs
where it fires "real-like," it is the **deployed AUDIO baseline that passes (0.98, 0.99) and SEQ that fails
(0.00, 0.01)**; SEQ NEVER clears 0.01 on any of 8 songs, AUDIO clears on 25%; SEQ ≤ AUDIO on every song.
⇒ **the lenient musicality gate ALSO ranks SEQ far below the deployed audio path.** The "AUC-vs-real too strict"
hypothesis is REFUTED — it is not a metric artifact; the seq head's free-run placement is genuinely worse than what
we ship. (Caveat: the critic is coarse/near-binary and `onset_override` puts the deployed pattern head slightly OOD
on BOTH arms equally; the binding by-ear gate (Rule 8) was NOT run — the critic gate failed first, the cheaper filter.)

## M1b-6 — the FAILURE MODE: a self-generated 16th-FLOOD (backbone collapse), by-ear + measured
**Probe:** `experiments/generation_typed/probe_seqonset_phase.py`. By-ear (the user, on the M1b export) the seq
charts read "bland, only 1/16s" — the chaos-OOD smear signature — and the user asked if chaos conditioning leaked.
It did NOT (radar=None in the rollout cond, the `generate()` call, AND the audio baseline — verified in code). But
the phase distribution (16th-grid shares, 8 Hard val) confirms the by-ear read is REAL — the seq head produces a
chaos-LIKE flood on its OWN:
| arm | quarter | 8th | 16th | density |
|---|---|---|---|---|
| REAL | 64 | 32 | **4** | 0.276 |
| audio head (raw) | 58 | 42 | 0 | =d_seq |
| audio + 16th-unlock | 57 | 26 | 17 | =d_seq |
| **SEQ head free-run** | **19** | **19** | **63** | 0.367 |

The SS head free-run is **62% 16th-offbeat with the quarter/8th backbone COLLAPSED** (19/19 vs real 64/32) — a
uniform off-grid smear (the H4/H16 chaos-flood failure), self-generated with NO chaos knob. The audio arms on the
SAME harness give sane backbone-heavy shares (controls clean) → this is the seq head's genuine free-run fixpoint, not
a harness bug. This is the MECHANISM behind M1b-4's free-run 16th-AUC **< 0.5** (it ranks 16th frames ABOVE backbone
frames) and precision 0.04: the head doesn't merely misplace 16ths, it INVERTS the rhythm — abandons the quarters it
can't author from audio+own-notes and floods the offbeats. **Methodology win (Rule 1 + Rule 8):** 16th-AUC and the
taste critic both compressed this to "worse" without naming it; the PHASE-SHARE metric SEES the property, and the
user's EAR caught it first. The by-ear gate was load-bearing — it surfaced the failure MODE the aggregate metrics hid.

## Verdict (M1b-4 + M1b-5 + M1b-6 together) — BANK fork (A): the cheap build is placement-hollow, confirmed three ways
Exact-match AUC (M1b-4: free-run ≤ audio floor, 16th precision 0.04), the metric-agnostic musicality critic (M1b-5:
SEQ ≪ deployed baseline), AND the by-ear-confirmed phase distribution (M1b-6: a self-generated 16th-flood, backbone
collapsed) all agree. Density-drift is broken (M1b-3) but placement quality is not recovered by the cheap
frozen-decoder + dropout-SS build — its free-run fixpoint is a backbone-less off-grid smear. Untested escalations (audio-anchor / true rollout-SS / joint retrain) remain the
EXPENSIVE path the arc sought to avoid, and the 4-way wall predicts low odds for fine placement. Fall to the
verified-intact governor stack's nearest-shippable (HANDOFF §3): perc-gate `harm_calib` re-A/B, or 1/16-jack OOD
`fatigue_penalty` 2→3 measurement.

## Repro
`/home/ybx/miniconda3/envs/stepmania-chart-gen/bin/python experiments/generation_typed/probe_seqonset_placement.py
--load_head` (env python directly, NOT `conda run`). `--tau`/`--robust_taus` to move the operating point; first run
trains+caches the pure-TF ceiling head. Critic gate: `probe_seqonset_critic.py --load_head` (needs the deployed model,
`cache/samples_v3`, the realism critic, the SS head). Needs the M1a caches + the M1b-3 SS head. Pairs with
`onset_ss_findings.md` (M1b-3, the density-drift break this re-scopes), `onset_frozenh_findings.md` (M1a ceiling).

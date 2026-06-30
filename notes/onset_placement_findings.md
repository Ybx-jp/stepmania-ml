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

## ⚠️ CORRECTION (M1b-7/8, 06-29) — "placement-hollow" was OVERSTATED; the truth is AUDIO-PARITY, not dead
User pushback (correct, an experiment-design catch): M1b-4/5/6 measured ONE under-tuned config (native `radar=None`,
the dmax=1.0 over-firing SS head, a global tau) and banked the build — but the 16th-flood is the KNOWN chaos-smear
failure (conditioning-mechanics §2), and the skill's OWN Evidence section is the canonical case where a backbone-
collapse committed as "the model can't" was overturned by the coherent-conditioning fair test. I committed model-blame
on the signature config artifact WITHOUT running the fair version (Rule 7) — and treated three metrics on ONE config as
robustness (they're one operating point measured three ways, NOT three independent tests; Rule 11).

Two fair tests followed:
- **M1b-7 manifold conditioning (`probe_seqonset_cond.py`):** condition the rollout on a backbone-heavy manifold groove
  (`build_target`, conditional-fill). NO effect — phase shares identical to the decimal. Diagnosed (Rule 11, dynamic
  range): the channel is LIVE but WEAK — groove changes the seq head's logits only ~3% (mean|Δlogit| 0.099 vs scale
  2.97), because the seq head reads the decoder's `h` and `h` barely encodes groove (the deployed onset head takes
  `radar` DIRECTLY — that's why the user's manifold fix worked THERE but doesn't transfer to this head's wiring).
  INCONCLUSIVE (OOD: head trained native) → the faithful fix is a radar-DIRECT conditioned head (a retrain).
- **M1b-8 phase-rebalance (`probe_seqonset_phasepen.py`):** a penalty on the seq head's 16th-offbeat logits.
  | arm | phase q/8/16 | precision | F1 |
  |---|---|---|---|
  | seq flood (b16=0) | 19/19/62 | 0.24 | 0.27 |
  | seq rebalanced (b16≥1) | 50/50/0 | 0.62 | **0.71** |
  | deployed audio head @ matched density | 55/28/17 | 0.61 | **0.70** |
  The flood DRAINS to a real-aligned backbone (precision 0.24→0.62) → the flood was a recoverable DECODE ARTIFACT, NOT
  an intrinsic dead end → **"placement-hollow / dead" was WRONG.** BUT calibrated against the deployed audio head, the
  rebalanced seq backbone reaches **F1 0.71 ≈ 0.70 = PARITY** — it reproduces the audio backbone without the
  sequence-context 16th advantage that was fork (A)'s whole point (the 0.87 teacher-forced ceiling does NOT survive
  free-run; the best free-run does is match audio, at 0% 16ths vs audio's 17%). The penalty is also binary (b16=1
  saturates → 0% 16ths; no graded control to hit real's 4%).

## M1b-9 (06-29) — the DECODE SURFACE is HEAD-SPECIFIC; the FAIR surface BUILT + playtested → path ALIVE, undertuned
The user's deeper reframe (correct): the whole prior comparison was a heavily-TUNED audio head vs an UNTUNED new head
on the audio head's OWN decode palette. The palette doesn't transfer — knobs BREAK/INVERT/are ABSENT for the seq head:
- **tau** — global density quantile assumes the audio head's non-causal calibrated `p_onset`; the seq head's concentrated
  in-loop logits make a fixed tau a per-song CLIFF (flood↔collapse). FIX = `seqonset_decode.selfcal_tau` (binary-search
  best-tracking to a target density; a quantile-iteration DIVERGED, collapsing a song to empty).
- **onset_phase_calib (16th-unlock)** — POLARITY FLIPS: the audio head 16th-UNDER-fires (the +1.0 unlock lifts it); the
  seq head 16th-OVER-fires → needs a DOWN-weight (the M1b-8 phase penalty).
- **rests** — the audio head is naturally silent in quiet (energy-tracking); the seq head NEVER pauses → needs an EXPLICIT
  rest valve sourced from the audio head's `p_onset` energy envelope (`seqonset_decode.build_rest_env`, the deployed
  breathing math). Built + measured (`probe_seqonset_rest.py`): rests/1k **1.95→3.9** toward real 5.1 at rest_gain~3.
- UNAFFECTED: per-note fatigue/jack (govern "where") — but the pattern head is OOD on the seq onset trajectory ("jumps").

FAIR-surface A/B regenerated (`export_seqonset_ab.py --rest_gain 3`, `~/sm-generated/seqonset_ab_fair`; all songs self-
cal'd to real density). **By-ear (the user): "it's better! still very linear."** A real improvement (it pauses now),
still clearly behind the deployed audio head. User's read: NOT a fair test yet — the tuning is unfinished, exactly as the
AUDIO decode was when it first landed (which took many hours of vibe-research to blossom).
**HOLD-RELEASE HYPOTHESIS (user, UNTESTED — the sharpest next lead):** the head may NOT have learned to REST — it may be
using a hold-release phantom note to stave off collapse. Worth a direct probe (does the "rest" coincide with hold tails?).

## CORRECTED VERDICT (supersedes the "hollow/dead/banked" framing below) — path ALIVE + undertuned; the fork is STRATEGIC
Fork (A) is NOT placement-hollow/dead. The 16th-flood was a measurement on the WRONG (audio-tuned) decode surface; a
head-appropriate surface (adaptive tau + inverted phase lever + rest valve) drains the flood to a real-aligned backbone
(precision 0.24→0.62 ≈ the audio head's 0.61) that now PAUSES and sits at real density. It is "better, still very linear"
— viable-but-early, **like the audio decode when it first landed.** The sequence-context 16th ADVANTAGE (M1a 0.84–0.89)
is reachable teacher-forced but does NOT yet survive free-run; whether a properly-tuned/retrained head reaches it is OPEN.
**The decision is now STRATEGIC — is the seq-onset path the right investment for THIS stage of the project? — not "is it
viable."** Open technical leads (next session): the hold-release-phantom-rest hypothesis; the per-song density cliff; the
"still linear" gap; a radar-DIRECT / phase-aware conditioned RETRAIN (the faithful fix — the inference-time manifold
channel is only a 3% echo, M1b-7). The M1b-4/5/6 sections below stand as RAW measurements; their FRAMING is superseded.

## (superseded framing) Verdict (M1b-4 + M1b-5 + M1b-6) — read with the CORRECTION above
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

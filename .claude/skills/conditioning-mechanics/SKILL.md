---
name: conditioning-mechanics
description: >
  Exact reference for how this project's chart-generator conditioning and decode math actually work, so
  probes, evals, and chart exports REPLICATE the real mechanism and their results match expectations. Use
  BEFORE writing or reviewing any code that SETS a conditioning knob (groove radar, continuous motif, discrete
  figure, CFG guidance, density / onset threshold, phase calibration) or MEASURES its effect — and before
  concluding a knob "doesn't work." Distilled from real misalignments in this project: mean-pin vs manifold
  conditional-fill, raw figure-mass vs radar-orthogonal knob, tau computed from unconditioned logits. Pairs
  with the experiment-design skill (which covers attribution); this one is the ground-truth on the math.
---

# Conditioning & decode mechanics

The single most expensive bug class here is a **probe/export that doesn't replicate the deployed conditioning
path** — you then measure the wrong thing and "learn" something false (e.g. "the chaos knob smears" / "the
candle knob did nothing"). This skill is the exact mechanism so the harness matches reality. **Golden rule: a
probe must build its conditioning EXACTLY as `export_typed_samples.py` / `generate()` do, and measure with the
SAME metric the conditioning targets.**

**Pairs with the `experiment-design` skill — use both, in order.** This skill = the MATH (what the deployed path
does); experiment-design = the DISCIPLINE for using it (attribution order HARNESS→DATA→MODEL; the pre-flight
checklist; don't blame the model for a harness bug). The mapping: experiment-design Rule 2 (match deployment) and
Rules 3–4 (in-distribution coherent inputs) are *satisfied* by replicating §1–§8 here; experiment-design's catalog
of attribution errors is the failure half, this skill's "Catalog of REAL misalignments" is the mechanism half of
the same bugs. Reach for experiment-design BEFORE a probe to pick a fair setup; reach for this to make the setup
literally match the code.

Model: `LayeredTypedChartGenerator` (`src/generation/typed_model.py`). Deliverable: `gen_motif_full` (42-dim
highres audio; radar + continuous motif + discrete figure). Pipeline per frame: **onset head** (audio-driven,
non-causal, which frames get a note) → **pattern head** (AR, which-panels, 15-way) → **type head** (per-panel
tap/hold/tail/roll). Frame = one 16th note; `tau` (onset threshold) sets density.

## 1. The conditioning vector `_cond` — what feeds what
`_cond(difficulty, radar, style, motif, figure)` returns a per-position vector, **added** to the sequence
embedding (additive conditioning):
```
c = diff_emb(difficulty) + (radar_proj(radar) | null_radar) + (style | null_style)      # (B,d)
c = c.unsqueeze(1)                                                                       # (B,1,d)
c = c + (motif_proj(motif)  | null_motif)        # (B,K)->(B,1,d)  OR  (B,T,K)->(B,T,d)  per-frame schedule
c = c + (figure_embedding(figure) | null_figure) # (B,)->(B,1,d)   OR  (B,T)->(B,T,d)    per-frame schedule
```
- Returns **(B,1,d)** (broadcast over time) normally, or **(B,T,d)** when motif/figure is a per-frame schedule.
- **ONSET HEAD IS DECOUPLED from motif & figure** (`onset_logits` calls `_cond(..., motif=None, figure=None)`).
  Motif/figure shape WHICH panels (a pattern-head concern); density/timing is the radar+onset's job. Coupling
  motif→onset made CFG inflate density — don't undo this. Onset sees difficulty + radar + style only.
- `null_*` are LEARNED CFG-dropout tokens, NOT zero and NOT "the mean". `motif=None` (null_motif) ≠ `motif=0`
  (motif_proj(0)=bias). Zero-init of `motif_proj`/`figure_embedding` is ONLY the warm-start no-op, not a default.

## 2. Groove radar + the MANIFOLD (the part most often gotten wrong)
Radar = 5 dims `[stream, voltage, air, freeze, chaos]` (normalized ~0–1). `radar_proj: 5→d`.
- **The dims are CORRELATED** (stream/voltage/air/chaos cluster r 0.71–0.92; freeze ~orthogonal). So setting one
  dim and pinning the rest at the mean is **OFF-MANIFOLD** and the model behaves degenerately.
- `--radar "chaos=0.9"` = pin the named dim, **others at the dataset mean** → OOD. Use only for a deliberate
  "see the raw reach" test. **At high single-dim values this SMEARS** (chaos=0.9 g3 → 16th-share 0.98, quarter
  backbone ~0 — a uniform off-grid flood, the H4/H16 failure).
- `--style "chaos=high"` = the **RadarManifold** (`cache/radar_manifold.npz`), the CORRECT path. `build_target`:
  1. **conditional-fill (Gaussian conditional)**: start at the mean `mu`, set fixed dims, then
     `x[free] = mu[free] + Σ[free,fixed] · Σ[fixed,fixed]⁻¹ · (fixed_val − mu[fixed])`. So fixing chaos high
     pulls the correlated dims (stream/voltage/air) UP too — a coherent profile, **NOT mean-pin**.
  2. **project to the covariance ellipsoid**: if Mahalanobis distance > the `project_quantile=0.90` of real, shrink
     along the ray `x = mu + (max_d/d)·(filled − mu)`. Keeps targets in-distribution.
  3. **density** = `E[density | radar, difficulty]` under the real joint Gaussian (source-chart-free).
- **Levels** (`resolve_value`): a LEVELS name (`low/mod/high` → a per-difficulty QUANTILE), `q0.9` (that
  quantile), or a bare float (raw 0–1). Because of (2), even `q0.99` is capped at the real spread — e.g. Hard
  chaos: q0.7→0.16, q0.9→0.28, q0.99→0.47 (real charts rarely exceed ~0.47 chaos).
- **CONSEQUENCE for chaos (crown jewel):** via the manifold, chaos stays coherent (backbone PRESERVED) but adds
  16ths only where the AUDIO affords them — on a quarter-heavy song it produces ~no 16ths (q0.99 g3 → 16th 0.09,
  backbone 0.50). The 16ths flood only OOD (mean-pin), which is a smear, not music. **To hear chaos, the SONG
  must afford it** (`--groove_select chaos`; H17 song-fit), and you trade some backbone for 16ths near the knee.
- A probe MUST build radar via `manifold.build_target(spec, difficulty)` to match `--style`; never hand-roll a
  radar vector with mean/zero fills.

## 3. CFG guidance (`--guidance g`)
Dual-path decode: run cond (real radar/style/motif/figure) and uncond (all `null_*`) in lockstep, blend every
logit stream (onset, pattern, type): `out = uncond + g·(cond − uncond)`. `g=1` off; ~1.4 musical; 2–3 strong;
>3 dissolves the backbone. `do_cfg` triggers if `g≠1 AND any(radar|style|motif|figure)` is set.
- **CRITICAL: `tau` must be computed from the SAME guided onset logits the decode uses.** A tau calibrated on
  unconditioned `p` lets conditioning (which raises `p` broadly) flood past it → wrong density. The exporter and
  any probe recompute `ol_guided = ol_u + g·(ol_cond − ol_u)` before the quantile.
- Guidance amplifies motif & figure on the DECODER, not onset (they're null on the onset path). So guidance does
  NOT change density via motif/figure — only via radar/style.

## 4. Continuous motif knobs (MotifBasis)
`cache/motif_basis.npz`. `encode(hist, radar)`: `hist_hat = z(radar)·ridgeW + b` (radar-explained figure mass);
`resid = hist − hist_hat`; `scores = z(resid)·components`; `knob = z(scores)` → 12 z-scored, **radar-ORTHOGONAL**
figure-contrast axes. **The knob REQUIRES the chart's radar to compute** (it removes what radar predicts).
- Knob meanings (signs matter): **k3 = step↔candle/cross (+ = candle)**, **k10 = jack↔trill (+ = trill)**,
  **k0 = jack↔sweep/staircase (+ = JACK, − = sweep)** — k0+ is a JACK detector, NOT sweep (sweep mushes with
  step/candle on the − side). Aliases in the exporter: candle=3, trill=10, jacksweep=0, bracket=1.
- Set as a GLOBAL `(B,K)` vector or a per-frame `(B,T,K)` SECTION schedule (piecewise-constant per ~64 frames;
  the local-motif lever). Decoder-only.
- **MEASURE the realized knob with `MotifBasis.encode_chart(generated, radar)`** — the SAME radar-orthogonal
  z-score. Do NOT use raw figure-family mass or dominant-figure-per-section: those are dominated by the song's
  baseline figure mix (e.g. trill-saturated Hard songs) and show a FALSE NULL even when the knob moved strongly.
  (Verified: candle conditioning reads ΔcandleK +0.65 / −2.00 across poles, but raw candle-mass was flat.)

## 5. Discrete figure tokens
7 classes `FIGURE_CLASSES = [sparse, jack, sweep, trill, candle, jump, step]`. `figure_token(section)` = the
dominant canonical W=3 figure family of a section. Conditioning = a per-section token schedule `(B,T)` →
`figure_embedding` → decoder (NOT onset). The "pick" of pick-then-realize.
- It's a SOFT per-frame bias: it nudges a figure's FREQUENCY but **cannot enforce a multi-frame SEQUENCE** (a
  sweep staircase). So sweep lift is modest/capped (~+0.05–0.08 realized-fraction; ΔsweepK ~0 at export
  settings) — established as the soft-realize ceiling. A strong sweep lever needs a structured realize (future).
- Measure with `figure_token` fractions of the GENERATED chart, vs a `figure=None` baseline on the same songs.

## 6. Onset threshold, density, and the phase grid
- `tau = quantile(sigmoid(guided onset logits), 1 − density)`. Density priority: `--target_density` >
  manifold `E[density|radar,diff]` > the source chart's own density (eval A/B only). Raising chaos at FIXED
  density forces quarter→offbeat REPLACEMENT (backbone collapse); real charts raise density WITH chaos
  (r +0.63), which the manifold density coupling reproduces — so let density float with the manifold.
  NOTE: `tau` sets the BASE density, but the per-frame onset decision is made IN the AR loop and can be raised by
  the STAMINA governor (§8c) — so realized density ≤ the tau target wherever sustained workload is high.
  NOTE (06-27): `onset_logit_scale` is a NO-OP under quantile thresholding — `p=sigmoid(scale·ol)` is monotonic,
  so it preserves the frame RANKING → the top-`density` frames are identical for any scale (confirmed: 0 frames
  differ at scale 0.5/2.0). There is NO "onset temperature" that changes WHICH onsets fire in deployment; it
  only bites under `onset_sample=True` (Bernoulli). The onset head's contribution to rhythm is WHERE it
  deterministically places (audio-only, non-causal → blocky/isolated-16th; see notes/jack_heaviness_findings.md).
- **Phase grid** (frame index `t`, 16th resolution): `t%4` → **0 = quarter, 2 = 8th, 1&3 = 16th-offbeat**.
  Backbone = quarter (+8th). "Chaos / syncopation" = 16th-offbeat share. Real Hard ~ quarter 0.7 / 8th 0.25 /
  16th 0.04; "real-like chaos" sits ~0.25 chaos-radar.
- Decode phase levers (all in `generate()`): `onset_phase_calib=(b8,b16)` adds logit offsets to 8th/16th frames
  BEFORE tau (the caller's tau MUST use the same offset) → 16th COUNT floats with audio per-song (the validated
  win). `onset_phase_alloc=(q,8,16)` forces fixed per-band SHARES (a quota — SMEARS; avoid). `onset_phase_penalty`
  subtracts from off-beat logits (a gate; doesn't rescue chaos because chaos MOVES notes off-beat).
  `onset_logit_offset=(B,T)` (added 06-28) = a per-FRAME content-driven onset logit offset — the hook for the
  ONSET PHRASE CALIBRATOR (e.g. the validated sparse-harm-in-quiet offset `gain·quiet_gate·harm` that un-buries a
  sparse melodic event in a lull). SAME tau-coupling rule as calib (the caller's tau MUST include it; the exporter's
  `--harm_calib` does). Phrasing (WHEN notes fire) is the ONSET head's job, not the pattern head's — see
  `notes/phrasing_coherence_findings.md` + the experiment-design lineage `onset-phrasing-calibrator-arc.md`.
  GATE-FEATURE caveat (06-28, by-ear): the `--harm_calib` quiet gate keys on **dim-0 total energy**, which MISSES
  an energy-LOUD melodic solo (a piano solo is perc-ABSENT not quiet) → it dumped boost onto a loud drum section
  ("1/16s after the piano solo"). Gate on **`perc_onset` dim-35 absence** to fire IN the solo (`probe_phrasing_
  coherence.py --quiet_feat perc`). Also: on weak-harm-channel songs the hand-gate is blunt; the GLOBAL
  `onset_phase_calib` reaches a melodic solo without locating it (head's own ranking).

## 7. Decode / playability (mandatory — see the `playtest` skill)
`hold_aware` automaton + `no_jump_during_hold` + `no_cross_during_hold` + `max_jack_run=2` (was 1; user-approved
2026-06-25 — allow a justified 2-note 16th jack, hard-forbid 3+) + `pattern_temperature ~1.0` (the DEPLOYED default per the `generation-defaults` skill; the old
~0.7 cap predates the governor — see the revisit below). Constraints act on
the FINAL playable symbols, NOT the pre-automaton pattern (a fix written against the pattern leaks because
`hold_aware` remaps it). Any new export/probe the user PLAYS must call `enforce_playability(gen_kwargs)`. The
graded escalation across spacings is the soft FOOT GOVERNORS (§8).
**`pattern_temperature` revisited (06-27, notes/jack_heaviness_findings.md):** the H2 0.6–0.85 cap (above which
arrows "over-randomize") PREDATES the fatigue governor. At the shipped 0.7 the pattern head is jack-HEAVY (len3
~2× real, len≥4 ~3–4×) because it's greedy → repeats the previous panel; raising temp REDUCES jacks AND raises
jumps toward real (both improve, no trade-off). With the fatigue governor ON, the jack tail stays bounded near
real (maxRun ~5) as temp rises to 1.0–1.5, while governor-OFF spikes (maxRun→22) — so the governor catches the
jacks the cap indirectly guarded. BUT transition-entropy (scramble) still climbs with temp regardless of the
governor (it bounds FATIGUE, not musical structure), and that metric can't separate good structured variety
from random scramble. So whether to raise `pattern_temperature` above 0.85 is a BY-EAR call (Rule 8); metrics
favor ~1.0–1.2 with the governor on. Root cause of the jacks = the pattern head (proximate) + the onset head's
blocky audio-only rhythm (contributing); see notes/jack_heaviness_findings.md.

## 8. Decode-time GOVERNORS — per-note FOOT model (placement) + per-region STAMINA/ARC (density)
TWO scopes, both decode-time in `LayeredTypedChartGenerator.generate`, both BPM-coupled (`bpm=`; `frame_hz =
BPM·4/60` = 16th-frames/sec; no bpm → silent):
- **Per-NOTE (8a jack, 8b fatigue):** act on `pat_logits` → govern WHICH-panels (footwork), NEVER note count.
  Measure with same-panel / jump-stream RUN-LENGTH vs REAL (`calib_foot_fatigue.py`), not raw figure mass.
- **Per-REGION (8c stamina + arc):** acts on the ONSET decision → governs DENSITY (thins notes where sustained
  workload is high; ceiling-only). Measure with the paired peak/rest density-window selectivity, NOT the mean
  (it's REDISTRIBUTION). Needs the foot model on (it supplies the cost signal).
RELEASE CENTER (`notes/governor_release_region.md`): per-note = `fatigue_penalty=2` (jack_penalty 0); stamina + arc
OFF by default. Full derivation + the failures in `notes/foot_fatigue_design.md`.
**WHEN↔WHERE ISOLATION (06-29, code-confirmed, `seq-onset-arc.md`):** the decode is STRICTLY one-way. `p_onset` is
PRECOMPUTED (audio-only, non-causal) → STAMINA thins it (8c, CEILING-only) → pattern head decides "where" → FATIGUE
adjusts "where" (8b). The onset HEAD sees neither governor; the onset DECISION is coupled ONLY to stamina (suppress-
only, reads realized foot-cost) and NEVER to fatigue. **The pattern head's "where" NEVER feeds the onset "when"** —
stamina is the lone where→when bridge, and it carries biomechanics, not musical structure. So the note-context
placement signal (16th-AUC 0.87 teacher-forced vs 0.66 audio-only) is structurally UNREACHABLE by any decode lever
from our own first pass (its onsets are audio-only-placed → "where" echoes audio); closing it needs a RETRAIN
(sequence-aware onset head). Re-confirmed 06-29 (`probe_seqcontext_c0.py`: deployed-C0 context 0.667 ≈ audio).

### 8a. Soft JACK governor (`jack_penalty`, OLD — exporter default now 0; SUPERSEDED by 8b) — single-foot
Accumulate `jack_exertion` over a same-panel single-run: on a repeat at gap `g` frames (≤ `jack_max_gap`=4),
`jack_exertion += (frame_hz/g)/jack_free_rate` (`jack_free_rate`=5). PERSISTS across empty frames; RESETS to 0 on
a different-panel single or a jump. Penalty to EXTEND: `pat_logits[single on jack_panel] -= jack_penalty ·
(jack_exertion + (frame_hz/since_onset)/jack_free_rate)` — escalates with run length + rate; a 2-note jack is
~free (accumulator starts at 0). Hard backstop `max_jack_run`=2 forbids a fresh single making a 3rd consecutive
16th-adjacent (`since_onset==1`) same-panel press. DENSITY-PRESERVING (re-routes to alternation). "solved the
unnatural jack problem" by ear. **GOTCHA:** it only watches the SINGLE on the jack panel → it nudges JUMPS up via
softmax (suppressing one single redistributes mass), and on jumpy songs it DISPLACES jacks into jumps (the felt
"consecutive jumps" were this, not intrinsic).

### 8b. Per-foot FATIGUE governor (`fatigue_penalty`, the RELEASE per-note default = 2.0; good range 1.5–3) — two-foot biomechanical model
Generalizes 8a (a jack = one foot stays & re-hits). State per chart: feet `f∈{L,R}` with `pos_f∈{L,D,U,R,∅}`
(= body orientation), `E_f` (exertion at last-hit time), `t_f`; plus same-panel run `(sp_run, sp_panel)`.
Per frame `t` (PAD_DIST = Euclidean on the cross `L=(-1,0) R=(1,0) U=(0,1) D=(0,-1)`):
```
Ẽ_f = E_f · exp(-(t-t_f)/τ)              τ = fatigue_tau·4 frames   (fatigue_tau=2 beats = half measure)
r_f = frame_hz / max(t-t_f, 1)           per-foot press rate
unit_f(p) = jack_weight  if pos_f==p (stay/jack)  else  travel_weight·d(pos_f,p)  (move)   [jack_weight 1.0 > travel_weight 0.6]
cost_f(p) = r_f·unit_f(p)·1[pos_f≠∅]  +  fs_add·1[other foot holds p]   (footswitch)
fs_add: runp=sp_run+1 → 0 (runp≤2, free) | footswitch_pen=4 (runp==3) | ∞ (runp≥4, hard cap)
fatigue(P) = min over the ≤2 footings of  max(Ẽ_L+cost_L, Ẽ_R+cost_R)    (player foots it the EASY way; crossovers when cheaper, NO surcharge)
pat_logits[P] -= ∞                                       if fatigue(P) ≥ fatigue_cap (30, unplayable)
              -= fatigue_penalty · relu(fatigue(P) − fatigue_free)   else   (fatigue_free=12 set HIGH → BARRIER/ceiling, NOT a downward pull)
```
After the chosen pattern: used feet `pos_f←p, E_f←Ẽ_f+cost_f, t_f←t` (idle feet keep state = lazy decay); a
footswitch LIFTS the displaced foot (both feet on one panel → the one that didn't act → ∅); `sp_run` +1 on
same-panel single / reset on new-panel or jump / persist on empty frames. E is BPM-coupled (cost ∝ rate) → a
fixed `fatigue_cap` auto-allows "fewer fast notes at higher BPM". Governor owns the CEILING; the difficulty/radar
conditioning owns where in the playable zone (NO lower bound — would fight the difficulty knob). Governs jacks
onto the human distribution (maxJackRun 6.2→4.1, real 3.5) with density held.

### 8c. Per-region STAMINA (Stage 2, `stamina_ceiling`) + breathing ARC (Stage 3, `stamina_breathe`) — DENSITY
Needs `fatigue_penalty` on (cost signal) and `bpm`; off by default (`stamina_ceiling=None`). The onset DECISION is
now made IN the AR loop (NOT precomputed) — a probe replicating onset must apply this per-frame or set stamina off.
`p_onset = sigmoid(guided onset logits)` is precomputed (all CFG/phase offsets baked in); per frame the EFFECTIVE
threshold is raised by a slow workload accumulator, shedding the LEAST-salient onsets. CEILING only (suppresses,
never adds; byte-identical to OFF below the ceiling); skipped under `onset_override`.
```
E_slow *= exp(-1/(stamina_tau·4))                         # GLOBAL slow accumulator, per-16th decay (stamina_tau=8 beats)
bump   = stamina_max_bump · tanh( (E_slow − ceiling_t[t])⁺ / stamina_scale )   # stamina_max_bump=0.45, scale=15
on_t   = onset[:,t]  &  ¬( p_onset[:,t] ≤ onset_threshold + bump )             # raise the bar when tired
# on each FIRED onset, add the REALIZED footing cost (decayed-to-now Ẽ from 8b):
E_slow += ((cE − Ẽ)·used).sum()      # the chosen footing's added per-foot exertion (a one-foot grind = big)
# HOLD-AWARE: during an open hold the held foot is pinned → the FREE foot does every note. Override the increment:
E_slow += rate_free · unit_free,  rate_free = frame_hz/since_onset,  unit_free = jack_weight (jack/first) | travel_weight·PAD_DIST[free_last, pp]
```
**ARC (Stage 3):** the ceiling BREATHES with a phrase-smoothed audio-energy envelope so it thins VERSES not climaxes:
```
env       = boxsmooth(p_onset, stamina_breathe_win≈96) ; z = zscore(env over valid frames)   # energy = the onset head's own p_onset
ceiling_t = ( stamina_ceiling · (1 + stamina_breathe · z[t]) ).clamp(min = stamina_breathe_floor · stamina_ceiling)
```
HIGH energy → high ceiling → no thin (keep the spicy notes); LOW → low ceiling → thin (rest). `stamina_breathe_floor`
(0.4) stops a low-energy OUTRO collapsing to ~0 = empty tail (the abrupt-ending bug). FLAT stamina (`breathe=0`)
DULLS the model's own arc (it thins the dense climaxes); breathing fixes + amplifies it.
**Knobs/ranges:** `stamina_ceiling` 15–50 (off=None; <10 dents REST too = a global cut; ≥200 ≡ OFF), `stamina_tau`
8, `stamina_scale` 15, `stamina_max_bump` 0.45; `stamina_breathe` 1.2–1.8 (0=flat), `stamina_breathe_floor` 0.4,
`stamina_breathe_win` 96. **MEASURE:** stamina = paired peak/rest density-window thinning (`diag_stamina.py`,
~20:1 at ceiling 25), holds = pinned vs non-pinned-dense frames (`diag_stamina_holds.py`), arc = corr(window-
density, window-energy) + climax-verse Δ (`diag_stamina_arc.py`); the energy must be the SAME p_onset, smoothed+
z-normed. NOT the density mean — the effect is REDISTRIBUTION (experiment-design Rule 1: a summary stat blind to
the property; the mean held while the shape moved). Stamina is also the canonical POSITIVE case of experiment-
design Rule 13 (global-quota anti-pattern): it's a per-frame emergent THRESHOLD, not an imposed density count —
contrast `onset_phase_alloc` (§6, a flat quota that SMEARS). VALIDATED + playtest-confirmed ("a tasteful edit").

### 8d. KNOWN GAPS — a probe must NOT assume these are modeled
- **HOLDS-BLINDNESS (per-NOTE only) — ⚠️ DISPUTED (user does not trust this claim; UNVERIFIED, re-test before
  relying on it):** the PATTERN penalty (8b) still does NOT pin the held foot (pinning it there
  REGRESSED — jacks explode non-monotonically; placement can't fix a COUNT problem). FIXED on the DENSITY side: the
  stamina cost (8c) IS hold-aware (free-foot grind). But on these charts holds aren't actually grinds (pinned frames
  ~0.14 dense, maxJackRun-in-holds 3 = human) so the effect is near-vacuous in practice. *(The "near-vacuous in
  practice" + the pinned-frame stats are the part flagged as untrusted — treat as a hypothesis, not a result.)*
- **ONSET-HEAD melodic under-placement (H-onset-perc-bias):** the onset head under-places on melodic-only sections
  (a piano solo reads sparse); NOT a governor knob — a feature/retrain thread. (The breathing energy itself is NOT
  percussion-biased on the tested songs — diag_breathe_energy refuted that; the gap is the onset head.)
- **BODY-TURN:** charges full per-foot travel for a coordinated rotation (ranking right, magnitude too high).
- **MODEL UNDER-JUMPS** these songs (6% vs real 31%) — a separate density/air thread; do NOT calibrate the
  governor to close that gap (the calib "dist-to-real" is dominated by it — the wrong target).

## THE ALIGNMENT CHECKLIST (run before any probe / eval / export)
1. **Radar:** built via `manifold.build_target` (matches `--style`)? Or a deliberate, labeled `--radar` OOD
   test? Never a hand-rolled mean/zero fill passed off as "the knob."
2. **tau:** computed from the SAME conditioned + guided onset logits the decode uses (and the same phase offset)?
3. **Motif measured** with `encode_chart(gen, radar)` (radar-orthogonal knob), not raw figure mass?
4. **Onset decoupling preserved** (motif/figure NOT fed to onset)?
5. **Density** sourced the same way as deployment (manifold E[density|·], not a stale source-chart density)?
6. **Decode settings match deployment** (pattern_temperature, guidance, max_len) — eval-vs-export mismatch is its
   own artifact (the eval used temp 1.0/radar=real; export uses temp 0.7).
7. **Song affords the axis** (`--groove_select <axis>`) — you can't test chaos on a quarter-heavy song.
8. **Sign/label check:** k0+ = jack not sweep; `null_motif` ≠ motif=0; `high` is a quantile not a raw value.
9. **Governors (§8):** passing `bpm=` (no bpm → silent)? Per-NOTE (8a/8b) measured with same-panel / jump-stream
   RUN-LENGTH vs REAL (not raw mass), NOT calibrated to "match real jump%" (model under-jumps — wrong target)?
   Per-REGION (8c stamina/arc) measured with paired peak/rest density windows (NOT the mean — it's redistribution),
   stamina needs `fatigue_penalty` on, and the onset decision is IN-loop (replicate the per-frame gate or set off)?

## Catalog of REAL misalignments (each cost a wrong conclusion)
- **mean-pin vs manifold conditional-fill** (this is why a chaos probe gave 16th 0.96 everywhere): set
  `radar=[0,0,0,0,c]` (or others-at-mean) instead of `build_target` → OOD smear, opposite of the deployed knob.
- **raw figure-mass vs radar-orthogonal knob**: dominant-figure-per-section showed candle Δ≈0 (false null);
  the knob z-score showed Δ+0.65/−2.00 (real). Match the metric to the conditioning target (exp-design Rule 1).
- **tau from unconditioned logits**: conditioning floods past a mean-calibrated threshold → density wrong.
- **eval-vs-export decode mismatch**: candle steered Δ+1.7/+3.9 at eval (temp 1.0, radar=real) but must be
  re-checked at export settings (temp 0.7, radar off) — characterize the EXPORTED charts to confirm it landed.
- **knob-0 sign**: pushing k0 "+" toward "sweep" actually pushes toward JACK; sweep is the − pole (and weak).
- **stale onset path (Stage-2)**: the onset decision is now made IN the AR loop (stamina gate), not a precomputed
  `(B,T)` mask. A probe that rebuilds onset as `p > tau` and skips the per-frame stamina bump silently diverges
  whenever stamina is on (exp-design Rule 2: match deployment). Replicate the §8c gate or set stamina off.
- **wrong probe population**: the breathing-energy probe run on the default Hard-song order gave a noisy NON-answer;
  re-run on the ACTUAL complaint songs (HSL/japa1, via `--match`) it flipped and REFUTED the percussion-bias
  hypothesis (exp-design Rules 5+11: bin the real reference / the population that actually exhibits the effect).

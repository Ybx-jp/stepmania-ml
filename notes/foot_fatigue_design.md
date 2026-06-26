# Per-foot fatigue model — design (decode-time governor that GENERALIZES the jack governor)

**Date:** 2026-06-25. **Branch:** `gen/foot-fatigue` (off `gen/h19-retrain`, which has the jack governor).
**Origin:** playtest — the jack governor ([[foot_exertion_findings]]) solved unnatural jacks but DISPLACED mass
into jumps (measured: jump rate 15.7→23.2%, longest jump-dense section 11→59). User: "we need a similar
mechanism for consecutive jumps, but with more nuance." The design conversation turned it into a unified
**per-foot fatigue simulator** where jacks and jumps are the same thing seen through two feet.

## The model
Track **two feet** with state, simulate how a player would dance each candidate pattern, accumulate per-foot
exertion that DECAYS over time, and penalize patterns that fatigue a foot. A jack = "one foot stays and
re-hits"; a jump = "two feet placed"; **one model**, so there is no jack-budget vs jump-budget to launder
between (dissolves the escape valve: `jack jack jump jack` keeps the same foot's exertion climbing).

State per sequence (batched):
- `foot_panel[2]` = panel under (left, right) foot (or none). This IS the body orientation — crossed vs not.
- `E[2]` = per-foot exertion accumulators (left, right), **exponential decay** with time constant `tau`.
- `t_last[2]` = frame of each foot's last hit (for rate).

Per note event (single or jump) at frame `t`:
1. **Assign** the arrow(s) to feet (policy below) — pick the assignment that MINIMIZES added exertion given
   current foot positions. (LOCKED: assume crossovers when they're lower-exertion, NO crossover surcharge — the
   user wants more crossovers; a chart that admits an easy crossed footing should read as easy.)
2. Per foot used, with rate `r = frame_hz / gap_frames` (gap to THAT foot's last hit; frame_hz = BPM*4/60):
   - **stay & re-hit** (same panel): `+= JACK_W * r`   (a 2-footed jack at high BPM is hard; ~0 at low BPM)
   - **move** panel→panel': `+= TRAVEL_W * dist(panel,panel') * r`   (foot-speed; distance × rate)
   - LOCKED: `JACK_W > TRAVEL_W` at equal r — a stay-and-re-hit is harder than a one-panel travel (no momentum
     to ride). dist = Euclidean on the cross (L=(-1,0) R=(1,0) U=(0,1) D=(0,-1)); adjacent=√2, L↔R=2.
3. **Decay then add:** `E[f] = E[f] * exp(-(t - t_last[f]) * frame_dt / tau) + cost[f]`; `t_last[f] = t`.
   LOCKED: exponential decay (NOT a hard window). `tau` configurable, default ≈ a half-measure (so a foot
   recovers across a phrase). NOTE: a single between jumps is NOT relief — the foot still travels to it AND time
   decay is the only relief, so a hard mid-stream single correctly stays costly.
4. **Penalty:** subtract `lambda_fat * max(E_left_after, E_right_after)` (the more-fatigued foot gates) from the
   candidate pattern's logit. Escalates with run length (E accumulates) and rate; eases with spacing (decay +
   the /gap rate term) — matching every nuance the user raised.

### Why this captures the user's nuances (each maps to a mechanism)
- LR→LR is a 2-footed jack at high BPM, free at low BPM → both feet "stay & re-hit", cost = JACK_W*r (rate-scaled).
- per-foot: LD→LU / LD→LR = L jacks (stay) while R travels D→U / D→R → assignment leaves L on L (stay cost) and
  moves R (travel cost); the two feet fatigue independently.
- LD→UR relieving at 1/4 vs 1/8 → travel cost ∝ r, and decay over the wider gap → cheaper at 1/4.
- LD→DR body-turn vs foot-swap by history → min-exertion assignment given foot_panel state: when the feet sit
  for a rotation it picks the body-turn (both feet move 1 step, low total); when a prior note pinned a foot it
  picks the L→R crossover (high). Body orientation = the foot-position state; no special-casing.

## Foot-assignment policy (the crux — LOCKED)
At each note, enumerate assignments (single: 2 feet; jump: 2 ways) and pick MIN added exertion given
`foot_panel`. No crossover surcharge. This makes the simulator assume the EASIEST footing — so the penalty
reflects the chart's best-case difficulty, which is the right thing to gate on (a chart is only as hard as its
easiest valid footing).

## Decode integration (plan)
Replace the single-foot jack governor in `typed_model.generate` with this two-foot model (jack governor = the
special case). Needs `bpm` (already threaded). Per frame, for each of the ≤15 candidate patterns, compute the
prospective `max(E)` after assignment and subtract `lambda_fat * that`. Cost: ≤2 assignments × 15 patterns × B,
cheap. Keep the hard `max_jack_run=2` backstop. New params: `fatigue_penalty` (lambda_fat), `fatigue_tau`,
`jack_weight`, `travel_weight`. Validate like the jack governor: same-panel + jump-stream + per-foot run
distributions vs real, density held, decay sanity. Then an A/B playtest.

## STRATEGIC — decode-now vs learned-v2, and why building now is not wasted
The user flags this may be **v2 territory**, and that the learned geometry/symmetry ([[geometry_feasible_region]]
— pad-symmetry equivariance) might let the MODEL reason about fatigue rather than a hand-coded decode simulator.
Resolution: **build the decode simulator now; it doubles as the fatigue CRITIC/oracle for v2.**
- NOW: a controllable decode-time penalty that ships value immediately (same proven pattern as the jack governor).
- V2: train a fatigue-aware generator (equivariant to the pad's L↔R/U↔D symmetry — fatigue is symmetric under
  mirroring, so an equivariant net learns it from far less data) using THIS simulator as the training signal /
  reward / eval. The hand-coded foot-physics becomes the differentiable-ish objective the learned model distills.
- So the simulator is the BRIDGE: decode governor today, training oracle tomorrow. Not throwaway v2 scaffolding.

## RELEASE note (the user hasn't forgotten)
The CURRENT release candidate (clean-retrain `gen_motif_full_fixed` + jack governor λ=1.5, both validated) is
assessable INDEPENDENT of this. The per-foot fatigue simulator is a fast-follow / v2 headline, not a release
blocker. Recommendation: assess the release on the current state; let foot-fatigue land on its own branch/PR.

## IMPLEMENTED + VALIDATED (2026-06-25, gen/foot-fatigue) — and the FIVE subtleties it took
`typed_model.generate(fatigue_penalty, fatigue_tau, jack_weight, travel_weight, fatigue_free, footswitch_pen, bpm)`.
Two-foot state (foot_panel/foot_E/foot_t) + same-panel run tracker (sp_run/sp_panel); per-frame, for each
candidate pattern, min-exertion foot assignment → penalty `λ·relu(max(E_after) − fatigue_free)`. Building it
surfaced five non-obvious failures, each caught by `diag_foot_fatigue.py` (8 songs, jump% / maxJumpStream /
maxJackRun / density), NOT by reading the code:
1. **Footswitch loophole.** A per-foot max(E) model is GAMED by alternating feet on one arrow (each foot hits at
   half rate → E never spikes) → 20+ note same-panel runs read as "cheap footswitches". (jumps 11.6→0.3%, jacks→23)
2. **Forbid-all-footswitch was too blunt** (user-corrected): a 2-note footswitch is legit; only 4+ should cap.
3. **Free threshold.** Penalizing per-NOTE cost crushed isolated jumps (→0.3%, fighting the air radar). Fix:
   `relu(E − fatigue_free)` — a rested jump passes, only accumulated streams are gated (mirrors jack_free_rate).
4. **Graded footswitch (user's rule): 2 free / 3 penalize (footswitch_pen) / 4 hard-cap**, by prospective
   same-panel run length sp_run+1.
5. **The lift bug.** A footswitch must LIFT the displaced foot; without it the state corrupts to "both feet on
   the panel" and the cap is bypassed via the stay path (jacks →21 again). Fix: if both feet read one panel, the
   foot that didn't act lifts (−1).
**Validated final** (max_jack_run=2, jack_penalty OFF, sweep λ): jacks BOUNDED (maxJackRun 6→5, was exploding to
21), jumps GENTLY reduced not crushed (11.6→6.6→4.5%), maxJumpStream ~5→4 (noisy at high λ), **density held 0.208
at every λ** (re-routes, never deletes). Regression test `test_fatigue_penalty_runs_and_preserves_density`; 36/36.
LESSON: a faithful foot simulator is mostly *bookkeeping* (lift, footswitch-count, free-zone) and every gap
becomes a loophole the decode exploits — the diag-on-real-data loop is what caught each one.

## NEXT (calibration + feel — metrics handing off to the ear)
- Calibrate on the EGREGIOUS rich-Hard set (where OFF maxJumpStream was 14, not 5) — the 8-song val set is too mild.
- Sweep `fatigue_penalty / fatigue_free / footswitch_pen / fatigue_tau / jack_weight:travel_weight` there.
- A/B PLAYTEST (fatigue OFF vs tuned) — does the per-foot governing FEEL natural? jump streams + 4-cap footswitch.
- Decide default + whether fatigue SUPERSEDES the shipped jack governor (it generalizes it) or runs alongside.

## CALIBRATION (2026-06-25, calib_foot_fatigue.py) — REFRAMED the governor's job (jacks, not jumps)
Swept λ×free on the EGREGIOUS rich-Hard set, targeting the REAL source charts of the SAME songs.
**REAL target: jump% 31.4 · maxJumpStream 10.2 · maxJackRun 3.5 · jack≥4 0.8% · dens 0.385.**
```
 lam free  jump%  jumpStrm  jackRun  jack>=4%   dens
  OFF   -    5.9     4.0      6.2      2.1     0.320
  2.0   6    1.5     1.1      3.9      0.7     0.320
  4.0   6    2.3     1.8      3.6      0.4     0.320
  2.0   8    2.7     1.9      4.1      1.3     0.320
```
**KEY FINDINGS (the calibration changed direction — experiment-design Rule):**
1. **The model UNDER-jumps, doesn't over-jump.** Real charts on these air-heavy songs = **31%** jumps with
   streams to 10; the model OFF = ~6%. The "consecutive jumps" the user felt were **INDUCED by the jack penalty**
   (it displaced suppressed jacks into jumps 15.7→23%), NOT intrinsic. So calibrating the governor to "match real
   jump%" optimizes the WRONG axis — the dist-to-real score is dominated by a gap the governor can't/shouldn't close.
2. **The governor's real value is JACKS:** maxJackRun 6.2→3.6 (real 3.5), jack≥4 2.1%→0.4% (real 0.8%) — lands
   right on the human distribution. The fatigue model = a good JACK governor that, unlike the jack penalty, does
   NOT displace into jumps (it penalizes both dimensions in one foot model). ⇒ **fatigue should REPLACE the jack
   penalty** — that dissolves the jump-displacement (the original motivation) while fixing jacks.
3. **The governor can't tell an EARNED jump stream from an unearned one** — it penalizes length; real charts have
   musical 10-streams. So keep λ GENTLE or it kills good streams too → a PLAYTEST is the only arbiter (offline blind).
4. **Separate thread (not the governor's job):** the model under-jumps AND under-densifies these rich songs
   (dens 0.320 vs real 0.385) — manifold density / air conditioning too low on jumpy songs.
**DECISION:** default λ=2, free=6, fatigue REPLACES jack_penalty. A/B = jack-penalty (current, has displacement)
vs fatigue (governs jacks, no displacement) on rich-Hard songs → does it FEEL cleaner on jacks while leaving
musical jump streams intact? (ab_fatigue_* sets.)

## FUTURE THREAD (parked 2026-06-25, user) — FATIGUE DYNAMICS should ARC, not sit flat
The governor enforces a CEILING; the conditioning owns WHERE in the playable zone a chart sits. But there's a
distinct, deeper idea: a good chart's exertion shouldn't be flat — it should **arc** (build → peak at the
musical climax → release), like a real charter paces difficulty to the song. This is the fatigue analog of H5
(generated density is structurally flat; real charts have an intro→build→climax@~80-90%→outro arc). Fatigue
ARCING = condition the fatigue target/ceiling on song STRUCTURE/energy so the hardest footwork clusters at
climaxes and eases in verses. NOT a static lower bound in the penalty (that fights difficulty conditioning) —
a time-varying target tied to audio novelty/energy. Connects: H5 (structure), the manifold density arc, and the
per-foot fatigue model as the difficulty currency that should follow the song. A v2/structural feature.

## CORRECTIONS (2026-06-25, user review of the math) — one landed, one reverted
- **BARRIER PENALTY (objective reshape) — DONE + VALIDATED.** Replaced `λ·relu(E−free)` (low free, monotone
  downward pull that jump-STARVED charts) with a one-sided CEILING: `penalty = ∞ if E≥fatigue_cap, else
  λ·relu(E−fatigue_free)` with fatigue_free set HIGH. Isolated barrier-only sweep (rich-Hard): free=8 gently
  governs (jackRun 6.2→4.1, jump% 5.9→2.7); free=18 ≈ OFF (silent). Tunable ceiling that doesn't flatten —
  exactly the shape wanted. Governor owns the ceiling; difficulty conditioning owns where in the zone (NO lower
  bound — would fight the difficulty knob). E is BPM-coupled (cost∝rate) so a fixed E_max auto-maps to "fewer
  fast notes at higher BPM" (correct). Params: fatigue_free (E_soft, def 12), fatigue_cap (E_max, def 30).
- **HOLD-PINNING (the holds-blindness bug) — ATTEMPTED + REVERTED, STILL OPEN.** Holds-blindness is real and
  serious (a stream during a hold is a ONE-foot grind — the model treats it as a shared two-foot load = inverted).
  First attempt (a foot on a `held` panel → cost ∞ → route taps to the free foot) REGRESSED NON-MONOTONICALLY:
  isolated A/B showed it produced maxJackRun 4→14 (more penalty → MORE jacks), root cause unidentified. Likely
  a state-corruption or greedy-foot-choice issue (the model doesn't look ahead to pick WHICH foot holds, so the
  free foot gets forced into one-foot grinds the penalty can't escape). REVERTED to keep the branch healthy.
  **Holds-blindness remains the top open problem for this model** — needs a different mechanism (e.g. choose the
  holding foot to keep the other free; or model the hold cost without forcing all assignment to one foot).

## ⚠️ CORRECTION (2026-06-25, user) — the "reach / affordability veto" below was an INVENTED reframe
The user's ACTUAL design was two sentences: **make the onset head hold-aware, and track the effort PER FOOT.**
i.e. during a hold one foot is pinned, so the free foot carries a one-foot grind, and THAT per-foot effort should
drive the onset thinning. The "LOCAL anaerobic reach/affordability veto — can THIS transition physically happen"
framing in the spec below was NOT the user's design; a prior session invented it, built a hard veto on ACCUMULATED
fatigue (`decayE + cost ≥ cap`), watched it hole-punch density 0.320→0.145, and then wrongly concluded "hard veto
is the wrong tool, the local layer is near-vacuous, skip it" — refuting a strawman it had invented. The real design
was never a reach veto. It is now BUILT correctly as the HOLD-AWARE E_slow increment (see "STAGE 2" → hold-aware
result): during a hold, charge the FREE foot's single-foot grind cost to the stamina accumulator. Read the spec
below as historical context, not as the design.

## TWO-TIMESCALE ONSET GOVERNOR (spec, 2026-06-25 — user-designed) — the holds-blindness + arc resolution
ROOT CAUSE of the hold-pinning regression (user diagnosed): the PATTERN penalty can only redistribute load, not
remove it. During a hold (one foot pinned) a one-foot WIDE stream costs MORE than a jack (travel_weight·2 >
jack_weight), so minimizing E picks the JACK — a stronger penalty picks it harder (the non-monotonicity). The
real lever for "this is too much to play" is the ONSET head (whether a note exists), not placement. So the same
exertion model must govern ONSET too — at TWO timescales (a greedy per-frame onset gate would punch incoherent
holes; "awkward note fallout"):
- **LOCAL / anaerobic (fast decay ~sub-measure):** can THIS transition physically happen? Foot speed / reach. A
  surgical HARD veto on the impossible note. The held-foot pin lives HERE (one foot ⇒ some notes unreachable).
- **SEMI-GLOBAL / aerobic STAMINA (slow decay ~phrase, trailing/exponential = a soft sliding window):** can you
  SUSTAIN this? When high, it raises onset tau over the UPCOMING stretch → COHERENT density thinning (onset head
  re-selects its most-salient notes), not hole-punching. AR-causal: trailing window thins forward = correct (you
  tire from past work). This IS the ARC: tie the stamina CEILING to AUDIO ENERGY/NOVELTY (user-chosen shape) →
  hard sections cluster at the climax, rest in verses. Arc emerges from a BREATHING ceiling (governor owns the
  ceiling; the breathing gives the shape) — unifies holds-blindness + jump-starving + structural flatness (H5).

### Architectural note
Onsets are PRECOMPUTED as a (B,T) mask before the pattern loop (typed_model.generate ~L480), so foot state isn't
available at onset-decision time. ⇒ the onset gate is an in-loop AFFORDABILITY VETO: with the held foot pinned in
the COST, if the cheapest footing's fatigue ≥ fatigue_cap (no affordable placement), drop the note
(`on_eff = onset[:,t] & affordable`). Hold-pinning moves HERE (removes the unplayable note) instead of the
pattern penalty (where it shuffled to jacks). The pattern penalty then foots only the affordable remainder → no
forced jacks. (This is why pinning alone regressed: without the onset gate, the pattern penalty was forced to
foot an unplayable stream and picked jacks.)

### STAGED BUILD (one change at a time)
1. ~~LOCAL affordability veto~~ **ATTEMPTED + REVERTED (2026-06-25).** Built it as a hard onset veto on
   `min_aff = decayE + cost ≥ cap` → DENSITY CRASHED 0.320→0.145 (>half the notes removed). This is EXACTLY the
   user's predicted "greedy accumulator → awkward note fallout": vetoing on ACCUMULATED fatigue punches holes
   through every dense section, not just impossible spots. LESSON (the user called it): a **hard per-note veto is
   the wrong tool for the stamina dimension.** Genuine INSTANTANEOUS impossibility is rare and mostly already
   handled by `no_jump_during_hold`; the holds problem is SUSTAINED one-foot load = STAMINA, which must modulate
   DENSITY coherently, never veto single notes. ⇒ Stage 1 (instantaneous hard veto) is near-vacuous; skip it.
2. **STAMINA → COHERENT tau MODULATION — BUILT + VALIDATED (2026-06-25, gen/foot-fatigue-stage2).** Trailing
   stamina accumulator E_slow (long τ), fed the REALIZED per-note footing cost; when E_slow > ceiling, RAISE the
   onset tau over the upcoming stretch → onset head sheds its LEAST-salient notes (coherent thinning, NOT
   hole-punching). See "STAGE 2 — BUILT + VALIDATED" section below.
3. **BREATHING ceiling**: stamina ceiling tied to audio energy/novelty → the ARC. **(Stage 3 — BUILT + VALIDATED
   2026-06-25; see "STAGE 3 — THE ARC" below.)**
Revised plan: the substance is in 2+3 (stamina tau-modulation + breathing ceiling). Foot model + barrier penalty
(jacks/jumps) stay as the per-note governor; stamina is the per-region density governor.

## STAGE 2 — BUILT + VALIDATED (2026-06-25, gen/foot-fatigue-stage2)
**The architectural refactor:** the onset mask was precomputed as a full `(B,T)` tensor BEFORE the AR loop, so an
in-loop stamina value couldn't reach the decision. Fixed by moving the onset DECISION into the loop: keep
`p_onset = sigmoid(...)` (B,T) precomputed (audio-driven, with all CFG/phase adjustments), but decide `on_t`
per-frame. `on_t = onset[:,t] & ~(tired)` where `tired = p_onset[:,t] <= onset_threshold + bump`. The gate is a
SUPPRESSION-only operation layered on TOP of whatever onset mode produced the base `onset` (threshold / Bernoulli /
phase_alloc) → all modes preserved; skipped under `onset_override` (caller owns onsets). Both consumers (`active`,
`on`) now read `on_t`, so a suppressed frame is a true rest everywhere downstream (foot model, jack trackers,
since_onset all key off `on`).

**The stamina math:** global `E_slow` (B,), per-frame slow decay `exp(-1/(stamina_tau·4))` (tau in beats, ~several
measures). On each onset, `E_slow += ((cE - decayE)·cu).sum()` = the REALIZED added foot exertion of the chosen
footing (reuses the per-note governor's committed footing — a one-foot grind dumps a bigger increment than
alternating feet at equal density, so E_slow integrates total load = rate × footing-difficulty, not raw count).
Effective threshold bump = `stamina_max_bump · tanh((E_slow - ceiling)⁺ / stamina_scale)`. A CEILING: below the
ceiling, output is BYTE-IDENTICAL to OFF (verified: ceiling 200 → identical onset count). Needs `fatigue_penalty`
(the foot model supplies the cost signal).

**Validation (diag_stamina.py, paired before/after on FIXED baseline-defined windows, 8 val songs, fatigue=2):**
the decisive test is the density PROFILE, not the mean. At `ceiling=25, scale=15, tau=8`: the top-decile
sustained-dense 4-measure windows thin 14% (peakΔ -0.039, 0.281→0.242) while moderate windows hold (restΔ -0.002)
= ~20:1 peak/rest selectivity. maxJackRun unchanged (6) → the per-note governor is untouched; the two layers
compose. This is the per-region density relief valve done RIGHT — coherent thinning of the genuinely-too-dense
spots, NOT the Stage-1 global hole-punch (which crashed density 0.320→0.145 everywhere). Knobs in `generate`:
`stamina_ceiling` (None=off), `stamina_tau=8`, `stamina_scale=15`, `stamina_max_bump=0.45`.

**HOLD-AWARE E_slow — BUILT (2026-06-25, the user's actual design).** During an open hold the held foot is pinned,
so the FREE foot does every note (a one-foot grind). The unpinned foot model under-counted this by alternating feet
→ a hold-stream looked no harder than an alternating stream. Now, during a hold, E_slow is fed the FREE foot's
single-foot grind cost: `rate_free × unit_free`, rate_free = `frame_hz / since_onset` (one foot every onset),
unit_free = jack_weight (same panel / first note) or `travel_weight × PAD_DIST[free_last, pp]`, in the same
exertion units as the normal increment. So a sustained one-foot stream during a hold raises stamina ~2× faster than
an equally-dense alternating stream. Placement is UNTOUCHED (this only re-attributes the COST that drives the onset
gate → no jack explosion, unlike the reverted pin-in-pattern-penalty).

**RESULT — correct, but the pathology is ABSENT under default conditioning (diag_stamina_holds.py frame-level
test, 12 songs).** The clean test = press-density thinning on PINNED grind-frames vs matched NON-pinned dense
frames. At ceiling 25: pinned 0.138→0.127 (−7.2% rel) vs non-pinned-dense 0.311→0.292 (−6.1% rel); at ceiling 12:
pinned −16.7% rel vs non-pinned −18.0% rel. So pinned frames thin at ~the SAME relative rate as non-pinned dense —
no preferential hold relief. The reason is in the baseline: pinned frames are only **0.138** dense — i.e. during a
hold the free foot presses on just ~14% of frames. **The model's default holds are NOT grinds** (consistent with
maxJackRun-in-holds = 3, human-level), so a correct hold-aware cost correctly has almost nothing to bite on. The
brutal "jack streams during holds" the user felt was under AGGRESSIVE conditioning (the trill +3 g2 playtest), so
that — not gentle defaults — is the venue to demonstrate the hold-aware cost. CARRY-FORWARD: stamina A/B under
chaos2_manifold_q99 conditioning (chaos=0.47 g3.0). The mechanism stays in (no-op when there's no grind, bites when
there is).

**PLAYTEST ✅ WIN (2026-06-25):** under aggressive chaos conditioning (--style chaos=q0.99 g3.0, density cranked
to 0.400), g50 on japa1 was "much more playable than off without being much different — a TASTEFUL EDIT, not a
rewrite"; HSL "felt the same"; "the stamina and fatigue system is DEFINITELY an improvement." H-stamina confirmed:
the onset thinning reads as RELIEF, not dropped notes, at the gentle end. The default-conditioned A/B was
imperceptible (correct — the chart wasn't over its ceiling); the relief is visible/felt under over-conditioning.

## STAGE 3 — THE ARC (breathing ceiling) — BUILT + VALIDATED (2026-06-25)
The `stamina_ceiling` becomes a per-frame schedule that BREATHES with a phrase-smoothed audio-energy envelope (the
onset head's own `p_onset`, box-smoothed over `stamina_breathe_win`~96 frames, z-normalized per song):
`eff_ceiling[t] = stamina_ceiling · (1 + stamina_breathe · z_energy[t])`. HIGH at climaxes (ceiling up → stamina
doesn't thin → the dense spicy notes survive), LOW in verses (ceiling down → thin → rest). Ceiling-only, NO lower
bound (never fights radar/difficulty). Knobs: `stamina_breathe` (0=off, ~1.2 validated), `stamina_breathe_win`.

**KEY FINDING (diag_stamina_arc.py, 10 Hard songs, plain generation; arc = corr(window-density, window-energy) +
climax-vs-verse density Δ):** the model is NOT structurally flat in this metric — its onset head already tracks
energy at corr 0.898. The problem is that FLAT stamina DULLS that arc: it thins the dense CLIMAXES (corr 0.898→
0.876, Δ 0.180→0.131). Breathing makes the thinning ARC-AWARE — protect climax, rest verses — recovering AND
amplifying past baseline: breathe=1.2 → corr 0.918 / Δ 0.185 (1.8 → 0.200) at held overall density 0.31
(REDISTRIBUTION, not a cut). ⇒ Stage 3 lets you run stamina relief AND keep/sharpen the arc (the user's "extra
room for that handful of spicy extra notes in the deserving sections"). The arc lever can only THIN (ceiling), so
it amplifies the arc by resting verses, not by adding climax notes beyond what the onset head places.

**PLAYTEST (2026-06-26) + two fixes:** the arc was "mostly good" but surfaced two issues.
(1) **Abrupt early endings (H-arc-end) — FIXED.** The breathing ceiling collapsed to ~0 at low-energy outros
(`clamp(min=1e-3)`) → max thinning → empty tail. Added `stamina_breathe_floor` (def 0.4): clamp the effective
ceiling to 0.4× base so low-energy/low-workload sections aren't emptied. Confirmed on real chaos charts: breathe1.8
floor=0 emptied HSL/Star Trail tails (last-10% dens 0.156/0.054→0.000); floor=0.4 restores them to the OFF baseline
(0.094/0.054). Small arc cost (corr 0.920→0.905, Δ 0.200→0.184). The residual end-fade is the pre-existing H5 one.
(2) **"Breathing ignores high-pitch/melodic energy" (H-arc-energy) — REFUTED at the signal level (diag_breathe_
energy.py).** On the actual playtest songs p_onset is NOT percussion-biased: c_harm (0.34) ≥ c_perc (0.30), and on
melodic-loud/perc-quiet frames p_onset is POSITIVE (+0.47, reads them as HIGH energy). So a perc+harm energy swap
would NOT help — not shipped. The "ignored piano solo" perception is the ONSET HEAD under-placing on melodic-only
sections (present at OFF, breathing inherits it) = a deeper feature/retrain thread (the user's "audio features need
tuning"), not a Stage-3 knob. **UNTESTED:** re-playtest the floored endings + the arc pacing feel.

## Open / to calibrate (not blocking the build)
- `JACK_W : TRAVEL_W` ratio + `lambda_fat` (sweep like the jack λ).
- `tau` default (half vs full measure) — expose configurable.
- dist metric (Euclidean vs a hand-tuned foot-travel table for U/D vertical awkwardness).
- whether `max(E)` or `sum(E)` gates (more-fatigued foot vs total).

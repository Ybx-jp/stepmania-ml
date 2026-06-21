# Chaos / syncopation — objective + conditioning-mechanism plan

*2026-06-21. Follows the H4 conclusion (`h4_offbeat_signal_findings.md`): chaos is NOT a feature problem.
Two feature retrains (chroma, high-res onset) failed to make chaos event-driven. This note scopes the
actual levers — the conditioning mechanism and the objective — and picks a first build.*

## The diagnosis, precisely

Chaos is a groove-radar scalar (0–1) injected as a **global per-sample conditioning vector added
identically to every frame** (`_cond = diff_emb + radar_proj(radar) + style`, broadcast over T), then
amplified by CFG. The model learned the only thing a global scalar can drive: **globally suppress on-beat
and raise off-beat onset probability uniformly → a smear** (~5% on-beat @ CFG2 vs real ~80–90%). It never
learned *which* off-beats deserve notes because (a) the local audio cue is weak (off-beat onset AUC 0.66,
mostly redundant with the coarse onset) and (b) frame-wise CE never rewards selective syncopation.

So two independent gaps: **mechanism** (chaos can only act globally) and **objective** (nothing rewards
selectivity). They need different fixes, in order.

## Why 2c (critic-guided fine-tune) can't fix chaos *first*

The taste critic scores current chaos output ~0.003 (REAL 0.82 > BASE 0.29 > CHAOS 0.003). Best-of-N
self-distillation (the 2c plan) needs *some* good draws to distill from — but every chaos draw is a smear,
so there's nothing good to select (same wall as Hard best-of-N). **2c lifts plain/overall taste (where a
real spread exists: base 0.29, some draws near-real); it cannot bootstrap chaos from a distribution that's
uniformly bad.** Conclusion: the model must first be *able* to produce selective syncopation (mechanism),
then the objective can reward it. **Gate before 2c (for chaos).**

## Design A — chaos × onset gate (the mechanism fix)  ⭐ first build

Make chaos *route* off-beats to salient frames instead of adding them uniformly. The existing chaos
behaviour is `p_chaos ≈ p_base + Δ`, where Δ uniformly lifts off-beats. We want Δ **modulated by local
saliency** so off-beats land where there's musical evidence.

### A1 — decode-time selective gate (NO retrain) — cheapest probe
Reuse `generate(onset_override=...)`: compute the onset decision externally and feed it in.
Per song, per frame t:
- `p_base`  = onset posterior at radar chaos=baseline (the model's "calm" read)
- `p_chaos` = onset posterior at radar chaos=0.9 (the smear)
- `sal[t]`  = local saliency, normalized in [0,1]. Candidates: the high-res onset dim41 (already in
  `cache/samples_v3`), OR `p_base` itself (the model's own onset evidence under no chaos — integrates ALL
  features, not just dim41, so likely a *stronger* saliency signal than dim41's 0.66).
- **Gate:** on-beat frames keep `p_base`; off-beat frames get `p_base + chaos_gain · sal[t] · (p_chaos −
  p_base)`. Then threshold at the target density. Net: chaos still adds off-beats, but preferentially
  where saliency is high → selective syncopation, density preserved.
- This is DIFFERENT from the failed `onset_phase_penalty` (which penalized ALL off-beats uniformly →
  empty). Here we *redistribute* off-beats to salient frames, not suppress them.
- Knobs: `chaos_gain` (how much syncopation), `sal` source (dim41 vs p_base vs blend), selection rule
  (soft reweight vs hard top-k off-beats by saliency at the target count).
- **Model-agnostic:** the gate operates on onset posteriors, so it works on the current shipped model
  (gen_style) — no dependence on the parked gen_highres.
- **Test:** offline = does off-beat placement become selective (on-beat% recovers toward real; generated
  off-beats correlate with sal)? Then playtest (does it FEEL like musical syncopation vs the smear?).

### A2 — trained gate (architecture change) — only if A1 shows promise
Add an explicit interaction term to the onset head: `onset_logit = base + radar_chaos · (w ⊙
local_onset_feat)`, trained. Principled (the model learns the saliency weighting) but needs a retrain;
gate it behind A1's result.

### A1 RESULT (2026-06-21, `experiments/realism_critic/chaos_gate.py`) — works offline, awaiting playtest
First gate (multiplicative boost on off-beat p_base) was WRONG: it swung the whole on-beat/off-beat
backbone bimodally (g=1→98% on-beat, g=12→0%) instead of adding accents, and p_base is ~blind to which
off-beats are salient (model trained to suppress them). **Fixed gate v2:** decouple phases — on-beat
backbone (top p_base) + off-beat accents (top **audio** saliency = high-res onset dim41); `chaos_frac` =
budget fraction on accents; total density fixed. Result (8 songs):

| | real | smear | f=0.15 | f=0.30 | f=0.45 | f=0.60 |
|---|---|---|---|---|---|---|
| on-beat% | 86% | 6.3% | 85% | 70% | 55% | 40% |
| selectivity (dim41 chosen/all off-beats) | — | 1.0× | 2.97× | 2.54× | 2.30× | 2.14× |

Smooth, density-preserving syncopation dial; accents land on ~2.1–3.0× louder audio events (event-driven,
NOT the smear's 1.0× uniform). **Caveat:** selectivity is vs dim41, the same signal used to select — so it
proves the audio HAS off-beat events to target (not flat), not that hitting them is musical; dim41 matches
real charter off-beat choices only ~66%.

**PLAYTEST VERDICT (2026-06-21, `playtest_log.md`): gated is PLAYABLE but NOT musical; the off-beats feel
arbitrary ("random chance"). smear was an unplayable wall.** So the gate is a strictly better chaos knob
(playable, controlled) but audio-selective placement is NOT the musicality lever. This was the
pre-registered "feels off despite landing on audio events" branch → **syncopation is groove/pattern, not
audio-placeable (H10).** The gate is *pointillist* (each off-beat chosen independently); musical
syncopation is a RHYTHMIC-PATTERN / periodicity property (repeated off-beat figures = a groove). A2
(training the gate in) is DROPPED — it would bake in the same pointillist placement. **Lever moves to
rhythmic-structure modeling**, OR deprioritize chaos for higher-ROI threads (see below).

## Design B — Stage 2c critic-guided fine-tuning (the objective fix, for general taste)
Distill the taste critic's preference into the generator so *single* draws improve (not just best-of-N).
B1 (best-of-N self-distillation): generate N, keep top-K by P(real), fine-tune CE on them, iterate.
- **Scope:** PLAIN/overall musicality, NOT chaos (see above). This is the natural culmination of the
  Stage-2 critic work and aligns with the project thesis (a learned taste metric that actually improves
  generation). Worth doing for general quality + the Hard tameness (if the gate/hands-filter give it good
  draws to distill).
- **Risk:** distribution collapse / reward-hacking the critic; mitigate with KL-to-base regularization,
  modest K, and re-checking the critic isn't being gamed (held-out real charts stay high).

## Recommended sequence
1. **A1 decode-time selective chaos gate** — cheap, no retrain, model-agnostic, directly tests "does
   selective off-beat placement (vs uniform smear) recover musical syncopation?" Try `sal = p_base` first
   (stronger than dim41). Offline selectivity check → playtest.
2. If A1 helps: **A2 trained gate** to bake it in; if A1 underwhelms (off-beats genuinely aren't
   placeable from available signal), that's strong evidence syncopation is groove/pattern-modeling
   (charter style) and the lever moves to richer pattern modeling / explicit groove templates.
3. **2c (B1)** for general taste/Hard — independent track, not chaos-specific.

Cross-refs: `h4_offbeat_signal_findings.md` (why not features), `stage2a_critic_findings.md` (critic =
taste metric; chaos 0.003), `playtest_log.md` (H4/H6), `stage2_realism_critic_plan.md` (2c original plan).

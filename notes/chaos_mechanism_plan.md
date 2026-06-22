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

## SYNTHESIS (2026-06-21) — chaos is the keystone; root is architectural

User intuition (endorsed): mastering chaos conditioning unlocks the other axes. Why, mechanistically:
the other radar dims (stream/voltage/air/freeze) are QUANTITY knobs — a global scalar satisfies them (and
`match_radar` steers them well, `conditioning_match_findings.md`). **Chaos is the only QUALITATIVE axis —
*which* off-beats are musically right — which a global scalar fundamentally can't express, so it smears.**
The capability that solves chaos (understanding musical rhythmic STRUCTURE) is what grounds every other
axis's musicality.

**Root (stacking the session's findings):** musical syncopation = a *periodic groove* (H10; ac_off real
0.19 vs null 0.01), largely NOT audio-cued (off-beat audio AUC 0.66, redundant; features don't fix it —
chroma ✗, high-res onset ✗×2). So groove is memory/context-based, not per-frame. **But the onset head
(which decides rhythm = *when* notes go) is NON-CAUSAL / per-frame / audio-driven** (chosen to avoid
AR-drift collapse). A per-frame head with no memory of prior placements **structurally cannot produce a
repeated rhythmic figure** — it hits on-beats (audio aligns) but can only *scatter* off-beats. **Chaos
smears by architecture.** (Ties to H11: rhythm needs memory; the onset head lacks it.) This is why per-frame
decode gates feel "arbitrary" (pointillist) and the global scalar only shifts the whole grid — neither makes
a groove.

**Lever = give rhythm generation structure/memory.** Decided path: probe FIRST (does *imposed periodicity*
feel musical → confirms the training TARGET) before a groove-aware-conditioning or periodicity-objective
retrain.

### Periodic-groove decode probe (2026-06-21, `chaos_groove_decode.py`): imposed periodicity is MECHANICAL
Imposed an audio-grounded repeated off-beat figure per section (top-3 salient off-beat slots, fired every
measure). Result (6 rich Hard songs): the groove hits ac16≈0.97 / ac4≈ac8≈0.59 **identically for every
song**, vs real's *varied* periodicity (ac4/8/16 ranging 0.2–0.76, song-specific). So a rigid "same slots
every measure" template = a **robotic loop**, not a musical groove. (Taste critic uninformative here —
reads ~0.00 for BOTH groove and baseline = its known Hard/dense over-rejection bias, not a verdict.)
**Conclusion: periodicity ALONE isn't musicality** (H10 is necessary, not sufficient) — real grooves are
periodic WITH variation/development/grounding, just as per-frame *selectivity* wasn't musicality either.
**Both decode routes (per-frame selective, rigid periodic) are now exhausted** → chaos needs LEARNED groove
generation (variation + grounding), i.e. the objective/architecture route, not a decode template. Playtest
(`chaos_groove_{baseline,groove}`, esp. deja loin) is the arbiter; offline predicts mechanical.

## NO-16THS LOCALIZATION (2026-06-21, `diag_no16ths.py`) — chaos=0 is "model produces no 16ths"; threads converge

Note-fraction by phase: REAL quarter/8th/16th = 70.7/25.2/**4.1**; gen_stage1 = 86.7/13.3/**0.0**;
gen_highres = 86.8/13.2/**0.0**. Model over-quarters, under-8ths, ZERO 16ths.
p_on by phase: gen_stage1 quarter/8th/16th = 0.616/0.434/**0.169** (16th>τ = **0.0%**); gen_highres
0.606/0.451/**0.204** (16th>τ 0.0%).

Localized — chaos=0 has FOUR converging causes:
1. **Posterior under-weights 16ths** (p_on 0.169 vs 0.43/0.62) — rhythm simpler than real.
2. **Density threshold structurally excludes 16ths** (16th>τ = 0%): 16ths are the lowest-p_on frames, so
   the top-density selection (quarters+8ths) never reaches them.
3. **High-res feature directionally right but insufficient** (16th p_on 0.169→0.204) and unengaged (H4-v2
   KL≈0 — 16ths too rare to move the average loss).
4. **Data is 16th-sparse** (filtered real = 4.1% 16ths; max-2/length filters cut the 16th-heavy charts).

**Keystone vindicated:** chaos (16ths) is where H4 (resolution) + constraint-relaxation (data) + objective
(16ths too rare) ALL converge. **Decode can't fix it** (16th posterior is weak AND 8th-resolved-noisy →
forcing 16ths = arbitrary, the gate lesson). **Chaos needs a RETRAIN** combining: high-res feature ENGAGED
(heavy 16th-frame loss weight, not the 3× off-beat — 16ths need a big weight), + ideally de-filtered
16th-richer data. Cheap targeted test: retrain gen_highres on the warm cache_v3 with a heavy 16th loss
weight (does the high-res feature, strongly incentivized, finally place 16ths?). Bigger: + de-filtered data.

## 16th-WEIGHTED RETRAIN (2026-06-21) — v3 FAILED (weighting bug), v4 WORKS: model now produces 16ths

Goal: force the model to produce 16ths (chaos=0 root). Both warm-start gen_stage1 + random-init high-res
column on warm cache_v3.
- **v3 (`train_highres_v3.py`, --w16 15 on all 16th FRAMES): FAILED.** 16ths stayed 0.1%; p_on@16th went
  DOWN (0.169→0.153). Bug: 16th positions are ~50% of frames and mostly EMPTY, so weighting all 16th frames
  amplified the "no-note" negatives → reinforced no-16ths.
- **v4 (`train_highres_v4.py`, weight POSITIVE 16th notes only = recall): WORKS.** note-fraction
  quarter/8th/16th: gen_stage1 86.7/13.3/**0.0** → v4 69.1/30.1/**0.8** (REAL 70.7/25.2/4.1). p_on@16th
  0.169→**0.415** (high-res feature ENGAGED at last), 16th>τ 0.0%→0.5%. crit_adj **0.969** (best),
  onset_F1 0.708. Early stopping fired @ epoch 4.

**The keystone cracks: the model went from STRUCTURALLY incapable of 16ths to producing them, with
real-matched rhythm balance (quarter-dominance 87%→69%).** chaos radar now > 0. Gap to real 4.1% is a
TUNING problem (16ths still mostly below the density threshold), not a wall. Levers to push toward 4.1%:
higher --w16, a phase-aware DECODE threshold (lower bar for 16ths so p_on 0.415 gets selected), more epochs
(early-stop@4 may cut 16th-recall short — val_total isn't the 16th metric), de-filtered 16th-rich data.
**NEXT: playtest gen_highres_v4 vs gen_stage1 — does the rebalanced rhythm + nascent 16ths feel more
musical/complex?**

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

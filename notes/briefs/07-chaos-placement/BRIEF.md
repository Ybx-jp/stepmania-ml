# Brief 07 — Chaos / 16th-placement: the keystone arc (AMOUNT solved, PLACEMENT bounded)

**Source notes (read in order):** `chaos_mechanism_plan.md` (keystone synthesis) → `selfsim_chaos_findings.md`
→ `chaos_conditioning_findings.md` → `phase_aware_threshold_findings.md` → `chaos_retrain_scope.md` →
`v7_additive_loss_design.md` → `chaos_placement_ceiling_SUPERSEDED.md` (⚠️ banner-superseded) →
`sequence_aware_onset_plan.md`
**Arc role:** the longest, most-revised thread. It contains **two superseded conclusions** — the single
most dangerous thing for a README claims audit. The net, durable result: **chaos AMOUNT (how many 16ths)
is a real shippable knob; chaos PLACEMENT (which 16th lands where) is bounded in this paradigm.** The
README's "chaos" honesty line lives or dies here. **Read this brief together with [[09-manifold-guidance]]**
— the *deployed* chaos dial was later reframed there.

---

## The narrative (with the supersession chain made explicit)

### Beat 1 — the keystone framing (`chaos_mechanism_plan.md`)

Chaos is the **only qualitative radar axis** (which off-beats are *musically* right), which a global
scalar fundamentally can't express:

> "**Chaos is the only QUALITATIVE axis — *which* off-beats are musically right — which a global scalar
> fundamentally can't express, so it smears.**"

A long chain of decode probes (selective gate, rigid periodic template) all play **mechanical/arbitrary**:

> "**PLAYTEST VERDICT: gated is PLAYABLE but NOT musical; the off-beats feel arbitrary**... syncopation is
> groove/pattern, not audio-placeable (H10)."

> "**periodicity ALONE isn't musicality** (H10 is necessary, not sufficient) — real grooves are periodic
> WITH variation/development/grounding."

The localization: chaos=0 because **the model produces literally zero 16ths** (gen_stage1 16th-share 0.0%
vs real 4.1%), with four converging causes (weak posterior, threshold excludes 16ths, feature unengaged,
data 16th-sparse). The fix that cracked it — weight **positive 16th notes only**:

> "**v4 (weight POSITIVE 16th notes only = recall): WORKS.**... quarter/8th/16th: 86.7/13.3/**0.0** → v4
> 69.1/30.1/**0.8**... p_on@16th 0.169→**0.415** (high-res feature ENGAGED at last)... **The keystone
> cracks: the model went from STRUCTURALLY incapable of 16ths to producing them.**"

### Beat 2 — don't INFER chaos, CONDITION it (`selfsim_chaos_findings.md`)

> "**Self-similarity is a dead end for chaos.**... **Section 16th density is barely audio-predictable —
> R² ≈ 0.06.**... WHERE a 16th goes is audio-driven (model nails it); HOW MANY a section gets is mostly
> charter style/artistic choice."

### Beat 3 — chaos conditioning WORKS post-high-res (`chaos_conditioning_findings.md`)

> "**Strong dynamic range:** realized 16th share 0.3% -> 26% across chaos 0->1, MONOTONIC."

> "**SPECIFIC, not a smear:** raising chaos DROPS quarter p_on... while 16ths flood in — trades quarters
> for 16ths like real charts getting busier. The H4/H6 smear raised everything uniformly; this does not...
> plain guidance=1 has full authority — NO CFG amplification needed."

> "**The paradox resolved:** Audio can't TELL the chaos level (selfsim R^2 0.06) but the chaos INPUT has
> near-total authority. Both true: don't infer chaos, CONDITION it."

### Beat 4 — the validated decode lever: `onset_phase_calib`, NOT the quota (`phase_aware_threshold_findings.md`)

A subtle but important result — the flat quota is provably *smearing*; the per-song-normalized calib is the win:

> "**The quota (`alloc`) is provably smearing**: std 0.1% (constant), and it DEGRADED corr 0.459→0.256."

> "**`calib` (per-phase LOGIT offset b16≈0.19 + per-song threshold) is the win**: variable (std 3.9),
> spans the full real range [0,24%], best volume-matched corr (0.389)."

> "**The ceiling is the MODEL, not decode.** Every method tops out at song-corr ~0.39–0.46. Strong
> frame-local (0.742), weak song-level."

**Shipped:** `onset_phase_calib=(b8,b16)`; `onset_phase_alloc` kept but documented as smearing.

### Beat 5 — the loss work: v6 FAILED, v7 additive WIN (`chaos_retrain_scope.md`, `v7_additive_loss_design.md`)

> "**v6 (ranking loss λ=1) stripped the 8th groove and flooded 16ths**... 50/3/48% <- 8ths NUKED, 16ths 3x
> overshoot."

> "**v7 = pure reweighted BCE (w8=1, w16=10)**... 16th-rate DOUBLED toward real (0.017->0.036), 8ths came
> DOWN... to ~real... rhythm-distribution error nearly halved (0.091->0.052, closest to real)."

Also a refuted sub-hypothesis: "**HPSS percussive/harmonic high-res onsets do NOT beat the mixed onset**
(all ~0.58)... **not a feature problem.**"

### Beat 6 — the SUPERSESSION: placement is sequence-determined, not audio-ambiguous

`chaos_placement_ceiling_SUPERSEDED.md` originally concluded placement was at an "audio-ambiguity ceiling
(~0.65–0.67)." **That conclusion is wrong and banner-superseded:**

> "⚠️ SUPERSEDED... the note SEQUENCE alone predicts real-16th placement at AUC **0.935** (vs audio 0.649).
> Placement is SEQUENCE-determined (run coherence), NOT audio-ambiguous."

### Beat 7 — but the sequence signal is REAL yet NOT cheaply exploitable (`sequence_aware_onset_plan.md`)

The final, durable verdict of the whole arc:

> "**AR onset explodes** (density 0.73 vs real 0.177)... **refinement is stable but can't bootstrap from
> v4's anti-correlated C0** (0.456, places 16ths in WRONG spots)... **Reaching the 0.935 ceiling requires
> good context (near-real notes) that audio alone can't produce.**"

> "**Step back: ship-state = v4 + amount-control (calib/conditioning); this rigorous bounding is the thesis
> deliverable.**"

---

## Audit hooks (reconcile README against these)

| README claim | Verbatim source | Verb precision |
|---|---|---|
| chaos is controllable | "realized 16th share 0.3% -> 26%... MONOTONIC"; calib "spans the full real range" | **measured** ✅ — but this is **AMOUNT** control. Say "chaos *amount*," not "chaos." |
| "chaos still smears off-grid" (old README line) | the **radar-CFG** chaos smears (H4/H6); the **trained chaos input** does NOT ("SPECIFIC, not a smear") | **AUDIT-CRITICAL & STALE-PRONE.** The v1 audit corrected row 19. The smear was the *old radar-CFG* mechanism; the *deployed* path doesn't smear. Don't ship the "smears" line as-is. |
| chaos = "in-distribution-bounded; 16th-share isn't the dial" (corrected line) | This arc says 16th-*amount* IS controllable; the "16th-share isn't the dial" reframing is the **manifold** conclusion → [[09-manifold-guidance]] | **Reconcile across briefs.** This arc: amount is a dial via calib/conditioning. The manifold arc: the *deployed* dial is conditional-fill staying in-distribution, and raw 16th-share is the wrong handle. Both true at different layers; the README's corrected line is the manifold framing. |
| 16th PLACEMENT quality | "placement is SEQUENCE-determined (AUC 0.935)" but "NOT cheaply exploitable... ship-state = v4 + amount-control" | **DO NOT claim placement is solved or audio-bounded.** The honest line: amount is controllable; placement excellence needs a different paradigm (learn the placement distribution from multiple human chartings). |
| methodology / "rigorous bounding" as a thesis point | "decode/features/architecture each refuted by a ~10-min probe BEFORE an expensive build" | **vouched** ✅ — strong case study for the evaluation thesis; safe to cite as *process*, not as a solved capability. |

**Verb-precision watch:** this arc has TWO superseded conclusions (the quota; the audio-ambiguity ceiling).
Any README number sourced from `*_SUPERSEDED.md` is automatically suspect — the "~0.65–0.67 audio ceiling"
framing is **refuted** (real placement signal is 0.935, sequence-borne). The safe, durable claims are: (1)
the model went from 0% to real-matched 16th *amount* (v4); (2) `onset_phase_calib` gives per-song-variable
chaos across the full real range; (3) placement excellence is an open frontier, not a shipped feature.

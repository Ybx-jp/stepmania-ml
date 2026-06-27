# Brief 08 — Phase 3: the joint generative paradigm (de-risked, PARKED — not shipped)

**Source notes:** `phase3_generative_design.md` → `phase3_prototype_findings.md`
**Arc role:** the "what would actually fix placement" research thread. It exists because [[07-chaos-placement]]
proved placement is sequence-determined but not cheaply exploitable. Phase 3 is a **research project, not a
fine-tune** — its gates passed, a prototype was built, but **the deployed track stayed the staged AR
generator.** For the README this matters mainly as a *negative*: do not present Phase 3 as a shipped
capability.

---

## The narrative

### Beat 1 — the design + why a new paradigm (`phase3_generative_design.md`)

The cleanest single explanation of the whole chaos arc lives here — placement is **inherently a
distribution**:

> "**Multi-charting DIVERGENCE check DONE**... onset-IoU median 0.60 and **16th-IoU median 0.325** —
> independent humans agree on only ~1/3 of 16th placements. **PLACEMENT AMBIGUITY IS REAL AND LARGE.**
> This is the cleanest explanation of the whole arc: there is NO single right 16th placement, so
> point-target training chased one arbitrary sample of a high-variance distribution → mush."

The gates that passed:

> "**OBJECTIVE GATE PASSED**... critic P(real) REAL 0.844 vs v4-gen 0.043... vs **shuf16 0.524**... **The
> critic SEES placement quality.**"

> "**DATA GATE**... 484 titles... appear in >=2 distinct packs... **The distribution data EXISTS.**"

The convergent design: a **joint generative** model (diffusion or mask-predict), structure-aware,
audio-conditioned, with a distribution-aware objective. Honestly costed:

> "Phase 3 = a research project, not a fine-tune. Current best playable stays gen_highres_v4."

### Beat 2 — the prototype: stable, but a generation-order flaw (`phase3_prototype_findings.md`)

> "**POSITIVE: joint generation is STABLE.** run-mean 1.00 ≈ real (1.02) — NO explosion (vs AR 5.7)... And
> the model LEARNED placement (TF AUC 0.849 >> audio 0.65)."

> "**NEGATIVE: naive generation starves 16ths (0% placed)**... the UNMASK ORDER: from all-masked there's no
> note-context... 16ths are CONTEXT-dependent... never become 'most confident' → never committed."

And a deeper finding that *reinforces the project's evaluation thesis*:

> "**evaluating placement QUALITY on GENERATED charts is unsolved** — the taste critic confounds placement
> with panel style, and you can't score against ONE real chart because placement is a distribution... **The
> EVALUATION is as hard as the generation.**"

---

## Audit hooks (reconcile README against these)

| README claim | Verbatim source | Verb precision |
|---|---|---|
| any "joint/diffusion/mask-predict" generation capability | "de-risked prototype"; "Current best playable stays gen_highres_v4"; staged track deployed | ⚠️ **Phase 3 is PARKED.** If the README mentions it, frame as *explored / de-risked research*, NOT a feature. The deployed model is the staged AR generator ([[10-motif-arc]] / [[00-meta]] HANDOFF). |
| placement is "inherently a distribution" / humans agree ~1/3 | "16th-IoU median 0.325 — independent humans agree on only ~1/3 of 16th placements" | **measured** ✅ but **N=15 comparable pairs** ("small; but 86%-differ is robust"). If cited, keep the sample-size honesty. Strong evaluation-thesis material. |
| critic "sees placement" | "critic P(real) REAL 0.844 vs shuf16 0.524 (+0.32)" | **measured** ✅ — the clean placement signal is the **shuf16 +0.32** (v4-gen's +0.80 "conflates placement with other deficits"). Use +0.32, not +0.80, as the placement-specific number. |

**Verb-precision watch:** Phase 3's gates *passed* and a prototype *ran*, but it is **not deployed and not
playtested as a product**. The durable, citable takeaways are conceptual (placement is a distribution; the
critic can see placement; evaluation is as hard as generation) — not "we built a diffusion chart generator."

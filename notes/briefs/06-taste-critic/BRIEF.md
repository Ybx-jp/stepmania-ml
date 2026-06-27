# Brief 06 — The taste critic (the README HEADLINE thesis)

**Source notes:** `stage2_realism_critic_plan.md` → `stage2a_critic_findings.md`
**Arc role:** the project's differentiated story — **"evaluation for hard-to-measure quality."** Every
prior offline metric (onset_F1, crit_adj, phase, structure, periodicity) is blind to musical *taste*;
this arc builds a learned metric that isn't. It is the README's climax, so its claims must be exact —
and the arc contains a **failed v1** that the README must not erase.

---

## The narrative

### Beat 1 — the motivation: the objective, not the model (`stage2_realism_critic_plan.md`)

> "the Stage-1 model plays 'mostly right' but has 'no sense of taste'... Diagnosis converged on the
> **training objective** — frame-wise CE rewards matching the reference token, never 'is this a coherent,
> tasteful chart.'"

The plan is explicitly staged with a **validation gate**: 2a build+validate the critic → 2b best-of-N
reranking → 2c critic-guided fine-tune, "*Stop here and check before escalating.*" It even predicts the
exact failure mode that v1 hits:

> "**Critic shortcuts** (density/length detector instead of taste) — mitigate with mismatched negatives
> + an ablation audit; this is the main thing to watch in 2a."

### Beat 2 — v1 FAILED: the critic learned the generator's fingerprint (`stage2a_critic_findings.md`)

This failure is load-bearing for the *story* and must survive into the README's narrative (it's why the
final critic is trustworthy):

> "**v1 (generated negatives) FAILED** — learned the generator fingerprint, taste metric scored
> backwards." Taste test ranked **CHAOS 0.680 > REAL 0.386 > BASE 0.123** — "**Ranking is BACKWARDS.**"

> "**Lesson:** A discriminator optimizes the *easiest* separating cue, not the one we want. 'Real vs
> our-generated' was easier to learn than 'tasteful vs not.'"

### Beat 3 — v2 SUCCEEDED: corrupted-real negatives isolate taste (`stage2a_critic_findings.md`)

The fix: negatives are REAL charts perturbed at **fixed density/timing** (scramble panels, or shift vs
audio), so the only remaining cue is taste — no generator in training.

> "**AUC 0.964** (epoch 12) separating real vs corrupted-real. P(real): real ≈ 0.79, panels ≈ 0.03,
> shift ≈ 0.05 → the critic confidently flags both scrambled-arrow and misaligned charts as fake. It
> learned arrow-choice taste AND audio-alignment."

> "**Taste metric... REAL 0.823 > BASE 0.290 > CHAOS 0.003** — matches the playtest exactly... **We now
> have a quantitative musicality signal** — the thing every prior metric couldn't see."

### Beat 4 — the critic is trustworthy at Hard; the generator is *tame* (Stage 2b)

A crucial cross-check that protects against over-claiming best-of-N:

> "**Real Hard scores 0.82 — as high as every other difficulty. The critic does NOT over-reject Hard.**...
> the gap is real. Generated Hard ≈ 0.02–0.12 vs real Hard ≈ 0.82 is a genuine gen-vs-real difference."

> "**2b (best-of-N) has little headroom at Hard** — *all* candidates are tame, so selection can't surface
> intensity that the generator never draws. Best-of-N is a low/mid-difficulty win, not a Hard fix."

Two instruments agreeing — the critic flags tameness, the user "expected more chaos" — is itself a
validation of the taste metric.

---

## Audit hooks (reconcile README against these)

| README claim | Verbatim source | Verb precision |
|---|---|---|
| Taste critic **AUC 0.964** | "AUC 0.964 (epoch 12)" | **measured** ✅ — separating real vs **corrupted-real** (NOT real vs generated; v1's real-vs-gen AUC was only 0.723). State the negative type. |
| **REAL 0.823 > BASE 0.290 > CHAOS 0.003** | verbatim | **measured** ✅ (64 val songs). The ordering "matches the playtest exactly" — this is the metric↔hands agreement, the thesis's keystone. |
| critic reuses Phase-1 backbone; corrupted-real negatives; **v1 scored backwards** | "warm-started from Phase-1"; "v1... learned the generator fingerprint, scored backwards" | **measured** ✅. The README SHOULD keep the v1-failure beat — it's what makes v2 credible. Don't sand it into "we built a critic." |
| best-of-N reranking | "best-of-N reranking" (2b) — built; Hard diagnostic run | ⚠️ **HONESTY BLOCKER (release criteria):** "No claims about un-playtested work (e.g. best-of-N reranking — built, not playtested)." Best-of-N is **low/mid-difficulty only**, has "little headroom at Hard," and was NOT validated as a play-feel win. The README must NOT claim best-of-N improves charts. Mention it as built/experimental at most. |
| critic = the "evaluation for hard-to-measure quality" thesis | "We now have a quantitative musicality signal — the thing every prior metric couldn't see" | **measured/vouched** ✅ — strongest support for the headline. Keep it as a *metric that ranks in playtest order*, not "solves musicality." |

**Verb-precision watch:**
- AUC 0.964 is **real-vs-corrupted-real**, an easier task than real-vs-arbitrary-chart. Don't imply it
  distinguishes good *generated* charts from bad ones at 0.964 — that's a different, harder distribution.
- The critic is a **judge**, not a fix. It made the *tameness* of Hard charts measurable; it did not make
  Hard charts good. 2c (critic-guided fine-tune) is "**bottlenecked by H4**" and was not completed.
- Cross-ref [[05-musical-features]] (why CE can't reward taste) and [[playtest_log]] (the playtest order
  the metric reproduces).

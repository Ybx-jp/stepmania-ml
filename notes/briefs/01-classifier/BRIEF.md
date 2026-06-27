# Brief 01 — Phase 1: the difficulty classifier (CLOSED)

**Source notes:** `notes/ordinal_experiment_findings.md`
**Arc role:** the project's *origin*. A 4-class difficulty classifier (Beginner/Easy/Medium/Hard)
whose audio encoder + difficulty judgment are later reused everywhere downstream — warm-started into
the generator, and reused again as the difficulty critic. Everything after this is generation; this is
the only "closed" phase.

---

## The narrative

Phase 1 was a head bake-off, not a from-scratch build. The question was whether an **ordinal**
(order-aware) head beats a plain softmax classifier on an inherently ordered label (Beginner < Easy <
Medium < Hard), and whether contrastive pretraining helps. A 6-way grid was run under identical seed/splits:

> "6-way comparison: `{standard, contrastive} × {classification, ordinal (scalar), ordinal_multi}`."
> — `ordinal_experiment_findings.md`

The result is a **multi-output** ordinal head winning narrowly, a **scalar** ordinal head collapsing
outright, and contrastive adding nothing:

> "**Winner: `standard_ordinal_multi`** — best on every metric (82.9% acc, 0.835 macro F1, lowest MAE
> and adjacent-error rate). New Phase 1 best, beating the prior contrastive baseline (~81.4% val)."

> "**The scalar proportional-odds head collapses** (~44% acc, ~49% adjacent errors)... Early-stopping
> fired at epoch 2 — it never trains. The single-scalar bottleneck + tight init is the culprit, not
> ordinal modeling itself."

> "**Ordinal structure helps only without the bottleneck.** `ordinal_multi` modestly beats
> classification (16.5% vs 18.3% adjacent, +1.7pp acc, +1.6pp F1)."

The original hypothesis (<15% adjacent-error) was **not** met by anyone — honestly flagged:

> "**Nobody cleared the <15% adjacent-error bar**, but `ordinal_multi` came closest at 16.5%. The
> threshold was aspirational; treat 16.5% as the current ceiling for this data/architecture."

There's also a self-correction worth noting (a guard against a misleading auto-printed verdict):

> "`compare.py`'s printed 'HYPOTHESIS VERDICT' only contrasts classification vs the *scalar* ordinal
> head... reports 'classification wins by ~31%.' That verdict is stale... read the table, not the verdict line."

---

## Audit hooks (reconcile README against these)

| README would claim | Verbatim evidence | Verb precision |
|---|---|---|
| Phase 1 **82.9% test acc, 0.835 macro F1** | "82.9% acc, 0.835 macro F1" (test set, 871 samples, seed 42, 20 epochs) | **measured** ✅ — a single closed test eval |
| ordinal head chosen / `standard_ordinal_multi` deployed | "Winner: `standard_ordinal_multi` — best on every metric" | **measured**, but the win over plain classification is *modest* (+1.6pp F1). Don't oversell "ordinal beats classification" — it's true only for the multi-output form; the scalar form collapsed. |
| classifier reused downstream | "Graduating to Phase 2... reusing the Phase 1 audio encoder backbone." | architecture fact, traced in later briefs ([[02-generative-foundation]], [[06-taste-critic]]) |

**Caveat for the README:** the headline accuracy is a *classifier* metric, not a generation metric.
If the README cites 82.9% / 0.835 it should be unambiguously labelled as the **Phase-1 difficulty
classifier**, not the chart generator. The checkpoint is `checkpoints/ordinal_exp/standard_ordinal_multi/best_val_loss.pt`.

**Numbers a reader could trip on:** test set = 871 samples; the comparison ran 20 epochs at seed 42.
These are small-scale; the macro F1 is the primary metric per project convention (imbalanced classes).

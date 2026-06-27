# Brief 00 — Meta: the playtest ledger (the validated instrument) + current deployment state

**Source notes:** `playtest_log.md` (1462 lines, the cross-cutting play-feel ledger) + `HANDOFF.md` (current
state). **Arc role:** not an experiment arc — the *spine that runs through all of them*. The playtest log is
**the project's only validated instrument for play-feel** ("offline metrics are blind to choreography/vibe,
proven all arc"). Every brief that says "playtest-confirmed" or "vouched" traces here. For the README, this
is the source of the central thesis (metrics can't see the quality that matters) and of the honest current
state of the deployed model.

---

## What the playtest log is, and why it's load-bearing

> "**METHODOLOGY (06-21, user directive): playtest sets must be GROOVE-VALIDATED.** A set only tests a
> hypothesis if its songs actually exercise the relevant axis — B4U with 3 holds can't test a hold fix."

The log records, newest-on-top, every hands-on play of a generated chart, plus standing hypotheses (H1–H20
and the H-stamina / H-arc family). It is the arbiter the offline metrics repeatedly *failed* to be — the
concrete demonstration behind the README's "evaluation for hard-to-measure quality" thesis.

## The standing hypotheses and their dispositions (the audit-relevant ones)

These are the play-feel claims the README's qualitative statements rest on. Disposition matters for verb
precision:

- **The breakthrough (H15, 2026-06-23):** "the project's **FIRST genuinely-good chart by ear**" — OH WORLD
  glitch, manifold-conditioned, g≈3.5. Real, but a *single operating point*, not a general guarantee. → [[10-motif-arc]]
- **H13 (exertion):** jacks confirmed by data AND felt; `max_jack_run=1` fix "AWESOME!" → CLOSED. → [[11-governor]]
- **H-stamina (governor Stage 2):** "a **tasteful edit, not a rewrite** … DEFINITELY an improvement" under
  cranked conditioning; **imperceptible on normal charts** (correct — opt-in relief valve). → [[11-governor]]
- **H-arc (governor Stage 3):** "mostly good" with two fixed/parked issues — abrupt endings FIXED
  (`stamina_breathe_floor`); the "ignored piano solo" REDIRECTED to an onset-head feature thread (not a
  governor bug). → [[11-governor]]
- **H18 (chaos / structure):** the corrected README framing (chaos in-distribution-bounded; the model tracks
  structure/accents; global phrase-planning is the open frontier) traces here + [[07-chaos-placement]] / [[09-manifold-guidance]].
- **H20 (vocabulary coverage gap):** the model over-produces jacks, under-covers ornamental footwork — a
  data/objective gap, not a knob. → [[10-motif-arc]]
- **Open/parked:** H-onset-perc-bias (onset head under-places on melodic-only sections); model under-jumps
  (6% vs real 31%) — both **not release blockers**, both **not** governor issues.

## Current deployment state (from `HANDOFF.md`, 2026-06-26)

- **Deployed generator:** `checkpoints/gen_motif_full_fixed/best_val.pt` (42-dim highres, the H19 clean retrain).
- **Decode-time governor:** Stage 1 per-note foot model (`fatigue_penalty=2`, **always on**); Stage 2 stamina
  relief + Stage 3 breathing arc (**opt-in**, near-no-op until conditioning is cranked past the human density
  envelope).
- **Mandatory playability** (code-enforced, never off): `hold_aware, no_jump_during_hold, no_cross_during_hold,
  max_jack_run=2, pattern_temperature≈0.7`.
- **Conditioning:** `--style` (manifold, in-distribution) is the path; **`--radar` is DISABLED** (mean-pin is
  off-manifold → errors). Motif/figure knobs exist internally but are **not exposed in `scripts/generate.py`**
  and `cache/motif_basis.npz` is **not shipped**.
- **Release vehicle:** **PR #41 — MERGED.** Verified against git this audit pass: `origin/main` head is
  `37257d1` ("Merge pull request #41"), and it contains the governor (`fatigue_penalty`). `HANDOFF.md` says
  "Not merged" because it was written *pre-merge* and is stale on that one point; `RELEASE_CRITERIA.md` ("PR
  #41 merged @ `37257d1`") is correct. **The tagged code does contain the governor** — the README's
  code-currency claim is safe. (The working branch `release/v0.1.0-prep` also has it: 7 `fatigue_penalty`
  refs in `typed_model.py`.)

---

## Audit hooks (reconcile README against these)

| README claim | Source | Verb precision |
|---|---|---|
| "playtest log" exists and backs qualitative claims | `playtest_log.md`, groove-validated methodology | **vouched** ✅ — the README rightly points here for play-feel the metrics can't capture. This IS the differentiated evaluation story. |
| any "playtest-confirmed" governor/motif claim | H13, H-stamina, H15 candle dispositions above | **vouched** ✅ — keep qualitative; these are single-user hands-on plays, not a study. Don't quantify. |
| "first genuinely good chart" / quality milestone | H15 OH WORLD entry | **vouched** ✅ but it's *one chart at one operating point*. Don't generalize to "the model makes great charts." |
| deployed model identity | HANDOFF: `gen_motif_full_fixed` | **measured** ✅ — if the README/model-card names a generator it should be this, not `gen_stage1/radar/style` (model-card lineage is a known-stale deferred item). |
| the tagged code contains the governor | **VERIFIED merged** — origin/main `37257d1` (PR #41 merge) contains `fatigue_penalty` | **measured** ✅ — safe to assert. (HANDOFF's "Not merged" is stale; it predates the merge.) |

**Verb-precision watch:** the playtest log is **single-user, qualitative, hands-on** — its power is that it
*catches what metrics miss*, not that it's a controlled study. Every "playtest-confirmed" should stay
qualitative. The open threads (perc-bias, under-jumps) are honestly **not fixed** — the README's honesty
section should not imply they are. Cross-references: every other brief links here for its play-feel evidence.

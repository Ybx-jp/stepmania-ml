# Brief 03 — Phase 2.5: the typed model (taps → holds, the layered head, playable export)

**Source notes (read in order):** `step_types.md` → `typed_model_findings.md` → `hold_aware_decode.md`
→ `hold_cross_decode.md` → `style_decoding.md` → `playable_samples.md`
**Arc role:** turns the binary-tap generator into one that emits the real step vocabulary (taps + holds)
and writes **playable** `.sm` files. This is where "it round-trips to a chart you can actually play in
StepMania" becomes true. Also the source of two README facts: **rolls are never generated** and the
**holds-at-the-real-rate** claim.

---

## The narrative

### Beat 1 — the data foundation, and the roll fact (`step_types.md`)

The typed representation is `(T,4)` over symbols `0=none 1=tap 2=hold-head 3=tail 4=roll-head`. The
load-bearing dataset finding:

> "**Rolls (symbol 4) do not occur anywhere in the dataset** (0 of 675 charts)... the effective
> generation vocabulary is **{none, tap, hold-head, tail}** (4 symbols)."

> "**Holds are well-represented** (in 380/410 external charts) but **rare per-cell** (~0.3–0.6%, roughly
> 1/20th the tap rate)."

This is the README's **"rolls 0 of 675 charts (not generated)"** claim — verbatim and exact.

### Beat 2 — a 3-attempt journey to make holds work (`typed_model_findings.md`)

This note is unusually honest about a multi-failure path; the README should not flatten it into "we
added holds." The journey:

> "Journey: broken weighted-CE per-panel (crit 0.53) → flat focal (0.69) → **layered head (0.79–0.84)**.
> The layered factorization (which-panels vs type) was the fix; the type sampling temperature is the
> hold-rate knob."

The repeated root cause is the SAME imbalance-conflation lesson from the onset head ([[02-generative-foundation]]):

> "**Root cause (deeper than weighting):** the per-panel 5-way head conflates *is this panel active*...
> with *what type* — the SAME imbalance-conflation we solved at the frame level with the onset head. No
> weighting scheme fixes a conflated objective."

The fix is the **layered head**: onset → 15-way *pattern* (which panels, warm-started from the binary
model) → 4-way *type* per active panel. And the hold-rate is a decode knob:

> "layered head, **no type-head class weighting** (calibrated focal), **sample the type** at generation
> with **type_temperature ≈ 0.4** → holds at ~the real rate (≈20:1), onset_F1 ≈ 0.76, crit_adj ≈ 0.79."

Cost is stated plainly: "Adding holds costs ~0.1 crit_adj vs the hold-free model (0.93)... ~0.79 with
real-rate holds is the working result."

### Beat 3 — holds become coherent spans, not noise (`hold_aware_decode.md`)

Stateless type sampling produced orphan heads/tails. A hold automaton (open on a head, close at the next
audio-driven note on that panel) fixed it:

> "**Orphans 56% → 3%** — holds are now genuine spans... **crit_adj 0.797 → 0.859** — difficulty
> fidelity *improved*... **onset_F1 up** (0.760 → 0.772); hold **median length 4 matches real**."

This is the README's **"holds at the real rate"** + structural-validity claim. Note the spans are
**audio-aligned** ("both endpoints are model-placed audio-driven notes").

### Beat 4 — two playtest-driven decode fixes (`style_decoding.md`, `hold_cross_decode.md`)

- **Sample the pattern head** — fixed a real playtest complaint ("almost always Left"):
  > "Greedy = 48% Left, **88% jacks** — exactly the playtest complaint." → sampling drops jack rate
  > "0.88 → 0.20 = real's 0.20", entropy "0.85 → 2.56 ≈ real 2.37", "onset_F1 unchanged."
- **`no_cross_during_hold`** — fixed the B4U "crossovers with one foot during a hold" complaint; the
  bipedal `hold_burst` metric "8.7% → 4.7% (real ~4.0%)" at "density 0.192 → 0.192 — unchanged (redirect,
  not delete)." ⚠️ Note status: "**Playtest pending**" at the time this note was written.

### Beat 5 — playable export closes the loop (`playable_samples.md`)

> "Generated `.sm` re-parses cleanly; e.g. 'Reach The Sky' Challenge: 211 steps, **28 holds = 28 tails**
> (valid), audio present in the folder → loads & plays in StepMania."

This is the ground truth behind the README's "writes back to `.sm`, re-parses with valid hold spans."

---

## Audit hooks (reconcile README against these)

| README claim | Verbatim source | Verb precision |
|---|---|---|
| **Rolls 0 of 675** charts (not generated) | "Rolls (symbol 4) do not occur anywhere in the dataset (0 of 675 charts)" | **measured** ✅ (dataset scan, not a generation claim — rolls aren't generated *because* there's no training signal) |
| Phase 2.5 typed **~0.77** onset_F1, "**holds at the real rate**" | onset_F1 0.772 (`hold_aware_decode.md`); type_temp 0.4 → "holds at ~the real rate (≈20:1)" | **measured** ✅, but "real rate" is *approximate* — the deployed hold-aware decode runs ~11:1 tap:hold, not exactly the real ~20:1. Say "near" the real rate, not "exactly." |
| holds round-trip, valid spans | "28 holds = 28 tails (valid)... loads & plays" | **measured** ✅ |
| crit_adj ~0.86 typed | "crit_adj 0.797 → 0.859" | **measured** ✅ — self-critic metric, same framing caveat as [[02-generative-foundation]] |

**Verb-precision watch:**
- The hold journey was a **3-attempt** path with two failures (crit 0.53 → 0.69 → 0.79). If the README
  implies holds "just worked," that overstates. The honest version: holds required re-deriving the
  factorized lesson at the panel level.
- `no_cross_during_hold` was **playtest-pending** when written — check [[playtest_log]] (cross-cutting,
  see [[00-meta]]) before the README claims it "feels more danceable." (It was later folded into the
  default decode; verify the playtest actually confirmed it.)
- Adding holds **lowers** crit_adj vs the hold-free model (0.93 → 0.79–0.86). README should not cite the
  hold-free 0.93 next to a "with holds" capability claim.

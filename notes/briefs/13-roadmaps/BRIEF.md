# Brief 13 — Roadmaps / standing constraints (augmentation; constraint relaxation; the "tame Hard" data cause)

**Source notes:** `augmentation_roadmap.md` → `constraint_relaxation_roadmap.md`
**Arc role:** the forward-looking, mostly-not-done threads — and, importantly for the README's **"Scope &
honesty notes,"** the authoritative statement of **what the data layer still constrains** (fixed BPM, max-2
simultaneous, 16th resolution) and the **second cause of "Hard feels tame"** (a data filter, not just the model).

---

## The narrative

### Beat 1 — augmentation: mirror is NEUTRAL; data quantity isn't the lever (`augmentation_roadmap.md`)

The one augmentation actually built (L↔R mirror) returned a clean null:

> "**RESULT... NEUTRAL.** Panel symmetry improved marginally (|L−R| 3.1%→2.0%)... onset_F1 flat... **taste
> P(real) slightly DOWN 0.419→0.377 (likely noise, certainly no gain).**"

> "**Disposition: keep the `--mirror` flag (free, harmless...) but it is NOT a quality lever.** ...consistent
> with the recurring 'the bottleneck is decode/objective/capacity, not data quantity' theme."

The one data lever that adds a *missing category* (not more-of-same) is the hands filter — which lives in the
next note.

### Beat 2 — constraint relaxation: what's still constrained, and the "tame Hard" data cause (`constraint_relaxation_roadmap.md`)

The status table is the README's ground truth for scope: holds **RELAXED**, multi-difficulty **RELAXED**,
rolls representable-but-untrained, **fixed BPM enforced**, **max-2 simultaneous enforced in data**, **16th
resolution enforced**, mines excluded.

The headline finding — the data's "Hard" is a tame subset:

> "`validate_pattern_quality` rejects any difficulty... with >2 simultaneous panels... this
> **disproportionately excludes real Hard charts** → the dataset's 'Hard' is the tame, hands-free subset →
> the model never sees hands and can't produce the intensity real 11s have."

But the honest correction on how much data that actually is:

> "**CORRECTION: the recoverable data is MUCH smaller than 55%.** The 55% was measured on RAW charts WITHOUT
> the song-length / BPM filters. Applying those... relaxing max-2 → 4 adds only: **Hard 910→997 (+87, +9.6%)**...
> Most hands-heavy expert/hardcore charts are LONG (>130s) and the length filter excludes them anyway."

And fixed BPM as deliberate, foundational scope:

> "**Fixed BPM — DEFER (heavy, foundational; do as a deliberate 'data layer v2').** Not a toggle. The whole
> pipeline rests on `hop_length = sr·60/(bpm·4)` → 'one audio frame = one 16th note.'"

---

## Audit hooks (reconcile README against these)

| README claim | Verbatim source | Verb precision |
|---|---|---|
| scope: fixed BPM / max-2 / 16th-grid / no mines | the constraint status table | **measured/accurate** ✅ — the README's "Scope & honesty notes" should match this table. holds + multi-difficulty are RELAXED (don't list them as limitations); fixed-BPM, max-2-in-data, 16th-res, mines are the real current limits. |
| "Hard feels tame" framing | "the dataset's 'Hard' is the tame, hands-free subset" + "the recoverable data is... only +9.6%" | **measured** ✅ — tame-Hard has **two** causes (on-gridness [[05-musical-features]]/[[07-chaos-placement]] AND the hands data filter here). Don't attribute it to one. And don't imply relaxing the filter would fix it — it's only ~+10% Hard data, and the bigger cull is the length filter. |
| data augmentation as a feature/strength | mirror "NEUTRAL... NOT a quality lever" | ⚠️ Do NOT claim augmentation improves quality. The `--mirror` flag exists but is a null result; the recurring theme is data *quantity* isn't the bottleneck. |
| the `max-2` parser note | `src/data/stepmania_parser.py:80` comment references this roadmap ("excludes 55% of real Hard charts") | The in-code 55% refers to the **raw** number; the **post-length-filter** number is ~+9.6%. If the README cites 55%, qualify it as raw-pre-filter, or it overstates the recoverable Hard data. |

**Verb-precision watch:** these are **roadmaps** — most items are *not done*. If the README mentions variable
BPM, finer resolution, rolls, mines, or augmentation, they must read as *future/deferred scope*, not
capabilities. The mirror null is a useful honesty data point (data quantity isn't the lever); the hands-filter
55%-vs-10% correction is exactly the kind of number a claims audit should pin to its *filtered* value.
Cross-ref [[06-taste-critic]] (the critic is *trustworthy* at Hard; the generator is tame for these two
independent reasons).

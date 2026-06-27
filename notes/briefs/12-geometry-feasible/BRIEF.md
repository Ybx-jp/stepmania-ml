# Brief 12 — Geometry / feasible region (the EASY-corner release gate + the v2 "map the region" framing)

**Source notes:** `difficulty_corner_findings.md` → `geometry_feasible_region.md`
**Arc role:** two things. (1) The **EASY-corner release gate** — the one boundary-walk actually run for v1
("make sure it can still generate easy songs"). (2) The **feasible-region theory** (product-of-experts /
manifold geometry / a real GDL door), explicitly **parked for v2**. This brief mostly protects the README
from over-claiming "the region of good settings is mapped" — it is not.

---

## The narrative

### Beat 1 — the EASY corner is healthy (`difficulty_corner_findings.md`)

A controlled difficulty sweep (audio fixed, only the difficulty token varied):

> "**The model KNOWS easy = sparse (not 'Hard minus notes').** nat_dens@0.5 rises monotonically
> 0.095→0.222→0.242→0.400 with the difficulty token alone... internalized in the onset posteriors, not faked
> at decode. This is the key result."

> "**Backbone SURVIVES and strengthens toward easy.** Quarter share 1.00/1.00/0.89/0.69... Easy/Beginner are
> 100% on-grid quarter — maximally coherent."

⚠️ Two caveats the README must respect:

> "**Low song diversity:** `collect()` took the first 6 val samples = effectively 2 distinct songs... rerun
> with `--songs` larger... before treating the numbers as population-level."

> "**Offline only** — this answers STRUCTURE... 'Does an easy chart read as *deliberately* musically easy' is
> a FEEL question → needs a playtest."

### Beat 2 — the feasible-region geometry, corrected from the project's own experiments (`geometry_feasible_region.md`)

What's literally true: "**The groove-radar manifold IS an ellipsoid, in the deployed code**"
(`cache/radar_manifold.npz`). But the tempting "three overlapping ellipsoids" picture is corrected:

> "**The pattern/motif space is NOT an ellipsoid**... the motif residual is **not low-rank** (>20 dims for
> 80% variance)... a flat *linear subspace*, not a Gaussian shell."

> "**The audio term is a CONSTRAINT, not a third ellipsoid.** This is H17 (song↔style fit)... the feasible
> 'good chart' set is **song-conditional**."

> "'Good charts = where all the soft constraints are simultaneously satisfied' is a **product of experts**...
> evaluated **per song**... its boundary is fuzzy and **moves with the audio**."

And an honest read on the GDL aspiration:

> "Having a latent manifold ≠ GDL... **BUT there is a real, non-forced GDL door here**: the dance pad has a
> genuine **symmetry group** — the L↔R mirror... The GDL move = make the **generator itself equivariant**...
> distinct from the (mis-attributed) ellipsoid framing."

### Beat 3 — the explicit v1-vs-v2 honesty correction

> "**What v1 actually has (don't overclaim it):** `governor_release_region.md` is a **table of per-knob ranges
> the user PERSONALLY VOUCHED FOR**... it is NOT a 'mapped region.' Calling it 'the region mapped' was an
> overclaim (user-corrected 2026-06-26)."

> "**The real region-mapping (v2):** the region of good settings is the JOINT feasible set across ALL
> conditioning knobs... determined by ACTUAL EXPERIMENTATION + MEASUREMENT... because the knobs are coupled
> and the feasible set is SONG-CONDITIONAL."

---

## Audit hooks (reconcile README against these)

| README claim | Verbatim source | Verb precision |
|---|---|---|
| the model can generate easy charts | "The model KNOWS easy = sparse... nat_dens 0.095→0.40... backbone survives" | **measured** ✅ offline, but on **~2 distinct songs** and **not playtested**. If cited, keep "offline structural check," not "validated." |
| "region of good settings mapped/characterized" | "it is NOT a 'mapped region'... Calling it 'the region mapped' was an overclaim" | ⚠️ **claim-precision (memory [[claim-precision]]):** the README/marketing must NOT say the operating region is "mapped," "characterized," or "measured." v1 has **vouched per-knob ranges** (governor only) + **one boundary-walk** (the easy corner). The joint region map is **v2, explicitly not done.** |
| geometric-DL / equivariance framing | "Having a latent manifold ≠ GDL... a real, non-forced GDL door... equivariant generator" | If the README makes a GDL claim, the honest version: the *radar manifold* is a literal ellipsoid (used), the L↔R mirror is baked in **post-hoc** (not an equivariant architecture). An equivariant generator is a **v2 research direction**, not a shipped property. |
| "good charts = on-manifold" | "Good charts ≈ the in-distribution / on-manifold region is supported... Failures are boundary-crossings" | **measured/vouched** ✅ — supported, and it's the theory the README's chaos/manifold framing rests on. Safe as a *framing*, not a *map*. |

**Verb-precision watch:** this whole brief is a guard against the single most likely README overstatement —
that the project has "mapped the feasible region of good settings." It has **one** boundary-walk (easy,
offline, 2 songs) and **vouched** governor ranges. The map is v2. Cross-ref [[11-governor]] (the vouched
table) and [[09-manifold-guidance]] (the shipped ellipsoid).

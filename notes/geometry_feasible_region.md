# The feasible-region geometry framing (parked for later — the "geometric DL" thread)

**Origin:** 2026-06-25, after the H15 playtest milestone ("the model might be ready"). The user proposed a
geometric picture of the generator and asked whether it's "legal to say" / whether it's their chance to apply
geometric deep learning. This note records the framing — refined to what's actually true in the code — so it can
be picked up as a research thread without re-deriving it. **It is NOT a release blocker** (the release gate is
the empirical boundary-walk; see [[difficulty_corner_findings]]). It is the *theory* the boundary-walk tests.

## The user's intuition
There is a geometric space (an ellipsoid?) formed by the coherent *overlap* of the **groove manifold**, the
**pattern manifold**, and the **audio manifold**, and any mixture of settings *inside* that space yields a very
good chart. (Spoken as a geometric-deep-learning aspiration.)

## What's literally true (not metaphor)
- **The groove-radar manifold IS an ellipsoid, in the deployed code.** `cache/radar_manifold.npz` = mean +
  covariance with a Mahalanobis projection onto the 0.90-quantile shell; conditional-fill is a Gaussian
  conditional. Conditioning radar already = constrained projection onto a literal ellipsoid. (See the
  conditioning-mechanics skill §2.)
- **"Good charts ≈ the in-distribution / on-manifold region" is supported.** Failures are boundary-crossings:
  OOD mean-pin smears (chaos=0.9 others-at-mean), CFG g>3 dissolves the backbone, chaos forced on a 1/4 song.
  Quality degrades smoothly as you approach/cross the boundary.

## Where the picture needs correcting (from our own experiments)
1. **The pattern/motif space is NOT an ellipsoid.** The Phase-1 mining proved the motif residual is **not
   low-rank** (>20 dims for 80% variance) — explicitly "no Gaussian motif manifold to bolt on." We condition
   through **12 radar-orthogonal knobs**, a flat *linear subspace*, not a Gaussian shell. So "overlap of three
   ellipsoids" overstates the middle term: it's **radar ellipsoid × motif subspace**, asymmetric.
2. **The audio term is a CONSTRAINT, not a third ellipsoid.** This is H17 (song↔style fit): same conditioning,
   OH WORLD sang at g3.5 while Deja loin fought it. Audio doesn't give free axes to slide along — it *gates*
   which radar/motif settings are realizable. So the feasible "good chart" set is **song-conditional**.
3. **Precise restatement:** the three structures don't share a space (radar 5-D, motif >20-D, audio its own).
   They're **coupled** (radar linearly predicts ~61% of motif mass; audio constrains realizable radar/density).
   "Good charts = where all the soft constraints are simultaneously satisfied" is a **product of experts**
   (each manifold a soft constraint; the good region is their joint feasible set), evaluated **per song**. The
   user's "ellipsoid where any mixture is good" is real — it's the in-distribution core — but its boundary is
   fuzzy and **moves with the audio**. That last part (song-conditional feasible set) is the genuinely novel,
   defensible claim.

## Is this "geometric deep learning"? (honest answer)
- What's described above is the **manifold hypothesis + a feasible region** = *latent geometry*, NOT GDL-the-field.
  Geometric Deep Learning (Bronstein/Bruna/Cohen/Veličković, the "Erlangen program of ML") is specifically about
  **architectures that respect symmetries** — equivariance (a transform of the input produces the matching
  transform of the output), graphs, groups, gauges. Having a latent manifold ≠ GDL.
- **BUT there is a real, non-forced GDL door here, and we're already halfway through it:** the dance pad has a
  genuine **symmetry group** — the L↔R mirror (already used by hand in `motif_codebook._MIRROR = [3,1,2,0]`, which
  canonicalizes LLLL↔RRRR while keeping UUUU distinct), and arguably U↔D, giving the 4 panels a Klein-four /
  dihedral structure. We currently bake the mirror in **post-hoc** (canonicalize figures). The GDL move = make the
  **generator itself equivariant** to the pad's symmetry group, so mirrored patterns are intrinsically the same
  figure. That is a legitimate application of the field to *this exact* problem — and the user's honest "chance",
  distinct from the (mis-attributed) ellipsoid framing.

## How the theory becomes empirical (the boundary-walk)
A release = "the good region is characterized to its BOUNDARIES", not "we found one great point in it" (OH WORLD
g3.5). We had stress-tested only the HARD corner. First boundary walked = the EASY corner
([[difficulty_corner_findings]]): **healthy** — the model knows easy=sparse (intrinsic density scales
0.095→0.40), backbone survives (Beginner/Easy 100% on-quarter), figures stay varied. Evidence the feasible
region extends cleanly to low difficulty. Remaining boundaries to map if we want the full region: low-density ×
each radar/motif axis, the audio-feasibility frontier (H17, the moving boundary), and the multi-knob interior
(do combined knobs stay coherent — partly seen in the combo_* playtests).

## V2 — MAP THE REGION OF GOOD SETTINGS (the real thing; explicitly NOT done in v1)
**What v1 actually has (don't overclaim it):** `notes/governor_release_region.md` is a **table of per-knob ranges
the user PERSONALLY VOUCHED FOR** (hands-on playtest + a few targeted offline sweeps), for the GOVERNOR knobs only.
That is a vouched-for shipping default + safe envelope — it is NOT a "mapped region." Calling it "the region
mapped" was an overclaim (user-corrected 2026-06-26).
**The real region-mapping (v2):** the region of good settings is the JOINT feasible set across ALL conditioning
knobs — radar (5 dims), continuous motif (12 knobs), discrete figure (7), CFG guidance, density/onset, phase calib,
AND the governor knobs (fatigue, stamina_ceiling, breathe) — not a per-knob 1-D union. It must be determined by
ACTUAL EXPERIMENTATION + MEASUREMENT (sweep the interior, walk the boundaries, measure goodness against real charts
AND by ear), because the knobs are coupled and the feasible set is SONG-CONDITIONAL (the audio gates which
settings are realizable — H17, the moving boundary above). This is the empirical instantiation of the
product-of-experts / feasible-region theory in this note. Scope: the boundary-walks listed above + the multi-knob
interior + the governor axes, run as a real sweep, not vouched-for ranges. Pairs with the GDL/equivariance v2 work
(a symmetry-respecting generator would reshape the region). Lives here with the rest of v2.

### Proposed starting method (v2 entry point — sweep × healthy song sample × best-of-N, critic-scored)
The way in: a **comprehensive sweep of the conditioning settings, run against a healthy sample of songs**, where each
(setting, song) cell is scored by **best-of-N** — generate N candidates, keep the best — and "best" is judged by
the **difficulty critic** (the Phase-1 LateFusionClassifier difficulty head: is the chart in the requested class)
**and** the **taste critic** (the v2 realism critic's P(real), `checkpoints/realism_critic/best_val.pt`,
`eval_taste.py`; AUC 0.964, ranks REAL 0.823 > BASE 0.290 > CHAOS 0.003). A cell is "good" only if a candidate
clears both gates: on-difficulty AND tasteful. This gives an automated goodness surface over the knob grid that we
then spot-check by ear, instead of vouching for ranges one knob at a time.

**PREREQUISITE — re-evaluate the taste critic against the LATEST decode machinery first. → DONE 2026-06-26,
[[taste_critic_transfer_findings]].** Result: the critic's ranking TRANSFERS to the current decoder (REAL 0.823 >
BASE 0.269 > CHAOS 0.228 at n=64; REAL>BASE 86% per-song; the old-machinery control reproduced 0.823/0.290/0.003
exactly, so the harness is faithful). The density-OOD worry is largely cleared (`corr(P(real),density)=-0.09` over
0.05–0.44). **BUT** the binding caveat for best-of-N is NOT density — it's that the critic is **near-binary** (only
14–30% of scores in the discriminating middle; generated candidates cluster on the low rail), a strong SEPARATOR
but weak GRADER. So best-of-N reranking likely needs the critic's low end recalibrated/temperature-rescaled before
it can rank *among generated candidates*, and two gates still remain: (a) confirm by ear that high-P(real)
generations actually play better than low-P(real) ones for the same song/setting (experiment-design Rule 8), and
(b) understand WHAT the critic measures before trusting it as the inner judge — the activation + saliency probe in
[[taste_critic_interpretability_plan]] (does it key on a coherent musical property like off-grid penalty/alignment,
or a density/fingerprint artifact?). The route is viable, conditional on the recalibration + those two gates.
Supporting evidence the critic tracks a real quality axis: the chaos-conditioning isolation
([[taste_critic_transfer_findings]]) — holding the model fixed, manifold chaos scored 0.228 vs mean-pin 0.028, so
the critic registered the conditioning redesign as more tasteful, matching play experience. (Original instruction kept below for the record.)

  *Original instruction:* before it is allowed to be the goodness signal for a sweep, re-run on charts decoded by
  the current machinery across the difficulty/density/knob range we intend to sweep, and confirm P(real) still
  ranks by play-feel there (not just on the old 0.2-density split). Done via
  `experiments/realism_critic/eval_taste_current.py` (mirrors `scripts/generate.py`'s decode path).

Design points so the harness measures the model, not itself (experiment-design skill):
- **"Healthy sample of songs" is load-bearing, not decoration.** The feasible set is **song-conditional** (audio
  gates which settings are realizable — H17, the moving boundary). A single song maps *its* slice, not the region.
  Sample across the audio axes that move the boundary (tempo, onset density, genre/energy) so the swept region is
  the *intersection* (settings good across songs) — and also report the *spread* (how much the boundary moves), which
  is itself the H17 result quantified.
- **Don't import the old "no headroom at Hard" finding as a constraint — re-test it.** [[stage2a_critic_findings]]
  §2b observed best-of-N had little headroom at Hard because *all candidates came out tame*. That was almost certainly
  an artifact of the **poorly-tuned conditioning of that era's generator**, not a property of best-of-N — the decode
  machinery has moved on a lot since (governor, motif arc, the highres retrain). Treat it as a hypothesis to re-check
  under current conditioning, not a known limit. If candidate diversity is still flat at Hard with the latest stack,
  *that* is a finding; assuming it up front would bias the sweep.
- **The critic's density range is a known unknown, not a settled caveat.** The 2026-06-20 critic was trained only on
  ~0.2-density charts and scored OOD-high-density chaos spuriously. Whether the *current* critic-vs-current-decoder
  pairing has that blind spot is exactly what the prerequisite re-evaluation answers — fold the density-range check
  into it rather than pre-restricting the sweep on a stale assumption.
- **Pick N and the selection metric before running** (don't tune them on the same sweep that reports the region), and
  **keep test-by-ear as a separate, once-only confirmation** of a handful of swept cells, not an inner loop.

This is the *automation* of the boundary-walk + multi-knob-interior scope above: the critics are the cheap inner
goodness proxy that lets the sweep cover the joint grid; the by-ear pass and the difficulty-corner walks remain the
ground truth that keeps the proxy honest — but the proxy itself has to clear the re-evaluation gate before it earns
that role.

## Threads this connects to
- [[two-generator-tracks]] (the manifold conditioning track), conditioning-mechanics skill (the exact math),
  experiment-design skill (don't mistake a rigged harness for a model defect when probing the boundary).
- **GDL/equivariance = a v2 research direction**, not a release gate. Park here; revisit if we want a principled
  redesign or a paper angle.
- **Region-of-good-settings empirical map = v2** (above): the joint, song-conditional feasible set across all
  conditioning + governor knobs, by real experiment. v1's governor range table is the seed, not the map.

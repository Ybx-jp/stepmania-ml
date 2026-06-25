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

## Threads this connects to
- [[two-generator-tracks]] (the manifold conditioning track), conditioning-mechanics skill (the exact math),
  experiment-design skill (don't mistake a rigged harness for a model defect when probing the boundary).
- **GDL/equivariance = a v2 research direction**, not a release gate. Park here; revisit if we want a principled
  redesign or a paper angle.

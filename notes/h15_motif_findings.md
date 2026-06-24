# H15 Phase 0 — motif de-risk gate: do NOTE-PATTERN motifs carry vibe, distinct from existing conditioning?

*2026-06-23. Probe: `experiments/generation_typed/diag_motif_gate.py` (no training). N=4606 real charts, 1399
files.* **Motif = a short window of the WHICH-PANELS note sequence over onsets** (`typed.panels_to_pattern`,
empties dropped) — a recurring spatial FIGURE, per `h15_motif_handoff.md`. *(A first pass drifted onto
frame/rhythm windows + a density-only floor and "rediscovered" the already-shipped freeze knob; that was an
experiment-design failure — wrong control + hypothesis drift — now logged in the `experiment-design` skill,
post-mortem §10 / Rules 15–16. This note is the faithful redo.)*

## Two questions
1. **Signal exists? (handoff gate, PASS/STOP)** Do note-pattern motif histograms separate groove buckets vs a
   SHUFFLED control?
2. **A DISTINCT lever?** Is the motif VOCABULARY something the radar we ALREADY condition on (density, jumps,
   holds, chaos-amount) can't already pin? Control = the FULL radar, NOT a strawman subset.

## PART 1 — SIGNAL EXISTS ✅ (note-pattern motifs separate groove, far above shuffled)
Equal-count rank tertiles, macroF1 (chance 0.333), motif-alone vs shuffled-label control:

| axis | W=3 | W=4 | W=6 | shuffled |
|------|-----|-----|-----|----------|
| chaos  | 0.639 | 0.594 | 0.520 | ~0.33 |
| stream | 0.713 | 0.661 | 0.549 | ~0.34 |
| freeze | 0.501 | 0.503 | 0.462 | ~0.34 |

Every axis/window beats shuffled by +0.12 to +0.37. Strongest at **short windows (W=3)**; decays by W=6 as
exact long windows go sparse (→ confirms the handoff's call for softer-than-exact matching). **Note-pattern
motif usage carries groove. Premise holds.**

## PART 2 — DISTINCT FROM EXISTING CONDITIONING (controlled against the FULL radar)
**(a) radar+difficulty → motif histogram: variance-weighted R² = 0.607.** The conditioning we already have
linearly pins ~61% of note-pattern motif usage (the common, quantity-driven mass — staircases scale with
stream/density, etc.). **~39% is residual the radar cannot express.**

**(b) k-NN style test (centered-cosine motif similarity, random ≈ 0):**
- random pairs (floor): **+0.006**
- radar-nearest-neighbour, different song (what conditioning to that profile targets): **+0.475**
- same-song **same-difficulty** charter-twins (would-be ceiling): **+0.245**

The striking part: **radar-twins are MORE motif-similar (+0.475) than same-song/same-difficulty charter-twins
(+0.245).** Two human charters of the *same song at the same difficulty* agree on motifs LESS than two charts
with the same groove profile. So motif choice has large charter spread at fixed groove → **motif vocabulary is
a DISTRIBUTION, not a deterministic point** (echoes the Phase-3 divergence gate: 16th-IoU median 0.325 — humans
agree on ~1/3 of placements).

## PART 3 — the crux, grounded in the artifact: signature figures the radar CAN'T see
Charts most concentrated on a single recurring motif — all at **near-identical (near-zero) radar** yet with
**completely different signature figures**:

| chart | signature motif | share | radar |
|-------|------|------|------|
| Party Rock Anthem | `[--U --U --U --U]` (Up jack) | 35% | all ≈ 0 |
| Don't Stop the Party | `[L-- L-- L-- L--]` (Left jack) | 33% | all ≈ 0 |
| DO ME | `[L-- --R L-- --R]` (L/R alternation) | 24% | all ≈ 0 |

**The radar provably cannot distinguish UUUU from LLLL from LRLR** — same radar point, different characteristic
figure. This is exactly the residual H15 targets.

## VERDICT — PASS, and the lever is the SIGNATURE MOTIF VOCABULARY (not freeze, not the common filler)
Reconciling the parts: the radar already captures the **common, quantity-driven** motif mass (R² 0.61;
groove-twins are motif-similar), but the **distinctive signature figures that make a style recognizable are
radar-invisible** (PART 3: same radar, different figure) **and high-variance across charters** (PART 2b). That
matches H16 exactly — *guidance/radar can set quantities and rhythm balance but "cannot ADD vocabulary."* So
the H15 lever is real and DISTINCT: give the pattern head a **characteristic motif vocabulary to SAMPLE from,
conditioned on style** — the recurring figure, not the note count, and not holds (already a radar knob).

## Steer for Phase 1 (representation)
- **Motif codebook over short note-pattern windows (W≈3–4; symmetry-collapse gives a small consistent lift).**
  Condition on a target motif **DISTRIBUTION**, not a point (a point target would mush — Phase-3 lesson);
  compose with the manifold (per-style motif stats, parallel to how density was added).
- **Target the SIGNATURE vocabulary, not the common filler** (radar already pins 61%) and **not freeze/holds**
  (already conditionable). Don't re-condition what the radar reaches.
- **Pair with the taste critic** (REAL>BASE>CHAOS) for selection — per-frame CE can't reward characteristic
  figures. Build the softer run/coherence metric the handoff flagged (exact W=6 already decays here).

## Caveats (Rule 9 — conditional)
- The residual vocabulary is partly distributional spread / charter idiosyncrasy, not a deterministic missing
  signal → condition & evaluate it DISTRIBUTIONALLY (best-of-N + critic), don't pin a point.
- Exact-window vocab is brittle past W=4; the Phase-1 rep should use a soft motif notion (cluster/codebook),
  which the gate's W=6 decay already motivates.

---

# Phase 1 mining gate — the codebook (2026-06-23, `diag_motif_codebook.py`)

Representation (user-chosen): **absolute which-panels token over onsets + L↔R-mirror fold** (panel map
`[3,1,2,0]`, the standard StepMania mirror — merges LLLL↔RRRR, keeps UUUU distinct so PART-3 signatures
survive; fixes the Phase-0 gate's `[3,2,1,0]` 180° bug). **Multi-scale** windows W∈{2,3,4,6}. Per-scale top-120
canonical codebook (418 motifs total); per-chart multi-scale distribution.

**Four gates (commit to training only if all pass):**
1. **INTERPRETABLE ✅** — frequent canonical motifs read as real figures: W=2 candles/crosses + jacks; W=3/4
   staircases/sweeps (`L-D-U-R`), trills (`L-R-L-R`), candles; W=6 long trills + extended sweeps.
2. **LOW-RANK ❌ (the important negative)** — PCA scree is slow: PC1 7.8%, PC2 6.0%, then a flat ~2% tail;
   **~12 PCs for 80%**. With the radar regressed OUT (GATE 3b), the residual is even flatter (5.0/3.2/2.7/…,
   **>20 dims for 80%**). **There is NO low-rank motif manifold** — motif-style is a broad vocabulary of many
   small independent style directions. → **Kills the original "carry 2–3 motif-style dims on the RadarManifold
   ellipsoid" plan; the Gaussian-manifold machinery can't faithfully model a 20+ dim flat residual.**
3. **RESIDUAL / DISTINCT FROM RADAR ✅** — PC1 is radar-REDUNDANT (maxcorr 0.84 with air/stream = the common
   filler the radar already pins). PC3–6 are all radar-ORTHOGONAL (maxcorr ≤0.17) AND interpretable: a
   sweep/staircase axis, a jump-figure axis, a candle/cross axis, a bracket/jack axis. These are the genuine
   H15 lever — the vocabulary the radar can't reach.
4. **STABLE ✅** — top-6 PCs reproduce across a random split-half at |cos| 0.91–0.98 (not overfit).

**Phase-1 design implication (revised by the gate):** condition on the **radar-ORTHOGONALIZED motif
distribution** (drop the radar-redundant PC1 so we don't re-condition what the radar already pins — the
exact rediscovery trap from the first Phase-0 pass), via a **distributional/embedding** mechanism — NOT a
low-rank manifold extension. Mechanism choice (named residual-PC knobs vs learned motif-encoder vs raw
distribution) is the open Phase-1 fork. Conditioning width ≈ the residual rank (top stable radar-orthogonal
directions, ~8–12 dims capture the reliably-reproducing structure even though full 80% needs >20).

## MotifBasis ARTIFACT BUILT (user chose: named residual-PC knobs)
`src/generation/motif_codebook.py::MotifBasis` (the motif analog of `cache/radar_manifold.npz`), fit + saved
to **`cache/motif_basis.npz`** by `experiments/generation_typed/fit_motif_basis.py`. Pipeline: per-scale
L↔R-folded codebook (418 motifs) → Ridge radar→motif-hist → PCA on the **residual** → keep top-12 axes that
are split-half STABLE (|cos|>0.7) and radar-orthogonal, each with a contrast label. `encode_chart(tensor,
radar)` → 12-d z-scored motif knobs (training target, autoencoder-style; at inference the user SETS a target).
**The 12 knobs are interpretable figure contrasts, all maxcorr_radar 0.00, stab 0.86–0.99:** jack↔sweep,
jump/bracket, step↔candle/cross, trill↔jump/bracket, jack↔trill, … (the auto-labeler is crude on some — the
stored pos/neg motif lists carry the real meaning).

## PHASE 2 — model wired + trained + steerability tested (2026-06-23)
Wired `motif_proj`+`null_motif` into `LayeredTypedChartGenerator._cond` (zero-init proj → warm-start is an
exact no-op at step 0; CFG-amplifiable like radar/style). `train_motif.py` warm-started from **gen_style**
(Track B, 23-dim — the OH WORLD lineage; see [[two-generator-tracks]]), self-conditioned each chart on its own
motif vector, CFG-dropout radar/style/motif. 12 epochs → `checkpoints/gen_motif/best_val.pt`, **val_total
1.213 (≈ warm-start; quality preserved)**.

**Steerability (eval_motif.py single-knob + eval_motif_transfer.py on-manifold exemplar, 20 songs):**
- **A real but PARTIAL lever.** **step↔candle (knob 3) steers well** (on-manifold exemplar transfer +0.27,
  own-axis Δ **+1.60** at g=3); **jack↔trill (knob 10) weakly** (Δ+0.51); **jack↔sweep (knob 0) not at all**
  (Δ+0.15, transfer +0.04). Quality intact throughout (onset_F1 0.73–0.77, density matched).
- **Two confounds tested (Rule 7):** (a) the H13 `max_jack_run=1` cap — REFUTED as the cause (disabling it
  didn't rescue knob 0); (b) single-knob ±3z is OFF-MANIFOLD (cross-talk 1.82 ≫ self 0.19 at g=3) — SUPPORTED;
  the on-manifold exemplar test helped (transfer rose, guidance amplified) but did NOT rescue the jack axes.
- **ROOT CAUSE (training signal, visible):** the motif vector reduced TRAIN pattern-loss only 1.054→1.038
  (~1.5%). It's a GLOBAL chart descriptor added to PER-FRAME conditioning → a global figure-mix barely helps
  predict which-panel-at-frame-t → weak per-frame gradient → weak learned control. (Same global-bottleneck
  that made style-transfer work for density but not finer structure.) Jacks fail worst because the AR pattern
  head was tuned to AVOID jacks (the old always-Left fix) so overcoming it needs signal the global vec lacks.

**Where this leaves H15:** the conditioning surface is sound (interpretable, radar-orthogonal, quality-safe)
and one characteristic-figure axis (candle/cross) genuinely steers with guidance. The PARTIAL result is a
GLOBAL-DESCRIPTOR limit, not a dead end. Two forward paths: (1) cheap — PLAYTEST the working candle knob (does
the measurable steer change the FEEL? — the validated instrument; offline metrics are blind to vibe); (2)
bigger — give the motif lever a per-frame/LOCAL signal (local motif targets, or a hierarchical
"pick-motif-then-realize" decode) so control isn't bottlenecked by a single global vector. Pair selection with
the taste critic (per-frame CE can't reward characteristic figures).

## PHASE 2b — base-swap test: motif on TRACK A (42-dim high-res base) (2026-06-24)
**Question (after reconciling the Track A/B divergence):** the two generators are the SAME class
(`LayeredTypedChartGenerator`); they differ ONLY in `audio_dim` (Track A 42-dim = 23 base +12 chroma +2 HPSS +4
metric-phase +1 high-res onset; Track B 23-dim). Conditioning experiments landed on B by PATH DEPENDENCE (the
gen_radar→gen_style→gen_motif warm-start chain was built on 23-dim), NOT architecture — radar already runs on A.
So: does a richer LOCAL-audio base (A carries high-res onset + metric phase) lift the GLOBAL motif descriptor's
realization, or is the Phase-2 partial result base-invariant (→ purely the global-descriptor mechanism)?
- `train_motif_hr.py` = train_motif.py with ONE change: warm-start **gen_highres_v4** (42-dim), cache/samples_v3.
  **STYLE OFF (radar+motif only)** — Track A's lineage never fed `reference=` so its style encoder is UNTRAINED
  (turning it on = a 2nd changed variable); and eval generates with style=null anyway, so off is both clean AND
  how the lever is used. 12 ep → `checkpoints/gen_motif_hr/best_val.pt`, **val_total 1.216** (< base 1.265; ≈
  Track B's 1.213 — quality preserved). `eval_motif.py` got `--ckpt`/`--highres` flags to drive both tracks.
- **RESULT (own-axis Δ at g=3, single-knob ±3z, 20 songs) — Track B → Track A:**
  - knob 3 step↔candle: **+1.60 → +2.57** (richer base AMPLIFIES the already-working axis; even g=1 Δ+1.18)
  - knob 10 jack↔trill: +0.51 → **+0.23** (NOT rescued; weaker)
  - knob 0 jack↔sweep: +0.15 → **−0.16** (NOT rescued; still dead)
  - Quality intact (onset_F1 0.68–0.73, density matched ~0.19).
- **CONCLUSION — base-INVARIANT for the failing axes ⇒ Phase-2 root cause CONFIRMED.** If richer per-frame
  audio could fix jacks it would have here; it didn't. It only sharpened the axis the per-frame head can already
  render (candle = a spatial figure). So the bottleneck is the GLOBAL-DESCRIPTOR mechanism, not the audio input.
  We've now ruled out BOTH the playability cap (Phase 2) AND the audio base (here) as the cause.
- **DECISION:** per-frame/LOCAL motif conditioning is now the EVIDENCED next step (the only lever that can move
  jacks), and **Track A is its home** (candle steers harder, 16th-note capability already present, quality safe).

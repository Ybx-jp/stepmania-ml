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
stored pos/neg motif lists carry the real meaning). **NEXT (Phase 2 = model):** add `motif_proj`+`null_motif`
to `LayeredTypedChartGenerator._cond` (alongside difficulty+radar+style) with CFG dropout; `train_motif.py`
warm-start from `gen_style`, self-condition each chart on its own motif-knob vector; then steerability eval +
playtest (does moving a motif axis change the FEEL?). Pair selection with the taste critic (per-frame CE can't
reward characteristic figures).

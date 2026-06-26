# Brief 09 — Manifold conditioning + guidance tuning (the SHIPPED `--style` path)

**Source notes:** `radar_manifold_findings.md` → `h14_guidance_sweep_findings.md` → `h16_harmonic_findings.md`
**Arc role:** explains the README's deployed conditioning line — **"`--style` is the in-distribution
(manifold) conditioning path; `--radar` is disabled (off-manifold)."** This is where raw radar steering
was found to go OOD and was replaced by a manifold-aware surface that ships as `cache/radar_manifold.npz`.
It also bounds guidance: CFG is a *trade*, not a *vibe unlock* — which hands the baton to the motif arc
([[10-motif-arc]]).

---

## The narrative

### Beat 1 — the radar manifold: 5 dims are really ~rank-2 (`radar_manifold_findings.md`)

> "The 5 radar dims are NOT independent knobs — they're ~rank-2: **Intensity cluster** stream/volt/air/chaos,
> pairwise r **0.71–0.92**... **freeze (holds) is orthogonal**, r ~**0.3**... chaos↔stream = **0.80** → 'high
> stream, low chaos' is the natural CONTRADICTION (the chaos OOD bug)."

The fix is a decode-time, no-retrain manifold surface: the user sets a loose spec over named axes;
**conditional-fill** completes the rest via the real Gaussian conditional, and **ellipsoid projection**
snaps extreme corners back inside the realizable envelope:

> "**Conditional-fill reveals hidden couplings the user doesn't have to know:** Power jumps auto-fills
> freeze 0.46... The user steers 2-3 axes; the rest come out real."

> "**Mahalanobis d ranks realizability cleanly**... A natural UI 'how unusual is this combo' meter."

And the **source-free density** piece — what makes generation work on a brand-new song with no reference
chart (this is what ships):

> "Density must come from **difficulty + style**, not a source... **Verified end-to-end**... density is
> driven by difficulty+style, source chart unused."

This manifold is **SHIPPED** as `cache/radar_manifold.npz` (per the INDEX: "Manifold now SHIPPED as
`cache/radar_manifold.npz` for dataset-free generation"). It is the mechanism behind `scripts/generate.py
--style`.

### Beat 2 — guidance is a CLIFF into the smear (`h14_guidance_sweep_findings.md`)

> "**The decision boundary is REAL and it's HIGH (between g=3 and g=5).**... the conditioning is roughly
> inert until a high guidance, then bites hard."

> "**But crossing it OVERSHOOTS into the H4 chaos smear (for chaos).** glitch_tech's off-beat-16th rate goes
> 0.00 → 0.07 (g=3) → **0.62 (g=5)**... **Guidance does NOT unlock glitch tech's vibe; it floods off-beats.**"

Axis-specific: "**freeze tolerates more guidance; chaos overshoots into smear well before it becomes
tasteful.**"

### Beat 3 — guidance is monotonic, not "harmonic"; the sweet spot is a KNEE (`h16_harmonic_findings.md`)

A playtest hypothesis (guidance has discrete coherent "nodes") tested against sampling noise and **refuted**:

> "**H16 (harmonic) NOT SUPPORTED.** No reproducible non-monotonic feature exceeds the seed std... **The real
> mechanism: guidance dissolves the QUARTER BACKBONE into 16ths**... **The sweet spot is a KNEE, not a node:**
> 'where the quarter backbone is still ~20–25%'. Song-dependent."

> "Guidance only TRADES backbone↔16ths along a monotonic curve — it cannot ADD the motif vocabulary that
> makes a style's *character*. So the vibe lever is NOT more/cleverer guidance; it's **H15 (motifs)**."

---

## Audit hooks (reconcile README against these)

| README claim | Verbatim source | Verb precision |
|---|---|---|
| "`--style` is the in-distribution (manifold) path; `--radar` disabled (off-manifold)" | "Radar-point conditioning is WHY chaos went OOD"; manifold "conditional-fill + project"; shipped `radar_manifold.npz` | **measured/shipped** ✅ — accurate. The manifold surface is real and shipped. Keep the framing that raw radar is disabled *because* point-conditioning goes off-manifold. |
| chaos is "in-distribution-bounded; musical on-manifold, degrades only past the boundary" (corrected line) | conditional-fill keeps requests inside the envelope; guidance past the knee "OVERSHOOTS into the H4 smear" | **AUDIT-CRITICAL** ✅ — this brief is the *source* of the v1 audit's corrected chaos framing (row 19). Accurate: musical while on-manifold, smears past the boundary. |
| source-free / dataset-free generation | "density is driven by difficulty+style, source chart unused"; ships `radar_manifold.npz` (256 KB) | **measured/shipped** ✅ — this is what makes `scripts/generate.py` work without the training dataset (release reproducibility item). |
| CFG / guidance as a controllability strength | guidance is a *trade* (backbone↔16ths), monotonic, "cannot ADD motif vocabulary"; sweet spot is a song-dependent knee | **Don't oversell guidance.** It's a per-song knee to tune, not a vibe dial. The "harmonic nodes" idea is **refuted**. |

**Verb-precision watch:** the manifold's realizability claims are **measured** on 1086 Hard charts
(Mahalanobis envelope), and the surface is shipped — safe. But "the manifold makes the ASK coherent, not
automatically the OUTPUT" (the note's own caveat): conditioning on an on-manifold target doesn't guarantee
the *rendered* chart nails the style — that rendering-fidelity test is the motif arc's job ([[10-motif-arc]]).
The taste critic is **FLOORED (~0) for forced off-song styles** (H14) — don't cite critic P(real) as a
style-quality measure here; it reads any off-song style as low-realism by design.

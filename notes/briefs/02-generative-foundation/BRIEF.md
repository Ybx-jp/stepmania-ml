# Brief 02 — Phase 2: the generative foundation (baseline → AR → factorized → calibrated decode)

**Source notes (read in this order):** `generation_baselines_findings.md` → `generation_transformer_findings.md`
→ `density_calibration.md` → `factorized_head_findings.md` → `focal_onset_findings.md` →
`per_difficulty_threshold.md` → `onset_calibration.md` → `hybrid_decode.md` → `kv_cache.md` →
`stage3_roadmap.md`
**Arc role:** the spine of the whole generator. This is where almost every numeric claim in the
README is set. The story is a clean four-beat: *establish the floor → an AR transformer clears it but
over-places → factorize onset-from-panel to fix density → settle the decode/calibration frontier.*

---

## The narrative

### Beat 1 — the floor: onset alignment is the hard axis (`generation_baselines_findings.md`)

Two deliberately dumb baselines set the bar. The lesson is that **difficulty conditioning is cheap;
onset *timing* is the real problem.**

> "**Density / difficulty conditioning is EASY.** The audio-blind n-gram reproduces real density
> exactly (0.211)... 50.8% exact, **97.7% adjacent**... So a high difficulty-fidelity number is cheap."

> "**Rhythmic onset alignment is the HARD, unsolved axis.**... the per-frame MLP, is terrible at
> onset_F1 (**0.053**)... **onset_F1 ≈ 0.05 is the number the Stage 2 transformer must beat.**"

This is the origin of the README's **"floor 0.053"** number and the framing that the project's real
contribution is *placing steps in time with the music*.

### Beat 2 — AR transformer clears the floor, but decoding is everything (`generation_transformer_findings.md`)

A 1.12M-param decoder reusing the Phase-1 audio encoder (warm-started, frozen 3 epochs). The headline
isn't the model, it's that **greedy decoding hides the whole result**:

> "Greedy argmax **collapses to all-empty** (onset_F1=0)... Sampling unlocks it."

> "**Stage 2 clears the floor on the hard axis.** onset_F1 0.053 → **0.30** (5.7×)... Greedy decoding
> hid this entirely; report sampled metrics."

But sampling over-places — the tradeoff that motivates everything next:

> "High temp (1.0) maximizes onset recall/F1 but over-places (density 0.536 >> 0.196)... Low temp +
> top_k (0.7/2) gives the best difficulty conditioning (crit_adj 0.984 ≈ n-gram)... at some onset recall."

This is the README's **"AR onset F1 0.300, 5.7× the floor"** and **"greedy collapses to empty."**

### Beat 3 — diagnose, then factorize (`density_calibration.md` → `factorized_head_findings.md`)

`density_calibration.md` is the most important *diagnostic* note in the project. A no-retrain probe
separated three confounded failure modes and found the real culprit is **not** localization:

> "**Localization is NOT the bottleneck.** onset ROC-AUC 0.813... the model genuinely knows where notes belong."

> "**The real failure is autoregressive drift (exposure bias), not just calibration.** Free-running @ τ
> **collapses to all-empty**... The decode is bistable: collapse (deterministic) or flood (stochastic),
> with no stable middle."

The prescribed fix — make onset **audio-driven and non-causal** so it can't drift — is exactly what the
factorized head implements, and it works:

> "**Both diagnosed problems are solved.** Onset alignment: onset_F1 **0.30 → 0.763** (2.5× over Stage
> 2, 14× the floor) at *correct* density. No collapse... Precision jumped 0.21 → 0.76."

> "Onset **ROC-AUC 0.950, PR-AUC 0.825**... **Recall 0.80 + precision 0.76 at real density** is the
> first genuinely usable generation operating point in the project."

This is the README's **big jump (0.30 → 0.76)** and the "audio-driven, non-causal onset predictor"
capability claim. **Architecture fact worth keeping straight:** the onset head does NOT read generated
tokens (that's *why* it doesn't collapse); only the *panel* head is autoregressive.

### Beat 4 — settle the objective and the decode frontier (focal → per-diff threshold → calibration → hybrid)

A cluster of no-retrain (and one retrain) decode studies. The single best operating point comes from
**focal loss on the onset head**:

> "**focal + per-difficulty threshold gives both** the high onset_F1 of thresholding (0.748 ≈ BCE's
> 0.756) **and** difficulty fidelity (crit_adj 0.927)... This is the best operating point in the project."
> — `focal_onset_findings.md`

The supporting decode levers:
- **Per-difficulty thresholds** are a free win: "Same onset_F1 (0.756)... but better difficulty
  fidelity: crit_exact 0.281 → **0.406**" (`per_difficulty_threshold.md`).
- **Calibration** (per-difficulty Platt) fixed the onset head's pos_weight over-confidence:
  "[[ECE]]... collapsed ~0.17 → ~0.01" and "Fitted scale a≈1.0 with bias c≈−1.9 confirms the
  over-confidence was a **pure bias offset from pos_weight**" (`onset_calibration.md`). This is the
  README's **"ECE ~0.17 → ~0.01"** claim — note the per-class numbers: Hard raw 0.173 → cal 0.013.
- The frontier is real, not a free lunch: "**The hybrid is a clean, tunable dial along the frontier —
  not a free lunch.** onset_F1 declines monotonically... m ≈ 0.20 is the best-balance" (`hybrid_decode.md`).

### Beat 5 — infra to make it full-length (`kv_cache.md`)

> "**Bit-identical to non-cached** `generate()`... 0/600 timesteps differ... **Speedup scales with
> length** (O(T²) → O(T)): T=1440: 33.4s → 3.6s (**9.2×**), **batch of 4**."

This is the README's **KV-cache claim**. ⚠️ **Audit-critical nuance:** the **3.6s figure is a batch of
4**, not a single song. The v1 audit flagged that the README omitted this. Confirm the README states it.

---

## Audit hooks (reconcile README against these)

| README claim | Verbatim source | Verb precision |
|---|---|---|
| Floor onset F1 **0.053** | "onset_F1 (**0.053**)" — per-frame audio MLP, `generation_baselines_findings.md` | **measured** ✅ |
| AR (sampled) onset F1 **0.300**, "**5.7×** the floor" | "onset_F1 0.053 → **0.30** (5.7×)" | **measured**; 5.7× is derived (0.300/0.053) |
| "**greedy collapses to empty**" | "Greedy argmax **collapses to all-empty** (onset_F1=0)" | **measured** ✅ |
| Factorized head onset F1 **0.763** (and **0.748** focal) | "onset_F1... **0.763**"; "focal + per-diff threshold... **0.748**" | **measured** ✅. NOTE: two numbers exist — 0.763 (BCE factorized) and 0.748 (focal). README "crit-adj 0.927" pairs with the **0.748 focal** row, not 0.763. Don't mix the rows. |
| "audio-driven, **non-causal** onset predictor, density immune to AR drift" | "audio-driven... does **not** read generated step tokens, so it can't collapse" | **measured** ✅ (the probe proved drift was the failure) |
| Per-difficulty **Platt/calibration: ECE ~0.17 → ~0.01** | "ECE... collapsed ~0.17 → ~0.01"; Hard 0.173 → 0.013 | **measured** ✅ |
| KV-cache **bit-identical**, **33.4s → 3.6s (9.2×)** at 1440 | "0/600 timesteps differ"; "33.4s → 3.6s (**9.2×**), **batch of 4**" | **measured** ✅ — but the **batch-of-4** qualifier MUST travel with the 3.6s, or the speed reads as per-song. |
| "full 1440-frame (~2 min)" generation | "Full 2-minute (1440-frame) chart generation is now practical" | **measured** ✅ |

**Verb-precision watch:** "crit_adj" (difficulty-critic adjacent agreement) is a **self-critic** metric
(the Phase-1 classifier reading the generator's own output), not a human/ground-truth measure. README
should not phrase critic numbers as "difficulty accuracy" without that framing. The README headline
generation number should be the **focal + per-diff-threshold** operating point (onset_F1 0.748, crit_adj
0.927) — the note calls it "the best operating point in the project."

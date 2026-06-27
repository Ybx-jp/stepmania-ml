# Brief 10 — The motif arc (H15→H19): the *which-figures* control axis (PARTIAL, candle ships strong)

**Source notes (read in order):** `h15_motif_handoff.md` → `h15_motif_findings.md` → `h15_local_motif_plan.md`
→ `h15_hierarchical_findings.md` → `h15_set_characterization.md` → `repr_integrity_findings.md` →
`h19_retrain_findings.md` → `note_patterns_and_motifs.md` (the consolidated home)
**Arc role:** the most recent *generation* work and the source of the **deployed model `gen_motif_full_fixed`**
(42-dim high-res, the H19 clean retrain). It adds a *which-figures* control axis (candle/trill/sweep) on top
of the groove-radar *how-much* axis. **The headline honesty point: this surface is PARTIAL** — candle steers
strongly (playtest-confirmed), trill modestly (with a fixed measurement confound), **jack↔sweep is the lone
dead axis**, and the knobs are **not exposed in `scripts/generate.py`.**

---

## The narrative

### Beat 1 — the bet, and the Phase-0 gate (`h15_motif_handoff.md`, `h15_motif_findings.md`)

Born from the project's "FIRST genuinely-good chart by ear" (OH WORLD glitch, g≈3.5), and the H16 finding
that guidance "cannot ADD vocabulary." The gate proved note-pattern motifs carry style *distinct from the radar*:

> "**SIGNAL EXISTS ✅** ... Every axis/window beats shuffled by +0.12 to +0.37."

> "radar+difficulty → motif histogram: variance-weighted R² = 0.607 ... **~39% is residual the radar cannot
> express.**"

> "**The radar provably cannot distinguish UUUU from LLLL from LRLR** — same radar point, different
> characteristic figure."

And a key negative that shaped the design — **there is no low-rank motif manifold**:

> "**LOW-RANK ❌ (the important negative)** ... **There is NO low-rank motif manifold** — motif-style is a
> broad vocabulary of many small independent style directions."

### Beat 2 — the global descriptor is too weak; localize it (`h15_motif_findings.md` Phase 2/2b)

> "**A real but PARTIAL lever.** step↔candle (knob 3) steers well ... **jack↔sweep (knob 0) not at all** ...
> **ROOT CAUSE (training signal):** the motif vector ... is a GLOBAL chart descriptor added to PER-FRAME
> conditioning → ... weak per-frame gradient → weak learned control."

> "**CONCLUSION — base-INVARIANT for the failing axes** ⇒ Phase-2 root cause CONFIRMED" (richer 42-dim audio
> sharpened candle but did **not** rescue jacks).

### Beat 3 — per-section (local) motif conditioning: the payoff (`h15_local_motif_plan.md`)

Gated before training (variance + leakage), chose a **64-frame section window**, and it delivered:

> "**HEADLINE: jack↔trill revived** ... a jack-family axis finally moves. Candle stronger. From TWO dead axes
> to ONE."

> "**CANDLE = genuine per-section control:** local_Δ ≥ global_Δ ... track_r +0.52→+0.70 ... quality intact.
> THE payoff of the whole local-motif effort."

> "**jack↔sweep is the one unmoved axis** (long-range staircase = multi-frame sequential coherence a section
> vector can't pin)."

Carry-forward model: `gen_motif_local2` (motif decoupled from the onset head → density-safe at high guidance).

### Beat 4 — discrete figure token: first sweep movement, but a soft-realize ceiling (`h15_hierarchical_findings.md`)

> "**WIN (qualified): the discrete sweep token gives the FIRST positive sweep movement in the H15 arc** —
> 0.05→0.13 ... **CONCLUSION:** a per-section token biases sweep FREQUENCY a little ... but cannot ENFORCE the
> L→D→U→R staircase SEQUENCE. ... The real lever is a STRUCTURED 'realize' (option 2), not more figure training."

Consolidated deliverable `gen_motif_full` = candle/trill (continuous) + modest figure sweep nudge.

### Beat 5 — a representation bug, found and fixed (`repr_integrity_findings.md`)

A pre-retrain audit caught a real data bug:

> "**★ HEADLINE BUG (FIXED) — typed converter dropped sub-16th notes via zero-overwrite** ... the later row's
> empty panels overwrote the earlier row's notes with zero ... Post-fix the gap collapses to −2.2 frames/chart
> ... tap share 0.192→0.211 (recovered notes)."

### Beat 6 — the H19 clean retrain → the deployed model (`h19_retrain_findings.md`)

> "**Verdict — strict win (or equal) on every lever, no regression** ... **Candle preserved** ... **Trill is
> now HONEST** (deployed learned from hold-tail-INFLATED targets and slightly over-produced it) ... **Sweep
> IMPROVED** (realized 0.09 vs 0.07, vs real 0.11) ... `gen_motif_full_fixed` is a clean, strictly-not-worse
> replacement."

⚠️ Honest complication: "the retrain did **NOT** reduce the during-holds activity" (the user's felt "jacks
during holds" is a *decoder* behavior, not the detector bug).

### Beat 7 — the consolidated status (`note_patterns_and_motifs.md`) — THE source of truth for the README

> "**Status: PARTIAL / unfinished as a control surface. Do not present it as done.**"

> "**Candle/cross — works, playtest-confirmed.** ... **Trill — steers**, but the offline trill metric is partly
> confounded ... **jack↔sweep — the lone dead/weak axis.**"

> "**Dependency note:** `cache/motif_basis.npz` is **not shipped** (unlike `cache/radar_manifold.npz`), and
> `scripts/generate.py` does **not** expose `--motif`/`--figure`."

Plus the H20 **coverage gap** (distinct from the steering gap):

> "The model's repertoire is **over-concentrated on jacks** and **under-covers ornamental footwork** ... a
> **data-coverage / objective** issue ... **not** a decode knob."

---

## Audit hooks (reconcile README against these)

| README claim | Verbatim source | Verb precision |
|---|---|---|
| section-level motif / figure control (candle/trill/sweep) | "PARTIAL / unfinished ... Do not present it as done"; candle "works, playtest-confirmed"; jack↔sweep "lone dead axis" | ⚠️ **DON'T claim a finished motif control surface.** Candle is the only *strong, playtest-confirmed* lever. Trill = modest + confounded. Sweep = soft-realize ceiling. |
| motif knobs as a user-facing feature | "`scripts/generate.py` does **not** expose `--motif`/`--figure` ... `cache/motif_basis.npz` is **not shipped**" | ⚠️ **AUDIT-CRITICAL.** The motif surface is NOT reachable from the shipped CLI. If the README implies a user can ask for "candles here, trills there," that overstates the shipped product. It's an internal/experimental lever. |
| deployed model identity | "`gen_motif_full_fixed` ... clean, strictly-not-worse replacement" (42-dim high-res, H19 retrain) | **measured** ✅ — this is THE deployed model. If the README/model-card names a generator, it should be `gen_motif_full_fixed`, not older `gen_stage1/radar/style`. (Release criteria flags the model-card lineage as stale — a deferred item.) |
| candle "playtest-confirmed" | "`motif_candle_neg` played audibly 'more linear'" | **vouched** ✅ (playtest) — qualitative, keep it qualitative. |
| trill steering strength | H19: "Trill is now HONEST ... g1 +0.32 vs +0.47" — measurement was ~10% inflated by a hold-tail artifact | **measured**, but the OLD numbers overstated trill by ~10%. Use the *fixed*-model framing; don't cite pre-H19 trill numbers. |

**Verb-precision watch:** the deployed chaos behavior on this model: "**pure radar chaos via the MANIFOLD
barely places 16ths** ... **`onset_phase_calib` is what actually PLACES the 16ths**" (set characterization) —
consistent with [[07-chaos-placement]] / [[09-manifold-guidance]]. The motif numbers are 16-song stochastic
evals "within ... noise" — directions are reliable, magnitudes are soft. Cross-ref [[playtest_log]] (H15
candle, H19 trill confound, H20 vocabulary) and [[00-meta]] HANDOFF for current deployment.

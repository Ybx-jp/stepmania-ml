# Brief 04 — Conditioning: the control knobs (pattern prefs → radar → CFG → style; + the transition negative)

**Source notes (read in order):** `conditioning_step1_pattern_prefs.md` → `conditioning_step2_radar.md`
→ `conditioning_step2b_cfg.md` → `conditioning_step3_style.md` → `conditioning_match_findings.md` →
`numeric_difficulty_conditioning_plan.md` → `h11_transitions_findings.md`
**Arc role:** the README's "**Controllability**" beat. This is where "groove-radar conditioning + CFG +
reference-chart style transfer" comes from. It also contains the project's most important **honest
negative** for the song-structure claim (h11 transition drift), which the v1 audit corrected.

---

## The narrative

### Beat 1 — decode-time pattern knobs (`conditioning_step1_pattern_prefs.md`)

Free knobs that touch *which panels*, not *where notes go*:

> "**Graded jump control** (0.02 ↔ 0.77), **crossovers → 0** on demand, **panel reshaping** (R 0.29→0.68)...
> **onset_F1 and difficulty critic unchanged** across all settings — pattern controls only touch which
> panels, not where notes go or the overall difficulty."

### Beat 2 — the trained groove-radar knob (`conditioning_step2_radar.md`)

A 5-dim groove radar (stream/voltage/air/freeze/chaos) trained as a conditioning input with CFG dropout:

> "Each dim moves its expected proxy in the right direction: **stream/voltage → density, air → jumps,
> freeze → holds, chaos → density/complexity**. The model learned to obey the radar profile."

Honestly hedged at the time: "**Working knob, modest magnitudes.** Effects are directionally correct at
guidance scale 1." That's what motivates CFG next.

### Beat 3 — CFG amplifies the radar (`conditioning_step2b_cfg.md`)

> "CFG makes every radar dimension a strong knob (~3–4× the plain-conditioning effect)... g=2–3 is a good
> range, higher overshoots (chaos→0.74 density is extreme)."

Cost is stated: "CFG runs a second decoder pass per step... generation is ~2× slower when
`guidance_scale != 1`."

### Beat 4 — reference-chart style transfer (`conditioning_step3_style.md`)

A `StyleEncoder` (masked-mean-pool → one latent, an autoencoder-style bottleneck) transfers *feel*:

> "The reference chart's feel transfers onto *different* audio: a sparse reference pulls density down, a
> dense one pulls it up, and CFG widens the gap monotonically (~2.5× from g=1 to g=3)." (g=3 gap **0.47**)

### Beat 5 — the crucial caveat: radar vs style do DIFFERENT jobs (`conditioning_match_findings.md`)

This is the note that should govern how the README describes the two knobs:

> "**`match_radar` wins decisively (0.81→0.44)**... **`reference_self`... does NOT help match the profile
> (0.83 ≈ baseline).** The StyleEncoder is a bottleneck... built for STYLE TRANSFER of holistic feel; it
> does not carry the quantitative groove dims... **this measures PROFILE match (radar distance), not FEEL.**"

So: **radar** = hit a target groove *profile*; **style** = transfer a holistic *feel*. They are not
interchangeable.

### Beat 6 — the planned-but-noted numeric-difficulty thread (`numeric_difficulty_conditioning_plan.md`)

A **plan**, not a result: switch difficulty conditioning from 4 name-classes to the numeric meter via a
thermometer encoding (reusing the ordinal lesson). The honesty point it bakes in:

> "**Difficulty is inherently fuzzy → soft control, by design.** Nobody beat 16.5% adjacent-class error
> in the ordinal experiment... meter conditioning is a gentle dial, not a precise selector."

### Beat 7 — the transition negative (`h11_transitions_findings.md`) — AUDIT-CRITICAL

This note is the strongest evidence governing the README's song-structure honesty claim. It has a
**robust** finding and two **non-robust** ones:

> "**Given the correct context, the model predicts transitions ~as well as steady-state.** Transitions
> are NOT a representational deficit." (robust — teacher-forced probe)

> "**The free-running under-transition effect is NOT robust — it's song-set-dependent.**... the big gap
> reported in Probe 2... was on a different 30-song set. The responsiveness metric is noisy across song sets."

> "**Buffered-sectional OVERSHOOTS badly**... Real transitions *evolve*; this *resets*."

> "**Verdict:** offline does NOT support the sectional approach... The robust H11 finding is only
> 'transitions aren't a representational deficit'; the drift gap and the reset/sectional fixes are
> inconclusive (noisy metric) or counterproductive (overshoot)."

---

## Audit hooks (reconcile README against these)

| README claim | Verbatim source | Verb precision |
|---|---|---|
| Controllability: groove-radar + **CFG** + reference-chart **style transfer** | radar "model learned to obey"; CFG "~3–4× the plain-conditioning effect"; style "feel transfers onto different audio" | **measured** ✅ on val sweeps. But each is a *directional* sweep on val songs, NOT a guarantee of arbitrary control. "Knob," "dial," "pull" — not "set exactly." |
| radar vs style framing | "match_radar wins... reference_self does NOT help match the profile" | If the README equates `--radar` and `--style`, that's WRONG: radar hits a *profile*, style transfers *feel*. They measure different things. |
| ⚠️ README says "`--radar` is disabled (off-manifold)" | The notes here show radar conditioning *trained and working*. The "disabled / off-manifold" decision comes LATER ([[09-manifold-guidance]] / [[10-motif-arc]] / [[12-geometry-feasible]]). | **Reconcile across briefs:** the *trained* radar knob works (this brief); the *deployed default* moved to the in-distribution `--style` (manifold) path because raw radar steering goes off-manifold. Don't let the README imply radar never worked — it works, but isn't the shipped default. |
| "no learned song-structure/climax awareness" (honesty note) | h11: "transitions are NOT a representational deficit"; but the free-running drift gap "is NOT robust"; sectional "overshoots" | **AUDIT-CRITICAL.** The v1 audit *corrected* this README line (row 18). The defensible statement: the model **tracks** local structure/accents given context; the **open frontier is global whole-song phrase planning**. Do NOT claim the model re-choreographs transitions reliably (not robust), and do NOT claim it's structure-blind (teacher-forced probe refutes that). |

**Verb-precision watch:** CFG amplification numbers (3–4×) are **measured on val sweeps**, but "overshoots"
past g≈3. The README should present CFG as a *strength dial with a usable range*, not unbounded control.
The numeric-difficulty thread is a **plan** — if the README implies meter-level difficulty selection, that
overstates (it's named in the plan as "a gentle dial, not a precise selector," and the build may not be
deployed — verify against the deployed model in [[10-motif-arc]]/[[00-meta]] HANDOFF before claiming it).

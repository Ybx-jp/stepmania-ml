# Brief 05 — Musical features (H1/H4/H5): the choreography axis, and the "features aren't the lever" verdict

**Source notes (read in order):** `feature_retrain_plan.md` → `stage1_musical_features_findings.md` →
`h4_offbeat_signal_findings.md` → `choreography_metrics_findings.md` → `groove_periodicity_findings.md`
**Arc role:** the project's hardest, most honest dead-end-that-taught-something. It set out to fix
musical blindness (arrow choice, syncopation, structure) by adding audio features. The features got
*used* but **did not** fix chaos/syncopation — which is precisely what redirected the project toward the
**taste critic** ([[06-taste-critic]]) and the **manifold/conditioning** rethink ([[07-chaos-placement]],
[[09-manifold-guidance]]). Anything the README says about "chaos," "musicality," or "song structure"
must reconcile against this arc.

---

## The narrative

### Beat 1 — the diagnosis: features are timbre+energy only (`feature_retrain_plan.md`)

> "The 23-dim audio features are **timbre + energy only**... There is **no melody/harmony (no
> chroma/pitch) and no source separation**... the audio encoder is a shallow 2-block Conv1D (receptive
> field ~5 frames, no global/structural view)."

The plan names three felt gaps — **H1** local choreography, **H4** chaos-as-uniform-smear, **H5**
flat/fading structure — and adds chroma+HPSS+metric-phase (23 → 41 dims).

### Beat 2 — Stage 1: features are USED, but offline musicality doesn't move (`stage1_musical_features_findings.md`)

> "**Offline verdict: no improvement; onset-phase slightly REGRESSED (more on-beat).**"

Crucially, the model genuinely consumes the new info — it's not a warm-start artifact:

> "**Chroma is used as heavily as the MFCCs** (KL 10.34)... **Metric phase is used (4.58)** but its net
> effect pushed onsets MORE on-beat (0.93→0.952)... **HPSS nearly ignored (0.29).**"

And the honesty framing the README depends on:

> "Chroma did **not** fix chaos: cranking the chaos radar dim still destroys the downbeat (≈6% on-beat)
> and smears notes uniformly."

> "Not a refutation of the feature hypothesis (H1)... The decisive test is the playtest of `gen_stage1`."

(Per the INDEX/memory, the **playtest of `gen_stage1` was a WIN even though metrics were blind** — the
metrics-can't-see-it story. Confirm in [[playtest_log]] / [[00-meta]].)

### Beat 3 — H4: chaos is NOT a feature problem (two retrains) (`h4_offbeat_signal_findings.md`)

This is the arc's keystone negative. It walks: the off-beat signal *is* mostly a **resolution** problem
at the feature level (high-res onset recovers AUC 0.527→0.662 at 16th-off) → so a high-res onset feature
was added and the model retrained **twice**:

> "**Result 5 — the warm-started retrain ran but DID NOT ENGAGE the feature.**... ablation KL = 0.0000...
> **There is essentially no gradient pressure to grow the new column.**"

> "**Result 6 — H4-v2 engaged the feature but it BARELY MATTERS; chaos still smears.**... under chaos@CFG=2,
> on-beat% = **4.4% (v2) vs 6.1% (stage1)** — both wildly over-syncopated vs real (~80-90% on-beat)."

> "**CONCLUSION (well-evidenced, two independent retrains): chaos/syncopation is NOT a feature problem.**...
> The real levers are **(a) the conditioning mechanism**... and **(b) the objective** — frame-wise CE
> never rewards tasteful syncopation, which is exactly why the Stage-2 **taste critic** is the thing that
> actually tracks musicality."

`gen_highres` / `gen_highres_v2` are **parked**.

### Beat 4 — the choreography metrics that finally agree with the hands (`choreography_metrics_findings.md`)

Two real findings on the which-arrow axis:

> "**KL(gen‖real) 0.024 ≈ KL(shuffle‖real) 0.025: the generator's panel transitions are no closer to real
> than a random panel-shuffle.**... rhythm ~right, **choreography unstructured.**"

> "**hold_burst = HIT:** gen 6.9% ≈ random null 7.0%, real 4.0%... **★ AGREES WITH THE HANDS:** this is
> exactly the B4U playtest complaint... First geometric metric shown to PREDICT a play-feel verdict."

This bipedal `hold_burst` metric is what motivated the `no_cross_during_hold` decode fix in
[[03-typed-model]].

### Beat 5 — groove periodicity: the metric that exposed a decode under-placement (`groove_periodicity_findings.md`)

A validated periodicity metric (`ac_off`, 15× its null) localizes the groove deficit to **decode**, not
architecture:

> "**ANSWER: NOT architectural — it's a DECODE under-placement of off-beats.**... selecting top off-beats
> by the model's own `p_on` yields **real-level groove (0.318 ≈ 0.300, 15× null)**. The onset head DOES
> carry groove."

…but the fix doesn't cleanly win, and the metrics disagree:

> "**Budget RAISED periodicity (ac_off, toward real) but LOWERED taste (critic)**... The two
> validated-ish metrics disagree... **Unresolved offline.**"

This is the cleanest in-project demonstration that **periodicity ≠ musicality** — directly motivating the
taste critic as the only judge that tracks felt quality.

---

## Audit hooks (reconcile README against these)

| README claim | Verbatim source | Verb precision |
|---|---|---|
| "the chaos knob still smears off-grid" (honesty note) | "chaos still smears... 4.4% on-beat vs real ~80-90%" (under the **radar-CFG** mechanism) | **AUDIT-CRITICAL.** The v1 audit *corrected* this (row 19) to "chaos is in-distribution-bounded; musical on-manifold." **Both can be true** because they're DIFFERENT mechanisms: radar-CFG chaos smears (this arc); the **deployed** chaos uses manifold conditional-fill ([[07-chaos-placement]], [[09-manifold-guidance]]). The README must not claim the *radar* chaos is musical — it isn't. It must attribute the "in-distribution" behavior to the manifold path. |
| musical features improve choreography | "features are USED (chroma KL 10.3) but offline musicality didn't move"; playtest of gen_stage1 = WIN | **measured** (feature-use) + **vouched** (playtest). Say features are *used*, not that they *measurably improved musicality*. The win is playtest-level, metrics blind. |
| chart tracks musical structure / H5 | feature plan: structure is "the model faithfully tracking audio energy, not a decode artifact — it has no notion of song *sections*" | Supports the corrected song-structure framing in [[04-conditioning]]: tracks *energy*, lacks *global section planning*. |
| "metrics can't capture play-feel" framing | "periodicity ≠ groove-that-feels-good"; budget decode: ac_off up, critic down, "Unresolved offline" | **measured/vouched** ✅ — this is the strongest evidence FOR the README's central thesis (evaluation for hard-to-measure quality). |

**Verb-precision watch:** the high-res onset feature is a **parked negative** — if the README lists
features the model uses, do NOT include the high-res onset as adopted (it's not the base; KL≈0 then
"barely matters"). The "chaos isn't a feature problem" conclusion is **well-evidenced (two retrains)** —
safe to state strongly. The groove decode fix is **unresolved offline** — do not claim groove was "fixed."

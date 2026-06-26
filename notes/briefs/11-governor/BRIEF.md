# Brief 11 — The biomechanical governor (foot fatigue / stamina / breathing arc) — the controllability beat

**Source notes (read in order):** `h13_exertion_findings.md` → `foot_exertion_findings.md` →
`foot_fatigue_design.md` (the 296-line design bible) → `governor_release_region.md`
**Arc role:** the README's **second controllability beat** (per the release criteria: present as a short
*qualitative* "tasteful edit" proof point, NOT a metric, and it does NOT displace the taste-critic climax).
A decode-time governor that makes generated charts physically danceable: per-note foot fatigue (default on),
per-region stamina relief (opt-in), and a breathing difficulty arc (opt-in). Playtest-confirmed.

---

## The narrative

### Beat 1 — H13: the model over-produces unplayable jacks (`h13_exertion_findings.md`)

Confirmed by data, not just feel:

> "**DEPLOYMENT 0.284 ≫ REAL 0.000 → H13 CONFIRMED by data**... The generator over-produces fast same-panel
> jacks by ~50×."

> "**Attribution (clean isolation):** ... it is the **PATTERN head (which-panel), not onset placement**...
> Given the *exact* notes a human patterned with ZERO jacks, the v4 pattern head still jacks 28% of fast pairs."

The fix `max_jack_run=1` gave "9× reduction, brutal 4+ jacks eliminated" and was **PLAYTEST-CONFIRMED**
("night in motion 'AWESOME!'") — promoted to the mandatory playability set.

### Beat 2 — soft, BPM-aware jack governor (`foot_exertion_findings.md`)

The hard cap had an 8th-note blind spot; replaced by a graded exertion penalty:

> "**Density IDENTICAL (0.208) at every λ** — the headline. The penalty re-routes long jacks to ALTERNATION,
> it does NOT delete notes (biomechanically correct: a human alternates feet instead of hammering)."

### Beat 3 — the unified per-foot fatigue model + every failure (`foot_fatigue_design.md`)

The jack governor "DISPLACED mass into jumps," so it was generalized into a **two-foot fatigue simulator**
(a jack and a jump are the same thing seen through two feet). The note is unusually honest about how much
*bookkeeping* it took — five subtleties (footswitch loophole, the lift bug, free threshold, graded
footswitch, blunt forbid):

> "**LESSON: a faithful foot simulator is mostly *bookkeeping***... every gap becomes a loophole the decode
> exploits — the diag-on-real-data loop is what caught each one."

The calibration **changed direction** — an experiment-design win worth preserving:

> "**The model UNDER-jumps, doesn't over-jump.** Real charts on these air-heavy songs = **31%** jumps... the
> model OFF = ~6%. The 'consecutive jumps' the user felt were **INDUCED by the jack penalty**... So
> calibrating the governor to 'match real jump%' optimizes the WRONG axis."

> "**The governor's real value is JACKS:** maxJackRun 6.2→3.6 (real 3.5), jack≥4 2.1%→0.4% (real 0.8%) — lands
> right on the human distribution... **fatigue should REPLACE the jack penalty.**"

**Stage 2 — stamina (per-region density relief):** a slow accumulator that raises onset tau over an
upcoming stretch → *coherent thinning*, not hole-punching. Note the failed Stage-1 hard veto first:

> "Built it as a hard onset veto... → **DENSITY CRASHED 0.320→0.145**... a **hard per-note veto is the wrong
> tool for the stamina dimension.**"

> "**PLAYTEST ✅ WIN (2026-06-25):** under aggressive chaos conditioning (density cranked to 0.400), g50 on
> japa1 was 'much more playable than off without being much different — a **TASTEFUL EDIT, not a rewrite**'...
> The default-conditioned A/B was imperceptible (correct — the chart wasn't over its ceiling)."

**Stage 3 — the breathing arc:** ceiling breathes with audio energy → protect climaxes, rest verses:

> "the model is NOT structurally flat in this metric — its onset head already tracks energy at corr 0.898.
> The problem is that FLAT stamina DULLS that arc... Breathing makes the thinning ARC-AWARE... breathe=1.2 →
> corr 0.918 / Δ 0.185... at held overall density (REDISTRIBUTION, not a cut)."

Plus an honest open problem: **holds-blindness** ("the top open problem for this model") — the hold-aware
E_slow cost was built, but "the pathology is ABSENT under default conditioning" (default holds aren't grinds).

### Beat 4 — the vouched ranges + release defaults (`governor_release_region.md`)

The release center, stated precisely (and with the right epistemic label):

> "⚠️ **SCOPE — this is NOT 'the region of good settings mapped.'** It is a **table of per-knob ranges the
> user PERSONALLY VOUCHED FOR** (hands-on playtest + a few targeted offline sweeps)... The joint region is V2."

Release center: `fatigue_penalty=2` (always on), stamina + breathe **opt-in** (near-no-op until a chart is
over its workload ceiling).

---

## Audit hooks (reconcile README against these)

| README claim | Verbatim source | Verb precision |
|---|---|---|
| governor = "a tasteful edit, playtest-confirmed" | "a **TASTEFUL EDIT, not a rewrite**" (stamina playtest); H13 "AWESOME!" | **vouched/playtest** ✅ — **KEEP IT QUALITATIVE.** Release criteria: "Governor claim stays qualitative... NOT dressed as a percentage." Do NOT attach a % improvement to the governor. |
| jacks made human-like | "maxJackRun 6.2→3.9, jack≥4 2.1%→0.7% (real 0.8%); density held" | **measured** ✅ offline — these numbers are real, but they describe *jack run-length distribution matching*, not "the governor improves charts by X%." If cited, frame as "matches the real Hard jack distribution," qualitatively. |
| governor structure (per-note foot + stamina + arc) | the three stages, all "BUILT + VALIDATED" | **measured + vouched** ✅ — accurate. But note the deployment posture: **fatigue_penalty=2 always on; stamina + breathe OPT-IN** (default near-no-op). Don't imply stamina/arc are always shaping output. |
| governor + difficulty | "the model UNDER-jumps (6% vs real 31%)... do NOT tune the governor to close it" | ⚠️ Do NOT claim the governor fixes density/jumps. It's a *ceiling* (playability), not a difficulty knob. The jump gap is a separate density/air thread. |
| "vouched-for ranges" vs "mapped region" | "NOT 'the region of good settings mapped'... a table of per-knob ranges the user PERSONALLY VOUCHED FOR" | ⚠️ **claim-precision (memory [[claim-precision]]):** vouched ≠ mapped. If the README/marketing says the governor's operating region is "mapped" or "characterized," that overstates. It's vouched per-knob ranges; the joint feasible region is a V2 item ([[12-geometry-feasible]]). |

**Verb-precision watch:** the governor's *numbers* (jack≥4 0.7% vs real 0.8%) are **measured offline**; the
governor's *value* ("tasteful edit") is **vouched by playtest**. Keep those two registers distinct. The
holds-blindness problem is **open** — don't claim the governor handles holds (the hold-aware cost is built
but has "almost nothing to bite on" under default conditioning). Cross-ref [[playtest_log]] (H-stamina,
H-arc-end) and [[00-meta]] HANDOFF (governor is the current arc; PR #41 was the release prep).

# Numeric-difficulty conditioning — design lessons + plan

*2026-06-21.* Switching the generator's difficulty conditioning from the 4 NAME classes
(Beginner/Easy/Medium/Hard) to the numeric METER (the .sm "feet" rating, `difficulty_value`). Driven by
the expert-data expansion (`constraint_relaxation_roadmap.md`): expert charts all collapse into "Hard",
so the 4-class signal can't separate tame-Hard from expert-Hard. Lessons drawn from the prior classifier
ordinal experiment (`ordinal_experiment_findings.md`) + the meter distribution.

## The data
Meter range **1–17**. The 4 classes overlap heavily — Beginner 1–7, Easy 1–10, Medium 2–15, **Hard 1–17**
(there are name=Hard charts with meter 1 = real label noise). Counts taper hard: meter 11=134, 12=141,
13=79, 14=34, 15=17, 16=6, 17=3. Parser falls back to `difficulty_value=0` for non-numeric meters
(stepmania_parser.py:372) — junk, must drop/impute.

## Lessons → design

1. **Encode the meter as a cumulative / "thermometer" vector, NOT a raw scalar or per-integer embedding.**
   The ordinal experiment's central result: the single-**scalar** proportional-odds head COLLAPSED (44%
   acc, early-stop epoch 2 — scalar bottleneck + tight init), while the **multi-output cumulative** head
   won. Transfer to conditioning INPUT: meter `d` → `[1×d, 0×(M−d)]` (M≈20), then `Linear(M, d_model)`.
   - Preserves ordinality: meter 9 and 10 share most of the vector → *close* (a per-integer `nn.Embedding`
     would make them unrelated).
   - Avoids the scalar bottleneck the experiment proved brittle.
   - The current 4-way `nn.Embedding` (no ordinality) is what we're replacing.

2. **The meter is more informative than the name-class but is itself noisy** (subjective, cross-pack
   inconsistent, name/meter disagreements). Mitigation: **pair meter with the objective groove radar**
   (density/stream/voltage are measurable difficulty proxies) — meter = intended difficulty, radar =
   measurable intensity. Don't condition on the subjective scalar alone. Drop/clean meter=0.

3. **High-meter sparsity is WHY numeric + expert-data are synergistic — but cap expectations.** Meters
   13+ are thin, 15–17 nearly empty. Expert-data expansion fills the hole, but the very top stays tiny →
   treat ~1–13 as the controllable range; expect to need CFG guidance to amplify the thin high-meter
   signal (like radar); don't promise sharp meter-16 control.

4. **Difficulty is inherently fuzzy → soft control, by design.** Nobody beat 16.5% adjacent-class error
   in the ordinal experiment; the model can't sharply separate adjacent difficulties. The thermometer's
   smoothness MATCHES that fuzziness; meter conditioning is a gentle dial, not a precise selector.

5. **Sequencing caveat (multi-variable):** meter conditioning re-plumbs `_cond` (replace `diff_embedding`
   nn.Embedding(4) with a thermometer projection) AND needs a retrain — coupling it with the expert-data
   rebuild. That's 3 changes at once (expert data + numeric difficulty + the new periodicity metric). Do
   them in one rebuild, but lean on the **periodicity/groove metric** to judge the combined effect, and
   keep `gen_stage1` (4-class, tame data) as the reference. Consider an intermediate isolation run if the
   combined result is ambiguous.

## Build order (agreed)
1. **Periodicity/groove metric FIRST** (cheap, no retrain) — operationalizes H10 (groove = repeated
   rhythmic figures = autocorrelation peaks at musical lags). Prerequisite to judge any of this; also
   informative now (does the chaos-gate scatter show flat autocorr while real grooves spike?).
   `notes/chaos_mechanism_plan.md`.
2. Expert-data cache rebuild (length→180 + `--allow_hands`) + numeric-meter conditioning (thermometer) +
   expert-aware critic retrain.
3. Evaluate with periodicity + phase-match-to-real-expert + expert-aware P(real) + playtest.

Cross-refs: `ordinal_experiment_findings.md`, `constraint_relaxation_roadmap.md`, `chaos_mechanism_plan.md`,
`conditioning_step2_radar.md` (radar conditioning the meter pairs with).

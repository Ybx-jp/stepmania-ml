# Chaos / 16th-PLACEMENT: a principled ceiling (audio ambiguity) — arc conclusion

The chaos arc set out to make generated charts produce musical 16th notes (chaos radar was 0). We can now
control the 16th AMOUNT but NOT the musical PLACEMENT, and we've established WHY, ruling out every lever:

## What works (AMOUNT / global control)
- High-res onset feature engaged (v4) → model can place 16ths at all (0%→). See [[chaos_mechanism_plan]].
- Chaos-radar CONDITIONING dials the global 16th amount 0.3%→26%, specifically, monotonically
  ([[chaos_conditioning_findings]]). Decode `onset_phase_calib` gives variable per-song amount.
- v7 reweighted BCE matches real's per-phase RATE distribution (additive) — right AMOUNT of 8ths+16ths.

## What does NOT work (PLACEMENT / which-16th-where) — all cheaply ruled out
- **Decode** (calib / conditioning): unburies 16ths but can't make them land musically.
- **Features** (`diag_16th_features.py`): HPSS percussive/harmonic high-res onsets + BPM do NOT beat the
  mixed onset at 16th localization (all ~0.58 raw; model already 0.67). Source separation adds no signal.
- **Architecture / context** (`diag_seqhead_probe.py`): onset classifiers with increasing receptive field
  plateau at 16th-AUC ~0.65 (pf 0.60 → k7x3 0.655 → dil 0.644); GLOBAL self-attention is NO better than
  per-frame (0.596). If long-range structure carried placement signal, attention would win — it doesn't.
  A sequence-aware onset head would NOT break the ceiling.

## Conclusion: 16th placement is AUDIO-AMBIGUITY-bound (~0.65–0.67), not a fixable model gap
WHICH 16th frame deserves a note is genuinely under-determined by the audio — the same drum fill has many
valid, equally-musical chartings (charter style). The model already extracts ~all the placement signal
present. This is WHY v7's distribution-correct 16ths still played "awkward" (06-22 playtest): there is no
single right placement to hit. AMOUNT is controllable; PLACEMENT is near an inherent ceiling.

## Standing decisions
- **Current best playable model = v4** (gen_highres_v4): coherent structure, sane conservative 16ths. Do
  NOT ship v6 (8th-collapse) or v7 (cold-start regression + awkward placement).
- Chaos AMOUNT control (calib / radar conditioning) is a real, shippable knob; chaos PLACEMENT excellence is
  not reachable from audio with this paradigm.
- **Methodology win:** decode/features/architecture each refuted by a ~10-min probe BEFORE an expensive
  build (the [[experiment-design]] discipline). This arc is a strong case study for the evaluation thesis
  (rigorously bounding a hard-to-measure quality axis). See [[marketing-track]].

## If revisited later (bigger pivots, not this paradigm)
- A fundamentally different generation paradigm (joint structure+placement, e.g. token-level AR over a
  richer representation, or learning from MULTIPLE human chartings to model the placement DISTRIBUTION
  rather than a point target — accepting ambiguity instead of fighting it).
- Constraint-relaxation / data-layer v2 (variable BPM, finer res) — see [[constraint_relaxation_roadmap]].

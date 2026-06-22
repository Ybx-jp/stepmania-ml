# Chaos conditioning WORKS (post-high-res) — the lever, validated

Follows [[selfsim_chaos_findings]] (chaos can't be inferred from audio -> it must be CONDITIONED) and the
H4/H6 history (chaos conditioning SMEARED — but that was pre-high-res, when the model had no local 16th cue
to attach conditioning to). Re-tested on gen_highres_v4 (high-res onset engaged, frame 16th AUC 0.742).

Cheap offline probe (experiments/generation_typed/diag_chaos_condition.py, NO generation): the onset head
takes the radar, so sweep ONLY the chaos dim (other 4 radar dims at the song's real values) and read the
posterior + threshold placement. 50 songs, guidance=1.0:

```
  chaos -> mean p_on by phase            chaos -> realized note share (per-song density threshold)
   chaos  p_q    p_8th  p_16th            chaos  quarter%  8th%   16th%
    0.00  0.625  0.518  0.316              0.00    78.7%   21.0%   0.3%
    0.50  0.564  0.617  0.556              0.25    32.1%   63.5%   4.4%   <- real-like
    1.00  0.587  0.626  0.598              0.50    15.2%   67.6%  17.2%
   (16th p_on delta +0.282)                1.00    11.5%   62.2%  26.2%
```

## Findings
- **Strong dynamic range:** realized 16th share 0.3% -> 26% across chaos 0->1, MONOTONIC. Ample control.
- **SPECIFIC, not a smear:** raising chaos DROPS quarter p_on (0.625->0.587) and collapses quarter share
  (79%->11%) while 16ths flood in — trades quarters for 16ths like real charts getting busier. The H4/H6
  smear raised everything uniformly; this does not.
- **Survives to placed notes** (not threshold-buried), and plain guidance=1 has full authority — NO CFG
  amplification needed (contradicts the old "chaos needs strong guidance" assumption; strong guidance was
  papering over a model that couldn't render chaos pre-high-res).
- **Real-like charts sit at chaos ~0.25** (4.4% 16ths, 8th-dominant). chaos=0 = no 16ths explains why
  generation at the song's (low) real chaos produced ~none; the knob was always there, just not turned up.

## The paradox resolved
Audio can't TELL the chaos level (selfsim R^2 0.06) but the chaos INPUT has near-total authority. Both true:
don't infer chaos, CONDITION it. The high-res feature unlocked this (local 16th cue for the conditioning to
key on) — the same feature whose absence made chaos smear in H4/H6. This validates the project thesis
("mastering chaos conditioning gives control over all axes").

## Relationship to calib
[[phase_aware_threshold_findings]] calib (decode-time per-phase offset) and chaos-radar conditioning are two
routes to more 16ths. Conditioning is the principled one (trained control, full 0->26% range, specific).
calib is the no-retrain decode nudge. Where you want chaos control, prefer turning up the chaos input.

## Next (recommended)
GLOBAL chaos is solved. The musical goal is LOCAL control — chaotic sections busy, calm sections clean
WITHIN one song. Next probe: PER-SECTION chaos conditioning — vary the chaos input over time (the
buffered-sectional idea), and check 16th density tracks the per-section chaos input. If section structure +
per-section chaos targets work, that's choreographic control of the whole intensity arc. See
[[chaos_mechanism_plan]], [[playtest_log]]. Playtest moderate global chaos (~0.3-0.5) to confirm the
posterior-level specificity translates to play-FEEL (the H4/H6 failure was a feel failure, not just a stat).

# no_cross_during_hold — decode fix for one-foot-stream-during-hold

*2026-06-21. `generate(no_cross_during_hold=True)`.* Closes the danceability gap the bipedal metric found
(`choreography_metrics_findings.md`): while a hold pins one foot, the free foot was being forced to
fast-cross panels (the B4U playtest complaint "crossovers and jacks with one foot during a hold"). The
existing `no_jump_during_hold` blocks *jumps* during a hold but allows single-tap *streams* on other
panels, so the free foot still scatters.

## The metric (motivation)
Bipedal kinematics `hold_burst` = rate of one-foot fast-crosses (different panel, ≤2-frame gap) while the
other foot is pinned by a hold. gen_stage1 **6.9% ≈ random null 7.0%**; real **4.0%**. Real choreography
avoids it; the generator produced it at random rates (and ~4× more during-hold notes overall).

## The constraint
Track the free foot's last panel + recency during an open hold. Graduated by speed:
- **16th gap** (≤1 frame since the free foot's last note): forbid ALL different-panel singles → the free
  foot stays on its panel (a jack), never crosses.
- **8th gap** (==2 frames): forbid only the OPPOSITE single (the worst, distance-2 cross).
It *redirects* (forces a jack) rather than deleting notes, so density is preserved. Needs `hold_aware=True`
(holds must be tracked). Implemented in the KV-cached decode loop; `free_last`/`free_gap` state.

## Result (20 val songs, gen_stage1, same seed off vs on)
- **hold_burst 8.7% → 4.7%** (real ~4.0%) — essentially matched real.
- **density 0.192 → 0.192** — unchanged (redirect, not delete).
- Test `tests/test_generation.py::test_no_cross_during_hold` (comparative: flag must not increase, and
  reduces, fast hold crosses); all 26 generation tests pass.

## Use / status
`export_typed_samples.py --no_cross_during_hold`. **Playtest pending:** does it FEEL more danceable than
the default (B4U-style awkwardness gone)? This is the meta-test — the bipedal metric predicted the B4U
hands complaint, so a fix that drops hold_burst to real *should* feel better; the playtest confirms the
metric→feel link both ways. Recommended decode if confirmed: add to the default
(pattern_temp 0.7, type 0.4, hold_aware, no_jump_during_hold, no_cross_during_hold).

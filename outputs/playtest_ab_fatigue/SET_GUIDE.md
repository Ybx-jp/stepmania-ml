# Jack-penalty vs Per-foot-fatigue A/B (~/sm-generated/ab_fatigue_*)

Same songs (rich Hard, seed 42 — the jump-heavy egregious set), same clean model (`gen_motif_full_fixed`),
same hard cap (max_jack_run=2). Two DIFFERENT jack/jump governors:

- **`ab_fatigue_A_jack`** = the CURRENT shipped jack penalty (`--jack_penalty 1.5`). Governs jacks but
  DISPLACES the mass into jumps (the thing you flagged).
- **`ab_fatigue_B_foot`** = the new PER-FOOT FATIGUE model (`--fatigue_penalty 2`, jack penalty OFF). Governs
  jacks AND jump streams in one biomechanical foot simulator → no displacement.

A/B the SAME song across the two folders.

## What changed (offline, on the actual installed charts)
| | A (jack penalty) | B (fatigue) |
|---|---|---|
| jump rate | 23.3% | 4.6% |
| **longest jump stream** | **59** | **3** |
| longest jack run | 4 | 5 |
| density | 0.346 | 0.346 |

A has a 59-note jump WALL (the displacement); B kills it (longest 3). Density identical — B re-routes, doesn't delete.

## The question for your ears (this is the real test)
- Does **B** feel cleaner — no jump walls — while jacks stay controlled?
- **The risk flips direction:** B's 4.6% jumps is FAR below real (~31% on these songs). So does B feel **too flat
  / jump-starved**, or pleasantly clean? (The model already under-jumps; the fatigue model is gentle but still
  reduces jumps.)
- **Crucial — earned vs unearned streams:** real hard charts DO have long jump streams (musical ones). Does B
  flatten jump streams that *should* be there (a moment that wants a jump burst), or only the ugly ones?

## Decision it gates
If B feels better (clean, not starved) → **fatigue REPLACES the jack penalty as the default** (it fixes the
displacement, the original motivation). If B feels jump-starved → lower λ, or the real issue is the model
under-jumping (separate density/air thread, not the governor). Both pad-playable.
Calibration detail: `notes/foot_fatigue_design.md`. Note: the model under-densifies these songs (0.346 vs real
0.385) — a separate thread.

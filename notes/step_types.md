# Step Types: Typed Chart Representation (Phase 2.5)

*2026-06-18. Branch `gen/step-types`. Data-layer foundation — model retrain is the follow-up.*

Extends generation beyond binary taps to the full step vocabulary (mines excluded):

    per-panel symbol: 0=none  1=tap  2=hold-head  3=hold/roll-tail  4=roll-head

A typed chart is a `(T, 4)` int array over these 5 symbols. Holds/rolls are head→tail with
empty rows implied between (matches `.sm`). **Additive** — the frozen binary classifier path
is untouched; this is a parallel representation for the generator.

## What landed (foundation)

- `StepManiaParser.convert_to_tensor_typed()` — `.sm` → typed `(T,4)` int8 (lossless; mines/
  unknown chars → 0). Sits alongside the frozen binary `convert_to_tensor`.
- `sm_writer` — `typed=True` on `tensor_to_sm` / `charts_to_sm` writes symbols 0..4 verbatim.
- `src/generation/typed.py` — symbol constants + `onset_mask` / `symbol_histogram` helpers.
- Tests: typed writer→parser round-trip (taps, holds, rolls, jumps) exact; helpers. 16 pass.
- **Validated on real charts**: typed round-trip is exact including holds.

## Key data finding: rolls are absent; holds are learnable-but-rare

Scanned 410 external + 265 community charts:

| symbol | external | community |
|---|---|---|
| tap | 5.88% | 16.34% |
| hold-head | 0.300% | 0.609% |
| tail | 0.300% | 0.623% |
| **roll-head** | **0.000% (0/410)** | **0.000% (0/265)** |

- **Rolls (symbol 4) do not occur anywhere in the dataset** (0 of 675 charts). They stay in the
  schema for lossless round-trip, but the model will have **zero examples** — the effective
  generation vocabulary is **{none, tap, hold-head, tail}** (4 symbols).
- **Holds are well-represented** (in 380/410 external charts) but **rare per-cell** (~0.3–0.6%,
  roughly 1/20th the tap rate). hold-head and tail counts match closely (each hold = one head +
  one tail), a good consistency check.

## Implications for the typed model (next step)

- **Panel head → per-panel 5-way symbol** (none/tap/hold-head/tail/roll-head) instead of the
  current 15-way binary-pattern head. Onset head (binary: any panel non-empty) is unchanged.
- **Hold rarity needs handling**: hold-head/tail are ~20× rarer than taps, so a plain per-panel
  softmax will under-predict them — likely needs class weighting or focal loss on the panel head
  (same lesson as the onset head).
- **Structural validity**: every hold-head must get a matching tail later on the same panel.
  Free per-frame prediction can produce orphan heads/tails; may need a post-process pass (pair
  heads→tails, drop orphans) or a constraint at decode. Start simple (per-frame), measure orphan
  rate, add pairing if needed.
- Roll symbol: keep in the head for schema completeness; expect it never fires (no training signal).

Code: `src/generation/typed.py`, `StepManiaParser.convert_to_tensor_typed`, `sm_writer(typed=True)`.

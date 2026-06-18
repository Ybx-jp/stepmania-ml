# Hold-State-Aware Decoding (Phase 2.5 polish)

*2026-06-18. Branch `gen/hold-aware-decode`. Decode-time only — no retraining.*

The layered typed generator's type sampling was **stateless**: it sampled hold-heads and tails
independently per frame, so ~half of all hold symbols were orphans (head with no tail, or vice
versa), salvaged crudely by `pair_holds`. Holds had no coherent span. A human author opens a hold
on a sustained note and closes it where that sound ends.

## Hold automaton (in `LayeredTypedChartGenerator.generate(hold_aware=True)`)

Per panel, track an open/closed state during autoregressive decode:
- a **hold/roll-head opens** a hold; the panel is then *occupied* (emits nothing on
  intermediate frames);
- the hold **closes (tail)** at the **next frame the model places a note on that panel** — which
  is audio-driven, so the stop lands on a real musical event.

Heads come from the audio-conditioned type head; spans run head → next-note-on-panel. Guarantees
validity (no orphans) by construction. `tail-on-free` is demoted to a tap; a head dangling at song
end is demoted to a tap by `pair_holds`.

## Result (64 val songs, type_temperature 0.4; real holds: mean 7.5 / median 4 frames)

| decoder | onset_F1 | density | tap:hold | raw orphan% | mean hold len | median | crit_adj |
|---|---|---|---|---|---|---|---|
| stateless | 0.760 | 0.194 | 13.0:1 | 56% | 6.8 | 4 | 0.797 |
| **hold-aware** | **0.772** | 0.199 | 11.4:1 | **3%** | 12.5 | 4 | **0.859** |

## Conclusions

1. **Orphans 56% → 3%** — holds are now genuine spans, not random adjacent head/tail pairs. The
   residual 3% is heads dangling at song end (demoted to taps).
2. **crit_adj 0.797 → 0.859** — difficulty fidelity *improved*, approaching the hold-free ceiling
   (0.93): coherent holds read as more correct difficulty than scattered hold noise.
3. **onset_F1 up** (0.760 → 0.772); hold **median length 4 matches real**. Mean is higher (12.5 vs
   7.5) — some long sustained holds where notes on a panel are sparse; right-skewed but musical.
4. Spans are **audio-aligned**: both endpoints are model-placed (audio-driven) notes, so holds
   start and stop on real musical events, as intended.

## Recommended decode (typed generator)

Layered head, no type-head weighting (calibrated focal), **type sampling @ type_temperature ≈ 0.4,
`hold_aware=True`**. Best typed result: onset_F1 0.77, crit_adj 0.86, holds at ~11:1 with coherent
audio-aligned spans, always playable. Cheaper than retraining; pure decode-time.

(Also fixed in this branch: `typed.py` pattern helpers `NUM_PATTERNS/panels_to_pattern/...` were
never committed during the layered work, so merged main's `typed_model.py` import was broken;
restored here.)

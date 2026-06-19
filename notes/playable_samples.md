# Playable Sample Export (typed hold-aware generator)

*2026-06-18. `experiments/generation_typed/export_typed_samples.py`.*

Closes the loop from metrics to *playable charts*. Loads the layered checkpoint and, per val
song, generates a **full-length** (up to 1440-frame / 2-min, KV-cached) typed chart conditioned
on the real audio + difficulty, with **hold-aware decoding** so holds are coherent audio-aligned
spans. Writes a StepMania song folder: the original audio + a `chart.sm` containing the generated
chart (as "Challenge") and the original (for in-game A/B), both with hold/tail symbols.

Decode: per-chart onset threshold matched to that chart's real density; type sampling @
`type_temperature 0.4`; `hold_aware=True`; `pair_holds` → always valid.

## Verified (8 exported samples)

- Generated `.sm` re-parses cleanly; e.g. "Reach The Sky" Challenge: 211 steps, **28 holds = 28
  tails** (valid), audio present in the folder → loads & plays in StepMania.
- Density matches the source chart exactly (threshold matched per song).
- Holds scale with difficulty: Beginner charts get few/none (appropriate), denser charts get more.
- Critic reads generated difficulty on-target or adjacent.

## Usage

```
python experiments/generation_typed/export_typed_samples.py \
    --data_dir data/ --audio_dir data/ --out_dir outputs/typed_samples \
    --num_songs 8 --type_temperature 0.4
```

Output: `outputs/typed_samples/NN_<title>/` each with the song audio + `chart.sm`
(Challenge = generated, plus the original difficulty). Drop a folder into your StepMania
`Songs/` directory to play. (`outputs/typed_samples/` is gitignored — it contains copied audio.)

This is the qualitative ground truth the metric-driven development couldn't provide: load a folder
and actually play the generated chart.

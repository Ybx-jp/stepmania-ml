# BYO audio acquisition + assembly tooling (generate.py front-door)

**Session 2026-07-10.** Turned `scripts/generate.py` into a one-command path from a YouTube link to a playable,
multi-difficulty StepMania song. Tooling, not a model change — the generator/decoder are untouched. Pairs with
[[byo-audio-bpm-footgun]] (alignment gotchas) and the wired offset detector (`byo_offset_detection_findings.md`).

## What shipped (all on `scripts/generate.py`, plus a standalone CLI)
- **`--audio` accepts a URL** (YouTube or any yt-dlp-supported). Resolved at the top of `main()` to a local Vorbis
  `.ogg` via `src/data/youtube_audio.py::fetch_to_cache`, cached by **video id** under
  `~/.cache/stepmania-chart-gen/youtube/<id>.ogg` — so charting the same song at several difficulties (or a whole
  pack) re-downloads at most **once**. `#TITLE` defaults to the video title. Same logic exposed as
  `scripts/pull_audio.py` (standalone puller). Vorbis, NOT Opus — StepMania's decoder wants Vorbis-in-Ogg.
- **`--trim-audio START[,END]`** (`src/data/audio_slice.py`): chart only a time range. Timestamps are `SS`, `M:SS`,
  or `H:MM:SS`. `0:04` = from 4s to end; `0:04,2:14` = the interior. Runs **BEFORE** generation on the local file,
  so bpm/duration/offset-detection/features all see the clip. Sample-accurate ffmpeg cut (output-side `-ss` after
  `-i` + `-t` duration), cached per `(source, range)`. End clamps to real duration; start past EOF errors.
- **`--sm_difficulty {Beginner,Easy,Medium,Hard,Challenge,Edit}`**: the StepMania difficulty SLOT written to the
  `.sm`. **DEFAULT CHANGED**: now follows `--difficulty` (e.g. `Hard`) instead of the old hardcoded `Challenge`.
  Pass `--sm_difficulty Challenge` for the old behavior. Needed so multiple difficulties of one song land in
  DISTINCT slots instead of colliding as duplicate `Challenge` entries.
- **`--append_to CHART.sm`**: splice a generated difficulty into an existing song's `.sm` as another slot instead
  of writing a new song folder. Target owns `#OFFSET`/`#BPMS`/title/audio; we contribute one `#NOTES` block.
  **Grid guards** refuse a mismatch: bpm (different beat spacing) or rows-per-measure (different subdivision /
  `--features`). Backs the target up to `<name>.sm.bak`.

## ★ The load-bearing gotcha: playback `#OFFSET` ≠ generation beat-anchor
Hit live this session (a merged Beginner played a half-beat off the tuned Easy). The rule:

- `generate.py --offset X` does **two** things: sets `anchor = (-X) mod beat_period` (the within-beat phase the
  model places notes on, via the feature-extraction skip) **and** writes `#OFFSET = -anchor`. So `--offset` is a
  GENERATION-grid control, not just a playback tag.
- A `.sm` has **ONE `#OFFSET`, shared by all difficulties**. Therefore every difficulty of a song MUST be generated
  with the **same `--offset` (same beat-anchor)**, or the charts sit on grids up to a half-beat apart and cannot
  all be in sync under the one shared offset.
- Editing `#OFFSET` in the `.sm` by hand (a playback fix) does NOT re-place notes; it slides the whole chart. That
  can correct a half-beat-slipped chart (the onsets are still correctly *spaced*), but a newly generated sibling
  difficulty must be built on that SAME anchor to match — do not pass the hand-tuned playback offset as `--offset`
  for the sibling unless it equals the original generation anchor.

Concretely this session: Easy was generated at auto-anchor **0.291s**; the Beginner was first (wrongly) generated
with `--offset -0.03` (the user's hand-tuned playback `#OFFSET`) → anchor **0.03s** → **0.261s ≈ a half-beat @129
BPM** apart. Fix = regenerate the Beginner with `--offset -0.29149…` (match the Easy's grid), share `#OFFSET -0.03`.
`--append_to`'s grid guards catch bpm/subdivision mismatches but CANNOT detect a wrong anchor (the note rows are
beat-coordinates); the ℹ️ message reminds the user to match `--offset`.

## Dependencies (installed into the `stepmania-chart-gen` env this session)
`yt-dlp` (pip) + `ffmpeg` (conda-forge, has `libvorbis`) on PATH; **`deno`** (conda-forge) recommended — it's the
JS runtime yt-dlp uses to reach YouTube's full-quality audio (without it, the fallback client caps quality and 403s
more). YouTube audio tops out ~130 kbps Opus/AAC regardless (source limit; fine for onset/tempo features).

## Recommended multi-difficulty workflow (label-free, grid-safe)
1. Generate the first difficulty normally (note the `#OFFSET` it wrote, i.e. `-anchor`).
2. For each additional difficulty: same `--bpm` and same `--offset` (the first chart's anchor = `-#OFFSET`), a
   distinct `--sm_difficulty`, and `--append_to <the first chart.sm>`. No hand-splicing, no relabeling.

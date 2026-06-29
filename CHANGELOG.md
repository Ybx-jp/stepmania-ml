# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/). Versioning is semantic; `0.x` is pre-1.0,
so interfaces may still change.

## [0.1.0] — 2026-06-29

First tagged release. An audio-conditioned, autoregressive StepMania chart generator with a
factorized onset/panel/type head, trained controllability (groove-radar conditioning, CFG,
reference-chart style transfer), a decode-time biomechanical governor, and a learned taste critic.
See the README for the results table and the honest-limitations section; every headline number
traces to a write-up under `notes/`.

### Added
- **Bring-your-own-audio generation** — `scripts/generate.py --audio song.ogg --difficulty Hard`
  writes a playable `.sm` from a single audio file, no dataset required (BPM auto-estimated;
  optional `--style`/`--bpm`). Replicates the canonical decode path (42-dim highres features,
  manifold density target, governor default `fatigue_penalty=2`, mandatory playability).
- The fitted groove manifold (`cache/radar_manifold.npz`, 256 KB derived stats) now ships, so
  dataset-free generation works out of the box.
- `LICENSE` (MIT) and this changelog.
- Smoke tests for the new generator (`tests/test_generate_cli.py`).

### Changed
- **Packaging** migrated from `setup.py` to `pyproject.toml` (PEP 621); version pinned `0.1.0`;
  description corrected (the project is autoregressive + factorized, not "diffusion models").
- **README** 0.1.0 pass: added the governor as a controllability beat; recalibrated the
  song-structure and chaos framings to match the playtest evidence (the model *does* track
  structure; chaos is in-distribution-bounded, not a standing "smear" defect); fixed a broken
  `--radar` example (→ `--style`, the manifold path); made the demo/install promises honest.
- The dataset-bound `export_typed_samples.py` is documented as the A/B **evaluation harness**,
  distinct from the new bring-your-own-audio entrypoint.

### Fixed
- `StepManiaParser._validate_phase1_requirements` (typed `-> bool`) raised `ValueError` on a
  no-BPM chart instead of rejecting it; now returns `False`.
- Four stale-contract test failures (audio feature dim 13→23, sample-key subset check, fixture
  parser song-length window). Full suite now green.

### Repo hygiene
- No secrets or copyrighted audio/data/weights are tracked (verified). Untracked 3 build logs that
  embedded song-library paths; scrubbed a personal interpreter path from 10 scripts; cleared
  notebook outputs.

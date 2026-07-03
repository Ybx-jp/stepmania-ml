# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/). Versioning is semantic; `0.x` is pre-1.0,
so interfaces may still change.

## [0.2.0] — 2026-07-02

Decode-quality release. The default generated chart changes materially (and for the better) on
the same model weights: same song + seed now produces different, more human footwork. Two new
decode-time controllability knobs ship, both validated by playtest **and** an independent metric.
No retrain — `checkpoints/gen_motif_full_fixed` is unchanged; every change is in the decode path.

### Added
- **`footswitch` decode knob** (default **off**) — forbids footswitch footing, so same-panel runs
  must be one-foot jacks and the model **alternates** instead. A diagnostic A/B (`--ab_footswitch`)
  first revealed that the "brutal 16th voltage" was dominantly a *footswitch strategy*, not intrinsic
  jacks (forbidding it collapses same-panel runs 81–85%); forbidding it played "much better, more
  creative." Now the canonical default.
- **`hold_stream_penalty` family** (`hold_stream_penalty=8`, `hold_stream_floor=0.45`,
  `hold_stream_win=16`) — suppresses hold-heads in dense *stream* sections (where the type head
  otherwise opens a hold whose pinned foot forces a jack), gated on local onset density so sparse
  musical holds are untouched. Playtest-tuned ("just right" on japa1).
- **Graded realism critic** (`checkpoints/realism_critic_graded`, `train_graded_critic.py`) — a
  non-saturating taste instrument (graded corrupted-real ladders + within-song margin-ranking loss)
  that fixes the deployed binary critic's ~94%-railed saturation on Hard generations. Reusable
  evaluation asset.
- Shared-RNG (common-random-number) A/B paths in the exporter (`--ab_hold_stream`, `--ab_footswitch`)
  that isolate a decode-knob's effect from sampling noise.

### Changed
- **Default decode behavior** — `footswitch=False` + the `hold_stream_penalty` defaults are now part
  of the canonical palette (`decode_defaults.CANONICAL_DECODE`), so a bare `generate` / export
  produces alternating (not footswitched) same-panel runs and no holds-in-streams. This is the
  user-visible headline of the release.
- **Decode config single-source** — the canonical palette, tau pipeline, and model/feature loaders
  now live in one place (`decode_defaults.py` + `decode_harness.py`); both entry points and all probes
  import them, so `scripts/generate.py` can no longer silently drift from the exporter.

### Fixed
- **The fast-song quality defect.** Faster Hard songs generated materially worse charts (BPM→quality
  Spearman −0.68, family-wise p 0.004). The two new decode defaults resolve it: rerunning the denoised
  BPM→quality probe flattens the slope to +0.11 (n.s.) with all songs improved. Decomposition attributes
  the critic-measurable win to `footswitch=False`; `hold_stream_penalty`'s win is by-ear (it is
  presence-blind to the realism critic by construction). Full record in
  `notes/hold_in_stream_findings.md` / `notes/quality_feature_attribution_findings.md`.

### Diagnostics / internal
- BPM→quality attribution thread: localized the defect to the pattern/type head at high density
  (governor, training-coverage, and onset-head all ruled out); established the reliability/ICC-first
  method (a single generation's quality score is ~46% sample noise — denoise before attributing).
- Choreography distance-to-real quality instrument and the harness-loader refactor tiers.

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

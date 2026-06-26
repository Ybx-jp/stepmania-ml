# Release criteria

How this project decides a version is ready to ship. Written for the **v0.1.0** release
(first ever), but the structure is meant to be reused for later versions — copy the
checklist, re-set the bars.

## What a release here actually is

A "release" is **three separable artifacts**, shipped in order — not one big bang:

1. **Code release** — a git tag (`vX.Y.Z`) + `LICENSE` + a finalized `README`. The
   foundation everything else links to. Lowest-stakes, do this first.
2. **Model release** — weights public on Hugging Face with a model card. Higher-stakes:
   once weights + card are public and indexed, the *claims* in them are out there.
3. **Narrative release** — the long-form writeup / LinkedIn post that drives inbound.

`0.x.y` deliberately signals **"pre-1.0: usable, with known limits, interface may still
change."** That matches the project's honesty posture (improved, not solved). A 0.1.0 is a
"here's a real, working thing with documented limits" release — not "this is finished."

---

## v0.1.0 — resolved scope decisions

| Decision | Choice |
|---|---|
| **Primary deliverable** | **Code release first** — tag `v0.1.0` on the repo. HF weights + writeup follow as later steps (0.1.x / 0.2). |
| **License** | **MIT** (permissive, matches the model-card placeholder, fits the portfolio goal). |
| **Narrative** | Keep the **taste-critic thesis** as the README headline (the differentiated "evaluation for hard-to-measure quality" brand). Add the **biomechanical fatigue governor** as a short *controllability* beat — present, but qualitative ("playtest-confirmed as a tasteful edit"), not a metric. The governor does **not** displace the evaluation climax. |

Why the narrative split: the taste critic and the governor are two different stories — the
critic is the *evaluation* thesis (the climax), the governor is a *craft / controllability*
proof point. The governor extends the README's "Controllability" beat; it does not replace
the spine. Full governor write-up (its own findings doc → model-card update) is a 0.1.x item
that lands when HF weights ship.

---

## v0.1.0 — the checklist

A release is **0.1.0-ready** when every box is checked.

### Legal / licensing  *(hard blocker for anything public)*
- [x] `LICENSE` file (MIT, © 2026 Jackson Porter) committed at repo root.
- [ ] README + (eventual) model-card license line say MIT, matching the file.
- [x] Training data redistribution: charts sourced from freely-available community packs on
      zenius-i-vanisher and search.stepmania.online; no song **audio** is shipped (only the
      learned weights + the `.sm` charts the model produces). Low risk for a 0.1.0 code release.
      (Re-confirm pack licenses before the *weights* go public in step 2.)

### Correctness / tests
- [x] **Full suite green: 57/57** (`pytest tests/`). The 4 previously-failing tests were fixed
      (they were stale assertions pinned to older data contracts + one validator that raised
      instead of returning `False`): audio feature dim 13→23, sample-key check loosened to a
      subset, fixture parser given a relaxed song-length window, and
      `_validate_phase1_requirements` now rejects (not crashes) on a no-BPM chart.

### Code currency  *(narrative depends on this)*
- [x] **PR #41 merged** (origin/main @ `37257d1`) — the tagged code now contains the fatigue
      governor.

### Documentation / narrative
- [x] README carries the governor as a short controllability beat (no stale silence about a
      major subsystem the code contains).
- [x] README's "Live demo / Sample charts" promises are honest (no "coming soon" that points
      at nothing the reader can reach).
- [x] **README claims audit** published (`readme-0.1.0-audit.md`): every significant claim traced
      to a `notes/` finding, source file, or test. Fixed a broken `--radar` example (→ `--style`),
      recalibrated the song-structure + chaos framings, added the batch-of-4 KV-cache nuance.

### Repo hygiene / packaging
- [x] Git-tracking privacy audit: no secrets, no copyrighted data/audio/weights tracked. Untracked
      3 path-leaking build logs; scrubbed the personal interpreter path from 10 scripts; cleared
      notebook outputs.
- [x] Migrated `setup.py` → `pyproject.toml` (PEP 621, version `0.1.0`, MIT, accurate description).

### Honesty / claims  *(the project's brand discipline)*
- [ ] Every number in the README traces to a real `notes/*_findings.md` — verbs match the
      evidence (vouched ≠ measured ≠ mapped).
- [ ] No claims about un-playtested work (e.g. best-of-N reranking — built, not playtested).
- [ ] Governor claim stays qualitative ("a tasteful edit, playtest-confirmed"), not dressed
      as a percentage.

### Reproducibility
- [x] End-to-end generation works from the README: `scripts/generate.py --audio song.ogg
      --difficulty Hard` produces a `.sm` that re-parses cleanly (verified; smoke-tested in
      `tests/test_generate_cli.py`). Dataset-free — added this pass, because the only prior
      entrypoint required the full training dataset. Ships the 256 KB groove manifold.
- [x] `pip install -e .` verified against the new `pyproject.toml`.

### Packaging / deliverable hygiene
- [ ] `CHANGELOG.md` (or release notes) for v0.1.0.
- [ ] `marketing/` stays gitignored (personal-brand material, not part of the research repo).
- [ ] Annotated git tag `v0.1.0` on merged `main`.

---

## Execution order (once boxes are ready)

1. ~~Pick + commit `LICENSE` (MIT).~~ **done**
2. ~~Merge PR #41 → `main`.~~ **done** (origin/main @ `37257d1`)
3. ~~Triage the failing tests.~~ **done** — fixed all 4; suite is 57/57 green.
4. README governor beat + claims audit + honest demo promises.
5. `CHANGELOG.md` / release notes.
6. Clean-env reproducibility check.
7. Annotated tag `v0.1.0`; (optional) GitHub Release from the tag.

**Deferred to later versions (not blockers):** HF weights + model-card lineage reconciliation
(the card currently lists the older `gen_stage1/radar/style` lineage, not the deployed
`gen_motif_full_fixed`); the long-form writeup; the HF Space; the data-license check for
redistributing weights.

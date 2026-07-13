# Golden decode fingerprints

`decode_fingerprints.json` pins the **decoded output** of the bare canonical exporter run on three
val songs chosen to cover distinct decode regimes (see `tools/decode_fingerprint.py` for the spec —
that module is the single source for the cases, songs, and fingerprint format):

| Song | T (frames) | Pins |
|---|---|---|
| A Stupid Barber | 2624 | short control — universal onset window must be a no-op |
| Giudecca | 3577 | just under the W3600 window boundary — pins the edge |
| Dead Heat | 4080 | window + tail hangover FIRE |

Two cases: `v2_deployed` (the bare run = deployed config) and `v1_legacy` (same songs through the
16th-grid stack — enforces every "v1 / subdiv=4 byte-identical no-op" claim; the legacy gates
deterministically admit only 2 of the 3 songs, which is itself pinned).

- **Enforced by** `tests/test_decode_golden.py` (marker `golden`, ~4 min GPU; skip fast runs with
  `pytest -m "not golden"`). The stack is deterministic end-to-end, so any mismatch = a behavior change.
- **Intended change?** Regenerate and commit the goldens WITH the change: `python tools/bless_golden.py`.
  The .json diff in the PR then documents exactly which songs/charts moved and how.
- **Machine-bound**: byte-determinism holds per machine + torch version (`_meta` records the blessing
  environment). Re-bless after an environment change; the test skips on machines without the artifacts.
- **Validated 2026-07-13**: two independent runs byte-identical (determinism); an `--onset_window 0`
  mutation changed ONLY Dead Heat while both controls stayed byte-identical (sensitivity + specificity).

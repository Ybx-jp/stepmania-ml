# experiments/ — layout and conventions

Research code lives here; **production code lives in `src/` and `scripts/`** and must never import from
`experiments/`. The dependency direction is one-way: experiments import `src`, never the reverse.

## Directory map

| Directory | What it is |
|---|---|
| `generation_typed/` | The active generator lab: probes/diagnostics for the deployed `LayeredTypedChartGenerator`, plus the two **live** trainers (`train_motif_figure.py` = legacy v1, `train_motif_figure_v2.py` = deployed v2 48th-grid) and a compat shim for the canonical exporter (now `scripts/export_typed_samples.py`). |
| `probes/` | Standalone cross-cutting probes (quality attribution, BPM decomposition, backbone/flip-point, v2 grid checks). Formerly at repo root; they import each other as siblings. |
| `realism_critic/` | Realism/taste critic training + evaluation. |
| `generation_factorized/`, `generation_transformer/`, `generation_baselines/` | Earlier generator architectures (superseded; kept for lineage). |
| `autotune/`, `density_calibration/` | Throughput benchmarking / one-off calibration. |
| `archive/` | Retired scripts kept for checkpoint reproducibility. **Append-only; don't modernize.** |

## Conventions (enforced by `tools/check_repo_layout.py`)

1. **No `.py` files at repo root.** New probes go in `experiments/probes/` (cross-cutting) or the relevant
   experiment dir. Run them **from the repo root**: `python experiments/probes/probe_X.py`.
2. **Naming:** `probe_*` = answers one question, writes a findings note; `diag_*` = inspects an existing
   model/dataset; `calib_*` = fits a knob value; `train_*` = produces a checkpoint; `eval_*` = compares
   checkpoints/decodes.
3. **Results:** tabular/log outputs go to `outputs/probe_results/` (NOT `cache/` — cache is for regenerable
   feature caches and fitted artifacts only). Generated chart sets go to `outputs/<set_name>/`.
4. **Findings:** every probe that concludes something writes/updates a `notes/*_findings.md` and is linked
   from `notes/INDEX.md` (and the relevant `experiment_lineage/` arc file). A probe without a findings note
   is a dead end for the next reader.
5. **One canonical decode:** any probe that generates charts imports `src/generation/decode_defaults.py` /
   `decode_harness.py` — never hand-rolls conditioning, tau, or the palette (see the
   `conditioning-mechanics` and `generation-defaults` skills).
6. **Retirement:** when a lineage is superseded, `git mv` its scripts to `experiments/archive/<dir>/` and
   note the replacement in the archive README. Never leave two "current" versions side by side.

# Archived training scripts (generation_typed lineage)

These are the **retired version-by-copy training scripts** of the typed chart generator. Each was the
live trainer for one step of the model lineage; they are kept (with full git history via `git mv`) so any
historical checkpoint remains reproducible, but **none of them trains a deployed model anymore**.

The live trainers stayed in `experiments/generation_typed/`:

- `train_motif_figure.py` — trained the **legacy v1 deployed model** `checkpoints/gen_motif_full_fixed`
  (42-dim `highres`, 16th grid).
- `train_motif_figure_v2.py` — trained the **current deployed model** `checkpoints/gen_motif_v2_48th_cont`
  (42-dim `highres_v2`, 48th grid).

## Script → checkpoint map (approximate lineage order)

| Script | Checkpoint(s) | Era |
|---|---|---|
| `train_stage1.py` | `gen_stage1`, `gen_stage1_mirror` | staged-generation prototype |
| `train_typed.py` | `gen_typed`, `gen_typed_focal` | first typed (tap/hold) model |
| `train_layered.py` | `gen_layered` | onset/pattern/type layered split |
| `train_radar.py` | `gen_radar` | groove-radar conditioning |
| `train_style.py` | `gen_style` | 23-dim style-conditioned model |
| `train_highres.py` … `train_highres_v7.py` | `gen_highres` … `gen_highres_v7` | 42-dim highres feature sweeps |
| `train_motif.py`, `train_motif_hr.py` | `gen_motif`, `gen_motif_hr` | continuous-motif conditioning |
| `train_motif_local.py` | `gen_motif_local`, `gen_motif_local2` | per-section motif schedules |
| `train_motif_consolidated.py`, `train_motif_figure_standalone.py` | consolidation experiments | pre-`gen_motif_full` |

Do not "fix up" or modernize these scripts — their value is that they are byte-faithful to the run that
produced their checkpoint. If you need to retrain something, start from a live trainer.

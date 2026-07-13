# experiments/probes/ — cross-cutting probes

Standalone probes formerly at repo root. They import each other as siblings and expect to be run **from
the repo root** (`python experiments/probes/probe_X.py`); each inserts the repo root and this directory
onto `sys.path` itself. Results → `outputs/probe_results/`; conclusions → `notes/*_findings.md`.

Grouped by investigative arc (see `.claude/skills/experiment-design/experiment_lineage/INDEX.md`):

## Quality–feature attribution / BPM slope (notes/quality_feature_attribution_findings.md, hold_in_stream_findings.md)
`probe_quality_features.py`, `probe_quality_variance.py`, `probe_quality_choreo.py`,
`probe_bpm_governor_ablation.py`, `probe_bpm_head_decomp.py`, `probe_bpm_hold_decomp.py`,
`probe_bpm_holdfix_decomp.py`, `probe_onset_head_bpm.py`, `probe_train_bpm_coverage.py`,
`probe_holdburst_dynamics.py`, `probe_holdstream_fix.py`, `probe_stream_holdjack.py`

## Good-settings region / backbone tolerance (notes/goodregion_findings.md, backbone_phase_findings.md, tolerance_formula_findings.md)
`probe_backbone_phase.py`, `probe_backbone_tolerance.py`, `probe_goodregion_sweep.py`,
`probe_flip_point.py`, `probe_flip_secondfactor.py`, `probe_tolerance_audio_density.py`,
`probe_real_phase_reference.py`, `probe_sb_variants.py`

## Chaos onset gate — closed negative (notes/chaos_onset_gate_scope.md)
`probe_chaos_onset_gate.py`

## Meter / 48th-grid (data-layer-v2) (notes/meter_4_4_assumption_scope.md, data_layer_v2_scope.md)
`probe_meter_equivariant_sb.py`, `probe_v2_alignment.py`, `probe_v2_bpm_misalignment.py`,
`probe_v2_context_fit.py`, `probe_v2_displacement.py`, `probe_v2_grid_emptiness.py`

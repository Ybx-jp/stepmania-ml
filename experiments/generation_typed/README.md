# experiments/generation_typed/ — the active generator lab

Everything here targets the deployed `LayeredTypedChartGenerator` stack. Files that belonged to
**closed arcs** were archived to [`../archive/generation_typed/`](../archive/generation_typed/README.md)
(2026-07-12 reorg; `git mv`, history preserved — the historical `notes/*_findings.md` still describe them).
Conventions: [`../README.md`](../README.md). Arc histories:
`.claude/skills/experiment-design/experiment_lineage/INDEX.md`.

## Infrastructure (living)

| File | Role |
|---|---|
| `export_typed_samples.py` | **Compat shim** → the canonical exporter now at `scripts/export_typed_samples.py` |
| `train_motif_figure.py` / `train_motif_figure_v2.py` | The two live trainers (legacy v1 / deployed v2 48th) |
| `warm_cache_v2.py` | Parallel feature-cache warmer (see `tools/cache.py status`) |
| `fit_motif_basis.py` | Fits `cache/motif_basis.npz` (the 12-axis radar-orthogonal motif knob) |
| `diag_radar_manifold.py` | Groove-manifold fit/diagnostic (`cache/radar_manifold.npz`; refit is retrain-gated) |

## ★ Taste-critic / decode-fix arc (ACTIVE — lineage `taste-critic-arc.md`)

`probe_universal_window.py` + `probe_universal_window_decoded.py` (the shipped W3600 window),
`probe_onset_window_sweep.py`, `probe_stamina_longsong.py`, `probe_subtail_position.py` (tail collapse),
`probe_harm_offline.py`, `probe_harm_fills_middle.py`, `probe_density_by_section.py` (empty-middles),
`probe_critic_catches_defects.py`, `probe_lick_vs_byebye.py`,
`probe_score_personal.py`, `probe_offgrid_personal.py` (personal-set / critic-grid diagnostics).

## Governor measurement instruments (arc shipped; these are the canonical meters — cond-mech §8)

`diag_stamina.py`, `diag_stamina_holds.py`, `diag_stamina_arc.py`, `diag_breathe_energy.py`,
`calib_foot_fatigue.py`, `diag_ar_stability.py`.

## Seq-onset fork (parked-ALIVE; decode surface is head-specific — lineage `seq-onset-arc.md`)

`seqonset_decode.py` (the head-appropriate decode surface), `export_seqonset_ab.py`, `gen_train_c0.py`,
`probe_seqonset_*.py` (9), `probe_seqcontext_*.py` (5), `diag_seqcontext_probe.py`,
`probe_recon_audio.py` / `probe_recon_critic.py` (the analysis-by-synthesis wall),
`probe_phrasing_coherence.py`, `probe_boundary_snap.py`, `probe_figure_snap.py`.

## v2 / meter arc (deployed; lineage `meter-grid-arc.md`)

`analyze_v2_envelope.py` (safety-envelope sweep). Earlier v2 probes live in `../probes/` (`probe_v2_*.py`).

## Misc

`probe_song_similarity.py` — song-similarity tooling (good-settings / quality arcs).

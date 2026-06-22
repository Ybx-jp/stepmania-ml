"""
Groove-validated song selection for playtest exports.

A playtest set only tests a hypothesis if its songs actually exercise the relevant axis (B4U with 3 holds
can't test a hold fix). This ranks a dataset's songs by their REAL chart's groove radar so exports pick
songs that meaningfully read on the axis under test, and reports the profile so the set is auditable.

Radar dims (normalized 0-1 via GrooveRadar.to_vector): [stream, voltage, air, freeze, chaos]
  stream/voltage -> density, air -> jumps, freeze -> holds, chaos -> off-grid/varied rhythm.
"""
import numpy as np

RADAR_DIMS = ['stream', 'voltage', 'air', 'freeze', 'chaos']
_IDX = {d: i for i, d in enumerate(RADAR_DIMS)}
# "rich" = strong across the intensity dims (deja-loin-like); chaos excluded (it's the broken axis and
# high chaos != better). min() favors balanced-strong over spiky.
_RICH_DIMS = ['stream', 'voltage', 'air', 'freeze']


def _score(vec, by):
    if by == 'rich':
        return float(min(vec[_IDX[d]] for d in _RICH_DIMS))
    return float(vec[_IDX[by]])


def select_by_groove(ds, n=None, by='rich', difficulty=None, min_val=0.0):
    """Rank ds.valid_samples by groove radar; return a list of sample INDICES (best first), deduped by
    chart_file, optionally filtered to a difficulty class and a minimum score on the ranking axis.

    by: 'rich' (min over stream/voltage/air/freeze) or a single dim name in RADAR_DIMS.
    difficulty: optional class index 0-3 (Beginner..Hard) to require.
    Use the returned order in place of range(len(ds.valid_samples)) when exporting.
    """
    assert by == 'rich' or by in _IDX, f"by must be 'rich' or one of {RADAR_DIMS}"
    scored, seen = [], set()
    for i, meta in enumerate(ds.valid_samples):
        if meta['chart_file'] in seen:
            continue
        if difficulty is not None and meta['difficulty_class'] != difficulty:
            continue
        vec = meta['groove_radar'].to_vector()
        s = _score(vec, by)
        if s < min_val:
            continue
        seen.add(meta['chart_file'])
        scored.append((s, i, vec))
    scored.sort(key=lambda x: -x[0])
    if n is not None:
        scored = scored[:n]
    return [i for _, i, _ in scored]


def radar_table(ds, indices):
    """Return a printable report of the radar profile of the selected songs (for export logging)."""
    lines = [f"{'song':<30} {'diff':<8} " + " ".join(f"{d[:4]:>6}" for d in RADAR_DIMS)]
    lines.append("-" * (30 + 8 + 7 * len(RADAR_DIMS)))
    for i in indices:
        meta = ds.valid_samples[i]
        vec = meta['groove_radar'].to_vector()
        title = (meta['chart'].title or '')[:29]
        lines.append(f"{title:<30} {meta['difficulty_name'][:7]:<8} " + " ".join(f"{v:>6.2f}" for v in vec))
    return "\n".join(lines)

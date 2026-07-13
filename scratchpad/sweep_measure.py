#!/usr/bin/env python3
"""Measure the defect-#3 A/B across a sweep dir of exported folders (Challenge=baseline block 0, Edit=fix block 1).

Reports per song:  baseline defect -> fix defect  | holds  | notes | frames-under-hold  (over-release guard).
Aggregate:  total defect base vs fix; hold-count preservation (the fix must SHORTEN monster holds, not DELETE holds).
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'experiments' / 'realism_critic'))
from measure_defect import parse_sm, freefoot_stream_runs
from bipedal_metrics import pinned_mask
from choreography_metrics import note_starts

SWEEP = sys.argv[1] if len(sys.argv) > 1 else 'outputs/holdrelease_sweep'
GAP, MINRUN = 6, 4


def stats(typed):
    ns = note_starts(typed); pin = pinned_mask(typed)
    runs = freefoot_stream_runs(typed, GAP, MINRUN)
    return dict(defect=len(runs), notes=int(ns.sum()),
                heads=int(((typed == 2) | (typed == 4)).sum()),
                held_frames=int(pin.any(1).sum()))


folders = sorted(p for p in Path(SWEEP).iterdir() if p.is_dir() and (p / 'chart.sm').exists())
print(f"sweep: {SWEEP}  ({len(folders)} songs)\n")
hdr = f"{'song':30s} {'defect b->f':>12s} {'holds b->f':>12s} {'notes b->f':>13s} {'heldfr b->f':>14s}"
print(hdr); print('-' * len(hdr))
agg = {'db': 0, 'df': 0, 'hb': 0, 'hf': 0, 'over': 0, 'byte_id': 0, 'defect_songs': 0}
for f in folders:
    sm = str(f / 'chart.sm')
    try:
        b = stats(parse_sm(sm, block=0)); fx = stats(parse_sm(sm, block=1))
    except Exception as e:
        print(f"{f.name[:30]:30s}  ERROR {e}"); continue
    name = f.name[:30]
    print(f"{name:30s} {b['defect']:5d} -> {fx['defect']:<4d} {b['heads']:5d} -> {fx['heads']:<4d} "
          f"{b['notes']:6d} -> {fx['notes']:<4d} {b['held_frames']:6d} -> {fx['held_frames']:<5d}")
    agg['db'] += b['defect']; agg['df'] += fx['defect']
    agg['hb'] += b['heads']; agg['hf'] += fx['heads']
    if b['defect'] > 0:
        agg['defect_songs'] += 1
    # OVER-RELEASE guard: on a song with NO baseline defect, the fix should barely touch holds.
    if b['defect'] == 0 and (fx['heads'] < b['heads'] - 1 or fx['held_frames'] < b['held_frames'] * 0.9):
        agg['over'] += 1
        print(f"    ^ OVER-RELEASE? no baseline defect but holds/held-frames dropped notably")

print('-' * len(hdr))
removed = agg['db'] - agg['df']
verdict = 'ELIMINATED' if agg['df'] == 0 else f'{removed} removed'
print(f"\nAGGREGATE ({len(folders)} songs, {agg['defect_songs']} with a baseline defect):")
print(f"  total defect runs:  baseline {agg['db']}  ->  fix {agg['df']}   ({verdict})")
print(f"  total hold heads:   baseline {agg['hb']}  ->  fix {agg['hf']}   "
      f"(delta {agg['hf']-agg['hb']:+d} = holds SHORTENED not deleted if ~0)")
print(f"  over-release flags: {agg['over']}  (holds cut on a defect-free song -> should be 0)")

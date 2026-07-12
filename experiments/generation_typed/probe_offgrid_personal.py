#!/usr/bin/env python3
"""How much of the HAND-MADE personal charts' placement is INVISIBLE to a 16th-grid critic?

Parse each personal chart on the 48th grid (for_v2, timesteps_per_beat=12). Phase = t % 12.
  16th-aligned cells  = {0,3,6,9}   (the ONLY cells a 16th-grid critic can represent)
  triplet cells       = {2,4,8,10}  (8th/16th-triplet family)
  pure-48th cells      = {1,5,7,11}  (grid_snap's veto set)
The 16th critic floors everything onto {0,3,6,9}; notes on the other 8 cells COLLAPSE -> lost.
"""
import warnings, os, sys, glob
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
from pathlib import Path
import numpy as np
ROOT = Path('/home/ybx/code/stepmania-chart-generator'); sys.path.insert(0, str(ROOT))
from src.data.stepmania_parser import StepManiaParser

PERSONAL = Path('/home/ybx/sm-personal')
SIX = {0, 3, 6, 9}; TRIP = {2, 4, 8, 10}; P48 = {1, 5, 7, 11}


def main():
    p = StepManiaParser.for_v2(subdiv=12, min_song_length=30.0, max_song_length=600.0,
                               min_bpm=40.0, max_bpm=320.0, max_simultaneous=4, gimmick_max_bpm=400.0)
    sms = sorted(glob.glob(str(PERSONAL / '**' / '*.sm'), recursive=True))
    per_song = []; all_offgrid = []
    for smf in sms:
        try:
            ch = p.parse_file(smf)
            if ch is None or not ch.note_data: continue
        except Exception:
            continue
        for nd in ch.note_data:
            if not nd.difficulty_name: continue
            try:
                t = p.convert_to_tensor_typed(ch, nd)
            except Exception:
                continue
            present = ((t == 1) | (t == 2) | (t == 4))
            rows = np.where(present.any(1))[0]
            if len(rows) < 32: continue
            ph = rows % 12
            n = len(rows)
            f16 = np.isin(ph, list(SIX)).sum() / n
            ftr = np.isin(ph, list(TRIP)).sum() / n
            f48 = np.isin(ph, list(P48)).sum() / n
            offgrid = 1 - f16
            per_song.append((Path(smf).parent.name, nd.difficulty_name, n, f16, ftr, f48, offgrid))
            all_offgrid.append(offgrid)

    per_song.sort(key=lambda r: -r[6])
    print(f"{'song':30s} {'diff':10s} {'notes':>6s} {'16th%':>6s} {'trip%':>6s} {'p48%':>6s} {'OFFGRID%':>8s}")
    print("-" * 80)
    for s, d, n, f16, ftr, f48, og in per_song:
        print(f"{s[:30]:30s} {d[:10]:10s} {n:>6d} {f16*100:>5.1f}% {ftr*100:>5.1f}% {f48*100:>5.1f}% {og*100:>7.1f}%")
    og = np.array(all_offgrid)
    print("-" * 80)
    print(f"n={len(og)} charts | OFF-16th-grid (invisible to a 16th critic): "
          f"mean {og.mean()*100:.1f}%  median {np.median(og)*100:.1f}%  max {og.max()*100:.1f}%")
    print(f"charts with >5% off-grid notes: {(og>0.05).sum()}/{len(og)}   "
          f">15%: {(og>0.15).sum()}/{len(og)}")


if __name__ == '__main__':
    main()

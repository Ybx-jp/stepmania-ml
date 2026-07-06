#!/usr/bin/env python
"""PROTOTYPE: a METER-EQUIVARIANT strong-beat feature (generalize SB off the hard Z/4 grid).

Hypothesis (verified in code): the deployed SB = onset mass on the "strong" coset of Z/4 (simple meter,
strong = t%4∈{0,2}). For triplet/compound songs the accents land on the 16th "weak" slots and DEFLATE SB —
SB is measuring the wrong metrical group.

This probe builds the geometry and VALIDATES it with NO generation / NO model:
1. From RAW audio, a FINE onset envelope (128-hop ≈5.8ms) — the 16th-hop onset_env cannot resolve a triplet.
2. Fold onsets into within-beat PHASE (using the chart's #BPMS/#OFFSET) at 12 cells/beat = LCM(4,3).
   - duple 16th cells = {0,3,6,9}; the 16th-OFFBEAT (duple-exclusive) cells = {3,9}
   - triple cells      = {0,4,8}; the triplet-OFFBEAT (triple-exclusive) cells = {4,8}
   - shared: beat {0}, eighth {6}
3. METER DETECTOR: triple if mass{4,8} > mass{3,9}. Meter-equivariant SB projects onto the WINNING group's
   strong coset: duple → {0,6}/{0,3,6,9} (≡ current SB); triple → {0}/{0,4,8}.
4. VALIDATE against an INDEPENDENT ground truth: the chart's triplet-note fraction (parsed from the .sm rows).
   If the audio detector's triple-preference tracks the chart's triplet content, the geometry RESOLVES.
"""
import warnings; warnings.filterwarnings('ignore')
import re, glob, os, argparse
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent))
# The detector now lives in the shared module so the decode pipeline reuses the SAME validated statistic
# this probe exercises (single source of truth). This probe stays the VALIDATION reference for it.
from src.data.meter_detect import _read as read, chart_triplet_frac, phase_hist, strong_readings

def build_title_index():
    idx = {}
    for f in glob.glob('data/**/*.sm', recursive=True):
        m = re.search(r'#TITLE:([^;]*);', read(f))
        if m: idx.setdefault(m.group(1).strip(), f)
    return idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--songs', nargs='*', help='explicit #TITLE list; else a curated triplet-spanning set')
    args = ap.parse_args()
    idx = build_title_index()

    # curated validation set: span chart triplet_frac from ~0.66 down to 0
    curated = ['subconsciousness', 'First of the Year (Equinox)', 'First of the year', 'Ishukan Communication',
               'MEANING OF LIFE', 'And Then We Kiss (Junkie XL Mix)', 'LOVE', 'Take It To The Morning Light',
               'BUMBLE BEE', 'Abyss', 'ONE TWO (LITTLE BITCH)', 'Heart Attack', 'IN BETWEEN', 'Deja loin',
               'AFRONOVA', 'Grand Chariot', 'OH WORLD', 'Taylor Swift', 'Dead Heat']
    titles = args.songs or curated

    print(f"{'song':34}{'chart_tri':>10}{'meter':>8}{'triple_pref':>13}{'SB_duple':>10}{'SB_eq':>8}")
    rows = []
    for t in titles:
        f = idx.get(t) or next((idx[k] for k in idx if t.lower() in k.lower()), None)
        if not f: print(f"{t:34}{'(no .sm)':>10}"); continue
        txt = read(f); ctf = chart_triplet_frac(txt)
        h = phase_hist(f)
        if h is None: print(f"{t:34}{ctf:>10.2f}{'(no audio/bpm)':>21}"); continue
        r = strong_readings(h)
        rows.append((ctf, r))
        print(f"{t[:33]:34}{ctf:>10.2f}{r['meter']:>8}{r['triple_pref']:>+13.2f}{r['sb_duple']:>10.2f}{r['sb_eq']:>8.2f}")

    if len(rows) >= 5:
        ctf = np.array([x[0] for x in rows]); tp = np.array([x[1]['triple_pref'] for x in rows])
        sbd = np.array([x[1]['sb_duple'] for x in rows]); sbe = np.array([x[1]['sb_eq'] for x in rows])
        print(f"\nVALIDATION (n={len(rows)}):")
        print(f"  audio triple_pref  vs  chart triplet_frac:  Spearman {spearmanr(tp, ctf)[0]:+.3f}  "
              f"(the detector resolves if strongly +)")
        print(f"  SB_duple (current) vs chart triplet_frac:    Spearman {spearmanr(sbd, ctf)[0]:+.3f}  "
              f"(deflation: current SB DROPS as triplets rise if −)")
        # on triplet songs (ctf>0.15) how much does the equivariant SB lift the reading?
        hi = ctf > 0.15
        if hi.sum():
            print(f"  on triplet songs (chart_tri>0.15, n={int(hi.sum())}): "
                  f"SB_duple mean {sbd[hi].mean():.2f} -> SB_eq mean {sbe[hi].mean():.2f} "
                  f"(equivariant reading recovers strong-beat mass the 16th grid buried)")

if __name__ == '__main__':
    main()

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
from fractions import Fraction
import numpy as np
from scipy.stats import spearmanr

SR, HOP = 22050, 128

def read(f):
    try: return open(f, encoding='utf-8', errors='ignore').read()
    except Exception: return ''

def build_title_index():
    idx = {}
    for f in glob.glob('data/**/*.sm', recursive=True):
        m = re.search(r'#TITLE:([^;]*);', read(f))
        if m: idx.setdefault(m.group(1).strip(), f)
    return idx

def chart_triplet_frac(txt):
    best = 0.0
    for m in re.finditer(r'#NOTES:(.*?);', txt, re.S):
        parts = m.group(1).split(':')
        if len(parts) < 6 or 'dance-single' not in parts[0]: continue
        tri = binv = 0
        for meas in parts[5].split(','):
            lines = [ln for ln in meas.splitlines() if ln.strip() and not ln.strip().startswith('//')]
            L = len(lines)
            for i, ln in enumerate(lines):
                if L and re.search(r'[124]', ln):
                    d = Fraction(i, L).denominator
                    tri += (d % 3 == 0); binv += (d % 3 != 0)
        if tri + binv > 50: best = max(best, tri / (tri + binv))
    return best

def bpm_map(txt):
    """Full #BPMS map -> (start_beats, start_times, bpms) for EXACT drift-free beat<->time.
    (Using only the first BPM over a multi-minute song drifts the phase into noise — the harness bug.)"""
    m = re.search(r'#BPMS:([^;]*);', txt)
    if not m or not m.group(1).strip(): return None
    segs = []
    for s in m.group(1).split(','):
        try:
            b, v = s.split('='); segs.append((float(b), float(v)))
        except Exception: pass
    segs = [s for s in sorted(segs) if s[1] > 0]
    if not segs: return None
    beats = np.array([s[0] for s in segs]); bpms = np.array([s[1] for s in segs])
    # cumulative audio time at each segment start (beat 0 anchored at t=0; global phase is irrelevant to meter)
    times = np.zeros(len(segs))
    for i in range(1, len(segs)):
        times[i] = times[i-1] + (beats[i] - beats[i-1]) * 60.0 / bpms[i-1]
    return beats, times, bpms

def time_to_beat(t, bm):
    beats, times, bpms = bm
    seg = np.searchsorted(times, t, side='right') - 1
    seg = np.clip(seg, 0, len(beats) - 1)
    return beats[seg] + (t - times[seg]) * bpms[seg] / 60.0

def get_offset(txt):
    m = re.search(r'#OFFSET:([^;]*);', txt)
    try: return float(m.group(1)) if m else 0.0
    except Exception: return 0.0

def music_path(sm_file, txt):
    m = re.search(r'#MUSIC:([^;]*);', txt)
    if not m or not m.group(1).strip(): return None
    p = os.path.join(os.path.dirname(sm_file), m.group(1).strip())
    return p if os.path.exists(p) else None

def phase_hist(sm_file):
    """12-cell within-beat onset-mass histogram from the fine onset envelope."""
    import librosa
    txt = read(sm_file); bm = bpm_map(txt); mus = music_path(sm_file, txt)
    if bm is None or not mus: return None
    y, sr = librosa.load(mus, sr=SR, mono=True)
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    t = np.arange(len(env)) * HOP / sr                      # frame times (s)
    beat = time_to_beat(t, bm)                              # EXACT beat via full BPM map (drift-free)
    phase = np.mod(beat, 1.0)                               # within-beat phase [0,1)
    cell = np.mod(np.round(phase * 12).astype(int), 12)     # nearest of 12 cells
    h = np.zeros(12)
    for c, e in zip(cell, env): h[c] += e
    return h

def strong_readings(h):
    """ROTATION-INVARIANT meter detection via the DFT of the within-beat onset histogram.
    Bin k of a 12-pt DFT = k cycles/beat: duple structure at {2 (eighth),4 (16th)}, triple at {3,6 (sextuplet)}.
    Magnitudes ignore absolute phase, so #OFFSET is irrelevant. The downbeat is recovered from bin-1's phase."""
    H = np.fft.rfft(h)                                   # bins 0..6
    mag = np.abs(H)
    duple = mag[2] + mag[4]; triple = mag[3] + mag[6]
    tp = (triple - duple) / (triple + duple + 1e-9)      # triple-preference in [-1,1]
    meter = 'triple' if tp > 0 else 'duple'
    # align the downbeat to cell 0 using the fundamental's phase, then read SB on the correct grid
    shift = int(np.round(np.angle(H[1]) / (2 * np.pi) * 12)) % 12
    ha = np.roll(h, -shift)
    sb_duple = (ha[0] + ha[6]) / (ha[0] + ha[3] + ha[6] + ha[9] + 1e-9)   # ≡ current SB (fine env, duple grid)
    sb_triple = ha[0] / (ha[0] + ha[4] + ha[8] + 1e-9)
    sb_eq = sb_triple if meter == 'triple' else sb_duple
    return dict(meter=meter, sb_duple=sb_duple, sb_triple=sb_triple, sb_eq=sb_eq, triple_pref=tp)

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

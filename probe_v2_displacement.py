"""data-layer-v2 phase-2 success criterion: does the 48th ROUND grid collapse the triplet DISPLACEMENT?

The meter thread measured floor-to-16th displacement (chart-triplet vs error rho+0.83, up to 0.083 beat / 33 ms).
This probe re-measures displacement under BOTH grids on triplet-rich charts and confirms v2 (round, subdiv=12)
drives triplet-note displacement to ~0. Chart-parse only; independent of audio/model.

displacement(note) = | true_beat - quantized_beat |, quantized_beat = round_or_floor(true_beat*subdiv)/subdiv.
"""
import re, glob
from fractions import Fraction
import numpy as np

FILES = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)


def chart_notes(txt):
    """Yield (true_beat, is_triplet) for every note in the densest dance-single chart."""
    best, best_tot = None, 0
    for m in re.finditer(r'#NOTES:(.*?);', txt, re.S):
        parts = m.group(1).split(':')
        if len(parts) < 6 or 'dance-single' not in parts[0]:
            continue
        notes, cur = [], 0.0
        for meas in parts[5].split(','):
            lines = [ln for ln in meas.splitlines() if ln.strip() and not ln.strip().startswith('//')]
            L = len(lines)
            if not L:
                cur += 4.0; continue
            bpl = 4.0 / L
            for i, ln in enumerate(lines):
                if re.search(r'[124]', ln):
                    beat = cur + i * bpl
                    notes.append((beat, Fraction(i, L).denominator % 3 == 0))
            cur += 4.0
        if len(notes) > best_tot:
            best, best_tot = notes, len(notes)
    return best or []


def disp(beat, subdiv, rnd):
    q = (np.round if rnd else np.floor)(beat * subdiv) / subdiv
    return abs(beat - q)


rows = []
for f in FILES:
    try:
        with open(f, encoding='utf-8', errors='ignore') as fh:
            notes = chart_notes(fh.read())
    except Exception:
        continue
    if len(notes) < 50:
        continue
    tri = np.array([n[1] for n in notes])
    if tri.sum() == 0:
        tf = 0.0
    else:
        tf = tri.mean()
    beats = np.array([n[0] for n in notes])
    d_floor4 = np.array([disp(b, 4, False) for b in beats])   # deployed grid
    d_round12 = np.array([disp(b, 12, True) for b in beats])  # v2 grid
    rows.append((tf, tri, d_floor4, d_round12))

tf = np.array([r[0] for r in rows])
# corpus-wide, triplet notes only
tri_all = np.concatenate([r[1] for r in rows])
f4_all = np.concatenate([r[2] for r in rows])
r12_all = np.concatenate([r[3] for r in rows])
tri_mask = tri_all.astype(bool)

print(f"charts: {len(rows)} | total notes: {len(tri_all):,} | triplet-family notes: {tri_mask.sum():,}")
print(f"\n--- DISPLACEMENT on TRIPLET-family notes (beats) ---")
print(f"  deployed floor@16th : mean {f4_all[tri_mask].mean():.4f}  max {f4_all[tri_mask].max():.4f}")
print(f"  v2 round@48th       : mean {r12_all[tri_mask].mean():.4f}  max {r12_all[tri_mask].max():.4f}")
print(f"  -> @150 BPM (0.4 s/beat): {f4_all[tri_mask].mean()*400:.1f} ms  ->  {r12_all[tri_mask].mean()*400:.1f} ms")

print(f"\n--- DISPLACEMENT on ALL notes ---")
print(f"  deployed floor@16th : mean {f4_all.mean():.5f}")
print(f"  v2 round@48th       : mean {r12_all.mean():.5f}")

# per-song mean displacement vs triplet fraction (the rho+0.83 relationship the meter thread found)
song_tf = tf
song_d4 = np.array([r[2].mean() for r in rows])
song_d12 = np.array([r[3].mean() for r in rows])
def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]
print(f"\n--- per-song displacement vs triplet_frac (Spearman) ---")
print(f"  deployed floor@16th : rho {spearman(song_tf, song_d4):+.3f}  (meter thread: +0.83 — triplet songs limp)")
print(f"  v2 round@48th       : rho {spearman(song_tf, song_d12):+.3f}  (should collapse toward 0)")
struct = song_tf >= 0.15
print(f"\n  structural triplet songs (tf>=0.15, n={struct.sum()}): "
      f"mean displacement {song_d4[struct].mean():.4f} -> {song_d12[struct].mean():.4f} beats "
      f"({song_d4[struct].mean()*400:.1f} -> {song_d12[struct].mean()*400:.1f} ms @150BPM)")

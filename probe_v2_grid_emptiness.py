"""data-layer-v2 emptiness check: under the UNIFORM 48th grid (A1), how much of the 3x context does the
corpus actually USE? Quantifies the A1 waste (every song pays 3x frames) vs payload (triplet-family notes
that the 16th grid mis-places and the 48th grid fixes). Chart-parse only, no generation.

A note is triplet-family (needs the finer grid) iff its reduced within-measure denominator is divisible by 3
(the validated classifier from probe_meter_equivariant_sb.chart_triplet_frac). On the 48th grid such a note
lands on a NEW cell (mod 3 != 0); a duple note stays on the old 16th sub-lattice (mod 3 == 0).
"""
import re, glob
from fractions import Fraction
import numpy as np

FILES = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)

def song_stats(txt):
    """Return (total_notes, triplet_notes, occ_cells_48, total_cells_48) over the DENSEST dance-single chart."""
    best = None
    for m in re.finditer(r'#NOTES:(.*?);', txt, re.S):
        parts = m.group(1).split(':')
        if len(parts) < 6 or 'dance-single' not in parts[0]:
            continue
        tot = tri = occ = cells = 0
        for meas in parts[5].split(','):
            lines = [ln for ln in meas.splitlines() if ln.strip() and not ln.strip().startswith('//')]
            L = len(lines)
            if not L:
                continue
            cells += 48                       # a 4/4 measure = 48 forty-eighth cells
            for i, ln in enumerate(lines):
                if re.search(r'[124]', ln):   # a tap/hold-head/roll present on this row
                    tot += 1
                    occ += 1
                    if Fraction(i, L).denominator % 3 == 0:
                        tri += 1
        if tot > 50 and (best is None or tot > best[0]):
            best = (tot, tri, occ, cells)
    return best

rows = []
for f in FILES:
    try:
        with open(f, encoding='utf-8', errors='ignore') as fh:
            s = song_stats(fh.read())
        if s:
            rows.append(s)
    except Exception:
        pass

tot = np.array([r[0] for r in rows]); tri = np.array([r[1] for r in rows])
occ = np.array([r[2] for r in rows]); cells = np.array([r[3] for r in rows])
frac = tri / tot                                   # per-song triplet-family note fraction
n = len(rows)

print(f"charts with a usable dance-single chart: {n}")
print(f"\n--- PAYLOAD (what the 48th grid buys) ---")
print(f"corpus-wide notes: {tot.sum():,} | triplet-family: {tri.sum():,} "
      f"= {100*tri.sum()/tot.sum():.2f}% of all notes land on NEW (triplet) cells")
print(f"per-song triplet-family fraction: mean {frac.mean()*100:.2f}%  median {np.median(frac)*100:.2f}%")

print(f"\n--- WASTE distribution (how many songs gain ~nothing from 3x context) ---")
for lo, hi, lbl in [(0, 1e-9, 'ZERO triplet notes'), (1e-9, 0.02, '0-2% (trace)'),
                    (0.02, 0.15, '2-15% (fills)'), (0.15, 1.01, '>=15% (structural)')]:
    k = ((frac >= lo) & (frac < hi)).sum() if lbl != 'ZERO triplet notes' else (frac == 0).sum()
    print(f"  {lbl:>22}: {k:>5} songs ({100*k/n:.1f}%)")

print(f"\n--- OCCUPANCY (grid sparsity, context we pay for either way) ---")
print(f"mean 48th-cell occupancy: {100*(occ/cells).mean():.2f}%  "
      f"(16th-grid equiv ~{100*(occ/cells).mean()*3:.1f}% — 48th triples the empty cells uniformly)")
print(f"\n--- COST/BENEFIT one-liner ---")
zero = (frac == 0).sum()
print(f"A1 pays 3x context on ALL {n} songs to correctly place the {100*tri.sum()/tot.sum():.1f}% of notes "
      f"that are triplet-family; {100*zero/n:.0f}% of songs have ZERO such notes (pure waste), "
      f"{100*(frac>=0.15).sum()/n:.0f}% are structural beneficiaries.")

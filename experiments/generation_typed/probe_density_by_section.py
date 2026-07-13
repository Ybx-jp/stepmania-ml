#!/usr/bin/env python3
"""Density-by-song-position for the 3-arm stamina probe vs the REAL hand-made chart (Rule 5 reference).

Reads .sm #NOTES blocks DIRECTLY (measure = a comma-separated block; a note = a row containing 1/2/4), so it
needs no audio and works uniformly on generated + real charts. Bins measures into position bins, normalizes each
profile to unit mean (compare SHAPE — where the song gets busy/sparse — not absolute difficulty), renders a
sparkline, and scores each arm's shape match to REAL (correlation + L1). The question: does the deployed GLOBAL
arm starve a section that LOCAL (the fix) keeps, and which best matches the human's section shape?
"""
import sys, re, glob
from pathlib import Path
import numpy as np

BARS = "▁▂▃▄▅▆▇█"
NBINS = 24
PERSONAL = Path('/home/ybx/sm-personal')
PROBE = Path('/home/ybx/sm-generated/stamina_probe')
# song -> its real .sm (user's hand-made)
REAL_SM = {
    'Calling': PERSONAL / "Yb's Home Cooked/Calling (Lose My Mind)",
    'Switch':  PERSONAL / "Hardcore Xtreme/Switch",
    'Bye Bye': PERSONAL / "Hardcore Xtreme/Bye Bye",
}


def note_blocks(sm_text):
    """Yield the note-grid string of each #NOTES/#NOTEDATA block (the part after the 5 colon-fields)."""
    txt = re.sub(r'//.*', '', sm_text)
    for m in re.finditer(r'#NOTES\s*:(.*?);', txt, re.DOTALL | re.IGNORECASE):
        body = m.group(1)
        parts = body.split(':')
        if len(parts) < 6:      # dance-single : author : difficulty : meter : radar : <grid>
            continue
        yield parts[-1]


def measures_density(grid):
    """grid -> per-measure note count + per-measure row count (measures split on ',')."""
    out = []
    for meas in grid.split(','):
        rows = [r.strip() for r in meas.strip().splitlines() if r.strip()]
        rows = [r for r in rows if re.fullmatch(r'[0-9MLF]{3,8}', r)]   # panel rows only
        if not rows:
            continue
        notes = sum(1 for r in rows if any(c in '124' for c in r))
        out.append((notes, len(rows)))
    return out


def profile_from_sm(sm_path, nbins=NBINS, pick='densest'):
    txt = Path(sm_path).read_text(encoding='utf-8', errors='ignore')
    blocks = list(note_blocks(txt))
    if not blocks:
        return None
    cand = [measures_density(g) for g in blocks]
    cand = [c for c in cand if c]
    if not cand:
        return None
    md = max(cand, key=lambda c: sum(n for n, _ in c)) if pick == 'densest' else cand[0]
    # per-measure note density (notes / rows), then resample measures -> nbins position bins
    dens = np.array([n / max(r, 1) for n, r in md], dtype=float)
    M = len(dens)
    binned = np.array([dens[int(i * M / nbins):max(int((i + 1) * M / nbins), int(i * M / nbins) + 1)].mean()
                       for i in range(nbins)])
    total = sum(n for n, _ in md)
    return binned, total, M


def spark(profile):
    p = profile - profile.min()
    p = p / (p.max() + 1e-9)
    return "".join(BARS[min(int(v * (len(BARS) - 1) + 0.5), len(BARS) - 1)] for v in p)


def norm_mean(p):
    return p / (p.mean() + 1e-9)


def find_real_sm(folder):
    g = sorted(glob.glob(str(Path(folder) / '*.sm')))
    return g[0] if g else None


def main():
    for song, realdir in REAL_SM.items():
        print("\n" + "=" * 92); print(f"  {song}"); print("=" * 92)
        rp = find_real_sm(realdir)
        real = profile_from_sm(rp) if rp else None
        profs = {}
        if real:
            profs['REAL  '] = real
        for arm in ('OFF', 'GLOBAL', 'LOCAL'):
            smp = PROBE / f"{song} {arm}" / "chart.sm"
            if smp.is_file():
                pr = profile_from_sm(smp)
                if pr: profs[f'{arm:6s}'] = pr
        if not profs:
            print("  (no charts)"); continue
        refN = norm_mean(profs['REAL  '][0]) if 'REAL  ' in profs else None
        print(f"  {'arm':7s} {'notes':>6s} {'meas':>5s}  density-by-position (each row normalized to its own mean)  "
              f"{'corr→REAL':>10s} {'L1→REAL':>8s}")
        for name, (prof, total, M) in profs.items():
            nm = norm_mean(prof)
            if refN is not None and name.strip() != 'REAL':
                c = float(np.corrcoef(nm, refN)[0, 1]); l1 = float(np.abs(nm - refN).mean())
                sc = f"{c:>+10.3f} {l1:>8.3f}"
            else:
                sc = f"{'—':>10s} {'—':>8s}"
            print(f"  {name} {total:>6d} {M:>5d}  {spark(prof)}  {sc}")


if __name__ == '__main__':
    main()

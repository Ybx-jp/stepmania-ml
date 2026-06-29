#!/usr/bin/env python3
"""PROBE: FIGURE-CHARACTER snap at section boundaries (2026-06-28).
Q: does the FIGURE-FAMILY character (jack/stream/trill/candle/jump/step mix) change crisply AT a section
   boundary, or bleed across? Density-ISOLATED (fractions over named W=3 figures, conditional on onsets) so it
   can't be the density-cascade H11 flagged. STEP 1 = REAL reference only (no generation): do real charts even
   snap figure-character at Foote boundaries? If not, the 'crisp figure structure' premise is moot.
"""
import warnings, os; warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import sys; sys.path.insert(0, "experiments/generation_typed")
import numpy as np
from probe_phrasing_coherence import load_songs
from diag_transitions_freerun import foote_boundaries, SSM_DIMS
from src.generation.motif_codebook import onset_tokens, name_figure, _canon
from src.utils.reproducibility import set_seed

FAMS = ["jack", "sweep/staircase", "trill", "candle/cross", "jump/bracket", "step"]
FI = {f: i for i, f in enumerate(FAMS)}
WSEG = 64   # frames per side of a transition window
W = 3

def char_vec(typed_seg):
    """(T,4) -> 6-dim figure-FAMILY fraction (density-isolated; over named W=3 windows). None if too few onsets."""
    toks = onset_tokens(typed_seg)
    if len(toks) < W + 3:
        return None
    v = np.zeros(6)
    for j in range(len(toks) - W + 1):
        v[FI[name_figure(_canon(toks[j:j + W]))]] += 1
    s = v.sum()
    return v / s if s > 0 else None

def mag(typed, c):
    a, b = char_vec(typed[c - WSEG:c]), char_vec(typed[c:c + WSEG])
    return None if a is None or b is None else float(np.abs(a - b).sum())

def responsiveness(typed, bnds, rng, T):
    bm = [m for b in bnds for m in [mag(typed, int(b))] if m is not None]
    rand = rng.integers(WSEG, T - WSEG, size=max(len(bnds) * 4, 40))
    rm = [m for c in rand for m in [mag(typed, int(c))] if m is not None]
    if not bm or not rm:
        return None
    return dict(at_bnd=float(np.mean(bm)), at_rand=float(np.mean(rm)),
                resp=float(np.mean(bm) - np.mean(rm)), n=len(bm))

def main():
    set_seed(42); rng = np.random.default_rng(42)
    match = "high school love,kneeso,deja loin,dancing lovers,pound the alarm,taylor swift,in between,first of the year"
    songs = load_songs(match, 1440)
    print(f"\nFIGURE-CHARACTER SNAP @ Foote boundaries — REAL charts ({len(songs)} Hard songs)")
    print("character = density-isolated figure-FAMILY fractions; resp = |Δchar|@boundary − @random (>0 = snaps)\n")
    print(f"{'song':<28} {'@bnd':>7} {'@rand':>7} {'resp':>7}  n")
    rs = []
    for s in songs:
        T = s['T']
        bnds = foote_boundaries(s['audio'][:, SSM_DIMS])
        r = responsiveness(s['real'], bnds, rng, T)
        if r:
            rs.append(r['resp'])
            print(f"{s['title'][:27]:<28} {r['at_bnd']:>7.3f} {r['at_rand']:>7.3f} {r['resp']:>+7.3f}  {r['n']}")
    print(f"\nMEDIAN resp = {np.median(rs):+.3f}  (mean {np.mean(rs):+.3f}; n_songs {len(rs)})")
    print("read: resp >> 0 → real DOES snap figure-character at boundaries (premise holds; generate model & compare)")
    print("      resp ≈ 0  → real does NOT change figure-family at section boundaries (premise moot; redirect)")

if __name__ == "__main__":
    main()

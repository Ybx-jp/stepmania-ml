#!/usr/bin/env python3
"""PROBE: BOUNDARY-SNAP decomposition (no-retrain, no-generation, 2026-06-28).
Q: is the onset head's WIDE/LAGGY density step at section boundaries a READOUT problem (raw p_onset is sharp,
   tau/allocation smears it) or a REPRESENTATION problem (the head's raw output is itself gradual)?
Method: at each Foote audio boundary, measure step WIDTH + LAG of three signals under IDENTICAL smoothing:
   p_onset (head's raw posterior) · realized (p>tau) · real chart density.  Compare to real as the reference.
"""
import warnings, os; warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import sys; from pathlib import Path
sys.path.insert(0, "experiments/generation_typed")
import numpy as np, torch
from probe_phrasing_coherence import (load_songs, calibrated_p_onset, axis1_boundary_snap, boxsmooth, I_ENERGY)
from diag_transitions_freerun import foote_boundaries, SSM_DIMS
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.playability_metrics import ACTIVE_SYMBOLS
from src.utils.reproducibility import set_seed

def summ(d):
    return f"w{d['width']:>4.0f}f lag{d['lag']:>+4.0f}f" if d else "   (none)"

def main():
    set_seed(42); dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    match = "high school love,kneeso,deja loin,dancing lovers,pound the alarm,taylor swift,in between,first of the year"
    songs = load_songs(match, 1440)
    m = LayeredTypedChartGenerator(audio_dim=songs[0]['audio'].shape[1], d_model=128, num_layers=4, onset_layers=2).to(dev)
    m.load_state_dict(torch.load("checkpoints/gen_motif_full_fixed/best_val.pt", map_location=dev)['model_state_dict']); m.eval()
    print(f"\nBOUNDARY-SNAP DECOMP  ({len(songs)} Hard songs; per-boundary WIDTH+LAG of the density step, box16 unless noted)")
    print("readout-vs-representation: praw(b4)=head raw sharpness | praw(b16)=head envelope | realized=after tau | REAL=reference\n")
    print(f"{'song':<26} {'praw_b4':>15} {'praw_b16':>15} {'realized_b16':>15} {'REAL_b16':>15}  bnds")
    agg = {k: [] for k in ('pb4','pb16','rz','re')}
    for s in songs:
        T = s['T']; feat = s['audio']
        audio = torch.from_numpy(feat).unsqueeze(0).to(dev); diff = torch.tensor([s['diff']], device=dev)
        p = calibrated_p_onset(m, audio, diff, dev)
        real_d = float((np.isin(s['real'], ACTIVE_SYMBOLS)).any(1).mean())
        tau = float(np.quantile(p, 1 - real_d)) if real_d > 0 else 0.5
        realized = (p > tau).astype(float)
        real_dens = np.isin(s['real'], ACTIVE_SYMBOLS).any(1).astype(float)
        bnds = foote_boundaries(feat[:, SSM_DIMS])
        a_pb4  = axis1_boundary_snap(boxsmooth(p, 4),  bnds)
        a_pb16 = axis1_boundary_snap(boxsmooth(p, 16), bnds)
        a_rz   = axis1_boundary_snap(boxsmooth(realized, 16), bnds)
        a_re   = axis1_boundary_snap(boxsmooth(real_dens, 16), bnds)
        for k, a in (('pb4',a_pb4),('pb16',a_pb16),('rz',a_rz),('re',a_re)):
            if a: agg[k].append((a['width'], a['lag']))
        print(f"{s['title'][:25]:<26} {summ(a_pb4):>15} {summ(a_pb16):>15} {summ(a_rz):>15} {summ(a_re):>15}  {len(bnds)}")
    print(f"\n{'MEDIAN':<26}", end="")
    for k in ('pb4','pb16','rz','re'):
        v = np.array(agg[k]); 
        print(f" {('w%.0ff lag%+.0ff'%(np.median(v[:,0]), np.median(v[:,1]))):>15}", end="")
    print("\n\nread: praw_b16 vs REAL_b16 width  -> REPRESENTATION gap if model >> real (head intrinsically wide);")
    print("      realized vs praw_b16          -> READOUT/tau adds smear if realized >> praw;")
    print("      lag(praw) vs lag(REAL)        -> does the HEAD respond late, or is it a Foote-boundary offset?")

if __name__ == "__main__":
    main()

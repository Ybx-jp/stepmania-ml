#!/usr/bin/env python3
"""OFFLINE half of the harm_calib #2 A/B: did harm_calib REDISTRIBUTE notes INTO the gated quiet regions on Bye Bye,
and do the TOTAL (dim-0 energy) vs PERC (dim-35 perc-onset) gates target DIFFERENT sections?

harm_calib is density-preserving (tau holds total), so the test is density INSIDE the gate vs outside, per arm."""
import warnings, os, sys, glob
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
from pathlib import Path
import numpy as np
ROOT = Path('/home/ybx/code/stepmania-chart-generator'); sys.path.insert(0, str(ROOT))
from src.data.stepmania_parser import StepManiaParser
from src.generation.decode_harness import make_feature_extractor

PROBE = Path('/home/ybx/sm-generated/stamina_probe')
SUBDIV = 12; QUIET_Q = 40.0
ARMS = ['GLOBAL', 'LOCAL', 'HARM-TOTAL', 'HARM-PERC']


def to_binary(t):
    t = np.asarray(t); return ((t == 1) | (t == 2) | (t == 4)).astype(np.float32)


def boxsmooth01(x, w=16):
    x = (x - x.min()) / (np.ptp(x) + 1e-9)
    return np.convolve(np.pad(x, w, mode='edge'), np.ones(2 * w + 1) / (2 * w + 1), mode='valid')


def gate_mask(feat_col):
    e = boxsmooth01(feat_col.astype(np.float64))
    return e < np.percentile(e, QUIET_Q)         # True = "quiet"/gate open


def main():
    P = StepManiaParser.for_v2(subdiv=SUBDIV, min_song_length=30.0, max_song_length=600.0,
                               min_bpm=40.0, max_bpm=320.0, max_simultaneous=4, gimmick_max_bpm=400.0)
    ext = make_feature_extractor("highres_v2").extractor
    # features from the GLOBAL chart's alignment (all arms share Bye Bye audio+bpm+offset)
    gsm = PROBE / "Bye Bye GLOBAL" / "chart.sm"
    ch = P.parse_file(str(gsm))
    ap = gsm.parent / ch.audio_file if (gsm.parent / ch.audio_file).exists() else Path(sorted(glob.glob(str(gsm.parent/'*.ogg')))[0])
    feats = ext.extract_from_chart(str(ap), ch).get_aligned_features()
    total_q = gate_mask(feats[:, 0])       # TOTAL-energy gate (deployed)
    perc_q = gate_mask(feats[:, 35])       # PERC-onset-absence gate (cond-mech §6)
    T0 = len(total_q)
    jac = (total_q & perc_q).sum() / max((total_q | perc_q).sum(), 1)
    print(f"Bye Bye | {T0} frames | TOTAL-quiet {total_q.mean()*100:.0f}% of song, PERC-absent {perc_q.mean()*100:.0f}% | "
          f"gate overlap (Jaccard) {jac:.2f}  (low => the two gates target DIFFERENT sections)\n")

    def onsets(arm):
        sm = PROBE / f"Bye Bye {arm}" / "chart.sm"
        if not sm.is_file(): return None
        c = P.parse_file(str(sm)); nd = next((n for n in c.note_data if n.difficulty_name), None)
        pres = to_binary(P.convert_to_tensor_typed(c, nd)).any(1)
        return pres

    print(f"{'arm':11s} {'notes':>6s} | {'dens IN total-quiet':>19s} {'OUT':>6s} {'ratio':>6s} | "
          f"{'dens IN perc-absent':>19s} {'OUT':>6s} {'ratio':>6s}")
    print("-" * 92)
    base = {}
    for arm in ARMS:
        pres = onsets(arm)
        if pres is None: continue
        T = min(len(pres), T0); pr = pres[:T]; tq = total_q[:T]; pq = perc_q[:T]
        din_t, dout_t = pr[tq].mean(), pr[~tq].mean()
        din_p, dout_p = pr[pq].mean(), pr[~pq].mean()
        base[arm] = (din_t, din_p)
        print(f"{arm:11s} {int(pr.sum()):>6d} | {din_t:>19.4f} {dout_t:>6.4f} {din_t/max(dout_t,1e-9):>6.2f} | "
              f"{din_p:>19.4f} {dout_p:>6.4f} {din_p/max(dout_p,1e-9):>6.2f}")

    if 'GLOBAL' in base:
        g_t, g_p = base['GLOBAL']
        print(f"\nΔ density INSIDE the gate vs GLOBAL baseline (the harm boost should RAISE its own gate's region):")
        for arm in ARMS:
            if arm in base and arm != 'GLOBAL':
                dt = base[arm][0] - g_t; dp = base[arm][1] - g_p
                print(f"  {arm:11s}  total-quiet {dt:+.4f} ({dt/max(g_t,1e-9)*100:+.0f}%)   "
                      f"perc-absent {dp:+.4f} ({dp/max(g_p,1e-9)*100:+.0f}%)")
        print("  (HARM-TOTAL should raise total-quiet; HARM-PERC should raise perc-absent; else the lever didn't land)")


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""PREDICT THE FLIP POINT: at what CFG guidance does a given song's 1/4 backbone flip to a 1/16 smear?
(2026-07-04, user thread.) The n=40 sweep showed 16th-ANCHORING falls off a CLIFF as guidance climbs (32/40
songs monotone cliffs), and the audio feature `env_strongbeat_frac` (SB) predicts the flip guidance g* at
Spearman +0.54 — but only R2~0.25 / resid ~0.58 guidance-units on the COARSE 5-pt, k=2 sweep. This probe pins
g* PRECISELY: a DENSER guidance grid + higher k, then a LOGISTIC-CLIFF fit per song whose inflection g0 is a
resolution-robust flip point (vs a noisy fixed-threshold crossing). Tests whether SB predicts g0 with a
calibrated band.

Fixed chaos=0.9 milestone --style spec (guidance is the OVERLOAD lever, H14 + the taste_grid referee). Deployed
FULL arm (mechanism = CFG-amplified chaos + the 16th-unlock, NOT the governor; backbone_phase_findings.md).
Metric = sixteenth_anchoring (the ear-validated OVERLOAD detector, goodregion_findings.md). Reuses the
deployed-faithful gen_typed + the anchoring metric. Song subset spans the SB range (from tolerance_audio_density.csv).
"""
import warnings
warnings.filterwarnings('ignore')
import argparse, sys, csv
from pathlib import Path
import numpy as np, torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.generation.decode_defaults import CANONICAL_DECODE
from src.generation.decode_harness import load_generator, DEPLOYED_CHECKPOINT
from src.generation.radar_manifold import RadarManifold
from probe_quality_features import load_val_dataset, build_songs
from probe_backbone_phase import gen_typed
from probe_backbone_tolerance import sixteenth_anchoring, on_grid_share

FULL_CALIB = CANONICAL_DECODE['onset_phase_calib']


def logistic_cliff(g, plateau, floor, g0, w):
    """anchoring(g) = floor + (plateau-floor) / (1 + exp((g-g0)/w)). g0 = inflection (the flip point);
    w>0 = cliff width (small = sharp). A high->low descending sigmoid in guidance."""
    return floor + (plateau - floor) / (1.0 + np.exp((g - g0) / np.clip(w, 1e-3, None)))


def fit_flip(gs, anch):
    """fit the logistic cliff; return (g0, width, plateau, floor, r2, ok). Fallback: 0.5-crossing interp."""
    gs = np.asarray(gs, float); a = np.asarray(anch, float)
    m = np.isfinite(a)
    gs, a = gs[m], a[m]
    def crossing(thr):
        if a[0] < thr: return gs[0]
        for i in range(1, len(gs)):
            if a[i] < thr:
                f = (a[i-1]-thr)/(a[i-1]-a[i]+1e-9); return gs[i-1]+f*(gs[i]-gs[i-1])
        return gs[-1] + 0.25
    if len(gs) < 4 or (a.max()-a.min()) < 0.2:
        return crossing(0.5), np.nan, a.max(), a.min(), np.nan, False
    from scipy.optimize import curve_fit
    p0 = [max(a[0], a.max()), min(a[-1], a.min()), float(np.median(gs)), 0.25]
    bounds = ([0.3, 0.0, gs[0]-0.5, 0.02], [1.2, 0.6, gs[-1]+0.5, 1.5])
    try:
        popt, _ = curve_fit(logistic_cliff, gs, a, p0=p0, bounds=bounds, maxfev=8000)
        pred = logistic_cliff(gs, *popt); r2 = 1 - ((a-pred)**2).sum()/max(1e-9, ((a-a.mean())**2).sum())
        plateau, floor, g0, w = popt
        return float(g0), float(w), float(plateau), float(floor), float(r2), True
    except Exception:
        return crossing(0.5), np.nan, a.max(), a.min(), np.nan, False


def select_songs(val_ds, sb_csv, n_build, n_pick, difficulty, max_len, seed):
    """build songs, attach SB from the audio-density CSV, pick n_pick spanning the SB range (dedup titles)."""
    songs = build_songs(val_ds, n_build, difficulty=difficulty, max_len=max_len)
    sb = {a['title']: float(a['env_strongbeat_frac']) for a in csv.DictReader(open(sb_csv))}
    seen = set(); pool = []
    for s in songs:
        if s['title'] in seen or s['title'] not in sb:
            continue
        seen.add(s['title']); s['SB'] = sb[s['title']]; pool.append(s)
    pool.sort(key=lambda s: s['SB'])
    if n_pick >= len(pool):
        return pool
    idx = np.linspace(0, len(pool)-1, n_pick).round().astype(int)   # even spread across SB
    return [pool[i] for i in sorted(set(idx))]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint', default=DEPLOYED_CHECKPOINT)
    p.add_argument('--n_build', type=int, default=40, help='#songs to build (must match the SB CSV order)')
    p.add_argument('--n_pick', type=int, default=16, help='#songs to sweep (spread across SB)')
    p.add_argument('--k', type=int, default=4, help='#gens/guidance (denoise the anchoring point)')
    p.add_argument('--max_len', type=int, default=768)
    p.add_argument('--cache_dir', default='cache/samples_v3')
    p.add_argument('--spec', default='chaos=0.9,voltage=0.7,air=0.5,freeze=0.5')
    p.add_argument('--guidance', default='1.0,1.25,1.5,1.75,2.0,2.25,2.5,3.0', help='DENSER grid in the cliff zone')
    p.add_argument('--sb_csv', default='cache/tolerance_audio_density.csv')
    p.add_argument('--out', default='cache/flip_point.csv')
    return p.parse_args()


def main():
    args = parse_args(); set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    guids = [float(x) for x in args.guidance.split(',')]
    print(f"device={device} | spec='{args.spec}' | guidance={guids} | k={args.k} | pick={args.n_pick} songs", flush=True)

    val_ds = load_val_dataset(args.data_dir, args.audio_dir, args.seed, args.cache_dir)
    songs = select_songs(val_ds, args.sb_csv, args.n_build, args.n_pick, 3, args.max_len, args.seed)
    manifold = RadarManifold.load(Path('cache/radar_manifold.npz'))
    model = load_generator(args.checkpoint, 42, device)
    print(f"selected {len(songs)} songs, SB range {songs[0]['SB']:.2f}-{songs[-1]['SB']:.2f}", flush=True)

    rows = []
    for i, s in enumerate(songs, 1):
        anch_curve, grid_curve = [], []
        for gg in guids:
            an, og = [], []
            for j in range(args.k):
                typed = gen_typed(model, s, args.spec, gg, manifold, device, FULL_CALIB, {}, args.seed + j)[0]
                an.append(sixteenth_anchoring(typed)); og.append(on_grid_share(typed))
            anch_curve.append(float(np.nanmean(an))); grid_curve.append(float(np.nanmean(og)))
        g0, w, plat, floor, r2, ok = fit_flip(guids, anch_curve)
        row = dict(title=s['title'], SB=s['SB'], bpm=s['bpm'], real_density=s['real_density'],
                   flip_g0=g0, cliff_w=w, plateau=plat, floor=floor, fit_r2=r2, fit_ok=int(ok))
        row.update({f'anch_g{g}': v for g, v in zip(guids, anch_curve)})
        row.update({f'ongrid_g{g}': v for g, v in zip(guids, grid_curve)})
        rows.append(row)
        cs = " ".join(f"{v:.2f}" for v in anch_curve)
        print(f"  [{i}/{len(songs)}] SB={s['SB']:.2f} {s['title'][:20]:20s} anch[{cs}] "
              f"-> g0={g0:.2f} w={w:.2f} r2={r2:.2f}{'' if ok else ' (fallback)'}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(args.out, 'w', newline='') as fh:
        w_ = csv.DictWriter(fh, fieldnames=keys); w_.writeheader()
        for r in rows: w_.writerow(r)
    print(f"\nwrote {args.out} ({len(rows)} songs)", flush=True)

    # ---- does SB predict the flip point g0? ----
    from scipy.stats import spearmanr, pearsonr, rankdata
    ok = [r for r in rows if r['fit_ok'] and np.isfinite(r['flip_g0'])]
    print(f"\n{'='*70}\nFLIP-POINT PREDICTION (n={len(ok)} clean logistic fits)\n{'='*70}")
    if len(ok) < 4:
        print(f"  too few clean fits ({len(ok)}) to correlate — see {args.out}. (fallback g0s are in the CSV.)")
        return
    SB = np.array([r['SB'] for r in ok]); G0 = np.array([r['flip_g0'] for r in ok])
    D = np.array([r['real_density'] for r in ok]); BPM = np.array([r['bpm'] for r in ok])
    rs, ps = spearmanr(SB, G0); rp, pp = pearsonr(SB, G0)
    print(f"  SB    -> g0:  Spearman {rs:+.3f}(p{ps:.4f})   Pearson {rp:+.3f}(p{pp:.4f})")
    print(f"  dens  -> g0:  Spearman {spearmanr(D,G0)[0]:+.3f}   bpm -> g0: {spearmanr(BPM,G0)[0]:+.3f}")
    b, a = np.polyfit(SB, G0, 1); pred = a + b*SB; resid = G0 - pred
    r2 = 1 - (resid**2).sum()/((G0-G0.mean())**2).sum()
    print(f"  OLS  g0 = {a:+.2f} + {b:+.2f}*SB   R2={r2:.3f}   resid_std={resid.std():.2f} guidance-units")
    # partial SB|density
    Z = np.column_stack([np.ones(len(SB)), rankdata(D)])
    rx = rankdata(SB) - Z@np.linalg.lstsq(Z, rankdata(SB), rcond=None)[0]
    ry = rankdata(G0) - Z@np.linalg.lstsq(Z, rankdata(G0), rcond=None)[0]
    print(f"  SB|density partial Spearman = {np.corrcoef(rx,ry)[0,1]:+.3f}")
    print(f"\n  (compare COARSE 5-pt/k2 sweep: Spearman +0.54, R2~0.25, resid~0.58. Denser+k{args.k} should tighten.)")


if __name__ == '__main__':
    main()

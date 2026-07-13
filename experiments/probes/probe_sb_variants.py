#!/usr/bin/env python
"""OPTIMIZE THE FIT cheaply: can a BETTER strong-beat feature beat the current env_strongbeat_frac (SB, dim13)?
(2026-07-04.) The deployed SB uses onset_env (dim13, coarse ~93ms hop). Test variants — especially on
highres_onset (dim34, ~5.8ms hop, built to resolve on-grid transients, H4) — and strong-beat CONTRAST (ratio
of on-beat to off-beat energy) vs the aggregate FRACTION. No generation/model — raw audio features only. Judged by
LOO-CV R2 vs the n=40 tolerance labels (cache/backbone_tolerance.csv)."""
import warnings; warnings.filterwarnings('ignore')
import sys, csv, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(1, str(Path(__file__).resolve().parent))  # probes dir (sibling probe imports)
from src.utils.reproducibility import set_seed
from probe_quality_features import load_val_dataset, build_songs

ONSET_ENV, HIGHRES_ONSET = 13, 34

def sb_features(env):
    """strong-beat features from a per-16th onset-strength envelope (T,)."""
    e = np.asarray(env, float); T = len(e); t = np.arange(T)
    strong = (t % 4) % 2 == 0
    f = {}
    f['frac'] = e[strong].sum() / (e.sum() + 1e-9)                       # aggregate mass fraction (= current SB)
    f['contrast'] = e[strong].mean() / (e[~strong].mean() + 1e-9)        # on/off mean ratio (peakiness)
    # quarter-only (t%4==0) vs the rest — the pure downbeat comb
    q = (t % 4) == 0
    f['q_frac'] = e[q].sum() / (e.sum() + 1e-9)
    f['q_contrast'] = e[q].mean() / (e[~q].mean() + 1e-9)
    return f

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--n', type=int, default=40)
    p.add_argument('--cache_dir', default='cache/samples_v3'); p.add_argument('--max_len', type=int, default=768)
    p.add_argument('--tol_csv', default='cache/backbone_tolerance.csv')
    args = p.parse_args(); set_seed(args.seed)
    val_ds = load_val_dataset(args.data_dir, args.audio_dir, args.seed, args.cache_dir)
    songs = build_songs(val_ds, args.n, difficulty=3, max_len=args.max_len)
    print(f"songs={len(songs)}", flush=True)
    feats = {}
    for s in songs:
        row = {}
        for dim, tag in [(ONSET_ENV, 'env'), (HIGHRES_ONSET, 'hi')]:
            for k, v in sb_features(s['audio'][:s['T'], dim]).items():
                row[f'{tag}_{k}'] = v
        feats[s['title']] = row
    tol = {r['title']: r for r in csv.DictReader(open(args.tol_csv))}
    T = [t for t in feats if t in tol]; n = len(T)
    def fc(k): return np.array([feats[t][k] for t in T])
    def tc(k): return np.array([float(tol[t][k]) for t in T])
    def loo_r2(x, y):
        x = np.asarray(x); pred = np.zeros(n)
        for i in range(n):
            tr = np.arange(n) != i
            b = np.polyfit(x[tr], y[tr], 1); pred[i] = b[0]*x[i]+b[1]
        return 1 - ((y-pred)**2).sum()/((y-y.mean())**2).sum()
    from scipy.stats import spearmanr
    names = [f'{tag}_{k}' for tag in ('env','hi') for k in ('frac','contrast','q_frac','q_contrast')]
    print(f"\nSingle-feature LOO-CV R2 vs tolerance (n={n}); env_frac = the deployed SB baseline:")
    print(f"{'feature':16}" + "".join(f"{t:>14}" for t in ['ongrid_tol','anch_tol','qrep_tol']))
    for nm in names:
        x = fc(nm)
        cells = "".join(f"  rho{spearmanr(x,tc(t))[0]:+.2f}/cv{loo_r2(x,tc(t)):+.2f}" for t in ['ongrid_tol','anch_tol','qrep_tol'])
        star = '  <-deployed SB' if nm == 'env_frac' else ''
        print(f"{nm:16}{cells}{star}")

if __name__ == '__main__':
    main()

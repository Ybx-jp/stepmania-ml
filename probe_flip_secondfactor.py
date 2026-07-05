#!/usr/bin/env python
"""SECOND-FACTOR HUNT for the flip-point g0, on the CLEAN k=4 labels (cache/flip_point_v2.csv).

Question: SB (env_strongbeat_frac) predicts the per-song flip guidance g0 only WEAKLY across 32 songs
(Spearman +0.29 clean / +0.39 censored, down from the small-n +0.72). The weakening is localized to a
HIGH-SB FORK: some high-SB songs resist (Take It, BUMBLE, Abyss) and some flip early (MEANING OF LIFE,
And Then We Kiss, LOVE). Is there a SECOND audio feature that explains the residual after SB?

DISCIPLINE (experiment-design):
- n is tiny (28 clean) vs 84 candidate dims => in-sample R2 is guaranteed to find a "winner" by chance.
  The JUDGE is the LOO-CV INCREMENT of a 2-feature model {SB, feat} over the SB-only model, NOT in-sample R2.
- A PERMUTATION NULL (shuffle g0, re-run the best-of-84 selection) gives the chance level for that increment.
  A real second factor must beat the null's upper tail.
- NEGATIVE CONTROL: real_density (the HANDOFF's known overfitter) should raise in-sample but NOT LOO-CV.
- Features = the SAME 84-dim pooling as cache/audio_fingerprints_highres.npz (mean|std of the 42 highres dims),
  recomputed from build_songs audio so titles align EXACTLY with the labels (no path join).
No generation / no model forward -- raw audio features vs the already-computed g0 labels.
"""
import warnings; warnings.filterwarnings('ignore')
import sys, csv, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils.reproducibility import set_seed
from probe_quality_features import load_val_dataset, build_songs
from scipy.stats import spearmanr

ONSET_ENV = 13

def sb_frac(env):
    e = np.asarray(env, float); t = np.arange(len(e)); strong = (t % 4) % 2 == 0
    return e[strong].sum() / (e.sum() + 1e-9)

def loo_cv_r2(X, y):
    """LOO-CV R2 for OLS with intercept. X: (n,p), y: (n,)."""
    n = len(y); pred = np.zeros(n)
    A = np.column_stack([X, np.ones(n)])
    for i in range(n):
        tr = np.arange(n) != i
        beta, *_ = np.linalg.lstsq(A[tr], y[tr], rcond=None)
        pred[i] = A[i] @ beta
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--n_build', type=int, default=60)
    p.add_argument('--cache_dir', default='cache/samples_v3'); p.add_argument('--max_len', type=int, default=768)
    p.add_argument('--flip_csv', default='cache/flip_point_v2.csv')
    p.add_argument('--n_perm', type=int, default=500)
    args = p.parse_args(); set_seed(args.seed)

    # --- labels ---
    rows = [r for r in csv.DictReader(open(args.flip_csv)) if r['title'] != 'title']
    lab = {r['title']: r for r in rows}

    # --- features: 84-dim pooled fingerprint + SB, recomputed per song (title-aligned) ---
    val_ds = load_val_dataset(args.data_dir, args.audio_dir, args.seed, args.cache_dir)
    songs = build_songs(val_ds, args.n_build, difficulty=3, max_len=args.max_len)
    feats = {}
    for s in songs:
        if s['title'] not in lab:
            continue
        a = s['audio'][:s['T']]                       # (T, 42)
        pooled = np.concatenate([a.mean(0), a.std(0)])  # (84,) == the fingerprint pooling
        feats[s['title']] = dict(sb=sb_frac(a[:, ONSET_ENV]), pooled=pooled)
    dim_names = [f'{p}_d{k}' for p in ('mean', 'std') for k in range(42)]

    T_all = [t for t in feats if t in lab]
    def sub(clean):
        return [t for t in T_all if (int(lab[t]['fit_ok']) == 1 or not clean)]

    for clean, tag in [(True, 'CLEAN flippers (fit_ok=1)'), (False, 'ALL incl. censored g0=3.25')]:
        T = sub(clean); n = len(T)
        sb = np.array([feats[t]['sb'] for t in T])
        g0 = np.array([float(lab[t]['flip_g0']) for t in T])
        dens = np.array([float(lab[t]['real_density']) for t in T])
        P = np.array([feats[t]['pooled'] for t in T])
        Pz = (P - P.mean(0)) / (P.std(0) + 1e-9)
        sbz = (sb - sb.mean()) / (sb.std() + 1e-9)

        print(f"\n{'='*74}\n{tag}  (n={n})\n{'='*74}")
        base_cv = loo_cv_r2(sbz.reshape(-1, 1), g0)
        print(f"  baseline SB-only:  Spearman {spearmanr(sb,g0)[0]:+.3f}   LOO-CV R2 {base_cv:+.3f}")

        # NEGATIVE CONTROL: density as 2nd factor
        Xd = np.column_stack([sbz, (dens-dens.mean())/(dens.std()+1e-9)])
        print(f"  +real_density (neg control):  LOO-CV R2 {loo_cv_r2(Xd,g0):+.3f}  "
              f"(in-sample R2 {1-((g0-np.column_stack([Xd,np.ones(n)])@np.linalg.lstsq(np.column_stack([Xd,np.ones(n)]),g0,rcond=None)[0])**2).sum()/((g0-g0.mean())**2).sum():+.3f})")

        # --- hunt: LOO-CV increment of {SB, feat_k} over SB-only, all 84 dims ---
        dcv = np.array([loo_cv_r2(np.column_stack([sbz, Pz[:, k]]), g0) - base_cv for k in range(84)])
        # descriptive: partial spearman(feat | SB) vs g0
        def partial_sp(x):
            # residualize both on SB (linear), spearman of residuals
            bx = np.polyfit(sbz, x, 1); rx = x - (bx[0]*sbz+bx[1])
            by = np.polyfit(sbz, g0, 1); ry = g0 - (by[0]*sbz+by[1])
            return spearmanr(rx, ry)[0]
        psp = np.array([partial_sp(Pz[:, k]) for k in range(84)])

        order = np.argsort(-dcv)
        print(f"\n  TOP-6 by LOO-CV increment (ΔCV = CV[SB+feat] − CV[SB]):")
        print(f"    {'feature':10}{'ΔCV':>9}{'CV[SB+f]':>10}{'partial_ρ(f|SB)':>17}")
        for k in order[:6]:
            print(f"    {dim_names[k]:10}{dcv[k]:>+9.3f}{base_cv+dcv[k]:>+10.3f}{psp[k]:>+17.3f}")

        # --- permutation null on the best ΔCV over 84 dims ---
        rng = np.random.default_rng(args.seed)
        best_null = np.empty(args.n_perm)
        for j in range(args.n_perm):
            yp = rng.permutation(g0)
            bcv = loo_cv_r2(sbz.reshape(-1, 1), yp)
            best_null[j] = max(loo_cv_r2(np.column_stack([sbz, Pz[:, k]]), yp) - bcv for k in range(84))
        best_real = dcv.max()
        pval = (1 + (best_null >= best_real).sum()) / (1 + args.n_perm)
        print(f"\n  PERMUTATION NULL (best-of-84 ΔCV under shuffled g0, {args.n_perm} perms):")
        print(f"    best REAL ΔCV = {best_real:+.3f}   null 95th pct = {np.quantile(best_null,0.95):+.3f}   "
              f"null max = {best_null.max():+.3f}")
        print(f"    p(best real ΔCV ≥ chance) = {pval:.3f}   "
              f"=> {'SIGNAL beyond chance' if pval<0.05 else 'NO reliable 2nd factor (within noise)'}")

if __name__ == '__main__':
    main()

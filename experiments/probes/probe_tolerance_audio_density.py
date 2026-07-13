#!/usr/bin/env python
"""DEPLOYABILITY CHECK for the tolerance formula (2026-07-04, user thread).

The n=40 tolerance sweep (probe_backbone_tolerance.py) found the ONLY significant per-song predictor of
backbone-collapse tolerance is `real_density` (rho~-0.37, orthogonal to audio-busyness, BPM null). But
`real_density` is the REFERENCE CHART's note density -> NOT available for an unseen song, and no existing
audio feature proxies it (best |rho|~0.45). A tolerance FORMULA usable on a new song needs an AUDIO-DERIVABLE
predictor.

This probe engineers audio-derivable density/concentration features and asks: does ANY of them predict
TOLERANCE directly (not just proxy real_density) as well as real_density does? We do NOT need to proxy
real_density; we need to predict tolerance from audio alone.

TOP HYPOTHESIS (mechanism-faithful): the deployed onset head's p_onset STRONG-BEAT MASS FRACTION. The
tolerance mechanism (goodregion_findings.md) is: at fixed manifold density, tau = quantile(p_onset, 1-D);
a song whose onset mass already sits ON strong beats (t%4 in {0,2}) keeps its 1/4 backbone when chaos is
cranked, while a song with mass SPREAD across the 16th grid smears sooner. So `ponset_strongbeat_frac` (and
concentration metrics: entropy, Gini, top-mass) should predict tolerance from audio alone.

Uses the SAME onset path as the deployed decode (conditioned_p_onset). Features are computed at the BASE
onset affordance (radar=None, guidance=1, calib off) = a pure audio+difficulty song property available at
deployment. No AR generation -> cheap (onset forward only). Merges with outputs/probe_results/backbone_tolerance.csv by title.
"""
import warnings, os
warnings.filterwarnings('ignore')
import argparse, sys, csv
from pathlib import Path
import numpy as np, torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(1, str(Path(__file__).resolve().parent))  # probes dir (sibling probe imports)
from src.utils.reproducibility import set_seed
from src.generation.decode_harness import conditioned_p_onset, load_generator, DEPLOYED_CHECKPOINT
from probe_quality_features import load_val_dataset, build_songs

ONSET_ENV_DIM = 13   # a42[:,13] = onset envelope (cache_features label)


def _entropy(p):
    """normalized Shannon entropy of a nonneg vector treated as a distribution. 1 = flat, 0 = one spike."""
    s = p.sum()
    if s <= 0 or len(p) < 2:
        return np.nan
    q = p / s
    q = q[q > 0]
    return float(-(q * np.log(q)).sum() / np.log(len(p)))


def _gini(p):
    """Gini concentration of a nonneg vector. 0 = uniform, ->1 = concentrated."""
    x = np.sort(np.asarray(p, float))
    n = len(x)
    if n < 2 or x.sum() <= 0:
        return np.nan
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def _topmass(p, frac):
    """fraction of total mass held by the top `frac` of frames (high = peaky)."""
    x = np.sort(np.asarray(p, float))[::-1]
    k = max(1, int(round(frac * len(x))))
    tot = x.sum()
    return float(x[:k].sum() / tot) if tot > 0 else np.nan


def audio_density_features(p_onset, env, bpm):
    """audio-derivable per-song scalars from the BASE p_onset (T,) and the onset envelope (T,)."""
    p = np.asarray(p_onset, float).ravel()
    T = len(p)
    t = np.arange(T)
    strong = (t % 4) % 2 == 0          # t%4 in {0,2} = quarter/8th (backbone); {1,3} = 16th-offbeat
    f = {}
    # --- p_onset shape (the tau surface the decode literally thresholds) ---
    f['ponset_mean'] = float(p.mean())
    f['ponset_cv'] = float(p.std() / (p.mean() + 1e-9))
    f['ponset_entropy'] = _entropy(p)                 # high = flat/spread  -> predict LOW tolerance
    f['ponset_gini'] = _gini(p)                        # high = peaky        -> predict HIGH tolerance
    f['ponset_top10_mass'] = _topmass(p, 0.10)
    f['ponset_top25_mass'] = _topmass(p, 0.25)
    # THE mechanism-faithful one: how much p_onset MASS already sits on strong beats
    f['ponset_strongbeat_frac'] = float(p[strong].sum() / (p.sum() + 1e-9))
    # strong-vs-offbeat MEAN ratio (density-normalized version of the above)
    off = ~strong
    f['ponset_strong_off_ratio'] = float(p[strong].mean() / (p[off].mean() + 1e-9))
    # --- onset-envelope OCCUPANCY (per-16th-frame occupancy, directly comparable to real_density) ---
    e = np.asarray(env, float).ravel()[:T]
    for q in (0.5, 0.7, 0.9):
        thr = np.quantile(e, q)
        f[f'env_occ_q{int(q*100)}'] = float((e > thr).mean())   # ~constant by construction? no: > vs >=, ties
    # absolute (bpm un-normalized) onset rate: env mass per SECOND, not per beat
    frame_hz = bpm * 4.0 / 60.0
    f['env_abs_rate'] = float(e.mean() * frame_hz)
    f['env_strongbeat_frac'] = float(e[strong].sum() / (e.sum() + 1e-9))
    return f


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint', default=DEPLOYED_CHECKPOINT)
    p.add_argument('--n', type=int, default=40, help='#Hard songs (match the tolerance CSV order)')
    p.add_argument('--max_len', type=int, default=768)
    p.add_argument('--cache_dir', default='cache/samples_v3')
    p.add_argument('--tol_csv', default='outputs/probe_results/backbone_tolerance.csv')
    p.add_argument('--out', default='outputs/probe_results/tolerance_audio_density.csv')
    return p.parse_args()


def main():
    args = parse_args(); set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device} | n={args.n} | BASE p_onset (radar=None, g=1, calib off) — audio+difficulty song property", flush=True)

    val_ds = load_val_dataset(args.data_dir, args.audio_dir, args.seed, args.cache_dir)
    songs = build_songs(val_ds, args.n, difficulty=3, max_len=args.max_len)
    model = load_generator(args.checkpoint, 42, device)
    print(f"songs={len(songs)} | computing base onset affordance features...", flush=True)

    rows = []
    for i, s in enumerate(songs, 1):
        audio = torch.from_numpy(s['audio']).unsqueeze(0).to(device)
        diff = torch.tensor([s['difficulty']], device=device)
        with torch.no_grad():
            memory = model.encode_audio(audio)
            p_onset = conditioned_p_onset(model, memory, diff, radar=None, style=None,
                                          guidance=1.0, phase_calib=(0.0, 0.0))   # base audio affordance
        p_onset = np.asarray(p_onset).ravel()[:s['T']]
        env = s['audio'][:s['T'], ONSET_ENV_DIM]
        feats = audio_density_features(p_onset, env, s['bpm'])
        feats.update(title=s['title'], bpm=s['bpm'], real_density=s['real_density'])
        rows.append(feats)
        if i % 10 == 0 or i == len(songs):
            print(f"  [{i}/{len(songs)}] {s['title'][:22]:22s} strongbeat_frac={feats['ponset_strongbeat_frac']:.3f} "
                  f"entropy={feats['ponset_entropy']:.3f} gini={feats['ponset_gini']:.3f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as fh:
        keys = ['title', 'bpm', 'real_density'] + [k for k in rows[0] if k not in ('title', 'bpm', 'real_density')]
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nwrote {args.out} ({len(rows)} songs)", flush=True)

    # ---- merge with tolerance CSV by title, correlate audio features vs tolerance ----
    tol = {r['title']: r for r in csv.DictReader(open(args.tol_csv))}
    merged = [(r, tol[r['title']]) for r in rows if r['title'] in tol]
    print(f"merged {len(merged)}/{len(rows)} songs with {args.tol_csv}")
    from scipy.stats import spearmanr
    feat_names = [k for k in rows[0] if k not in ('title',)]
    targets = ['ongrid_tol', 'anch_tol', 'qrep_tol']

    def fcol(name): return np.array([float(a[name]) for a, _ in merged])
    def tcol(name): return np.array([float(b[name]) for _, b in merged])

    print("\n" + "=" * 78)
    print("AUDIO-DERIVABLE feature  vs  TOLERANCE  (Spearman; the bar to clear = real_density ~ -0.37)")
    print("=" * 78)
    # reference: real_density itself
    for t in targets:
        rho, p = spearmanr(fcol('real_density'), tcol(t))
        print(f"  [ref] real_density   vs {t:11s}  rho={rho:+.3f} p={p:.3f}")
    print("-" * 78)
    ranked = []
    for f in feat_names:
        if f in ('real_density', 'bpm'):
            continue
        x = fcol(f)
        if np.nanstd(x) == 0:
            continue
        best = 0.0; detail = []
        for t in targets:
            rho, p = spearmanr(x, tcol(t), nan_policy='omit')
            detail.append((t, rho, p)); best = max(best, abs(rho))
        # also: does it proxy real_density?
        rd, pd = spearmanr(x, fcol('real_density'), nan_policy='omit')
        ranked.append((best, f, detail, rd, pd))
    for best, f, detail, rd, pd in sorted(ranked, reverse=True):
        ds = "  ".join(f"{t.split('_')[0]}={rho:+.2f}(p{p:.2f})" for t, rho, p in detail)
        flag = '  <== BEATS density-bar' if best >= 0.37 else ''
        print(f"  {f:24s} {ds}   | vs_density rho={rd:+.2f}{flag}")
    print("\nGOAL: an audio-only feature with |rho| >= ~0.37 on tolerance = a DEPLOYABLE tolerance predictor.")


if __name__ == '__main__':
    main()

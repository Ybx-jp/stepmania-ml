#!/usr/bin/env python
"""DISCOVER tolerance(song) = f(song features): how far can a song be cranked (CFG guidance, at the fixed
milestone HIGH-chaos --style spec) before its 1/4 backbone collapses to 1/16? (2026-07-03, user thread.)

The user CONFIRMED (playtest) that the charts they LIKE RETAIN a 1/4 backbone -> quarter-share retention is an
EAR-VALIDATED, OFFLINE, taste-aligned membership signal for the good-settings region. The conditioning target is
~song-invariant (manifold caps Hard chaos ~0.44), so per-song tolerance differences come purely from each song's
AUDIO structure interacting with the fixed conditioning -> a clean regression target. This probe measures the
per-song backbone-collapse point across a guidance sweep and correlates it with song features, to surface the
"matrix of influential song features and their interactions with conditioning" (the formula to be derived).

Deployed FULL arm only (mechanism already attributed in backbone_phase_findings.md: it's CFG-amplified chaos
conditioning + the 16th-unlock, NOT the governor). Offline: phase_shares, no ears, no critic.
"""
import warnings, os
warnings.filterwarnings('ignore')
import argparse, sys
from pathlib import Path
import numpy as np, torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(1, str(Path(__file__).resolve().parent))  # probes dir (sibling probe imports)
from src.utils.reproducibility import set_seed
from src.generation.decode_defaults import CANONICAL_DECODE
from src.generation.decode_harness import load_generator, DEPLOYED_CHECKPOINT
from src.generation.radar_manifold import RadarManifold
from probe_quality_features import load_val_dataset, build_songs, cache_features
from probe_backbone_phase import gen_typed, q_s16   # reuse the deployed-faithful generation + phase metric

FULL_CALIB = CANONICAL_DECODE['onset_phase_calib']
ONSET_ENV_DIM = 13   # audio onset-envelope channel (cache_features labels a42[:,13] onset_env)


def _onsets(typed):
    """per-frame boolean: is there an ONSET here? tap(1)/hold-head(2)/roll(4); EXCLUDES hold tails (3)."""
    return np.isin(np.asarray(typed), (1, 2, 4)).any(1)


def on_grid_share(typed):
    """fraction of ONSETS on a STRONG beat position (t%4 in {0,2} = quarter or 8th) vs the weak 16th-offbeats
    (t%4 in {1,3}). LOW => the pulse has PHASE-SHIFTED onto the 16th grid (the 'quarter notes basically gone,
    spine of 1/16-offset notes' failure seen on Deja loin g3). nan if no onsets."""
    on = _onsets(typed); idx = np.where(on)[0]
    if idx.size == 0:
        return np.nan
    return float(np.mean((idx % 4) % 2 == 0))   # t%4 in {0,2}


def quarter_representation(typed, onset_env, active_q=0.5, window=0):
    """% of ACTIVE downbeats (t%4==0) carrying an onset EXACTLY on the beat (window=0; a note a 16th off is NOT
    the downbeat — a ±1 window miscounts a 1/16-offset spine as coverage, the Deja loin g3 trap). 'active' =
    AUDIO onset-envelope above its active_q quantile (song-intrinsic -> avoids chart circularity). nan if none."""
    on = _onsets(typed); T = len(on)
    env = np.asarray(onset_env)[:T]
    thr = np.quantile(env, active_q)
    db = np.arange(0, T, 4)
    active = db[env[db] > thr]
    if active.size == 0:
        return np.nan
    return float(np.mean([on[max(0, d - window):d + window + 1].any() for d in active]))


def sixteenth_anchoring(typed):
    """fraction of 16th-offbeat onsets (t%4 in {1,3}) 'connected' to the grid = at least one FLANKING beat frame
    (t-1/t+1, always a quarter/8th) carries an onset. High = coherent runs into beats; low = floating off-grid
    smear. nan if no 16th onsets."""
    on = _onsets(typed); T = len(on)
    s16 = [t for t in range(T) if t % 4 in (1, 3) and on[t]]
    if not s16:
        return np.nan
    return float(np.mean([(t - 1 >= 0 and on[t - 1]) or (t + 1 < T and on[t + 1]) for t in s16]))


def tolerance_scalars(curve):
    """curve = sorted [(g, q_share)]. Returns (q_at_max_g, g_star) where g_star = guidance at which q first
    crosses BELOW 0.5 (linear-interp), censored to [g_min, g_max]. Higher q_at_max / higher g_star = MORE tolerant."""
    gs = [g for g, _ in curve]; qs = [q for _, q in curve]
    q_at_max = qs[-1]
    if qs[0] < 0.5:
        return q_at_max, gs[0]                                  # already collapsed at the gentlest crank
    for i in range(1, len(curve)):
        if qs[i] < 0.5:                                         # crossing between gs[i-1] and gs[i]
            t = (qs[i-1] - 0.5) / (qs[i-1] - qs[i] + 1e-9)
            return q_at_max, gs[i-1] + t * (gs[i] - gs[i-1])
    return q_at_max, gs[-1]                                     # never collapses in range (censored high)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint', default=DEPLOYED_CHECKPOINT)
    p.add_argument('--n', type=int, default=24, help='#Hard songs')
    p.add_argument('--k', type=int, default=2, help='#gens/cell (phase is fairly stable)')
    p.add_argument('--max_len', type=int, default=768)
    p.add_argument('--cache_dir', default='cache/samples_v3')
    p.add_argument('--spec', default='chaos=0.9,voltage=0.7,air=0.5,freeze=0.5')
    p.add_argument('--guidance', default='1.0,1.5,2.0,2.5,3.0')
    p.add_argument('--out', default='outputs/probe_results/backbone_tolerance.csv')
    return p.parse_args()


def main():
    args = parse_args(); set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    guids = [float(x) for x in args.guidance.split(',')]
    print(f"device={device} | spec='{args.spec}' | guidance={guids} | n={args.n} songs | k={args.k}")

    val_ds = load_val_dataset(args.data_dir, args.audio_dir, args.seed, args.cache_dir)
    songs = build_songs(val_ds, args.n, difficulty=3, max_len=args.max_len)
    manifold = RadarManifold.load(Path('cache/radar_manifold.npz'))
    model = load_generator(args.checkpoint, 42, device)
    print(f"songs={len(songs)}")

    rows = []
    for i, s in enumerate(songs, 1):
        env = s['audio'][:, ONSET_ENV_DIM]
        qrep_c, anch_c, grid_c = [], [], []      # per-guidance curves: quarter-rep(strict), 16th-anchor, on-grid-share
        for gg in guids:
            qr, an, og = [], [], []
            for j in range(args.k):
                typed = gen_typed(model, s, args.spec, gg, manifold, device, FULL_CALIB, {}, args.seed + j)[0]
                qr.append(quarter_representation(typed, env)); an.append(sixteenth_anchoring(typed))
                og.append(on_grid_share(typed))
            qrep_c.append((gg, np.nanmean(qr))); anch_c.append((gg, np.nanmean(an))); grid_c.append((gg, np.nanmean(og)))
        qrep_tol = float(np.nanmean([q for _, q in qrep_c]))   # strict downbeat coverage, mean over sweep
        anch_tol = float(np.nanmean([q for _, q in anch_c]))
        grid_tol = float(np.nanmean([q for _, q in grid_c]))   # on-grid share = 1 - phase-shift; the direct signal
        _, g_star = tolerance_scalars(qrep_c)
        row = dict(title=s['title'], bpm=s['bpm'], real_density=s['real_density'],
                   qrep_tol=qrep_tol, ongrid_tol=grid_tol, anch_tol=anch_tol, qrep_g_star=g_star,
                   qrep_at_maxg=qrep_c[-1][1], ongrid_at_maxg=grid_c[-1][1], anch_at_maxg=anch_c[-1][1])
        row.update({f'qrep_g{g}': q for g, q in qrep_c})
        row.update({f'ongrid_g{g}': q for g, q in grid_c})
        row.update({f'anch_g{g}': q for g, q in anch_c})
        row.update(cache_features(s['audio']))
        rows.append(row)
        print(f"  [{i}/{len(songs)}] {s['title'][:20]:20s} bpm={s['bpm']:.0f}  "
              f"Qrep(strict)@g1={qrep_c[0][1]:.2f}->g3={qrep_c[-1][1]:.2f}  "
              f"onGrid@g1={grid_c[0][1]:.2f}->g3={grid_c[-1][1]:.2f}  16thAnch@g3={anch_c[-1][1]:.2f}")

    import csv
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as fh:
        keys = list({k for r in rows for k in r})
        lead = ['title', 'bpm', 'real_density', 'qrep_tol', 'ongrid_tol', 'anch_tol', 'qrep_g_star',
                'qrep_at_maxg', 'ongrid_at_maxg', 'anch_at_maxg']
        keys = lead + [k for k in keys if k not in lead]
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nwrote {args.out} ({len(rows)} songs)")

    # ---- first-pass attribution: which SONG FEATURES predict tolerance (q@g3)? Spearman rank corr ----
    from scipy.stats import spearmanr
    cand = ['bpm', 'real_density', 'onset_env_mean', 'onset_env_cv', 'onset_rate_mean',
            'perc_mean', 'harm_mean', 'perc_harm_ratio', 'highres_onset_mean', 'highres_onset_cv']
    # REAL high-chaos envelope (probe_real_phase_reference, Q4 n=176): on_grid~0.85, anchor~0.73, qrep~0.68.
    # These are the real-ANCHORED discriminators (degenerate global-smear -> ~0.00). Higher = more real-like.
    print("\n*** real high-chaos (Q4) targets: on_grid~0.85  anchor~0.73  qrep~0.68  (degenerate smear -> 0.00) ***")
    for tname, tkey in [('ON-GRID share (backbone kept vs phase-shift; real Q4~0.85)', 'ongrid_tol'),
                        ('16th-ANCHORING (coherent runs vs global smear; real Q4~0.73)', 'anch_tol'),
                        ('quarter-REPRESENTATION strict (downbeat coverage; real Q4~0.68)', 'qrep_tol')]:
        tol = np.array([r[tkey] for r in rows])
        spread = np.nanmax(tol) - np.nanmin(tol)
        print("\n" + "=" * 70 + f"\nTOLERANCE = {tname}  [range {np.nanmin(tol):.2f}-{np.nanmax(tol):.2f}, "
              f"spread {spread:.2f}{' — near-constant, NON-discriminative' if spread < 0.15 else ''}]")
        print("  vs SONG FEATURES — Spearman (|rho| sorted):")
        scored = []
        for c in cand:
            x = np.array([r.get(c, np.nan) for r in rows])
            if np.isfinite(x).sum() > 5 and np.nanstd(x) > 0:
                rho, p = spearmanr(x, tol, nan_policy='omit'); scored.append((abs(rho), rho, p, c))
        for arho, rho, p, c in sorted(scored, reverse=True):
            print(f"    {c:18s} rho={rho:+.3f}  p={p:.3f}{'  <-- influential' if p < 0.05 else ''}")
    print(f"\n(first pass, n={len(rows)} songs. Interactions + a fitted formula come next once the influential set is clear.)")


if __name__ == '__main__':
    main()

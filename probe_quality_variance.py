#!/usr/bin/env python3
"""SIGNAL-vs-NOISE decomposition of per-song generator quality (+ the denoiser it implies).

The attribution target m_gen(song) = the GRADED-critic score of ONE canonical generation. But generation is
STOCHASTIC (pattern_sample/type_sample resample a different chart each time), so
    m_gen = true_per_song_quality  +  generation-stochasticity noise  +  critic imprecision.
If the per-song quality SIGNAL is small vs the sample-to-sample NOISE, NO audio feature can explain quality — the
null is structural, not "we picked bad features". This probe measures that directly by generating K charts per
Hard song with the GRADED critic (which — unlike the saturated binary critic where all K railed to ~0 — has the
dynamic range to show real within-song spread), then:
  - VARIANCE DECOMPOSITION: within-song var (resampling) vs between-song var (real quality) -> ICC = the fraction of
    score variance that is genuine per-song signal, and the reliability of the K-AVERAGED (denoised) target
    (Spearman-Brown). ICC is the CEILING on any feature attribution.
  - DENOISE: the per-song mean over K generations = the noise-reduced target; re-run the feature attribution on it
    (family-wise permutation floor) — the fairest possible read.
  - SPLIT-HALF reliability of the per-song quality estimate as an independent check.

Uses the GRADED critic (checkpoints/realism_critic_graded) score = logit margin. Deployed canonical decode via the
shared harness helpers (Rule 14). Usage:
    python probe_quality_variance.py --data_dir data/ --audio_dir data/ --n 24 --k 8
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, csv, sys
from pathlib import Path
import numpy as np, torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.models import LateFusionClassifier
from probe_quality_features import (load_val_dataset, build_songs, canonical_gen, critic_score,
                                    cache_features, raw_descriptors, spearman, load_generator, DEPLOYED_CHECKPOINT)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--difficulty', type=int, default=3)
    p.add_argument('--n', type=int, default=24, help='#Hard songs'); p.add_argument('--k', type=int, default=8, help='#generations/song')
    p.add_argument('--max_len', type=int, default=768)
    p.add_argument('--checkpoint', default=DEPLOYED_CHECKPOINT)
    p.add_argument('--critic', default='checkpoints/realism_critic_graded/best_val.pt')
    p.add_argument('--out', default='cache/quality_variance_hard.csv')
    return p.parse_args()


def main():
    args = parse_args(); set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device} | critic={args.critic} | n={args.n} songs x k={args.k} generations")
    val_ds = load_val_dataset(args.data_dir, args.audio_dir, args.seed)
    songs = build_songs(val_ds, args.n, args.difficulty, args.max_len)
    print(f"songs={len(songs)}")
    model = load_generator(args.checkpoint, 42, device)
    ck = torch.load(args.critic, map_location=device, weights_only=False)
    critic = LateFusionClassifier(ck['config']).to(device); critic.load_state_dict(ck['model_state_dict']); critic.eval()

    rows = []
    for n, s in enumerate(songs, 1):
        a23 = s['audio'][:, :23]
        m_real = critic_score(critic, a23, s['real'], device)[1]
        margins = [critic_score(critic, a23, canonical_gen(model, s, device), device)[1] for _ in range(args.k)]
        margins = np.array(margins)
        row = {'title': s['title'], 'bpm': s['bpm'], 'real_density': s['real_density'], 'm_real': m_real,
               'm_gen_mean': float(margins.mean()), 'm_gen_sd': float(margins.std(ddof=1)),
               **{f'g{j}': float(margins[j]) for j in range(args.k)}}
        row.update(cache_features(s['audio'])); row.update(raw_descriptors(s['audio_file']))
        rows.append(row)
        print(f"  [{n}/{len(songs)}] {s['title'][:24]:24s} m_gen={margins.mean():+.2f}±{margins.std(ddof=1):.2f} "
              f"[{margins.min():+.2f},{margins.max():+.2f}]  m_real={m_real:+.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"wrote {args.out}")

    # ---- VARIANCE DECOMPOSITION ------------------------------------------------------------------------------
    G = np.array([[r[f'g{j}'] for j in range(args.k)] for r in rows])       # (N, K) margins
    N, K = G.shape
    song_means = G.mean(1)
    within_var = float(G.var(1, ddof=1).mean())                            # avg within-song variance (resampling)
    between_var = float(song_means.var(ddof=1))                            # variance of the per-song means
    # correct between_var for the sampling noise it contains: E[var(means)] = true_between + within/K
    between_var_corr = max(between_var - within_var / K, 0.0)
    icc = between_var_corr / (between_var_corr + within_var) if (between_var_corr + within_var) > 0 else 0.0
    rel_mean = (K * icc) / (1 + (K - 1) * icc) if icc > 0 else 0.0         # Spearman-Brown: reliability of the K-mean
    print("\n" + "=" * 74)
    print("  VARIANCE DECOMPOSITION — is there a stable per-song quality SIGNAL to attribute?")
    print("=" * 74)
    print(f"  within-song sd (generation resampling): {np.sqrt(within_var):.3f}")
    print(f"  between-song sd (raw of per-song means): {np.sqrt(between_var):.3f}  "
          f"(noise-corrected true between-sd: {np.sqrt(between_var_corr):.3f})")
    print(f"  ICC (single generation) = {icc:.3f}   <- fraction of a SINGLE score that is real per-song signal")
    print(f"  reliability of the {K}-gen MEAN (Spearman-Brown) = {rel_mean:.3f}  <- ceiling: obs |r| attenuated by ~sqrt(this)")
    verdict = ("STRUCTURAL NULL — quality is mostly sample noise; no feature CAN explain it" if icc < 0.10 else
               "signal exists but noisy — denoising (K-mean) should sharpen attribution" if icc < 0.4 else
               "strong per-song signal — attribution should work on the denoised mean")
    print(f"  => {verdict}")

    # ---- SPLIT-HALF reliability (independent check) ----------------------------------------------------------
    h1 = G[:, :K // 2].mean(1); h2 = G[:, K // 2:].mean(1)
    print(f"\n  split-half: spearman(first {K//2}-gen mean, last {K-K//2}-gen mean) = {spearman(h1, h2):+.3f}  "
          f"(low => the per-song quality estimate itself is unreliable)")

    # ---- DENOISED ATTRIBUTION: features vs the K-averaged per-song quality ------------------------------------
    tgt = song_means
    exclude = {'title', 'm_real', 'm_gen_mean', 'm_gen_sd'} | {f'g{j}' for j in range(K)}
    featkeys = [k for k in keys if k not in exclude]
    ranked = []
    for k in featkeys:
        v = np.array([r.get(k, np.nan) for r in rows], float)
        if np.sum(np.isfinite(v)) < max(6, N // 2) or np.nanstd(v) < 1e-6: continue
        rsp = spearman(v, tgt)
        if np.isfinite(rsp): ranked.append((k, rsp, v))
    ranked.sort(key=lambda x: -abs(x[1]))
    thr = 1.96 / np.sqrt(N)
    print("\n" + "=" * 74)
    print(f"  DENOISED attribution — feature -> Spearman(feature, {K}-gen-mean quality)  [n={N}; |r|>~{thr:.2f} p<.05 uncorr]")
    print("=" * 74)
    tag = lambda k: 'raw ' if k.startswith('raw_') else ('d## ' if (len(k) > 3 and k[1:3].isdigit()) else 'cur ')
    for k, rsp, _ in ranked[:14]:
        print(f"    {tag(k)}{k:24s} r={rsp:+.3f}")
    if ranked:
        obs = abs(ranked[0][1]); V = [v for _, _, v in ranked]; rng = np.random.default_rng(0)
        nm = np.array([max(abs(spearman(v, rng.permutation(tgt))) for v in V) for _ in range(2000)])
        pfw = (nm >= obs).mean()
        print(f"\n  family-wise permutation: best |r|={obs:.3f}, null-max mean {nm.mean():.3f}/95th {np.quantile(nm,.95):.3f} "
              f"-> p_fw={pfw:.3f}  {'SIGNAL' if pfw < 0.05 else 'noise floor'}")
    print("  NOTE: if the ICC/reliability is low, the denoised attribution is also capped — read the decomposition first.")


if __name__ == '__main__':
    main()

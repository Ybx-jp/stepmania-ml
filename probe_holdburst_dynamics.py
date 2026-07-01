#!/usr/bin/env python3
"""PRE-REGISTERED single-hypothesis test of the ONE lead from the choreography probe.

Lead (notes/quality_feature_attribution_findings.md, follow-up): the generator's HOLD-BURST defect (fast
one-foot-cross-during-a-hold — the B4U play-feel complaint) correlated (family-wise p=0.027, ONE of 3 targets)
with a cluster of `d##_std` = z-scored spectral/chroma TIME-VARIABILITY. Concerns: truncation-sensitive artifact,
absent from interpretable features, marginal after multi-target correction.

HYPOTHESIS (pre-registered, directional): audio spectral/chroma/timbral DYNAMICS (time-variability) over the
window the generator saw predict the hold-burst defect NEGATIVELY — i.e. on spectrally/harmonically STATIC songs
the model has less structure to anchor footwork and commits MORE awkward held-note bursts.

DESIGN (de-artifacts all three concerns):
  - ONE target: `g_holdburst_excess`, REUSED from cache/quality_choreo_hard.csv (no regeneration; the choreo run
    is reproducible and build_songs gives the identical song order — paired by index, title-asserted).
  - Features recomputed from RAW audio on the FIRST-T-FRAMES window the generator conditioned on (NOT the
    full-song z-scored cache) — removes the z-score/truncation mismatch that made d##_std murky.
  - Pre-specified SMALL feature set + a composite dynamics index → family-wise permutation over just this set,
    no 3-target penalty.

Usage: python probe_holdburst_dynamics.py --data_dir data/ --audio_dir data/ --difficulty 3 --n 64
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, csv, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from probe_quality_features import load_val_dataset, build_songs, spearman


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--difficulty', type=int, default=3)
    p.add_argument('--n', type=int, default=64); p.add_argument('--max_len', type=int, default=768)
    p.add_argument('--choreo_csv', default='cache/quality_choreo_hard.csv')
    p.add_argument('--out', default='cache/holdburst_dynamics.csv')
    return p.parse_args()


def window_dynamics(audio_file, T, bpm, sr=22050):
    """Raw-audio DYNAMICS features on the first-T-16th-frames window the generator saw. All are time-variability
    ('flux'/std) measures — the de-artifacted, interpretable version of the d##_std cluster."""
    import librosa
    dur = T * 60.0 / (bpm * 4.0)                       # T sixteenth-notes -> seconds
    try:
        y, _ = librosa.load(audio_file, sr=sr, mono=True, duration=max(dur, 1.0))
    except Exception as e:
        print(f"    (load failed {os.path.basename(audio_file)}: {e})"); return {}
    if y.size < sr: return {}
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) + 1e-9
    chroma = librosa.feature.chroma_stft(S=S**2, sr=sr, tuning=0.0)
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    cent = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    flux = lambda M: float(np.mean(np.abs(np.diff(M, axis=1)).sum(0)))   # mean frame-to-frame L1 change
    return {
        'spec_flux':        flux(S / S.sum(0, keepdims=True)),           # normalized spectral change rate
        'chroma_flux':      flux(chroma),                                # harmonic (pitch-class) change rate
        'speccontrast_std': float(np.mean(contrast.std(1))),             # spectral-contrast time-variability (d16-22)
        'mfcc_flux':        float(np.mean(np.abs(np.diff(mfcc[1:], axis=1)))),  # timbral change rate (skip c0/energy)
        'centroid_std':     float(cent.std()),                          # brightness time-variability
    }


PRESPEC = ['spec_flux', 'chroma_flux', 'speccontrast_std', 'mfcc_flux', 'centroid_std']  # the pre-registered set


def main():
    args = parse_args(); set_seed(args.seed)
    # 1. reuse the already-computed hold-burst target (title + value per row, in build_songs order)
    crows = list(csv.DictReader(open(args.choreo_csv)))
    print(f"choreo csv: {len(crows)} rows")
    # 2. rebuild the SAME song list (parse only, NO generation) to recover audio_file paths in the same order
    val_ds = load_val_dataset(args.data_dir, args.audio_dir, args.seed)
    songs = build_songs(val_ds, args.n, args.difficulty, args.max_len)
    m = min(len(songs), len(crows))
    songs, crows = songs[:m], crows[:m]
    # 3. index-pair + title sanity check
    mism = [i for i in range(m) if songs[i]['title'] != crows[i]['title']]
    assert not mism, f"song/csv order mismatch at rows {mism[:5]} — cannot pair by index"
    print(f"paired {m} songs by index (titles match)")

    rows = []
    for i, (s, cr) in enumerate(zip(songs, crows), 1):
        feats = {'title': s['title'], 'bpm': s['bpm'], 'T': s['T'], 'real_density': s['real_density'],
                 'holdburst_excess': float(cr['g_holdburst_excess'])}
        feats.update(window_dynamics(s['audio_file'], s['T'], s['bpm']))
        rows.append(feats)
        if i % 10 == 0 or i == m: print(f"  [{i}/{m}] features computed")

    keys = list(rows[0].keys())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"wrote {args.out}")

    hb = np.array([r['holdburst_excess'] for r in rows])
    have = [k for k in PRESPEC if all(k in r for r in rows)]
    F = {k: np.array([r[k] for r in rows], float) for k in have}
    # composite dynamics index = mean z of the pre-specified flux/variability features
    z = lambda a: (a - a.mean()) / (a.std() + 1e-9)
    dyn = np.mean([z(F[k]) for k in have], 0)
    print("\n" + "=" * 72)
    print(f"  PRE-REGISTERED TEST — Spearman(feature, hold-burst excess)   [n={len(rows)}]")
    print(f"  hypothesis: DYNAMICS correlate NEGATIVELY (static audio -> more hold-burst defect)")
    print("=" * 72)
    res = [('dynamics_composite', spearman(dyn, hb))] + [(k, spearman(F[k], hb)) for k in have]
    for k, r in sorted(res, key=lambda x: -abs(x[1])):
        star = ' *' if abs(r) > 1.96 / np.sqrt(len(rows)) else ''
        print(f"    {k:20s} r={r:+.3f}{star}")
    # family-wise permutation over the pre-specified set (+ the composite)
    tested = {'dynamics_composite': dyn, **F}
    obs = max(abs(spearman(v, hb)) for v in tested.values())
    rng = np.random.default_rng(0); nmax = []
    for _ in range(5000):
        p = rng.permutation(hb)
        nmax.append(max(abs(spearman(v, p)) for v in tested.values()))
    nmax = np.array(nmax); pfw = (nmax >= obs).mean()
    print(f"\n  family-wise (over {len(tested)} pre-specified tests): best |r|={obs:.3f}, "
          f"null-max mean {nmax.mean():.3f}/95th {np.quantile(nmax,0.95):.3f} -> p_fw={pfw:.3f}  "
          f"{'CONFIRMED' if pfw < 0.05 else 'not significant'}")
    # primary endpoint: the composite, DIRECTIONAL (hypothesis says negative) one-sided permutation p
    r_dyn = spearman(dyn, hb)
    null_dyn = np.array([spearman(dyn, rng.permutation(hb)) for _ in range(5000)])
    p_one = (null_dyn <= r_dyn).mean()   # P(null as-or-more-negative than observed)
    print(f"  PRIMARY composite r={r_dyn:+.3f}, one-sided (negative) permutation p={p_one:.3f}  "
          f"{'supports H' if (r_dyn < 0 and p_one < 0.05) else 'does NOT support H'}")


if __name__ == '__main__':
    main()

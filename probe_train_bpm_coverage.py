#!/usr/bin/env python3
"""Is the fast-song generator-quality defect a TRAINING-COVERAGE effect? (no-generation diagnostic)

Finding: faster Hard songs -> worse generations (BPM r=-0.68), and it's INTRINSIC (governor ruled out). One intrinsic
candidate: the generator saw FEW fast Hard charts in training, so it's under-trained there. This bins the TRAIN split
by BPM (Hard charts) and overlays the per-bin generator QUALITY (from the n=30 variance run) — the decisive test:
  low-coverage bins COINCIDE with low-quality bins  -> coverage is (part of) the cause
  fast bins are WELL-populated but still low-quality -> coverage RULED OUT -> the onset/pattern head (next probe)

Reads train metadata (BPM, difficulty) from the parser's valid_samples WITHOUT feature extraction (parse-only, no
generation). Usage: python probe_train_bpm_coverage.py --data_dir data/ --audio_dir data/
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, csv, sys
from pathlib import Path
import numpy as np, yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import discover_chart_files, create_data_splits, create_datasets

BINS = [0, 120, 140, 160, 180, 1000]
FAST_CUT = 165.0                      # where the quality curve rolls off (>~165 BPM)
HARD = 3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--variance_csv', default='cache/quality_variance_hard.csv')  # per-song bpm + m_gen_mean (n=30)
    return p.parse_args()


def hard_bpms(ds):
    """BPMs of the Hard (dance-single) charts in a parsed dataset — from valid_samples, NO feature extraction."""
    out = []
    for meta in ds.valid_samples:
        if int(meta['difficulty_class']) == HARD:
            try:
                out.append(float(meta['chart'].bpm))
            except Exception:
                pass
    return np.array(out)


def main():
    args = parse_args(); set_seed(args.seed)
    cf = discover_chart_files(args.data_dir)
    train_files, val_files, _ = create_data_splits(cf, random_state=args.seed)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    # parse-only (feature_extractor=None, cache_dir=None) -> valid_samples populated, NO extraction/generation
    train_ds, val_ds, _ = create_datasets(train_files=train_files, val_files=val_files, test_files=[],
                                          audio_dir=args.audio_dir, max_sequence_length=msl,
                                          feature_extractor=None, cache_dir=None)
    tr = hard_bpms(train_ds); va = hard_bpms(val_ds)
    print(f"\ntrain Hard charts: {len(tr)} | val Hard charts: {len(va)}")
    print(f"train Hard BPM: median {np.median(tr):.0f}  mean {tr.mean():.0f}  "
          f"quantiles(10/25/50/75/90) {np.round(np.quantile(tr,[.1,.25,.5,.75,.9]),0)}")
    print(f"  FAST (>{FAST_CUT:.0f} BPM) share of train Hard: {(tr>FAST_CUT).mean()*100:.1f}%  "
          f"(val Hard: {(va>FAST_CUT).mean()*100:.1f}%)")

    # per-BPM-bin: train coverage + val generator quality (from the variance run)
    Q = list(csv.DictReader(open(args.variance_csv)))
    qb = np.array([float(r['bpm']) for r in Q]); qq = np.array([float(r['m_gen_mean']) for r in Q])
    print("\n" + "=" * 74)
    print("  BPM bin        train Hard (n / %)     val quality (m_gen_mean, n)   ")
    print("=" * 74)
    bin_cov, bin_q = [], []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        ntr = int(((tr >= lo) & (tr < hi)).sum()); ptr = ntr / len(tr) * 100
        m = (qb >= lo) & (qb < hi); vq = qq[m].mean() if m.any() else np.nan
        lbl = f"{lo:>3}-{hi if hi<1000 else '∞':<4}"
        print(f"  {lbl:12s}   {ntr:5d} / {ptr:5.1f}%          "
              + (f"{vq:+.2f}  (n={m.sum()})" if m.any() else "   —   (n=0)"))
        bin_cov.append(ptr); bin_q.append(vq)
    # correlation: does coverage track quality across bins?
    bc = np.array(bin_cov); bq = np.array(bin_q); ok = np.isfinite(bq)
    if ok.sum() >= 3 and np.std(bc[ok]) > 0 and np.std(bq[ok]) > 0:
        r = np.corrcoef(np.argsort(np.argsort(bc[ok])), np.argsort(np.argsort(bq[ok])))[0, 1]
        print(f"\n  spearman(bin train-coverage, bin quality) over {ok.sum()} bins = {r:+.2f}  "
              f"(+ => sparse bins are the low-quality ones = coverage supported)")

    # ---- SONG-LEVEL test (cleaner than coarse bins): does LOCAL train coverage predict quality beyond BPM? -----
    # local_cov(song) = # train Hard charts within +/-15 BPM of the song. Partial out BPM.
    def sp(x, y):
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 6 or np.std(x[ok]) < 1e-9 or np.std(y[ok]) < 1e-9: return np.nan
        return np.corrcoef(np.argsort(np.argsort(x[ok])), np.argsort(np.argsort(y[ok])))[0, 1]
    def partial(x, y, z):
        rx, ry, rz = [np.argsort(np.argsort(a)).astype(float) for a in (x, y, z)]
        res = lambda a, b: a - (np.polyfit(b, a, 1)[0] * b + np.polyfit(b, a, 1)[1])
        return np.corrcoef(res(rx, rz), res(ry, rz))[0, 1]
    loc = np.array([((tr >= b - 15) & (tr <= b + 15)).sum() for b in qb])   # local train-Hard density at each val song
    print("\n" + "=" * 74)
    print("  SONG-LEVEL (n={}): does LOCAL train coverage (#train Hard within ±15 BPM) explain quality?".format(len(qb)))
    print("=" * 74)
    print(f"  spearman(local_coverage, quality)          = {sp(loc, qq):+.3f}")
    print(f"  spearman(bpm, quality)                     = {sp(qb, qq):+.3f}")
    print(f"  partial(local_coverage, quality | bpm)     = {partial(loc, qq, qb):+.3f}   <- ~0 => coverage adds NOTHING beyond BPM")
    print(f"  partial(bpm, quality | local_coverage)     = {partial(qb, qq, loc):+.3f}   <- stays strong => BPM is the driver")
    print("\n  READ: if the fast bins (>165) hold a large share of train Hard AND still score low, coverage is NOT the")
    print("  cause -> the intrinsic defect is the onset/pattern head on fast/dense audio (the next probe).")


if __name__ == '__main__':
    main()

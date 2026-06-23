#!/usr/bin/env python3
"""
Groove-radar MANIFOLD realizability probe + manifold FIT/persist (for manifold-aware conditioning,
notes/radar_manifold_findings.md). The 5 radar dims are coupled (stream/volt/air/chaos cluster r 0.7-0.9;
freeze orthogonal) -> a radar POINT with one dim cranked while others are pinned is OFF-MANIFOLD (the chaos
OOD bug). The conditioning surface (src/generation/radar_manifold.py::RadarManifold) takes a PARTIAL/loose
user spec over named axes, (b) conditional-fills unspecified dims via the real Gaussian conditional, then
(c) projects onto the covariance ellipsoid -> a coherent on-manifold 5-vec for the existing radar+CFG path.

This script FITS the manifold over all real charts, SAVES it to cache/radar_manifold.npz (loaded by the
exporter), and for a set of named seed styles reports realizability: each target's Mahalanobis distance
(off-manifold-ness) + typicality percentile, whether it needed projection, and its nearest real charts
(does the combo exist, and what does it feel like?).

  python experiments/generation_typed/diag_radar_manifold.py [--difficulty 3]
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, sys
from pathlib import Path
import numpy as np, yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.data_splits import create_data_splits, create_datasets
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
from src.generation.radar_manifold import RadarManifold, DIMS

MANIFOLD_PATH = PROJECT_ROOT / "cache/radar_manifold.npz"
DIFF_NAMES = {0: 'Beginner', 1: 'Easy', 2: 'Medium', 3: 'Hard'}

# Seed styles: (name, spec). Unspecified dims are conditional-filled. Edit/extend freely.
SEED_STYLES = [
    ("Stream machine   (ex1)", "stream=high,chaos=low,air=low"),     # contradictory: stream~chaos r.80
    ("Power jumps      (ex2)", "air=high,voltage=high,stream=mod"),   # on the intensity manifold
    ("Glitch tech",            "chaos=high,air=low,stream=mod"),      # syncopation WITHOUT jumps
    ("Hold ballad",            "freeze=high,stream=low,chaos=low"),   # the orthogonal freeze axis, sparse
    ("Freeze storm",           "freeze=high,stream=high,air=high"),   # both axes maxed -- a rare corner?
    ("Pure minimal",           "stream=low,voltage=low,air=low,chaos=low,freeze=low"),
    ("Chaos flood",            "chaos=high,stream=high"),             # high-end manifold (sanity: realizable)
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--difficulty', type=int, default=3, help='difficulty bucket to probe (default Hard=3)')
    args = ap.parse_args()
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    tf, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                   use_metric_phase=True, use_highres_onset=True))
    tr, va, _ = create_datasets(train_files=tf, val_files=vf, test_files=[], audio_dir="data/",
                                max_sequence_length=msl, feature_extractor=ext, cache_dir='cache/samples_v3')
    titles = [m['chart'].title or Path(m['chart_file']).stem for ds in (tr, va) for m in ds.valid_samples]
    titles = np.array(titles)
    mani = RadarManifold.from_loaded_datasets(tr, va)
    mani.save(MANIFOLD_PATH)
    print(f"fit + saved manifold ({len(mani.vectors)} charts) -> {MANIFOLD_PATH}\n")

    DIFF = args.difficulty
    sel = mani.difficulties == DIFF
    titles_b = titles[sel]                                    # aligned to the bucket order in nearest()
    _, mu, _, _, real_d = mani._fit(DIFF)
    print(f"=== {DIFF_NAMES.get(DIFF, DIFF)} manifold (n={sel.sum()}) ===")
    print("mean: " + "  ".join(f"{d}={mu[i]:.3f}" for i, d in enumerate(DIMS)))
    print(f"real Mahalanobis d: median {np.median(real_d):.2f}, {int(mani.project_quantile*100)}th pct "
          f"{np.quantile(real_d, mani.project_quantile):.2f}  (lower = more typical)\n")

    for name, spec in SEED_STYLES:
        vec, info = mani.build_target(spec, DIFF)
        f = info['filled']
        print(f"▶ {name}   set: {spec}")
        print("   filled : " + " ".join(f"{DIMS[i]} {f[i]:.2f}" for i in range(5)))
        proj = (f"  -> PROJECTED to {info['max_mahalanobis']:.2f}: "
                + " ".join(f"{DIMS[i]} {vec[i]:.2f}" for i in range(5))) if info['projected'] else ""
        dens = f"   density~{info['density']:.3f} notes/frame (manifold, source-chart-free)" if info['density'] else ""
        print(f"   Mahalanobis d {info['mahalanobis']:.2f}  "
              f"({info['typicality_pct']:.0f}% of real {DIFF_NAMES.get(DIFF,'')} are MORE typical){proj}")
        print(dens) if dens else None
        for i, d in mani.nearest(f, DIFF, k=3):
            print(f"      ~ {titles_b[i][:30]:<32} [" +
                  " ".join(f"{mani.vectors[sel][i][j]:.2f}" for j in range(5)) + f"]  d={d:.2f}")
        print()


if __name__ == '__main__':
    main()

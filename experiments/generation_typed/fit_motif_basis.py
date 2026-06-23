#!/usr/bin/env python3
"""
Fit + persist the H15 MotifBasis (named radar-orthogonal motif-style knobs) to cache/motif_basis.npz, and
print the knobs. The motif analog of diag_radar_manifold.py's manifold fit. No training.

  /home/ybx/miniconda3/envs/stepmania-chart-gen/bin/python \
      experiments/generation_typed/fit_motif_basis.py [--K 12]
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
from src.generation.motif_codebook import MotifBasis

OUT = PROJECT_ROOT / "cache/motif_basis.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=12)
    args = ap.parse_args()

    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    tf, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                   use_metric_phase=True, use_highres_onset=True))
    tr, va, _ = create_datasets(train_files=tf, val_files=vf, test_files=[], audio_dir="data/",
                                max_sequence_length=msl, feature_extractor=ext, cache_dir='cache/samples_v3')

    basis = MotifBasis.from_loaded_datasets(tr, va, K=args.K)
    basis.save(OUT)
    print(f"fit + saved -> {OUT}\n")
    print(basis.describe())

    # round-trip + a sanity encode on a real chart
    b2 = MotifBasis.load(OUT)
    m0 = tr.valid_samples[0]
    v = b2.encode_chart(m0["chart_tensor"], m0["groove_radar"].to_vector())
    assert v.shape == (basis.K,)
    print(f"\nround-trip OK; example chart motif-knob vector (z): "
          + " ".join(f"{x:+.2f}" for x in v))


if __name__ == "__main__":
    main()

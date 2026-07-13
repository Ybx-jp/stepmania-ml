#!/usr/bin/env python3
"""
H15 sanity check (no training): do real charts at a motif KNOB's extremes actually PLAY like the named figure?

For each named knob, encode every real chart -> 12-d motif-knob vector, then list the songs at the + and -
extremes with their actually-most-used motifs. If knob 'jack<->sweep' high charts are visibly jack-heavy and
low charts sweep-heavy, the knob is a meaningful, steerable axis -> safe to condition on. (Grounds the basis
in artifacts; experiment-design Rule 8.)

  python \
      experiments/generation_typed/sanity_motif_knobs.py [--knobs 0 1 3 10]
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
from src.generation.motif_codebook import MotifBasis, motif_str

BASIS = PROJECT_ROOT / "cache/motif_basis.npz"


def top_motifs(basis, hist, n=3):
    order = np.argsort(hist)[::-1]
    out = []
    for j in order[:n]:
        if hist[j] <= 0:
            break
        W, c = basis.col_meta[j]
        out.append(f"[{motif_str(c)}]{hist[j]*100:.0f}%")
    return "  ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knobs", type=int, nargs="+", default=[0, 1, 3, 6, 10])
    ap.add_argument("--n", type=int, default=4, help="charts per extreme")
    args = ap.parse_args()

    basis = MotifBasis.load(BASIS)
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    tf, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                   use_metric_phase=True, use_highres_onset=True))
    tr, va, _ = create_datasets(train_files=tf, val_files=vf, test_files=[], audio_dir="data/",
                                max_sequence_length=msl, feature_extractor=ext, cache_dir='cache/samples_v3')

    V, hists, titles, diffs, seen = [], [], [], [], set()
    for ds in (tr, va):
        for m in ds.valid_samples:
            ct = m.get("chart_tensor"); cfk = m.get("chart_file", "")
            if ct is None or cfk in seen:
                continue
            seen.add(cfk)
            radar = m["groove_radar"].to_vector()
            h = basis.chart_histogram(ct)
            V.append(basis.encode(h, radar)); hists.append(h)
            titles.append((m['chart'].title or Path(cfk).stem)[:32]); diffs.append(m.get('difficulty_name', '?'))
    V = np.array(V); hists = np.array(hists); titles = np.array(titles); diffs = np.array(diffs)
    print(f"encoded {len(V)} distinct real charts through MotifBasis ({basis.K} knobs)\n")

    for k in args.knobs:
        a = basis.axis_info[k]
        print("=" * 92)
        print(f"KNOB {k}  '{a['label']}'   (stab {a['stability']:.2f}, radar |corr| {a['maxcorr']:.2f})")
        print(f"   + loads: " + "  ".join(f"[{motif_str(c)}]" for _, c in a['pos']))
        print(f"   - loads: " + "  ".join(f"[{motif_str(c)}]" for _, c in a['neg']))
        order = np.argsort(V[:, k])
        print("  --- HIGH (+) extreme ---")
        for i in order[::-1][:args.n]:
            print(f"    {V[i,k]:+5.1f}z  {titles[i]:<33} {diffs[i][:6]:<7} {top_motifs(basis, hists[i])}")
        print("  --- LOW (-) extreme ---")
        for i in order[:args.n]:
            print(f"    {V[i,k]:+5.1f}z  {titles[i]:<33} {diffs[i][:6]:<7} {top_motifs(basis, hists[i])}")
        print()


if __name__ == "__main__":
    main()

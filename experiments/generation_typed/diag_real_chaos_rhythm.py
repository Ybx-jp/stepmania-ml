#!/usr/bin/env python3
"""
What does REAL chaos look like rhythmically? (playtest 06-22: density-matched chaos conditioning collapsed
the quarter backbone into an 8th main line = the H4/H6 smear. The radar 'chaos' is DEFINED as a sum of
off-beat notes: quarter=0, 8th=0.5, 16th=1.0 -- so "raise chaos" literally means "more 8ths/16ths".)

Hypothesis: real charts get high chaos by ADDING off-beats on top of the quarter backbone (more notes =
higher DENSITY), keeping quarters as the backbone. We forced FIXED density while raising chaos -> the model
could only REPLACE quarters with 8ths (grid-fining smear). If real high-chaos charts have higher density
AND keep quarters as the plurality, then chaos must be applied WITH a density increase, not at fixed density
-- and the model's quarter->8th flip under density-matched conditioning is the artifact.

Bin real charts (val) by chaos-radar; per bin report n, density, and quarter/8th/16th SHARE + the quarter
NOTE-RATE (quarters per frame -- does the backbone itself survive, in absolute terms?).

  python experiments/generation_typed/diag_real_chaos_rhythm.py
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, sys
from pathlib import Path
import numpy as np, yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits
from src.data.dataset import StepManiaDataset
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_charts', type=int, default=300)
    ap.add_argument('--min_difficulty', type=int, default=2, help='Medium+ only (chaos lives at higher diff)')
    args = ap.parse_args()
    set_seed(42)
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    _, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                   use_metric_phase=True, use_highres_onset=True))
    ds = StepManiaDataset(chart_files=vf[:args.num_charts * 2], audio_dir="data/", max_sequence_length=msl,
                          feature_extractor=ext, cache_dir='cache/samples_v3')

    rows = []  # (chaos, density, q%, 8th%, 16th%, quarter_rate, diff)
    for i in range(len(ds.valid_samples)):
        if len(rows) >= args.num_charts:
            break
        meta = ds.valid_samples[i]
        if meta['difficulty_class'] < args.min_difficulty:
            continue
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None:
            continue
        typed = np.asarray(ds.parser.convert_to_tensor_typed(meta['chart'], nd))
        note = (typed != 0).any(1); T = len(note)
        if note.sum() < 64 or T < 256:
            continue
        t = np.arange(T); n = int(note.sum())
        q = note[t % 4 == 0].sum(); e = note[t % 4 == 2].sum(); s = note[(t % 4 == 1) | (t % 4 == 3)].sum()
        chaos = float(meta['groove_radar'].chaos)  # raw (0-200+), not normalized
        rows.append((chaos, n / T, 100 * q / n, 100 * e / n, 100 * s / n, q / T, meta['difficulty_class']))

    R = np.array(rows); chaos = R[:, 0]
    # bins by chaos quartile
    qs = np.quantile(chaos, [0, 0.25, 0.5, 0.75, 1.0])
    print(f"\n=== Real charts: rhythm vs chaos ({len(R)} charts, Medium+) ===")
    print(f"  raw chaos range [{chaos.min():.1f}, {chaos.max():.1f}]  (radar normalizes /200)")
    print(f"\n  {'chaos bin':<18} {'n':>4} {'density':>8} {'quarter%':>9} {'8th%':>7} {'16th%':>7} {'quarter/frame':>14}")
    for b in range(4):
        lo, hi = qs[b], qs[b + 1]
        sel = (chaos >= lo) & (chaos <= hi if b == 3 else chaos < hi)
        if sel.sum() == 0:
            continue
        m = R[sel].mean(0)
        print(f"  [{lo:6.1f},{hi:6.1f}]  {int(sel.sum()):>4} {m[1]:>8.3f} {m[2]:>8.1f}% {m[3]:>6.1f}% "
              f"{m[4]:>6.1f}% {m[5]:>13.3f}")
    # correlations
    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])
    print(f"\n  corr(chaos, density)      = {corr(chaos, R[:,1]):+.3f}   (real chaos comes WITH more notes?)")
    print(f"  corr(chaos, quarter share)= {corr(chaos, R[:,2]):+.3f}")
    print(f"  corr(chaos, quarter/frame)= {corr(chaos, R[:,5]):+.3f}   (does the quarter BACKBONE survive?)")
    print(f"  corr(chaos, 16th share)   = {corr(chaos, R[:,4]):+.3f}")
    print("\n  If density RISES with chaos and quarter/frame stays ~flat (backbone preserved) while 16th")
    print("  share rises -> real chaos ADDS off-beats on top of the backbone. Then fixed-density chaos")
    print("  conditioning is wrong by construction (forces quarter->8th replacement = the smear).")


if __name__ == '__main__':
    main()

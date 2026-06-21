#!/usr/bin/env python3
"""
Cheap offline gate before any retrain: does the high-res onset dim KEEP its off-beat discrimination
once it goes through the real AudioFeatureExtractor pipeline (slice -> high-res detect -> max-pool per
16th cell -> _normalize_envelope -> assemble into the full feature vector)?

Runs the actual extractor with use_highres_onset=True (+ Stage-1 flags so dim ordering matches a retrain),
pulls the new LAST dim out of get_aligned_features(), and recomputes per-phase note-vs-no-note ROC-AUC.
Should reproduce diag_highres_onset.py's standalone result (~0.66 16th-off). If it does, the retrain is
de-risked.
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import sys, glob
from pathlib import Path
import numpy as np, yaml
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits
from src.data.dataset import StepManiaDataset, DIFFICULTY_NAMES
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig

MAX_LEN = 1440
N_CHARTS = 60
MIN_DIFFICULTY_CLASS = 2
PHASES = ['on-beat', '8th-off', '16th-off']


def phase_mask(T):
    t = np.arange(T)
    return {'on-beat': (t % 4 == 0), '8th-off': (t % 4 == 2), '16th-off': (t % 2 == 1)}


def main():
    set_seed(42)
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    _, val_files, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    # Full Stage-1 + new dim, so the assembled vector matches what a retrain would see.
    ext = AudioFeatureExtractor(AudioFeatureConfig(
        use_chroma=True, use_hpss_onsets=True, use_metric_phase=True, use_highres_onset=True))
    ds = StepManiaDataset(chart_files=val_files, audio_dir="data/", max_sequence_length=msl,
                          feature_extractor=ext)

    auc = {ph: [] for ph in PHASES}
    align = {p: [] for p in range(4)}
    total_dim = None
    used = 0
    for i in range(len(ds.valid_samples)):
        if used >= N_CHARTS:
            break
        meta = ds.valid_samples[i]
        if meta['difficulty_class'] < MIN_DIFFICULTY_CLASS:
            continue
        chart = meta['chart']
        feats = ext.extract_from_chart(meta['audio_file'], chart)
        if feats is None:
            continue
        arr = feats.get_aligned_features()  # (T, D); new dim is appended LAST
        total_dim = arr.shape[1]
        nd = next((n for n in chart.note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        typed = np.asarray(ds.parser.convert_to_tensor_typed(chart, nd))
        T = min(arr.shape[0], typed.shape[0], MAX_LEN)
        hr = arr[:T, -1]                       # the high-res onset dim, as the model would receive it
        has_note = (typed[:T] != 0).any(axis=1).astype(np.int8)
        t = np.arange(T)
        for p in range(4):
            if (t % 4 == p).any(): align[p].append(hr[t % 4 == p].mean())
        for ph, m in phase_mask(T).items():
            lab = has_note[m]
            if lab.min() == lab.max(): continue
            auc[ph].append(roc_auc_score(lab, hr[m]))
        used += 1

    print(f"\nReal charts: {used} (difficulty >= {DIFFICULTY_NAMES[MIN_DIFFICULTY_CLASS]}); "
          f"assembled feature dim = {total_dim} (new high-res onset is dim {total_dim-1})")
    print("[ALIGNMENT] mean high-res dim by t%4 (downbeat should dominate): "
          + "  ".join(f"{p}={np.mean(align[p]):.3f}" for p in range(4)))
    print("\nPer-song ROC-AUC of the new dim, AS ASSEMBLED in the full feature vector:")
    print(f"{'phase':<10} {'AUC':>8}")
    print("-" * 20)
    for ph in PHASES:
        print(f"{ph:<10} {np.mean(auc[ph]):>8.3f}")
    print("-" * 20)
    print("Target (standalone, diag_highres_onset.py): on-beat 0.654, 8th-off 0.596, 16th-off 0.662.")
    print("Match -> feature survives the real pipeline; retrain de-risked. Mismatch -> debug normalization/align.")


if __name__ == '__main__':
    main()

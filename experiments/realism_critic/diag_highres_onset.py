#!/usr/bin/env python3
"""
H4 sub-fork: is the weak per-frame onset signal a RESOLUTION problem or a REPRESENTATION problem?

The shipped onset_env is computed at the chart-aligned hop (~one 16th note) with n_fft=2048 (~93ms),
so transients are smeared across the 16th grid. Here we recompute onset detection at HIGH resolution
(small hop) and max-pool into each 16th-grid cell, then re-score note-vs-no-note AUC by metric phase —
the same harness as diag_offbeat_signal.py. We also try librosa's superflux (transient-emphasized).

  high-res max-pooled AUC >> shipped AUC  -> RESOLUTION problem: a sharper onset feature would help
                                            (cheap fix: swap the onset extraction).
  high-res ~ shipped (both ~0.5 off-beat) -> REPRESENTATION problem: broadband onset just doesn't carry
                                            offbeat-placement; need a different signal (per-band/drum
                                            onset) OR offbeat placement isn't audio-onset-determined at
                                            all (lever = AR pattern/rhythm model, not features).
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import sys, glob
from pathlib import Path
import numpy as np, yaml, librosa
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits
from src.data.dataset import StepManiaDataset, DIFFICULTY_NAMES
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig

SR = 22050
HOP_HR = 128         # ~5.8ms high-res onset hop
N_FFT_HR = 512       # ~23ms window (vs shipped 2048/~93ms)
MAX_LEN = 1440
N_CHARTS = 60
MIN_DIFFICULTY_CLASS = 2
PHASES = ['on-beat', '8th-off', '16th-off']


def phase_mask(T):
    t = np.arange(T)
    return {'on-beat': (t % 4 == 0), '8th-off': (t % 4 == 2), '16th-off': (t % 2 == 1)}


def gridcell_maxpool(onset_hr, hop, T):
    """Max-pool a high-res (hop=HOP_HR) onset envelope into T grid cells of `hop` samples each."""
    out = np.zeros(T, dtype=np.float32)
    for t in range(T):
        lo = (t * hop) // HOP_HR
        hi = max(lo + 1, ((t + 1) * hop) // HOP_HR)
        seg = onset_hr[lo:hi]
        out[t] = seg.max() if seg.size else 0.0
    return out


def main():
    set_seed(42)
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    _, val_files, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig())
    ds = StepManiaDataset(chart_files=val_files, audio_dir="data/", max_sequence_length=msl,
                          feature_extractor=ext)  # no cache; we re-decode audio ourselves anyway

    variants = ['hr_flux', 'hr_superflux']
    auc = {v: {ph: [] for ph in PHASES} for v in variants}
    align = {p: [] for p in range(4)}
    used = 0
    for i in range(len(ds.valid_samples)):
        if used >= N_CHARTS:
            break
        meta = ds.valid_samples[i]
        if meta['difficulty_class'] < MIN_DIFFICULTY_CLASS:
            continue
        chart = meta['chart']
        af = meta['audio_file']
        if not os.path.exists(af):
            continue
        nd = next((n for n in chart.note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        try:
            y, _ = librosa.load(af, sr=SR, offset=0, duration=None)
        except Exception:
            continue
        start = int(chart.offset * SR) if chart.offset > 0 else 0
        end = start + int(chart.song_length_seconds * SR)
        start = max(0, min(start, len(y))); end = max(start, min(end, len(y)))
        y = y[start:end]
        hop = chart.hop_length
        T = min(len(y) // hop, MAX_LEN)
        if T < 64: continue
        typed = np.asarray(ds.parser.convert_to_tensor_typed(chart, nd))[:T]
        T = min(T, typed.shape[0])  # typed can be a frame short
        typed = typed[:T]
        has_note = (typed != 0).any(axis=1).astype(np.int8)

        flux_hr = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP_HR, n_fft=N_FFT_HR)
        sflux_hr = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP_HR, n_fft=N_FFT_HR,
                                                max_size=3)  # superflux: transient-emphasized
        feats = {'hr_flux': gridcell_maxpool(flux_hr, hop, T),
                 'hr_superflux': gridcell_maxpool(sflux_hr, hop, T)}
        # normalize per song (max) like the shipped feature
        for k in feats:
            mx = feats[k].max()
            if mx > 1e-8: feats[k] = feats[k] / mx

        pm = phase_mask(T)
        oe = feats['hr_flux']; t = np.arange(T)
        for p in range(4):
            if (t % 4 == p).any(): align[p].append(oe[t % 4 == p].mean())
        for ph, m in pm.items():
            lab = has_note[m]
            if lab.min() == lab.max(): continue
            for v in variants:
                auc[v][ph].append(roc_auc_score(lab, feats[v][m]))
        used += 1

    print(f"\nReal charts: {used} (difficulty >= {DIFFICULTY_NAMES[MIN_DIFFICULTY_CLASS]}); "
          f"high-res hop={HOP_HR} (~{1000*HOP_HR/SR:.1f}ms), n_fft={N_FFT_HR}")
    print("[ALIGNMENT] mean hr-onset by t%4 (downbeat should dominate): "
          + "  ".join(f"{p}={np.mean(align[p]):.3f}" for p in range(4)))
    print("\nPer-song ROC-AUC (mean over songs), high-res onset max-pooled into 16th cells:")
    hdr = f"{'phase':<10} " + " ".join(f"{v:>14}" for v in variants)
    print(hdr); print("-" * len(hdr))
    for ph in PHASES:
        print(f"{ph:<10} " + " ".join(f"{np.mean(auc[v][ph]):>14.3f}" for v in variants))
    print("-" * len(hdr))
    print("Compare to shipped onset_env (diag_offbeat_signal.py): on-beat ~0.57, off-beat ~0.53.")
    print(">> shipped -> resolution problem (swap onset feature).  ~ shipped -> representation problem.")


if __name__ == '__main__':
    main()

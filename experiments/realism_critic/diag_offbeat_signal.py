#!/usr/bin/env python3
"""
H4 fork diagnostic: is the off-beat-saliency signal ABSENT from the audio features, or
PRESENT-but-unused by the model?

Each audio frame is one 16th-note grid position (audio is BPM-aligned). For REAL charts, split
frames by metric phase and ask: within a phase bucket, does an audio feature distinguish
note-frames from no-note-frames? (ROC-AUC, no model involved.)

Phase buckets (timesteps_per_beat=4 -> 4 sixteenths per beat):
  on-beat  : t % 4 == 0   (quarter)
  8th-off  : t % 4 == 2    (the "and")
  16th-off : t % 2 == 1    (t%4 in {1,3})

Features probed (indices in the 41-dim vector):
  onset_env (13), onset_rate (14), perc_onset/HPSS (35), harm_onset/HPSS (36)

Read:
  off-beat AUC >> 0.5  -> the audio KNOWS which offbeats deserve notes; model's failure to render
                          event-driven syncopation is an OBJECTIVE/CONDITIONING problem (chaos is a
                          global bias that ignores this local signal). Fixable without new features.
  off-beat AUC ~ 0.5   -> the features can't resolve salient offbeats; need a richer/higher-res
                          audio representation. Features are the bottleneck.
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import sys, glob
from pathlib import Path
from collections import defaultdict
import numpy as np, yaml
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits
from src.data.dataset import StepManiaDataset, DIFFICULTY_NAMES
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig

MAX_LEN = 1440
N_CHARTS = 80
FEATS = {'onset_env': 13, 'onset_rate': 14, 'perc_onset': 35, 'harm_onset': 36}
# Only count charts dense enough to actually contain off-beat notes (skip near-empty Beginner).
MIN_DIFFICULTY_CLASS = 2  # Medium+ ; syncopation lives where it's dense


def phase_mask(T):
    t = np.arange(T)
    return {'on-beat': (t % 4 == 0), '8th-off': (t % 4 == 2), '16th-off': (t % 2 == 1)}


def main():
    set_seed(42)
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    _, val_files, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True, use_metric_phase=True))
    ds = StepManiaDataset(chart_files=val_files, audio_dir="data/", max_sequence_length=msl,
                          feature_extractor=ext, cache_dir='cache/samples_v2')

    # Per-song AUC, averaged across songs (controls for cross-song normalization differences).
    # For each feature also try a 3-frame max window (the STFT window ~93ms can blur the 16th grid,
    # so the onset peak for a frame may land on a neighbor).
    phases = ['on-beat', '8th-off', '16th-off']
    persong = {ph: {f: [] for f in FEATS} for ph in phases}
    persong_w = {ph: {f: [] for f in FEATS} for ph in phases}  # windowed (max over t-1..t+1)
    note_share = {ph: [] for ph in phases}
    align = {p: [] for p in range(4)}      # mean onset_env at t%4 == p (per song)
    align_dn = {'downbeat': [], 'other': []}  # t%16==0 vs rest
    used = 0
    for i in range(len(ds.valid_samples)):
        if used >= N_CHARTS:
            break
        meta = ds.valid_samples[i]
        if meta['difficulty_class'] < MIN_DIFFICULTY_CLASS:
            continue
        sample = ds[i]; T = min(int(sample['mask'].sum().item()), MAX_LEN)
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        audio = sample['audio'][:T].numpy().astype(np.float32)
        typed = ds.parser.convert_to_tensor_typed(meta['chart'], nd)[:T]
        has_note = (np.asarray(typed) != 0).any(axis=1).astype(np.int8)  # (T,)
        # Alignment sanity: mean onset strength by metric phase (aligned => downbeat dominates).
        oe = audio[:, FEATS['onset_env']]; t = np.arange(T)
        for p in range(4):
            mp = (t % 4 == p)
            if mp.any(): align[p].append(oe[mp].mean())
        align_dn['downbeat'].append(oe[t % 16 == 0].mean())
        align_dn['other'].append(oe[t % 16 != 0].mean())
        pm = phase_mask(T)
        for ph, m in pm.items():
            if m.sum() == 0: continue
            lab = has_note[m]
            if lab.min() == lab.max():  # need both classes in this song+phase to score
                continue
            note_share[ph].append(lab.mean())
            for f, idx in FEATS.items():
                x = audio[:, idx]
                xw = np.maximum.reduce([np.roll(x, s) for s in (-1, 0, 1)])  # 3-frame max
                persong[ph][f].append(roc_auc_score(lab, x[m]))
                persong_w[ph][f].append(roc_auc_score(lab, xw[m]))
        used += 1

    def show(title, store):
        print(f"\n{title}")
        hdr = f"{'phase':<10} {'note%':>6} {'nsongs':>6} " + " ".join(f"{f:>11}" for f in FEATS)
        print(hdr); print("-" * len(hdr))
        for ph in phases:
            ns = 100 * float(np.mean(note_share[ph]))
            n = len(store[ph]['onset_env'])
            cells = " ".join(f"{np.mean(store[ph][f]):>11.3f}" for f in FEATS)
            print(f"{ph:<10} {ns:>5.1f}% {n:>6} " + cells)

    print(f"\nReal charts: {used} (difficulty >= {DIFFICULTY_NAMES[MIN_DIFFICULTY_CLASS]})")
    print("\n[ALIGNMENT CHECK] mean onset_env by metric phase (aligned => t%4==0 downbeat dominates):")
    print("  t%4:  " + "  ".join(f"{p}={np.mean(align[p]):.3f}" for p in range(4)))
    print(f"  downbeat(t%16==0)={np.mean(align_dn['downbeat']):.3f}  other={np.mean(align_dn['other']):.3f}"
          f"  ratio={np.mean(align_dn['downbeat'])/np.mean(align_dn['other']):.3f}")
    print("Per-song ROC-AUC (mean over songs): does the feature distinguish note vs no-note WITHIN a phase?")
    show("[exact frame]", persong)
    show("[3-frame max window: tolerant of STFT smear across the 16th grid]", persong_w)
    print("\noff-beat AUC >> 0.5 -> signal present, model under-uses it (objective/conditioning fix).")
    print("off-beat AUC ~ 0.5  -> features can't resolve offbeats (need richer audio representation).")


if __name__ == '__main__':
    main()

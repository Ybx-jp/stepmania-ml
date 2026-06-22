#!/usr/bin/env python3
"""
Feature-informativeness probe for 16th PLACEMENT (notes/v7_additive_loss_design.md; user 06-22: the ceiling
is musical LINKAGE, not a flat density knob). Does a musically-specific cue beat the current MIXED high-res
onset (16th-localization AUC ~0.67) at predicting which 16th frames are real notes? Cheap, NO retrain.

Candidates (computed from audio, max-pooled to the 16th grid like the shipped high-res onset):
  mixed_hr   : current high-res onset (baseline = the model's existing 16th cue)
  perc_hr    : HPSS-PERCUSSIVE high-res onset (drum transients -> drum-driven 16th runs/fills)
  harm_hr    : HPSS-HARMONIC high-res onset (synth/lead accents)
  chroma_flux: positive chroma flux at the grid hop (melodic note attacks)
Per-song scalar: BPM vs real 16th-rate (real charts thin 16ths at high BPM?).

Metric: 16th-localization AUC — at 16th-phase frames (t%4 in {1,3}), real-16th-NOTE (1) vs no-note (0),
score = feature value. AUC > 0.67 (mixed_hr) => the cue carries placement signal the model lacks -> a
FEATURE retrain is the lever. All ~0.67 => not a feature problem (architecture / audio ambiguity).

  python experiments/generation_typed/diag_16th_features.py --num_songs 25 --min16 0.05
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, sys
from pathlib import Path
import numpy as np, librosa, yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits
from src.data.dataset import StepManiaDataset
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig

SR = 22050; HOP_HR = 128; NFFT_HR = 512


def pooled_onset(y, sr, grid_hop, T):
    """onset strength at the fine hop, max-pooled into each 16th-grid cell -> (T,) (same as the shipped one)."""
    o = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_HR, n_fft=NFFT_HR)
    out = np.zeros(T, dtype=np.float32)
    for t in range(T):
        lo = (t * grid_hop) // HOP_HR; hi = max(lo + 1, ((t + 1) * grid_hop) // HOP_HR)
        seg = o[lo:hi]; out[t] = seg.max() if seg.size else 0.0
    return out


def auc(scores, labels):
    labels = labels.astype(int); n1 = labels.sum(); n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    order = np.argsort(scores); rank = np.empty(len(scores), float); rank[order] = np.arange(len(scores))
    return (rank[labels == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_songs', type=int, default=25)
    ap.add_argument('--max_len', type=int, default=1024)
    ap.add_argument('--min_difficulty', type=int, default=2)
    ap.add_argument('--min16', type=float, default=0.05)
    args = ap.parse_args()
    set_seed(42)
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    _, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    # MUST match cache/samples_v3 (42-dim) or the dataset re-extracts audio for every song. We only use the
    # dataset for metadata (chart / audio_file / radar); features come fresh from librosa below.
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                   use_metric_phase=True, use_highres_onset=True))
    ds = StepManiaDataset(chart_files=vf[:args.num_songs * 8], audio_dir="data/", max_sequence_length=msl,
                          feature_extractor=ext, cache_dir='cache/samples_v3')
    print(f"dataset ready: {len(ds.valid_samples)} valid samples", flush=True)

    feats = ['mixed_hr', 'perc_hr', 'harm_hr']   # chroma_flux dropped: chroma_* segfaults in this env (numba)
    pool = {f: [] for f in feats}; lab = []   # pooled over 16th frames across songs
    bpm_x, rate16_y = [], []
    used = 0
    for i in range(len(ds.valid_samples)):
        if used >= args.num_songs:
            break
        meta = ds.valid_samples[i]
        if meta['difficulty_class'] < args.min_difficulty:
            continue
        chart = meta['chart']
        nd = next((n for n in chart.note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None:
            continue
        typed = np.asarray(ds.parser.convert_to_tensor_typed(chart, nd))
        note = (typed != 0).any(1); T0 = len(note); t = np.arange(T0)
        n = int(note.sum())
        if n < 32:
            continue
        s16 = (note & ((t % 4 == 1) | (t % 4 == 3))).sum() / max(n, 1)
        if s16 < args.min16:
            continue
        ap_ = meta.get('audio_file')
        if not ap_ or not os.path.exists(ap_):
            continue
        try:
            y_full, sr = librosa.load(ap_, sr=SR)
            st = max(0, int(chart.offset * sr)) if chart.offset > 0 else 0
            en = max(st, min(st + int(chart.song_length_seconds * sr), len(y_full)))
            y = y_full[st:en]
            if len(y) < sr:
                continue
            T = min(T0, args.max_len); grid_hop = chart.hop_length
            y_harm, y_perc = librosa.effects.hpss(y)
            fvals = {
                'mixed_hr': pooled_onset(y, sr, grid_hop, T),
                'perc_hr': pooled_onset(y_perc, sr, grid_hop, T),
                'harm_hr': pooled_onset(y_harm, sr, grid_hop, T),
            }
        except Exception as e:
            print(f"  (skip {meta['difficulty_name']}: {type(e).__name__})", flush=True); continue
        is16 = ((np.arange(T) % 4 == 1) | (np.arange(T) % 4 == 3))
        y16 = note[:T][is16]
        for f in feats:
            pool[f].append(fvals[f][is16])
        lab.append(y16)
        bpm_x.append(float(chart.bpm)); rate16_y.append((note[:T] & is16).sum() / max(note[:T].sum(), 1))
        used += 1
        print(f"  ..{used}/{args.num_songs} done", flush=True)

    labels = np.concatenate(lab)
    print(f"\n\n=== 16th-PLACEMENT feature probe ({used} chaotic songs, {len(labels)} 16th frames) ===")
    print(f"  16th-localization AUC (real-16th-note vs no-note, at 16th frames):")
    base = auc(np.concatenate(pool['mixed_hr']), labels)
    for f in feats:
        a = auc(np.concatenate(pool[f]), labels)
        tag = "  <- BASELINE (current model cue)" if f == 'mixed_hr' else (f"  (+{a-base:+.3f} vs mixed)" if not np.isnan(a) else "")
        print(f"  {f:<12} {a:.3f}{tag}")
    # combined perc+harm (simple logistic-free proxy: z-sum) vs mixed
    Z = []
    for f in ['perc_hr', 'harm_hr']:
        v = np.concatenate(pool[f]); Z.append((v - v.mean()) / (v.std() + 1e-8))
    print(f"  {'perc+harm':<12} {auc(np.sum(Z, 0), labels):.3f}  (z-sum, rough 'percussive+harmonic cues')")
    bx, ry = np.array(bpm_x), np.array(rate16_y)
    if bx.std() > 0:
        rb = np.argsort(np.argsort(bx)); rr = np.argsort(np.argsort(ry))
        print(f"\n  BPM vs real 16th-rate: Spearman {np.corrcoef(rb, rr)[0,1]:+.3f}  (BPM range [{bx.min():.0f},{bx.max():.0f}])")
    print("\n  perc_hr (or combined) AUC >> 0.67 -> musical-linkage FEATURE is the lever (retrain with it).")
    print("  all ~0.67 -> not a feature problem (architecture / audio ambiguity); reweighted-BCE rate fix is the cap.")


if __name__ == '__main__':
    main()

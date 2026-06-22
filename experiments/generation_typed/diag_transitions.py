#!/usr/bin/env python3
"""
H11 probe: does the model fail at TRANSITIONS (section boundaries)? Mechanistic prediction — the onset
head is audio-driven (adapts instantly), but the PATTERN head is autoregressive (momentum: it continues
the prior section's motif), so at a section boundary the choreography should LAG while onset stays fine.

Test: teacher-forced per-frame loss vs distance to audio section-boundaries.
  - boundaries = Foote novelty peaks on a self-similarity matrix of timbre (MFCC) + harmony (chroma).
  - onset loss = per-frame BCE(onset_logits, real onset); pattern loss = per-frame CE(pattern_logits,
    real which-panels) on note frames. Teacher-forced (real chart fed as context).
  - accumulate loss in a window around each boundary, normalized per song.
Prediction (H11): PATTERN loss spikes just after boundaries and decays; ONSET loss stays flat.
Also reports the song-START (cold-start) loss vs the song mean.
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, sys
from pathlib import Path
import numpy as np, torch, yaml
import torch.nn.functional as F
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits
from src.data.dataset import StepManiaDataset
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.typed import panels_to_pattern

GEN_CKPT = "checkpoints/gen_stage1/best_val.pt"
L = 16          # Foote half-kernel (4 beats)
W_BEFORE, W_AFTER = 8, 24  # window around a boundary (frames)
# timbre + harmony feature dims for the self-similarity matrix
SSM_DIMS = list(range(0, 13)) + list(range(23, 35))  # MFCC(13) + chroma(12)


def foote_boundaries(feat):
    f = feat - feat.mean(0, keepdims=True)
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
    S = f @ f.T
    a = np.arange(2 * L)
    sign = np.where((a[:, None] < L) == (a[None, :] < L), 1.0, -1.0)   # checkerboard
    gw = np.exp(-((a[:, None] - L + .5) ** 2 + (a[None, :] - L + .5) ** 2) / (2 * (L / 2) ** 2))
    ker = sign * gw
    T = len(feat); nov = np.zeros(T)
    for t in range(L, T - L):
        nov[t] = (S[t - L:t + L, t - L:t + L] * ker).sum()
    nov = np.maximum(nov, 0)
    pos = nov[nov > 0]
    prom = pos.std() if pos.size else 0.0
    pk, _ = find_peaks(nov, distance=L, prominence=max(prom, 1e-6))
    return pk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_songs', type=int, default=30); ap.add_argument('--max_len', type=int, default=1024)
    ap.add_argument('--min_difficulty', type=int, default=2)  # Medium+ (more structured)
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    _, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True, use_metric_phase=True))
    ds = StepManiaDataset(chart_files=vf[:args.num_songs * 4], audio_dir="data/", max_sequence_length=msl,
                          feature_extractor=ext, cache_dir='cache/samples_v2')
    gen = LayeredTypedChartGenerator(audio_dim=41, d_model=128, num_layers=4, onset_layers=2).to(device)
    gen.load_state_dict(torch.load(GEN_CKPT, map_location=device)['model_state_dict']); gen.eval()

    rel = np.arange(-W_BEFORE, W_AFTER + 1)
    prof = {'onset': np.zeros(len(rel)), 'pattern': np.zeros(len(rel))}
    cnt_o = np.zeros(len(rel)); cnt_p = np.zeros(len(rel))
    start_o, start_p, mean_o, mean_p, nb, used = [], [], [], [], 0, 0
    for i in range(len(ds.valid_samples)):
        if used >= args.num_songs: break
        meta = ds.valid_samples[i]
        if meta['difficulty_class'] < args.min_difficulty: continue
        sample = ds[i]; T = min(int(sample['mask'].sum().item()), args.max_len)
        if T < 256: continue
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        typed = np.asarray(ds.parser.convert_to_tensor_typed(meta['chart'], nd))[:T]
        if (typed != 0).any(1).sum() < 32: continue
        audio = sample['audio'][:T].unsqueeze(0).to(device)
        states = torch.from_numpy(typed).long().unsqueeze(0).to(device)
        diff = torch.tensor([meta['difficulty_class']], device=device)
        mask = torch.ones(1, T, dtype=torch.bool, device=device)
        with torch.no_grad():
            ol, pl, _ = gen(audio, states, diff, mask)
        onset_t = (states != 0).any(-1).float()
        o_loss = F.binary_cross_entropy_with_logits(ol[0], onset_t[0], reduction='none').cpu().numpy()  # (T,)
        active = (states != 0)
        pat_t = torch.from_numpy(panels_to_pattern(active.cpu().numpy())).clamp(min=0).to(device)  # (1,T)
        p_loss = F.cross_entropy(pl[0], pat_t[0], reduction='none').cpu().numpy()  # (T,)
        note = (typed != 0).any(1)
        p_loss = np.where(note, p_loss, np.nan)  # pattern loss only meaningful on note frames

        # boundaries from timbre+harmony SSM
        bnds = foote_boundaries(sample['audio'][:T, SSM_DIMS].numpy())
        # per-song normalization (so songs with different baselines are comparable)
        mo = np.nanmean(o_loss); mp = np.nanmean(p_loss)
        mean_o.append(mo); mean_p.append(mp)
        for b in bnds:
            for j, r in enumerate(rel):
                t = b + r
                if 0 <= t < T:
                    prof['onset'][j] += o_loss[t] / mo; cnt_o[j] += 1
                    if not np.isnan(p_loss[t]):
                        prof['pattern'][j] += p_loss[t] / mp; cnt_p[j] += 1
        nb += len(bnds)
        # cold-start: first 16 frames vs song mean
        start_o.append(np.nanmean(o_loss[:16]) / mo); start_p.append(np.nanmean(p_loss[:16]) / mp)
        used += 1

    po = prof['onset'] / np.maximum(cnt_o, 1)
    pp = prof['pattern'] / np.maximum(cnt_p, 1)
    print(f"\n=== H11 transition probe ({used} songs, {nb} boundaries, L={L}) ===")
    print("teacher-forced loss around audio section-boundaries, normalized to each song's mean (1.0 = avg).")
    print("Prediction: PATTERN spikes after the boundary (rel>=0) & decays; ONSET stays ~flat.\n")
    print(f"{'rel_frame':>9} {'onset':>7} {'pattern':>8}")
    for j, r in enumerate(rel):
        mark = "  <- boundary" if r == 0 else ""
        if r in (-8, -4, 0, 2, 4, 8, 12, 16, 24):
            print(f"{r:>9} {po[j]:>7.2f} {pp[j]:>8.2f}{mark}")
    print(f"\nboundary window [0..+8] mean:  onset {po[(rel>=0)&(rel<=8)].mean():.2f}  pattern {pp[(rel>=0)&(rel<=8)].mean():.2f}  (1.0=song avg)")
    print(f"cold-start (first 16 frames) mean: onset {np.mean(start_o):.2f}  pattern {np.mean(start_p):.2f}")


if __name__ == '__main__':
    main()

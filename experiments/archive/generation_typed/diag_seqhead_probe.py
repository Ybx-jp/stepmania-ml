#!/usr/bin/env python3
"""
De-risk the sequence-aware onset-head bet (notes/v7_additive_loss_design.md / playtest_log 06-22): is the
16th-localization ceiling (~0.67) an ARCHITECTURE limit (a wider-context head extracts more) or AUDIO
AMBIGUITY (the same fill has many valid chartings — inherently ~0.67)? Cheap, NO generator retrain.

Train small onset classifiers on the cached 42-dim features with INCREASING temporal receptive field, all
else equal, and measure held-out (val) 16th-localization AUC (real-16th-note vs no-note, at 16th frames):
  pf     : kernel=1, pure PER-FRAME (no context)
  k7     : one kernel-7 conv (~RF 7)
  k7x3   : three kernel-7 convs (~RF 19)
  dil    : dilated conv stack (kernel 3, dil 1/2/4/8 -> ~RF 31)
  attn   : pointwise + 1 self-attention layer (GLOBAL context)
Reads:
  16th-AUC RISES with context and exceeds ~0.67 -> ARCHITECTURE lever (build the sequence-aware head).
  16th-AUC FLAT across context -> AUDIO AMBIGUITY ceiling; a fancier head won't help -> stop chasing placement.

  python experiments/generation_typed/diag_seqhead_probe.py --epochs 6
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig

AD = 42


class OnsetProbe(nn.Module):
    def __init__(self, kind, d=64):
        super().__init__()
        self.kind = kind
        if kind == 'pf':
            self.net = nn.Sequential(nn.Conv1d(AD, d, 1), nn.ReLU(), nn.Conv1d(d, 1, 1))
        elif kind == 'k7':
            self.net = nn.Sequential(nn.Conv1d(AD, d, 7, padding=3), nn.ReLU(), nn.Conv1d(d, 1, 1))
        elif kind == 'k7x3':
            self.net = nn.Sequential(nn.Conv1d(AD, d, 7, padding=3), nn.ReLU(),
                                     nn.Conv1d(d, d, 7, padding=3), nn.ReLU(),
                                     nn.Conv1d(d, d, 7, padding=3), nn.ReLU(), nn.Conv1d(d, 1, 1))
        elif kind == 'dil':
            self.net = nn.Sequential(
                nn.Conv1d(AD, d, 3, padding=1, dilation=1), nn.ReLU(),
                nn.Conv1d(d, d, 3, padding=2, dilation=2), nn.ReLU(),
                nn.Conv1d(d, d, 3, padding=4, dilation=4), nn.ReLU(),
                nn.Conv1d(d, d, 3, padding=8, dilation=8), nn.ReLU(), nn.Conv1d(d, 1, 1))
        elif kind == 'attn':
            self.proj = nn.Conv1d(AD, d, 1)
            self.attn = nn.TransformerEncoderLayer(d, 4, dim_feedforward=2 * d, batch_first=True)
            self.out = nn.Linear(d, 1)

    def forward(self, x):  # x: (B, T, AD)
        if self.kind == 'attn':
            h = torch.relu(self.proj(x.transpose(1, 2)).transpose(1, 2))  # (B,T,d)
            h = self.attn(h)
            return self.out(h).squeeze(-1)  # (B,T)
        return self.net(x.transpose(1, 2)).squeeze(1)  # (B,T)


def auc(scores, labels):
    labels = labels.astype(int); n1 = labels.sum(); n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    order = np.argsort(scores); rank = np.empty(len(scores), float); rank[order] = np.arange(len(scores))
    return (rank[labels == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)


def collect(ds, cap, max_songs):
    out = []
    for i in range(min(len(ds.valid_samples), max_songs)):
        s = ds[i]; meta = ds.valid_samples[i]
        T = int(s['mask'].sum().item())
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None:
            continue
        typed = np.asarray(ds.parser.convert_to_tensor_typed(meta['chart'], nd))
        T = min(T, cap, typed.shape[0])
        if T < 128:
            continue
        out.append((s['audio'][:T, :AD].numpy().astype(np.float32), (typed[:T] != 0).any(1).astype(np.float32)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=6); ap.add_argument('--max_len', type=int, default=1024)
    ap.add_argument('--max_train', type=int, default=1500); ap.add_argument('--max_val', type=int, default=400)
    ap.add_argument('--bs', type=int, default=16); ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    tf, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                   use_metric_phase=True, use_highres_onset=True))
    tr_ds, va_ds, _ = create_datasets(train_files=tf, val_files=vf, test_files=[], audio_dir="data/",
                                      max_sequence_length=msl, feature_extractor=ext, cache_dir='cache/samples_v3')
    print("collecting...", flush=True)
    train = collect(tr_ds, args.max_len, args.max_train); val = collect(va_ds, args.max_len, args.max_val)
    print(f"train={len(train)} val={len(val)} songs", flush=True)
    rng = np.random.default_rng(42)

    def batches(data, shuffle):
        idx = np.arange(len(data));
        if shuffle: rng.shuffle(idx)
        for i in range(0, len(idx), args.bs):
            chunk = [data[j] for j in idx[i:i + args.bs]]
            T = max(len(a) for a, _ in chunk); B = len(chunk)
            X = np.zeros((B, T, AD), np.float32); Y = np.zeros((B, T), np.float32); M = np.zeros((B, T), bool)
            for b, (a, y) in enumerate(chunk):
                X[b, :len(a)] = a; Y[b, :len(y)] = y; M[b, :len(y)] = True
            yield (torch.from_numpy(X).to(device), torch.from_numpy(Y).to(device), torch.from_numpy(M).to(device))

    # pos_weight for the ~20% onset rate
    posrate = np.mean([y.mean() for _, y in train]); pw = torch.tensor((1 - posrate) / posrate, device=device)
    print(f"onset rate {posrate:.3f}, pos_weight {float(pw):.2f}\n", flush=True)
    print(f"  {'head':<8} {'val onset-AUC':>14} {'val 16th-AUC':>14}   (16th = real-16th-note vs no-note @ 16th frames)")
    results = {}
    for kind in ['pf', 'k7', 'k7x3', 'dil', 'attn']:
        set_seed(42)
        m = OnsetProbe(kind).to(device); opt = torch.optim.Adam(m.parameters(), lr=args.lr)
        for ep in range(args.epochs):
            m.train()
            for X, Y, M in batches(train, True):
                opt.zero_grad()
                logit = m(X)
                loss = nn.functional.binary_cross_entropy_with_logits(logit[M], Y[M], pos_weight=pw)
                loss.backward(); opt.step()
        # eval
        m.eval(); ps, ys, is16s = [], [], []
        with torch.no_grad():
            for X, Y, M in batches(val, False):
                p = torch.sigmoid(m(X)).cpu().numpy()
                B, T = Y.shape; t = np.arange(T)
                i16 = ((t % 4 == 1) | (t % 4 == 3))[None, :].repeat(B, 0)
                mm = M.cpu().numpy()
                ps.append(p[mm]); ys.append(Y.cpu().numpy()[mm]); is16s.append(i16[mm])
        P = np.concatenate(ps); Yv = np.concatenate(ys); I16 = np.concatenate(is16s)
        a_all = auc(P, Yv); a_16 = auc(P[I16], Yv[I16])
        results[kind] = a_16
        print(f"  {kind:<8} {a_all:>14.3f} {a_16:>14.3f}", flush=True)
    print(f"\n  16th-AUC by context: " + "  ".join(f"{k}={results[k]:.3f}" for k in results))
    rise = results['attn'] - results['pf']
    print(f"  attn - per-frame = {rise:+.3f}")
    print(f"  RISES with context & beats ~0.67 -> ARCHITECTURE lever (build sequence-aware head).")
    print(f"  FLAT -> AUDIO AMBIGUITY ceiling; a fancier head won't help -> stop chasing 16th placement.")


if __name__ == '__main__':
    main()

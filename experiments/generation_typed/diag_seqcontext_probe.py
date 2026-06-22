#!/usr/bin/env python3
"""
CORRECTED de-risk (the prior diag_seqhead_probe tested AUDIO context only; user 06-22: placement depends on
the NOTE SEQUENCE — preceding/following notes — not just audio). Question: can CAUSAL note-sequence context
(preceding real notes, teacher-forced) predict real-16th PLACEMENT better than audio alone (~0.65)?

Three onset predictors, all predict onset[t] (note vs no-note); measure held-out 16th-localization AUC
(real-16th-note vs no-note, at 16th frames):
  audio   : non-causal conv on the 42-dim audio features (= current head's info; baseline ~0.65)
  seq     : CAUSAL conv on the preceding per-panel notes (notes[<t]) ONLY — no audio. Tests run-coherence:
            given the rhythm so far, is the next 16th predictable? (strictly past -> no leakage)
  both    : audio (non-causal) + causal note-sequence

Reads:
  seq / both 16th-AUC >> audio (~0.65) -> placement is SEQUENCE-DETERMINED (run coherence), NOT audio-
    ambiguity -> a sequence-aware onset head is justified (collapse is a training problem, not predictability).
  seq / both ~ audio -> even the chart context can't predict 16ths -> ambiguity is real.

  python experiments/generation_typed/diag_seqcontext_probe.py --epochs 6
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

AD = 42; NP = 4


def causal_conv(cin, cout, k, dil):
    # left-pad by (k-1)*dil so output[t] depends ONLY on input[<=t] (strictly past once input is shifted)
    return nn.Sequential(nn.ConstantPad1d(((k - 1) * dil, 0), 0.0), nn.Conv1d(cin, cout, k, dilation=dil))


class Probe(nn.Module):
    def __init__(self, kind, d=64):
        super().__init__()
        self.kind = kind
        if kind in ('audio', 'both'):
            self.audio = nn.Sequential(nn.Conv1d(AD, d, 3, padding=1, dilation=1), nn.ReLU(),
                                       nn.Conv1d(d, d, 3, padding=2, dilation=2), nn.ReLU(),
                                       nn.Conv1d(d, d, 3, padding=4, dilation=4), nn.ReLU())
        if kind in ('seq', 'both'):
            self.s1 = causal_conv(NP, d, 3, 1); self.s2 = causal_conv(d, d, 3, 2)
            self.s3 = causal_conv(d, d, 3, 4); self.s4 = causal_conv(d, d, 3, 8)
        din = d * (2 if kind == 'both' else 1)
        self.out = nn.Sequential(nn.Conv1d(din, d, 1), nn.ReLU(), nn.Conv1d(d, 1, 1))

    def seq_feat(self, notes_prev):  # notes_prev: (B,T,4) = real notes shifted +1 (strictly past)
        h = notes_prev.transpose(1, 2)
        for c in (self.s1, self.s2, self.s3, self.s4):
            h = torch.relu(c(h))
        return h  # (B,d,T)

    def forward(self, audio, notes_prev):
        feats = []
        if self.kind in ('audio', 'both'):
            feats.append(self.audio(audio.transpose(1, 2)))
        if self.kind in ('seq', 'both'):
            feats.append(self.seq_feat(notes_prev))
        h = torch.cat(feats, 1) if len(feats) > 1 else feats[0]
        return self.out(h).squeeze(1)  # (B,T)


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
        notes = (typed[:T] != 0).astype(np.float32)              # (T,4) per-panel note
        out.append((s['audio'][:T, :AD].numpy().astype(np.float32), notes))
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
        idx = np.arange(len(data))
        if shuffle: rng.shuffle(idx)
        for i in range(0, len(idx), args.bs):
            chunk = [data[j] for j in idx[i:i + args.bs]]
            T = max(len(a) for a, _ in chunk); B = len(chunk)
            X = np.zeros((B, T, AD), np.float32); N = np.zeros((B, T, NP), np.float32); M = np.zeros((B, T), bool)
            for b, (a, n) in enumerate(chunk):
                X[b, :len(a)] = a; N[b, :len(n)] = n; M[b, :len(n)] = True
            Y = (N.sum(-1) > 0).astype(np.float32)               # onset target
            Nprev = np.zeros_like(N); Nprev[:, 1:] = N[:, :-1]    # CAUSAL: notes shifted +1 (strictly past)
            yield (torch.from_numpy(X).to(device), torch.from_numpy(Nprev).to(device),
                   torch.from_numpy(Y).to(device), torch.from_numpy(M).to(device))

    posrate = np.mean([(n.sum(-1) > 0).mean() for _, n in train]); pw = torch.tensor((1 - posrate) / posrate, device=device)
    print(f"onset rate {posrate:.3f}\n", flush=True)
    print(f"  {'predictor':<8} {'val onset-AUC':>14} {'val 16th-AUC':>14}")
    res = {}
    for kind in ['audio', 'seq', 'both']:
        set_seed(42); m = Probe(kind).to(device); opt = torch.optim.Adam(m.parameters(), lr=args.lr)
        for ep in range(args.epochs):
            m.train()
            for X, Np, Y, M in batches(train, True):
                opt.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(m(X, Np)[M], Y[M], pos_weight=pw)
                loss.backward(); opt.step()
        m.eval(); ps, ys, i16s = [], [], []
        with torch.no_grad():
            for X, Np, Y, M in batches(val, False):
                p = torch.sigmoid(m(X, Np)).cpu().numpy(); B, T = Y.shape; t = np.arange(T)
                i16 = ((t % 4 == 1) | (t % 4 == 3))[None].repeat(B, 0); mm = M.cpu().numpy()
                ps.append(p[mm]); ys.append(Y.cpu().numpy()[mm]); i16s.append(i16[mm])
        P = np.concatenate(ps); Yv = np.concatenate(ys); I = np.concatenate(i16s)
        res[kind] = (auc(P, Yv), auc(P[I], Yv[I]))
        print(f"  {kind:<8} {res[kind][0]:>14.3f} {res[kind][1]:>14.3f}", flush=True)
    print(f"\n  16th-AUC: audio={res['audio'][1]:.3f}  seq={res['seq'][1]:.3f}  both={res['both'][1]:.3f}")
    print(f"  seq/both >> audio -> placement is SEQUENCE-determined (run coherence) -> sequence-aware onset head justified.")
    print(f"  seq/both ~ audio -> chart context can't predict 16ths either -> ambiguity is real.")


if __name__ == '__main__':
    main()

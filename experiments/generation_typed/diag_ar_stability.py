#!/usr/bin/env python3
"""
AR-stability de-risk (notes/sequence_aware_onset_plan.md): the seq-context probe showed onset placement is
sequence-determined (16th-AUC 0.935 teacher-forced). But teacher-forcing doesn't prove the model survives
AUTOREGRESSIVE rollout (own predictions) -- that's the exact failure that killed AR-onset in Stage 2
(collapse to empty). Does an AUDIO-ANCHORED sequence-aware onset head stay stable + coherent under AR?

Onset model predicts onset[t] from audio[t] (non-causal conv, the ANCHOR) + causal onset-history[<t].
Train teacher-forced; then roll out AR (feed own Bernoulli samples). Variants:
  both : audio + causal onset-history   (the proposed head)
  seq  : onset-history only (NO audio anchor) -- expected to collapse/drift
Measure on AR rollout vs REAL: density (collapse->0 or explode?), run-length structure (coherent runs?).

Reads:
  both AR density ~ real & run-lengths ~ real -> audio anchor keeps it STABLE + COHERENT -> build the head.
  both collapses (density->0) -> drift survives the anchor -> scheduled sampling MANDATORY before integration.
  seq collapses but both stable -> confirms the audio anchor is what prevents collapse (the design insight).

  python experiments/generation_typed/diag_ar_stability.py --epochs 6
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


def causal(cin, cout, k, dil):
    return nn.Sequential(nn.ConstantPad1d(((k - 1) * dil, 0), 0.0), nn.Conv1d(cin, cout, k, dilation=dil))


class OnsetAR(nn.Module):
    def __init__(self, kind, d=64):
        super().__init__()
        self.kind = kind
        if kind in ('audio', 'both'):
            self.audio = nn.Sequential(nn.Conv1d(AD, d, 3, padding=1), nn.ReLU(),
                                       nn.Conv1d(d, d, 3, padding=2, dilation=2), nn.ReLU(),
                                       nn.Conv1d(d, d, 3, padding=4, dilation=4), nn.ReLU())
        if kind in ('seq', 'both'):
            self.s1 = causal(1, d, 3, 1); self.s2 = causal(d, d, 3, 2)
            self.s3 = causal(d, d, 3, 4); self.s4 = causal(d, d, 3, 8)
        din = d * (2 if kind == 'both' else 1)
        self.out = nn.Sequential(nn.Conv1d(din, d, 1), nn.ReLU(), nn.Conv1d(d, 1, 1))

    def forward(self, audio, onset_prev):  # onset_prev: (B,T,1) real/own onset shifted +1
        f = []
        if self.kind in ('audio', 'both'):
            f.append(self.audio(audio.transpose(1, 2)))
        if self.kind in ('seq', 'both'):
            h = onset_prev.transpose(1, 2)
            for c in (self.s1, self.s2, self.s3, self.s4):
                h = torch.relu(c(h))
            f.append(h)
        return self.out(torch.cat(f, 1) if len(f) > 1 else f[0]).squeeze(1)  # (B,T)


def run_stats(onset):  # onset: 1d binary -> (mean_run, %isolated, %in_run>=4) over onset runs
    runs = []; c = 0
    for x in onset:
        if x: c += 1
        elif c: runs.append(c); c = 0
    if c: runs.append(c)
    if not runs: return (0.0, 0.0, 0.0)
    runs = np.array(runs)
    return (runs.mean(), 100 * (runs == 1).mean(), 100 * (runs >= 4).sum() / max(len(onset), 1) * 1.0)


def collect(ds, cap, n):
    out = []
    for i in range(min(len(ds.valid_samples), n)):
        s = ds[i]; meta = ds.valid_samples[i]; T = int(s['mask'].sum().item())
        nd = next((x for x in meta['chart'].note_data if x.difficulty_name == meta['difficulty_name']
                   and x.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        typed = np.asarray(ds.parser.convert_to_tensor_typed(meta['chart'], nd))
        T = min(T, cap, typed.shape[0])
        if T < 128: continue
        out.append((s['audio'][:T, :AD].numpy().astype(np.float32), (typed[:T] != 0).any(1).astype(np.float32)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=6); ap.add_argument('--max_len', type=int, default=512)
    ap.add_argument('--max_train', type=int, default=1500); ap.add_argument('--roll_songs', type=int, default=10)
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
    train = collect(tr_ds, args.max_len, args.max_train); roll = collect(va_ds, args.max_len, args.roll_songs * 3)[:args.roll_songs]
    print(f"train={len(train)} roll={len(roll)} songs", flush=True)
    rng = np.random.default_rng(42)

    def batches(data):
        idx = np.arange(len(data)); rng.shuffle(idx)
        for i in range(0, len(idx), args.bs):
            chunk = [data[j] for j in idx[i:i + args.bs]]; T = max(len(a) for a, _ in chunk); B = len(chunk)
            X = np.zeros((B, T, AD), np.float32); Y = np.zeros((B, T), np.float32); M = np.zeros((B, T), bool)
            for b, (a, y) in enumerate(chunk):
                X[b, :len(a)] = a; Y[b, :len(y)] = y; M[b, :len(y)] = True
            Op = np.zeros((B, T, 1), np.float32); Op[:, 1:, 0] = Y[:, :-1]
            yield (torch.from_numpy(X).to(device), torch.from_numpy(Op).to(device),
                   torch.from_numpy(Y).to(device), torch.from_numpy(M).to(device))

    pr = np.mean([y.mean() for _, y in train]); pw = torch.tensor((1 - pr) / pr, device=device)
    real_dens = np.mean([y.mean() for _, y in roll]); real_runs = np.mean([run_stats(y)[0] for _, y in roll])
    real_iso = np.mean([run_stats(y)[1] for _, y in roll])
    print(f"onset rate {pr:.3f}; REAL roll: density {real_dens:.3f}, mean-run {real_runs:.2f}, isolated {real_iso:.0f}%\n", flush=True)
    print(f"  {'head':<6} {'AR density':>11} {'mean-run':>9} {'isolated%':>10}  (REAL: d={real_dens:.3f} run={real_runs:.2f} iso={real_iso:.0f}%)")
    for kind in ['both', 'seq']:
        set_seed(42); m = OnsetAR(kind).to(device); opt = torch.optim.Adam(m.parameters(), lr=args.lr)
        for ep in range(args.epochs):
            m.train()
            for X, Op, Y, M in batches(train):
                opt.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(m(X, Op)[M], Y[M], pos_weight=pw)
                loss.backward(); opt.step()
        # AR rollout (Bernoulli, feed own predictions)
        m.eval(); dens, runs, iso = [], [], []
        with torch.no_grad():
            for a, y in roll:
                T = len(y); A = torch.from_numpy(a).unsqueeze(0).to(device)
                act = np.zeros(T)                      # actual generated onset
                og = torch.zeros(1, T, 1, device=device)  # og[t] = act[t-1] (causal input)
                for t in range(T):
                    p = torch.sigmoid(m(A, og)[0, t])
                    s = float(torch.bernoulli(p)); act[t] = s
                    if t + 1 < T:
                        og[0, t + 1, 0] = s
                dens.append(act.mean()); r = run_stats(act); runs.append(r[0]); iso.append(r[1])
        print(f"  {kind:<6} {np.mean(dens):>11.3f} {np.mean(runs):>9.2f} {np.mean(iso):>9.0f}%", flush=True)
    print(f"\n  both density~real & runs~real -> audio-anchored AR is STABLE+COHERENT -> build the head.")
    print(f"  both density->0 -> drift survives anchor -> scheduled sampling mandatory.")
    print(f"  seq collapses but both stable -> the audio anchor prevents collapse (design confirmed).")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Phase-3 generation prototype (notes/phase3_generative_design.md): the core paradigm test. AR onset EXPLODES;
frozen-context refinement is stable but can't bootstrap from v4's bad C0. A JOINT generative model
(mask-and-predict: mask cells, predict from the rest, iterate) generates FROM SCRATCH on the learned chart
manifold -- no AR self-feedback loop, no dependence on a bad first pass. Does it produce STABLE, real-shaped
onset placement?

Onset-only prototype. Model predicts onset[t] from audio[t] (non-causal) + a PARTIAL onset context
(revealed frames = real onset, masked = 0) + a mask-indicator channel. Train with RANDOM masking (predict
masked onsets). Generate by ITERATIVE confidence-based unmasking from all-masked (MaskGIT-style). Measure the
generated chart vs real: density (stable? not exploded/collapsed), phase distribution, run-length; + the
teacher-forced AUC sanity (does it learn placement given partial real context).

Reads:
  generated density ~ real, phase dist ~ real, run-length ~ real, NO explosion/collapse -> joint generation
    is STABLE + real-shaped FROM SCRATCH (solves AR + refinement-bootstrap) -> paradigm viable -> next: panels
    + critic eval.
  exploded / collapsed / phase off -> joint gen doesn't fix it either.

  python experiments/generation_typed/diag_maskpredict_proto.py --epochs 8
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


class MaskPredict(nn.Module):
    """audio (non-causal) + partial onset context [revealed_onset, is_revealed] (non-causal) -> onset logit."""
    def __init__(self, d=96):
        super().__init__()
        self.audio = nn.Sequential(nn.Conv1d(AD, d, 3, padding=1), nn.ReLU(),
                                   nn.Conv1d(d, d, 3, padding=2, dilation=2), nn.ReLU(),
                                   nn.Conv1d(d, d, 3, padding=4, dilation=4), nn.ReLU(),
                                   nn.Conv1d(d, d, 3, padding=8, dilation=8), nn.ReLU())
        self.ctx = nn.Sequential(nn.Conv1d(2, d, 7, padding=3), nn.ReLU(),
                                 nn.Conv1d(d, d, 7, padding=6, dilation=2), nn.ReLU(),
                                 nn.Conv1d(d, d, 7, padding=12, dilation=4), nn.ReLU())
        self.out = nn.Sequential(nn.Conv1d(2 * d, d, 1), nn.ReLU(), nn.Conv1d(d, 1, 1))

    def forward(self, audio, ctx2):  # audio (B,T,AD); ctx2 (B,T,2)=[revealed_onset, is_revealed]
        a = self.audio(audio.transpose(1, 2)); c = self.ctx(ctx2.transpose(1, 2))
        return self.out(torch.cat([a, c], 1)).squeeze(1)


def phase_frac(onset):
    t = np.arange(len(onset)); n = max(int(onset.sum()), 1)
    return (100 * onset[t % 4 == 0].sum() / n, 100 * onset[t % 4 == 2].sum() / n,
            100 * onset[(t % 4 == 1) | (t % 4 == 3)].sum() / n)


def run_mean(o):
    runs = []; c = 0
    for x in o:
        if x: c += 1
        elif c: runs.append(c); c = 0
    if c: runs.append(c)
    return float(np.mean(runs)) if runs else 0.0


def auc(s, l):
    l = l.astype(int); n1 = l.sum(); n0 = len(l) - n1
    if n1 == 0 or n0 == 0: return float('nan')
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(len(s))
    return (r[l == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)


def collect(ds, cap, n):
    out = []
    for i in range(min(len(ds.valid_samples), n)):
        s = ds[i]; meta = ds.valid_samples[i]; T = int(s['mask'].sum().item())
        nd = next((x for x in meta['chart'].note_data if x.difficulty_name == meta['difficulty_name']
                   and x.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        typed = np.asarray(ds.parser.convert_to_tensor_typed(meta['chart'], nd)); T = min(T, cap, typed.shape[0])
        if T < 128: continue
        out.append((s['audio'][:T, :AD].numpy().astype(np.float32), (typed[:T] != 0).any(1).astype(np.float32)))
    return out


@torch.no_grad()
def generate(m, audio, T, K, device):
    """iterative confidence unmasking from all-masked -> binary onset (T,)."""
    A = torch.from_numpy(audio).unsqueeze(0).to(device)
    onset = np.zeros(T); revealed = np.zeros(T, bool)
    for step in range(K):
        ctx = np.stack([onset * revealed, revealed.astype(np.float32)], -1)[None]
        p = torch.sigmoid(m(A, torch.from_numpy(ctx.astype(np.float32)).to(device)))[0].cpu().numpy()
        conf = np.maximum(p, 1 - p); conf[revealed] = -1            # only consider masked
        n_rev = int(np.ceil(T / K)) if step < K - 1 else T
        order = np.argsort(conf)[::-1]; pick = order[:max(0, n_rev - revealed.sum())]
        onset[pick] = (p[pick] > 0.5).astype(float); revealed[pick] = True
        if revealed.all(): break
    return onset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=8); ap.add_argument('--max_len', type=int, default=768)
    ap.add_argument('--max_train', type=int, default=1500); ap.add_argument('--gen_songs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=16); ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--steps', type=int, default=10)
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
    train = collect(tr_ds, args.max_len, args.max_train); gen = collect(va_ds, args.max_len, args.gen_songs * 2)[:args.gen_songs]
    print(f"train={len(train)} gen={len(gen)} songs", flush=True)
    rng = np.random.default_rng(42)
    pr = np.mean([y.mean() for _, y in train]); pw = torch.tensor((1 - pr) / pr, device=device)
    m = MaskPredict().to(device); opt = torch.optim.Adam(m.parameters(), lr=args.lr)

    def batch():
        idx = rng.choice(len(train), min(args.bs, len(train)), replace=False)
        ch = [train[j] for j in idx]; T = max(len(a) for a, _ in ch); B = len(ch)
        X = np.zeros((B, T, AD), np.float32); Y = np.zeros((B, T), np.float32)
        C = np.zeros((B, T, 2), np.float32); LM = np.zeros((B, T), bool)   # loss mask = masked & valid
        for b, (a, y) in enumerate(ch):
            t = len(y); X[b, :t] = a; Y[b, :t] = y
            r = rng.uniform(0.15, 1.0); rev = (rng.random(t) > r)          # reveal (1-r) fraction
            C[b, :t, 0] = y * rev; C[b, :t, 1] = rev.astype(np.float32)
            LM[b, :t] = ~rev                                               # predict the MASKED frames
        return (torch.from_numpy(X).to(device), torch.from_numpy(C).to(device),
                torch.from_numpy(Y).to(device), torch.from_numpy(LM).to(device))
    nb = (len(train) + args.bs - 1) // args.bs
    for ep in range(args.epochs):
        m.train()
        for _ in range(nb):
            X, C, Y, LM = batch()
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(m(X, C)[LM], Y[LM], pos_weight=pw)
            loss.backward(); opt.step()

    # teacher-forced AUC sanity: reveal 50% real context, predict masked-16th placement
    m.eval(); tf_auc = []
    with torch.no_grad():
        for a, y in gen:
            T = len(y); A = torch.from_numpy(a).unsqueeze(0).to(device)
            rev = rng.random(T) > 0.5
            C = np.stack([y * rev, rev.astype(np.float32)], -1)[None].astype(np.float32)
            p = torch.sigmoid(m(A, torch.from_numpy(C).to(device)))[0].cpu().numpy()
            t = np.arange(T); m16 = ((t % 4 == 1) | (t % 4 == 3)) & (~rev)
            if m16.sum() > 4: tf_auc.append(auc(p[m16], y[m16]))

    real_ph = np.mean([phase_frac(y) for _, y in gen], 0); real_d = np.mean([y.mean() for _, y in gen])
    real_run = np.mean([run_mean(y) for _, y in gen])
    gen_ph, gen_d, gen_run = [], [], []
    for a, y in gen:
        g = generate(m, a, len(y), args.steps, device)
        gen_ph.append(phase_frac(g)); gen_d.append(g.mean()); gen_run.append(run_mean(g))
    gp = np.mean(gen_ph, 0)
    print(f"\n=== mask-predict prototype ({len(gen)} songs, {args.steps} unmask steps) ===")
    print(f"  teacher-forced 16th-AUC (given 50% real context): {np.nanmean(tf_auc):.3f}  (sanity; ceiling ~0.93)")
    print(f"  {'source':<10} {'density':>8} {'run-mean':>9} {'q%':>6} {'8th%':>6} {'16th%':>6}")
    print(f"  {'REAL':<10} {real_d:>8.3f} {real_run:>9.2f} {real_ph[0]:>5.0f}% {real_ph[1]:>5.0f}% {real_ph[2]:>5.0f}%")
    print(f"  {'gen(mask)':<10} {np.mean(gen_d):>8.3f} {np.mean(gen_run):>9.2f} {gp[0]:>5.0f}% {gp[1]:>5.0f}% {gp[2]:>5.0f}%")
    print(f"\n  gen density~real & run~real & phase~real -> joint gen STABLE+real-shaped from scratch -> paradigm")
    print(f"  viable (next: panels + critic). vs AR(exploded d0.73 run5.7) / refinement(bootstrap-limited 0.666).")


if __name__ == '__main__':
    main()

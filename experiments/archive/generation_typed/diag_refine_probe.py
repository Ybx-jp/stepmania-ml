#!/usr/bin/env python3
"""
Refinement-mechanism de-risk (notes/sequence_aware_onset_plan.md): the AR onset explodes (runaway self-
feedback). Iterative refinement breaks the loop into passes over a FROZEN chart: condition onset on a fixed
approximate chart (non-causal, neighbors both sides), refine, freeze, refine again. Question: does a
non-causal frozen-context refiner RECOVER good placement from a degraded chart, and does iterating CONVERGE
(stable) instead of exploding?

Setup (denoising): train refiner onset[t] = f(audio[t] non-causal, NOISY-chart-context[t] non-causal). The
noisy context simulates a rough first pass (drop real notes -- esp. 16ths, like the audio-only head's
16th-under -- + add spurious off-beats). Target = real onset. Then ITERATE on held-out songs: C0 = corrupt(
real); C_k = threshold(refiner(audio, C_{k-1}), real density). Measure per pass: 16th-localization AUC
(refiner p_on vs real-16th), run-length mean (coherence), density (stability), and change ||C_k - C_{k-1}||.

Reads:
  refined 16th-AUC >> input C0 AUC, rises toward ~0.9, run-mean -> real, density stable, passes CONVERGE
    -> frozen-context refinement WORKS (no explosion; captures the sequence signal) -> build it (+ critic).
  AUC flat / runs explode / no convergence -> refinement doesn't recover placement -> reassess.

  python experiments/generation_typed/diag_refine_probe.py --epochs 6
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
from src.generation.typed_model import LayeredTypedChartGenerator

AD = 42
V4_CKPT = "checkpoints/gen_highres_v4/best_val.pt"


class Refiner(nn.Module):
    """audio (non-causal) + NON-CAUSAL context-conv over a frozen chart -> onset logit."""
    def __init__(self, d=64):
        super().__init__()
        self.audio = nn.Sequential(nn.Conv1d(AD, d, 3, padding=1), nn.ReLU(),
                                   nn.Conv1d(d, d, 3, padding=2, dilation=2), nn.ReLU(),
                                   nn.Conv1d(d, d, 3, padding=4, dilation=4), nn.ReLU())
        # non-causal symmetric context conv (sees neighbors BOTH sides of the frozen chart)
        self.ctx = nn.Sequential(nn.Conv1d(1, d, 7, padding=3), nn.ReLU(),
                                 nn.Conv1d(d, d, 7, padding=6, dilation=2), nn.ReLU(),
                                 nn.Conv1d(d, d, 7, padding=12, dilation=4), nn.ReLU())
        self.out = nn.Sequential(nn.Conv1d(2 * d, d, 1), nn.ReLU(), nn.Conv1d(d, 1, 1))

    def forward(self, audio, ctx):  # audio (B,T,AD); ctx (B,T,1) frozen chart onset
        a = self.audio(audio.transpose(1, 2)); c = self.ctx(ctx.transpose(1, 2))
        return self.out(torch.cat([a, c], 1)).squeeze(1)


def corrupt(onset, rng, p_drop16=0.5, p_drop=0.2, p_add=0.06):
    """simulate a rough first pass: drop real notes (16ths harder) + add spurious off-beats."""
    T = len(onset); t = np.arange(T); is16 = (t % 4 == 1) | (t % 4 == 3)
    out = onset.copy()
    drop = rng.random(T) < np.where(is16, p_drop16, p_drop)
    out[drop & (onset > 0)] = 0.0
    add = (rng.random(T) < p_add) & (onset == 0) & (t % 4 != 0)   # spurious off-beat adds
    out[add] = 1.0
    return out.astype(np.float32)


def auc(s, l):
    l = l.astype(int); n1 = l.sum(); n0 = len(l) - n1
    if n1 == 0 or n0 == 0: return float('nan')
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(len(s))
    return (r[l == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)


def run_mean(onset):
    runs = []; c = 0
    for x in onset:
        if x: c += 1
        elif c: runs.append(c); c = 0
    if c: runs.append(c)
    return float(np.mean(runs)) if runs else 0.0


def collect(ds, cap, n, v4=None, device=None):
    """returns (audio, real_onset, c0). c0 = v4's audio-only onset thresholded to real density (the REAL
    rough first pass) if v4 given, else None (caller uses synthetic corruption)."""
    out = []
    for i in range(min(len(ds.valid_samples), n)):
        s = ds[i]; meta = ds.valid_samples[i]; T = int(s['mask'].sum().item())
        nd = next((x for x in meta['chart'].note_data if x.difficulty_name == meta['difficulty_name']
                   and x.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        typed = np.asarray(ds.parser.convert_to_tensor_typed(meta['chart'], nd)); T = min(T, cap, typed.shape[0])
        if T < 128: continue
        audio = s['audio'][:T, :AD].numpy().astype(np.float32); onset = (typed[:T] != 0).any(1).astype(np.float32)
        c0 = None
        if v4 is not None:
            d = float(onset.mean()); rad = torch.from_numpy(meta['groove_radar'].to_vector().astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                A = torch.from_numpy(audio).unsqueeze(0).to(device)
                p = torch.sigmoid(v4.onset_logits(v4.encode_audio(A), torch.tensor([meta['difficulty_class']], device=device), radar=rad))[0].cpu().numpy()
            c0 = (p > np.quantile(p, 1 - d)).astype(np.float32)   # real audio-only rough pass
        out.append((audio, onset, c0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=6); ap.add_argument('--max_len', type=int, default=768)
    ap.add_argument('--max_train', type=int, default=1500); ap.add_argument('--eval_songs', type=int, default=60)
    ap.add_argument('--bs', type=int, default=16); ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--passes', type=int, default=3)
    ap.add_argument('--real_c0', action='store_true',
                    help='use v4 audio-only onset (thresholded to real density) as the rough first pass C0 '
                         '(the REAL error distribution) instead of synthetic corruption.')
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    tf, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                   use_metric_phase=True, use_highres_onset=True))
    tr_ds, va_ds, _ = create_datasets(train_files=tf, val_files=vf, test_files=[], audio_dir="data/",
                                      max_sequence_length=msl, feature_extractor=ext, cache_dir='cache/samples_v3')
    v4 = None
    if args.real_c0:
        v4 = LayeredTypedChartGenerator(audio_dim=AD, d_model=128, num_layers=4, onset_layers=2).to(device)
        v4.load_state_dict(torch.load(V4_CKPT, map_location=device)['model_state_dict']); v4.eval()
        print("real_c0: using v4 audio-only onset as the rough first pass", flush=True)
    print("collecting...", flush=True)
    train = collect(tr_ds, args.max_len, args.max_train, v4, device)
    ev = collect(va_ds, args.max_len, args.eval_songs * 2, v4, device)[:args.eval_songs]
    print(f"train={len(train)} eval={len(ev)} songs", flush=True)
    rng = np.random.default_rng(42)
    pr = np.mean([y.mean() for _, y, _ in train]); pw = torch.tensor((1 - pr) / pr, device=device)

    m = Refiner().to(device); opt = torch.optim.Adam(m.parameters(), lr=args.lr)
    def batches():
        idx = np.arange(len(train)); rng.shuffle(idx)
        for i in range(0, len(idx), args.bs):
            ch = [train[j] for j in idx[i:i + args.bs]]; T = max(len(a) for a, _, _ in ch); B = len(ch)
            X = np.zeros((B, T, AD), np.float32); Y = np.zeros((B, T), np.float32)
            C = np.zeros((B, T, 1), np.float32); M = np.zeros((B, T), bool)
            for b, (a, y, c0) in enumerate(ch):
                X[b, :len(a)] = a; Y[b, :len(y)] = y; M[b, :len(y)] = True
                C[b, :len(y), 0] = c0 if c0 is not None else corrupt(y, rng)
            return (torch.from_numpy(X).to(device), torch.from_numpy(C).to(device),
                    torch.from_numpy(Y).to(device), torch.from_numpy(M).to(device))
    n_batches = (len(train) + args.bs - 1) // args.bs
    for ep in range(args.epochs):
        m.train()
        for _ in range(n_batches):
            X, C, Y, M = batches()
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(m(X, C)[M], Y[M], pos_weight=pw)
            loss.backward(); opt.step()

    real_run = np.mean([run_mean(y) for _, y, _ in ev]); real_dens = np.mean([y.mean() for _, y, _ in ev])
    print(f"\nREAL: density {real_dens:.3f}, run-mean {real_run:.2f}\n", flush=True)
    print(f"  {'pass':<8} {'16th-AUC':>9} {'run-mean':>9} {'density':>9} {'Δ vs prev':>10}")
    m.eval()
    # per song: iterate refinement on a frozen context, measure
    auc_p = [[] for _ in range(args.passes + 1)]; run_p = [[] for _ in range(args.passes + 1)]
    den_p = [[] for _ in range(args.passes + 1)]; chg_p = [[] for _ in range(args.passes + 1)]
    with torch.no_grad():
        for a, y, c0 in ev:
            T = len(y); t = np.arange(T); is16 = (t % 4 == 1) | (t % 4 == 3); d = y.mean()
            A = torch.from_numpy(a).unsqueeze(0).to(device)
            C0 = c0 if c0 is not None else corrupt(y, rng)         # rough first pass (real v4 onset if --real_c0)
            # pass 0 = input chart quality (AUC of C0 itself as a 'score' is binary; use the refiner's p on C0)
            prev = C0
            for k in range(args.passes + 1):
                if k == 0:
                    p = C0.astype(np.float32)                      # input; AUC of binary input
                    Ck = C0
                else:
                    ctx = torch.from_numpy(prev).view(1, T, 1).to(device)
                    p = torch.sigmoid(m(A, ctx))[0].cpu().numpy()
                    tau = np.quantile(p, 1 - d); Ck = (p > tau).astype(np.float32)
                auc_p[k].append(auc(p[is16], y[is16])); run_p[k].append(run_mean(Ck)); den_p[k].append(Ck.mean())
                chg_p[k].append(np.abs(Ck - prev).mean()); prev = Ck
    for k in range(args.passes + 1):
        tag = 'C0(input)' if k == 0 else f'refine {k}'
        print(f"  {tag:<8} {np.nanmean(auc_p[k]):>9.3f} {np.mean(run_p[k]):>9.2f} {np.mean(den_p[k]):>9.3f} "
              f"{np.mean(chg_p[k]):>10.3f}", flush=True)
    print(f"\n  AUC rises C0->refine & run-mean -> real ({real_run:.2f}) & density stable & Δ->0 (converges)")
    print(f"  -> frozen-context refinement WORKS -> build it (+ critic-guided loop). Flat/diverge -> reassess.")


if __name__ == '__main__':
    main()

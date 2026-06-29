#!/usr/bin/env python3
"""PROBE (2026-06-29): ANALYSIS-BY-SYNTHESIS viability — does g(chart -> audio onset/energy features) reconstruct
the REAL audio better from the REAL chart than from a differently-placed chart? If yes, the inverse model carries
the 16th-placement signal the FORWARD audio->onset head can't (0.65), and can serve as a CRITIC/likelihood in a
P(chart|audio) ∝ P(audio|chart)·P(chart) loop (the taste critic = the prior). User idea, 06-29.

WHY this can beat the forward 0.65: forward predicts each 16th POINTWISE -> throws away joint structure. Recon is
GLOBAL/JOINT: a coherent 16th-RUN makes a distinctive audio texture (a roll); the same count scattered does not. So
even when each 16th is locally ambiguous, the configuration is determined — and recon error reads it.

TARGET = the 5 placement-relevant audio dims (onset/energy envelopes; pitch/timbre/grid-phase EXCLUDED — placement
can't explain them): [13 onset_env, 14 onset_rate, 35 perc_onset, 36 harm_onset, 41 highres_onset].

CONTRASTS (recon MSE of REAL audio, lower=better):
  real chart        : the correct placement (train-distribution)
  c0 chart          : the DEPLOYED generator's placement (same song, ~same density) — the deployment question
  corrupted-real    : real chart with 16ths randomly RELOCATED (density+panels preserved) — isolates placement
METRIC SUBSETS: all valid frames / 16th frames (t%4∈{1,3}) / DISPUTED frames (real vs c0 disagree on a note).
READ: real ≪ c0 (esp. 16th/disputed) -> g sees placement -> analysis-by-synthesis VIABLE.
      real ≈ c0 but real ≪ corrupted -> placement matters but c0 already explains audio ~as well as real.
      real ≈ corrupted -> audio is placement-blind -> idea DEAD (clean bound).
"""
import warnings, os, sys, argparse
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
from pathlib import Path
import numpy as np, torch, torch.nn as nn
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils.reproducibility import set_seed

TRAINC0 = PROJECT_ROOT / "cache/seqctx_trainc0_cache.npz"
EVALC0  = PROJECT_ROOT / "cache/seqctx_c0_cache.npz"
TGT = [13, 14, 35, 36, 41]   # onset/energy audio dims
NP = 4


class TCN(nn.Module):
    """Non-causal dilated temporal conv: receptive field ~31 frames (sees a 16th RUN as texture)."""
    def __init__(self, cin=NP, c=64, cout=len(TGT)):
        super().__init__()
        def block(ci, co, d): return nn.Sequential(
            nn.Conv1d(ci, co, 3, padding=d, dilation=d), nn.ReLU(), nn.BatchNorm1d(co))
        self.net = nn.Sequential(block(cin, c, 1), block(c, c, 2), block(c, c, 4), block(c, c, 8))
        self.head = nn.Conv1d(c, cout, 1)
    def forward(self, x):                 # x: (B,T,NP)
        return self.head(self.net(x.transpose(1, 2))).transpose(1, 2)   # (B,T,len(TGT))


def corrupt_placement(chart, rng):
    """Relocate 16th-frame onset rows to random OTHER 16th slots (density+panel content preserved, runs destroyed)."""
    out = chart.copy(); T = chart.shape[0]
    pos16 = np.array([t for t in range(T) if t % 4 in (1, 3)])
    occ = np.array([t for t in pos16 if chart[t].any()])
    if len(occ) < 2: return out
    rows = chart[occ].copy()
    out[pos16] = 0.0                                   # clear all 16th slots
    dest = rng.choice(pos16, size=len(occ), replace=False)
    out[dest] = rows                                   # scatter the same rows to random 16th slots
    return out


def batches(data, bs, key, device, rng=None, shuffle=False, corrupt=False):
    idx = np.arange(len(data))
    if shuffle and rng is not None: rng.shuffle(idx)
    crng = np.random.default_rng(0)
    for i in range(0, len(idx), bs):
        chunk = [data[j] for j in idx[i:i+bs]]; T = max(a.shape[0] for a, _, _ in chunk); B = len(chunk)
        X = np.zeros((B, T, NP), np.float32); Yt = np.zeros((B, T, len(TGT)), np.float32); M = np.zeros((B, T), bool)
        for b, (audio, real, c0) in enumerate(chunk):
            ch = {'real': real, 'c0': c0}[key]
            if corrupt: ch = corrupt_placement(real, crng)
            t = min(audio.shape[0], ch.shape[0])       # min guard (mismatched-song control has unequal lengths)
            X[b, :t] = ch[:t]; Yt[b, :t] = audio[:t, TGT]; M[b, :t] = True
        yield (torch.from_numpy(X).to(device), torch.from_numpy(Yt).to(device), torch.from_numpy(M).to(device))


def train_g(train, device, epochs, bs, lr):
    set_seed(42); g = TCN().to(device); opt = torch.optim.Adam(g.parameters(), lr=lr); rng = np.random.default_rng(0)
    for ep in range(epochs):
        g.train()
        for X, Yt, M in batches(train, bs, 'real', device, rng, shuffle=True):
            opt.zero_grad(); pred = g(X)
            loss = ((pred - Yt) ** 2)[M].mean(); loss.backward(); opt.step()
    return g


def recon_mse(g, data, key, device, bs, corrupt=False):
    """Per-frame MSE over TGT dims; returns (all, 16th, disputed) means vs the REAL audio target."""
    g.eval(); errs_all, errs_16, errs_disp = [], [], []
    with torch.no_grad():
        for bi, (X, Yt, M) in enumerate(batches(data, bs, key, device, corrupt=corrupt)):
            e = ((g(X) - Yt) ** 2).mean(-1).cpu().numpy()         # (B,T) per-frame MSE
            mm = M.cpu().numpy(); B, T = e.shape; t = np.arange(T)
            is16 = ((t % 4 == 1) | (t % 4 == 3))[None].repeat(B, 0)
            # disputed: real-vs-c0 note disagreement (recompute from the cache rows in this batch)
            chunk = data[bi*bs:bi*bs+B]
            disp = np.zeros((B, T), bool)
            for b, (_, real, c0) in enumerate(chunk):
                tt = real.shape[0]; disp[b, :tt] = (real.any(-1) != c0.any(-1))
            errs_all.append(e[mm]); errs_16.append(e[mm & is16]); errs_disp.append(e[mm & disp])
    return (np.concatenate(errs_all).mean(), np.concatenate(errs_16).mean(),
            np.concatenate(errs_disp).mean() if sum(len(x) for x in errs_disp) else float('nan'))


def load(p): d = np.load(p, allow_pickle=True)['data']; return [(a, r, c) for a, r, c in d]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=15); ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    a = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train, val = load(TRAINC0), load(EVALC0)
    print(f"train={len(train)} | eval={len(val)} | target dims={TGT}\n", flush=True)
    g = train_g(train, device, a.epochs, a.bs, a.lr)
    # POSITIVE CONTROLS (experiment-design Rule 11): the predict-mean floor + does g depend on WHICH song's chart?
    mean_mse = float(np.mean([((au[:, TGT] - au[:, TGT].mean(0, keepdims=True)) ** 2).mean() for au, _, _ in val]))
    val_mis = [(val[(i + 1) % len(val)][0], val[i][1], val[i][2]) for i in range(len(val))]  # audio_{i+1} vs chart_i
    print(f"  predict-mean baseline MSE (no-skill floor): {mean_mse:.4f}", flush=True)
    print(f"  {'chart fed to g':<18} {'recon-all':>10} {'recon-16th':>11} {'recon-disputed':>15}", flush=True)
    r_real = recon_mse(g, val, 'real', device, a.bs)
    r_c0   = recon_mse(g, val, 'c0',   device, a.bs)
    r_corr = recon_mse(g, val, 'real', device, a.bs, corrupt=True)
    r_mis  = recon_mse(g, val_mis, 'real', device, a.bs)               # POSITIVE CONTROL: wrong-song audio target
    for name, r in [('real', r_real), ('c0 (deployed)', r_c0), ('corrupted-real', r_corr),
                    ('MISMATCH-song', r_mis)]:
        print(f"  {name:<18} {r[0]:>10.4f} {r[1]:>11.4f} {r[2]:>15.4f}", flush=True)
    g_learned = r_mis[0] - r_real[0]
    print(f"\n  POSITIVE CONTROL: mismatch−real recon-all = {g_learned:+.4f}  "
          f"(must be ≫0, and real ≪ predict-mean {mean_mse:.4f}, else g didn't learn chart→audio → probe void).", flush=True)
    print("\n  INTERPRET (recon of REAL audio; lower=better):", flush=True)
    print(f"   16th-frame:  real={r_real[1]:.4f}  c0={r_c0[1]:.4f}  corrupted={r_corr[1]:.4f}", flush=True)
    print(f"   disputed:    real={r_real[2]:.4f}  c0={r_c0[2]:.4f}", flush=True)
    d_c0 = 100 * (r_c0[1] - r_real[1]) / r_real[1]; d_cor = 100 * (r_corr[1] - r_real[1]) / r_real[1]
    print(f"   c0 is {d_c0:+.1f}% worse than real @16th; corrupted is {d_cor:+.1f}% worse.", flush=True)
    print("   >> real ≪ c0 (& disputed) -> g SEES placement -> analysis-by-synthesis VIABLE (build the critic).", flush=True)
    print("   >> real ≈ c0 ≈ corrupted -> audio placement-blind -> DEAD (clean bound).", flush=True)


if __name__ == "__main__":
    main()

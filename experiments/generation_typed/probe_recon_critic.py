#!/usr/bin/env python3
"""PROBE v2 (2026-06-29): ANALYSIS-BY-SYNTHESIS as a CONTRASTIVE MATCHING CRITIC (the right object).
v1 (probe_recon_audio.py) regressed audio from chart -> VOID (positive control failed: a binary chart has no
amplitude info, so g couldn't beat predict-mean). The likelihood a synthesis loop needs is a COMPATIBILITY SCORE
D(chart, audio) -> "do this placement and this audio go together?", not a generator. Well-posed + self-controlling.

D = chart-encoder (TCN over notes) + audio-encoder (TCN over the 5 onset/energy dims) -> concat -> joint TCN ->
MASKED mean-pool -> logit. NO BatchNorm (the v1 padding bug). Trained BCE: positives = real (chart_i, audio_i);
negatives = (a) CORRUPTED-PLACEMENT (corrupt(chart_i), audio_i) [same density+panels+audio, 16ths scrambled] and
(b) MISMATCH-SONG (chart_i, audio_j) [sanity].

DECISIVE METRIC (held-out 28 eval songs, paired AUC, higher=more separable):
  AUC(real vs MISMATCH-song)  = POSITIVE CONTROL — must be ≫0.5 or D is broken (probe void).
  AUC(real vs CORRUPTED-plc)  = THE MEASUREMENT — does the AUDIO distinguish coherent placement from scrambled?
    ≫0.5 -> audio carries the 16th-placement signal jointly -> analysis-by-synthesis VIABLE (build the critic).
    ≈0.5 (while control fires) -> audio is placement-blind beyond density -> idea DEAD (clean, controlled bound).
"""
import warnings, os, sys, argparse
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
from pathlib import Path
import numpy as np, torch, torch.nn as nn
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils.reproducibility import set_seed
from diag_seqcontext_probe import auc
from probe_recon_audio import TGT, corrupt_placement, load

NP = 4


def tcn(cin, chans, dils):
    layers, c = [], cin
    for co, d in zip(chans, dils):
        layers += [nn.Conv1d(c, co, 3, padding=d, dilation=d), nn.ReLU()]; c = co
    return nn.Sequential(*layers)


class Critic(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.ce = tcn(NP, [c, c, c], [1, 2, 4])
        self.ae = tcn(len(TGT), [c, c, c], [1, 2, 4])
        self.joint = tcn(2 * c, [c, c // 2], [8, 1])
        self.head = nn.Linear(c // 2, 1)
    def forward(self, chart, audio, mask):                 # (B,T,NP),(B,T,len(TGT)),(B,T)
        h = torch.cat([self.ce(chart.transpose(1, 2)), self.ae(audio.transpose(1, 2))], 1)
        h = self.joint(h).transpose(1, 2)                  # (B,T,c/2)
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)    # MASKED mean-pool (no padding leak)
        return self.head(pooled).squeeze(-1)               # (B,)


def make_examples(data, rng, kinds=('pos', 'corr', 'mis'), rand_mis=False):
    """Each item -> (chart, audio, label). pos=real match; corr=corrupted placement; mis=mismatch song.
    rand_mis: pick a RANDOM other song's audio (training diversity) vs the deterministic (i+1) (eval repro)."""
    ex = []
    for i, (audio, real, c0) in enumerate(data):
        if 'pos' in kinds:  ex.append((real, audio, 1))
        if 'c0' in kinds:   ex.append((c0, audio, 0))                  # deployed generator's placement (coherent, ~same density)
        if 'corr' in kinds: ex.append((corrupt_placement(real, rng), audio, 0))
        if 'mis' in kinds:
            j = (i + 1) % len(data)
            if rand_mis:
                j = int(rng.integers(len(data)));  j = (j + 1) % len(data) if j == i else j
            ex.append((real, data[j][0], 0))
    return ex


def batches(ex, bs, device, shuffle, rng=None):
    idx = np.arange(len(ex))
    if shuffle and rng is not None: rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [ex[j] for j in idx[i:i + bs]]; B = len(chunk)
        T = max(min(c.shape[0], a.shape[0]) for c, a, _ in chunk)
        C = np.zeros((B, T, NP), np.float32); A = np.zeros((B, T, len(TGT)), np.float32)
        M = np.zeros((B, T), bool); Y = np.zeros(B, np.float32)
        for b, (ch, au, lab) in enumerate(chunk):
            t = min(ch.shape[0], au.shape[0]); C[b, :t] = ch[:t]; A[b, :t] = au[:t, TGT]; M[b, :t] = True; Y[b] = lab
        yield (torch.from_numpy(C).to(device), torch.from_numpy(A).to(device),
               torch.from_numpy(M).to(device), torch.from_numpy(Y).to(device))


def score(model, ex, device, bs):
    model.eval(); ss, ys = [], []
    with torch.no_grad():
        for C, A, M, Y in batches(ex, bs, device, shuffle=False):
            ss.append(torch.sigmoid(model(C, A, M)).cpu().numpy()); ys.append(Y.cpu().numpy())
    return np.concatenate(ss), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--train_negs', default='mis', help="comma kinds for TRAINING negatives: mis|corr|mis,corr "
                    "(default mis = FORCE the audio path; corr lets D shortcut to chart-coherence)")
    a = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train, val = load(PROJECT_ROOT / "cache/seqctx_trainc0_cache.npz"), load(PROJECT_ROOT / "cache/seqctx_c0_cache.npz")
    tkinds = ('pos',) + tuple(a.train_negs.split(','))
    print(f"train={len(train)} | eval={len(val)} | audio dims={TGT} | train negatives={tkinds[1:]}\n", flush=True)
    rng = np.random.default_rng(0)
    model = Critic().to(device); opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    for ep in range(a.epochs):
        ex = make_examples(train, rng, kinds=tkinds, rand_mis=True); model.train()
        for C, A, M, Y in batches(ex, a.bs, device, shuffle=True, rng=rng):
            opt.zero_grad(); loss = nn.functional.binary_cross_entropy_with_logits(model(C, A, M), Y)
            loss.backward(); opt.step()

    # eval: paired AUC, real(pos) vs each negative type, on held-out songs
    s_pos, _  = score(model, make_examples(val, rng, ('pos',)),  device, a.bs)
    s_c0, _   = score(model, make_examples(val, rng, ('c0',)),   device, a.bs)
    s_corr, _ = score(model, make_examples(val, rng, ('corr',)), device, a.bs)
    s_mis, _  = score(model, make_examples(val, rng, ('mis',)),  device, a.bs)
    def pair_auc(pos, neg):
        s = np.concatenate([pos, neg]); y = np.concatenate([np.ones_like(pos), np.zeros_like(neg)]); return auc(s, y)
    auc_mis  = pair_auc(s_pos, s_mis)
    auc_corr = pair_auc(s_pos, s_corr)
    auc_c0   = pair_auc(s_pos, s_c0)
    print(f"  mean D-score: real={s_pos.mean():.3f}  c0={s_c0.mean():.3f}  corrupted={s_corr.mean():.3f}  mismatch={s_mis.mean():.3f}", flush=True)
    print(f"\n  AUC(real vs MISMATCH-song) = {auc_mis:.3f}   <- POSITIVE CONTROL (must be ≫0.5)", flush=True)
    print(f"  AUC(real vs DEPLOYED-C0)   = {auc_c0:.3f}   <- DEPLOYMENT measurement (can the audio critic beat our generator?)", flush=True)
    print(f"  AUC(real vs CORRUPTED-plc) = {auc_corr:.3f}   <- THE MEASUREMENT (fine placement, density held)", flush=True)
    if auc_mis < 0.65:
        print("  !! positive control WEAK -> D barely learned even song-matching -> probe UNDERPOWERED, not a verdict.", flush=True)
    else:
        print("  control fired. Read the measurement:", flush=True)
        print("   ≫0.5 -> audio distinguishes coherent vs scrambled 16ths -> analysis-by-synthesis VIABLE.", flush=True)
        print("   ≈0.5 -> audio placement-blind beyond density -> idea DEAD (controlled bound).", flush=True)


if __name__ == "__main__":
    main()

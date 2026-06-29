#!/usr/bin/env python3
"""M0 GATE: the MATCHED own-output seq-onset refiner probe (2026-06-29).
Re-opens the 06-22 wall on the one untested arm: train the note-context net ON deployed-C0 context AND eval ON
C0 context (vs the 06-28 MISMATCHED train-real/eval-C0 = 0.667, and vs 06-22's anti-correlated v4 C0 = 0.666).

Arms (TARGET = real onset always; only the note-CONTEXT source varies):
  audio              : audio-only                                  -> ~0.65 floor
  both_real          : train+eval on REAL context                  -> ~0.87 POSITIVE CONTROL (must fire)
  both_c0_mismatched : train on REAL, eval on C0 (= the 06-28 run)  -> ~0.667 reference
  both_c0_MATCHED    : train on C0,  eval on C0                     <-- THE MEASUREMENT (own-output refiner)

Read: MATCHED climbs past ~0.667 toward 0.87 -> deployed C0 has structure the refiner can exploit -> own-output
      refiner ALIVE -> build M1.   MATCHED ~ 0.667 -> wall is C0-independent -> refiner DEAD -> pivot.

Data (self-consistent snapshot, NOT the stale seqctx_train_cache):
  train = cache/seqctx_trainc0_cache.npz  (audio, real, c0)  800 current-split charts  [gen_train_c0.py]
  eval  = cache/seqctx_c0_cache.npz       (audio, real, c0)  28 deployed-C0 Hard val songs  [06-28]
"""
import warnings, os, sys, argparse
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
from pathlib import Path
import numpy as np, torch, torch.nn as nn
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils.reproducibility import set_seed
from diag_seqcontext_probe import Probe, auc
from probe_seqcontext_c0 import _batches, eval_model, AD, NP

TRAINC0 = PROJECT_ROOT / "cache/seqctx_trainc0_cache.npz"
EVALC0  = PROJECT_ROOT / "cache/seqctx_c0_cache.npz"


def train_model(kind, train, train_ctx, device, epochs, bs, lr, pw):
    """kind: 'audio'|'both'. train_ctx: 'real'|'c0' (the note-context the net is TRAINED on)."""
    set_seed(42); m = Probe('audio' if kind == 'audio' else 'both').to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr); rng = np.random.default_rng(0)
    for _ in range(epochs):
        m.train()
        for X, Np, Y, M in _batches(train, bs, train_ctx, rng, True, device):
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(m(X, Np)[M], Y[M], pos_weight=pw)
            loss.backward(); opt.step()
    return m


def load(path):
    d = np.load(path, allow_pickle=True)['data']
    return [(a, r, c) for a, r, c in d]   # (audio, real, c0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=8); ap.add_argument('--bs', type=int, default=12)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert TRAINC0.exists(), f"missing {TRAINC0} — run gen_train_c0.py --extract/--shard/--merge first"
    train = load(TRAINC0); val = load(EVALC0)
    posrate = np.mean([(r.sum(-1) > 0).mean() for _, r, _ in train]); pw = torch.tensor((1 - posrate) / posrate, device=device)
    dens_c0 = np.mean([(c.sum(-1) > 0).mean() for _, _, c in train])
    print(f"train={len(train)} (audio+real+C0) | eval={len(val)} | onset-rate real={posrate:.3f} c0={dens_c0:.3f}\n", flush=True)
    print(f"  {'arm':<20} {'onset-AUC':>10} {'16th-AUC':>10}", flush=True)
    res = {}

    m_audio = train_model('audio', train, 'real', device, args.epochs, args.bs, args.lr, pw)
    res['audio'] = eval_model(m_audio, val, 'real', device, args.bs)          # ctx ignored for audio
    print(f"  {'audio':<20} {res['audio'][0]:>10.3f} {res['audio'][1]:>10.3f}", flush=True)

    m_real = train_model('both', train, 'real', device, args.epochs, args.bs, args.lr, pw)
    res['both_real']          = eval_model(m_real, val, 'real', device, args.bs)   # ceiling / positive control
    res['both_c0_mismatched'] = eval_model(m_real, val, 'c0',   device, args.bs)   # = the 06-28 number
    print(f"  {'both_real':<20} {res['both_real'][0]:>10.3f} {res['both_real'][1]:>10.3f}", flush=True)
    print(f"  {'both_c0_mismatched':<20} {res['both_c0_mismatched'][0]:>10.3f} {res['both_c0_mismatched'][1]:>10.3f}", flush=True)

    m_c0 = train_model('both', train, 'c0', device, args.epochs, args.bs, args.lr, pw)
    res['both_c0_MATCHED'] = eval_model(m_c0, val, 'c0', device, args.bs)          # THE MEASUREMENT
    print(f"  {'both_c0_MATCHED':<20} {res['both_c0_MATCHED'][0]:>10.3f} {res['both_c0_MATCHED'][1]:>10.3f}", flush=True)

    a, cr = res['audio'][1], res['both_real'][1]
    mm, mt = res['both_c0_mismatched'][1], res['both_c0_MATCHED'][1]
    print(f"\n  16th-AUC: audio={a:.3f}  mismatched-C0={mm:.3f}  MATCHED-C0={mt:.3f}  real(ceiling)={cr:.3f}", flush=True)
    print(f"  POSITIVE CONTROL: both_real ({cr:.3f}) must be >> audio ({a:.3f}); else underpowered (re-check N).", flush=True)
    gap = (mt - a) / max(cr - a, 1e-6)
    print(f"  MATCHED recovers {100*gap:.0f}% of the (real-context − audio) gap  (vs mismatched {100*(mm-a)/max(cr-a,1e-6):.0f}%).", flush=True)
    print(f"  >> climbs past mismatched toward real -> own-output refiner ALIVE (build M1).", flush=True)
    print(f"  >> sits at ~mismatched/audio -> wall C0-independent -> refiner DEAD (pivot).", flush=True)


if __name__ == "__main__":
    main()

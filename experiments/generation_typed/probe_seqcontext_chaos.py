#!/usr/bin/env python3
"""STAGE-1 DE-RISK for the chaos×onset-gate RETRAIN (notes/chaos_onset_gate_scope.md Phase 1): is the note-context
placement signal present specifically on HIGH-CHAOS charts — where the tolerance failures live — or only on tame ones?
(2026-07-04; lineage seq-onset-arc.md + good-settings-region-arc.md.)

WHY. Phase-0 killed the AUDIO-keyed decode gate: off-beat placement isn't in audio (DESMEAR crushed GC's loved 16ths
and HSL's smear IDENTICALLY). M1a (`probe_seqcontext_frozenh.py`) showed the placement signal IS in the frozen
decoder's `h` (note-context, 16th-AUC 0.89 vs audio 0.66) — but POOLED across chaos. The retrain (`p_onset = σ(readout(h)
+ w·chaos·offbeat(h))`) only works if `h` predicts placement ON HIGH-CHAOS charts. This stratifies M1a's val 16th-AUC by
the real chart's off-beat share (= chaos; H4/referee) to test exactly that.

READ:
  frozen_h 16th-AUC stays HIGH on HIGH-chaos val (≈ pooled 0.89, >> audio) -> the placement signal is there where the
     tolerance failures happen -> the learned chaos gate can PLACE the off-beats -> retrain DE-RISKED (train Stage 2).
  frozen_h 16th-AUC COLLAPSES on HIGH-chaos (toward audio)                  -> h predicts placement only on tame charts;
     the retrain can't place high-chaos off-beats either -> STOP, rethink before the train.
Stratify by REAL 16th-offbeat SHARE (base rate), scored by 16th-AUC (which-frame rank) -> different quantities, not
circular. Reuses the frozen-h caches + heads (experiment-design Rule 6/11; capacity-matched controls intact).

  python experiments/generation_typed/probe_seqcontext_chaos.py
"""
import warnings, os; warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils.reproducibility import set_seed
from src.generation.typed_model import LayeredTypedChartGenerator
from probe_seqcontext_frozenh import (HRead, HReadConv, load_or_extract, precompute_h, batches,
                                      AD, NP, DMODEL, CKPT)
from diag_seqcontext_probe import Probe, auc


def real_s16(song):
    """real 16th-offbeat share = fraction of a chart's onset frames landing on t%4 in {1,3} (= chaos; H4/referee)."""
    on = (song['typed'] != 0).any(-1)
    idx = np.where(on)[0]
    return float(np.mean((idx % 4 == 1) | (idx % 4 == 3))) if idx.size else 0.0


def train_head(kind, train, device, epochs, bs, lr, pw):
    """Train ONE readout head on the full train set; return it + its forward (so we can eval on several val strata)."""
    set_seed(42)
    if kind == 'frozen_h':
        m = HRead(DMODEL).to(device); fwd = lambda X, Np, H: m(H)
    elif kind == 'frozen_h_conv':
        m = HReadConv(DMODEL).to(device); fwd = lambda X, Np, H: m(H)
    else:
        m = Probe(kind).to(device); fwd = lambda X, Np, H: m(X, Np)
    opt = torch.optim.Adam(m.parameters(), lr=lr); rng = np.random.default_rng(0)
    for _ in range(epochs):
        m.train()
        for X, Np, H, Y, M in batches(train, bs, rng, True, device):
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(fwd(X, Np, H)[M], Y[M], pos_weight=pw)
            loss.backward(); opt.step()
    m.eval()
    return m, fwd


@torch.no_grad()
def eval_16th_auc(fwd, val, device, bs):
    """16th-localization AUC (note vs no-note WITHIN 16th-offbeat frames) over a val subset, pooled."""
    rng = np.random.default_rng(0); ps, ys = [], []
    for X, Np, H, Y, M in batches(val, bs, rng, False, device):
        p = torch.sigmoid(fwd(X, Np, H)).cpu().numpy(); B, T = Y.shape; t = np.arange(T)
        m16 = ((t % 4 == 1) | (t % 4 == 3))[None].repeat(B, 0) & M.cpu().numpy()
        ps.append(p[m16]); ys.append(Y.cpu().numpy()[m16])
    P, Yv = np.concatenate(ps), np.concatenate(ys)
    return auc(P, Yv), int(Yv.sum()), len(Yv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_train', type=int, default=800); ap.add_argument('--max_val', type=int, default=400)
    ap.add_argument('--max_len', type=int, default=1024); ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--bs', type=int, default=12); ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tc = PROJECT_ROOT / "cache/seqctx_frozenh_train.npz"; vc = PROJECT_ROOT / "cache/seqctx_frozenh_val.npz"
    if not (tc.exists() and vc.exists()):
        raise SystemExit("frozen-h caches missing; run probe_seqcontext_frozenh.py first to build them.")

    model = LayeredTypedChartGenerator(audio_dim=AD, d_model=DMODEL, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device)['model_state_dict'], strict=False); model.eval()
    train = load_or_extract(None, args.max_train, args.max_len, tc, hard_only=False)
    val = load_or_extract(None, args.max_val, args.max_len, vc, hard_only=True)
    print(f"  computing frozen decoder h (teacher-forced)...", flush=True)
    precompute_h(model, train, device, args.max_len); precompute_h(model, val, device, args.max_len)

    # stratify val (Hard) by real 16th-offbeat share = chaos
    s16 = np.array([real_s16(s) for s in val]); med = float(np.median(s16))
    lo = [s for s, v in zip(val, s16) if v <= med]; hi = [s for s, v in zip(val, s16) if v > med]
    print(f"\nval Hard n={len(val)} | real 16th-share median={med:.3f} | "
          f"LOW-chaos n={len(lo)} (s16≤{med:.2f}) · HIGH-chaos n={len(hi)} (s16>{med:.2f})", flush=True)
    print(f"  HIGH-chaos mean real s16={np.mean([real_s16(s) for s in hi]):.3f}  "
          f"LOW-chaos mean real s16={np.mean([real_s16(s) for s in lo]):.3f}", flush=True)

    posrate = np.mean([(s['typed'] != 0).any(-1).mean() for s in train])
    pw = torch.tensor((1 - posrate) / posrate, device=device)
    print(f"\n  16th-AUC by chaos stratum (train once on {len(train)}, eval per stratum):", flush=True)
    print(f"  {'predictor':<14} {'ALL':>8} {'LOW-chaos':>10} {'HIGH-chaos':>11}", flush=True)
    res = {}
    for kind in ['audio', 'both', 'frozen_h', 'frozen_h_conv']:
        m, fwd = train_head(kind, train, device, args.epochs, args.bs, args.lr, pw)
        aa = eval_16th_auc(fwd, val, device, args.bs)[0]
        al = eval_16th_auc(fwd, lo, device, args.bs)[0]
        ah = eval_16th_auc(fwd, hi, device, args.bs)[0]
        res[kind] = (aa, al, ah)
        label = 'both_real' if kind == 'both' else kind
        print(f"  {label:<14} {aa:>8.3f} {al:>10.3f} {ah:>11.3f}", flush=True)

    aH = res['audio'][2]; bH = res['both'][2]; fH = res['frozen_h_conv'][2]
    print(f"\n  POSITIVE CONTROL (Rule 11): both_real >> audio on HIGH-chaos (else underpowered).", flush=True)
    print(f"  HIGH-chaos 16th-AUC: audio={aH:.3f}  frozen_h(conv)={fH:.3f}  both_real={bH:.3f}", flush=True)
    if bH - aH > 0.05:
        gap = max(bH - aH, 1e-6)
        print(f"  frozen_h recovers {100*(fH-aH)/gap:.0f}% of the note-context gap ON HIGH-CHAOS charts.", flush=True)
        print(f"  DE-RISK: high (≈pooled 0.89, >> audio) -> placement signal present WHERE tolerance fails -> the learned", flush=True)
        print(f"  chaos gate can place high-chaos off-beats -> TRAIN Stage 2. Collapses toward audio -> STOP, rethink.", flush=True)
    else:
        print(f"  !! both_real did NOT clear audio on HIGH-chaos -> underpowered on this stratum; raise --max_train "
              f"or widen the split before interpreting frozen_h.", flush=True)
    print(f"\n  BOUNDARY (Rule 9/10): teacher-forced on REAL notes = the UPPER bound; gen-time DRIFT (the seq-onset "
          f"binding gate) is Stage-2's risk, NOT settled here.", flush=True)


if __name__ == '__main__':
    main()

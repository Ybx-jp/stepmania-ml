#!/usr/bin/env python3
"""FAIR-TEST #2: can a PHASE-REBALANCE penalty pull the seq head's 16th-FLOOD back to a MUSICAL backbone? (2026-06-29)
The manifold-conditioning fair test (`probe_seqonset_cond.py`) found the groove reaches the seq head only as a 3%
echo (it reads `h`, not radar directly) — inconclusive. This is the DIRECT lever: subtract a penalty from the seq
head's 16th-offbeat logits before tau (mirrors the deployed `onset_phase_penalty`), sweep it, and ask not just
"does the 16th-share drop" (Rule 1 — shares can mislead) but "does the recovered BACKBONE land on the RIGHT frames"
(realized onset precision/recall vs real). No retrain.

READ:
  16th-share drops toward real AND onset precision-vs-real RISES with it -> the flood is rebalanceable to a real-
    aligned backbone -> a cheap decode lever (or a phase-aware retrain) is the path; bank was premature.
  16th-share drops but precision STAYS low / falls -> suppressing 16ths just exposes a misplaced backbone (the head
    ranks NO frames well free-run) -> rebalancing alone won't save it; the placement deficit is real.

  /home/ybx/miniconda3/envs/stepmania-chart-gen/bin/python experiments/generation_typed/probe_seqonset_phasepen.py
"""
import warnings, os; warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, sys
from pathlib import Path
import numpy as np, torch
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils.reproducibility import set_seed
from src.generation.typed_model import LayeredTypedChartGenerator
from probe_seqcontext_frozenh import AD, DMODEL, CKPT, HReadConv, load_or_extract
from probe_seqonset_rollout import rollout, _runlen
from probe_seqonset_phase import phase_shares
from probe_seqonset_placement import _prf
from probe_seqonset_critic import audio_onset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8); ap.add_argument('--cap', type=int, default=512)
    ap.add_argument('--tau', type=float, default=0.55)
    ap.add_argument('--b16', type=str, default='0,1,2,3,5', help='16th-offbeat logit penalties to sweep')
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val = load_or_extract(None, 400, args.cap, PROJECT_ROOT / "cache/seqctx_frozenh_val.npz", hard_only=True)[:args.n]
    model = LayeredTypedChartGenerator(audio_dim=AD, d_model=DMODEL, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device)['model_state_dict'], strict=False); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    head = HReadConv(DMODEL).to(device)
    head.load_state_dict(torch.load(PROJECT_ROOT / "cache/seqonset_ss_head.pt", map_location=device)); head.eval()
    for p in head.parameters():
        p.requires_grad_(False)

    real_ph = np.mean([phase_shares((s['typed'][:min(s['T'], args.cap)] != 0).any(-1)) for s in val], 0)
    real_d = np.mean([float((s['typed'][:min(s['T'], args.cap)] != 0).any(-1).mean()) for s in val])
    print(f"\n  seq head free-run, sweeping the 16th-offbeat phase penalty (tau={args.tau}).", flush=True)
    print(f"  REAL Hard ref: phase {real_ph[0]:.0f}/{real_ph[1]:.0f}/{real_ph[2]:.0f}, density {real_d:.3f}\n", flush=True)
    print(f"  {'b16':>5} {'quarter':>8} {'8th':>6} {'16th':>6} {'dens':>6} | {'prec':>5} {'rec':>5} {'F1':>5}  (onsets vs real)", flush=True)
    rows = []
    for b16 in [float(x) for x in args.b16.split(',')]:
        phs, ds, preds, reals = [], [], [], []
        for s in val:
            T = min(s['T'], args.cap)
            on = rollout(model, head, s['audio'][:T].astype(np.float32), s['diff'], T, args.tau, device,
                         phase_pen=(0.0, b16))
            phs.append(phase_shares(on)); ds.append(float(on.mean()))
            preds.append(on); reals.append((s['typed'][:T] != 0).any(-1))
        q, e, s16 = np.mean(phs, 0)
        pr, rc, f1 = _prf(np.concatenate(preds), np.concatenate(reals))
        rows.append((b16, q, e, s16, np.mean(ds), pr, rc, f1))
        print(f"  {b16:>5.1f} {q:>8.1f} {e:>6.1f} {s16:>6.1f} {np.mean(ds):>6.3f} | {pr:>5.2f} {rc:>5.2f} {f1:>5.2f}", flush=True)

    # BASELINE BRACKET (Rule 11): the DEPLOYED audio onset head (+16th-unlock) on the SAME precision/recall-vs-real
    # metric, density-matched to the best phase-pen seq density. Does the rebalanced seq head BEAT the audio backbone
    # it would replace, or just reproduce it? (real-vs-real = 1.0 is the trivial ceiling.)
    R = np.array(rows)
    best_d = float(R[np.argmax(R[:, 7])][4])
    a_phs, a_preds, a_reals = [], [], []
    for s in val:
        T = min(s['T'], args.cap)
        A42 = torch.from_numpy(s['audio'][:T].astype(np.float32)).unsqueeze(0).to(device)
        diff = torch.tensor([s['diff']], device=device)
        aon = audio_onset(model, A42, diff, T, best_d, device).cpu().numpy()
        a_phs.append(phase_shares(aon)); a_preds.append(aon); a_reals.append((s['typed'][:T] != 0).any(-1))
    aq, ae, a16 = np.mean(a_phs, 0); apr, arc, af1 = _prf(np.concatenate(a_preds), np.concatenate(a_reals))
    print(f"  {'AUD':>5} {aq:>8.1f} {ae:>6.1f} {a16:>6.1f} {best_d:>6.3f} | {apr:>5.2f} {arc:>5.2f} {af1:>5.2f}   <- DEPLOYED audio head @ matched density", flush=True)

    base = R[0]; best = R[np.argmax(R[:, 7])]                       # best-F1 row
    print(f"\n  flood (b16=0): 16th {base[3]:.0f}%, onset-precision {base[5]:.2f}, F1 {base[7]:.2f}.", flush=True)
    print(f"  best phase-pen (b16={best[0]:.0f}): 16th {best[3]:.0f}%, precision {best[5]:.2f}, F1 {best[7]:.2f}.", flush=True)
    print(f"  DEPLOYED audio head @ matched density: precision {apr:.2f}, F1 {af1:.2f}.", flush=True)
    rebalanceable = best[5] - base[5] >= 0.08 and best[3] <= 2.0 * max(real_ph[2], 1)
    if not rebalanceable:
        print(f"  => the flood does NOT rebalance to a real-aligned backbone -> placement deficit is real.", flush=True)
    elif best[7] >= af1 + 0.03:
        print(f"  => REBALANCEABLE and the seq backbone BEATS the deployed audio head (F1 {best[7]:.2f} > {af1:.2f}) -> sequence", flush=True)
        print(f"     context adds placement value -> fork (A) re-opens with a real target; scope a phase-aware/conditioned head.", flush=True)
    elif best[7] >= af1 - 0.03:
        print(f"  => REBALANCEABLE but the seq backbone only MATCHES the audio head (F1 {best[7]:.2f} ≈ {af1:.2f}) -> the flood was a", flush=True)
        print(f"     decode artifact (bank was premature), but the rebalanced head adds NO placement value over the deployed", flush=True)
        print(f"     audio backbone it would replace -> re-open as 'not dead' but the WIN is unproven; a graded 16th lever + by-ear decides.", flush=True)
    else:
        print(f"  => REBALANCEABLE but the seq backbone is WORSE than the audio head (F1 {best[7]:.2f} < {af1:.2f}) -> draining the", flush=True)
        print(f"     flood just reaches a backbone the audio head already does better. No value over deployed.", flush=True)
    print(f"\n  BOUNDARY: realized precision/rec vs real, free-run, governors off, binary penalty (no graded 16th control;", flush=True)
    print(f"  real has 4% 16ths, this nukes to 0). By-ear is the binding gate if a lever lands.", flush=True)


if __name__ == '__main__':
    main()

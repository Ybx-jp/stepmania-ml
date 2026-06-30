#!/usr/bin/env python3
"""M1b-3: NOTE-DROPOUT SCHEDULED SAMPLING for the frozen-h onset head, then re-run the drift gate. (2026-06-29)
Lineage seq-onset-arc.md. M1b (`probe_seqonset_rollout.py`) found the teacher-forced-trained head COLLAPSES free-run
(density 0.000): trained where note-context was always real/dense, it can't fire when its OWN context is sparse.

FIX (cheap scheduled-sampling approximation): manufacture the sparse-context regime in PARALLEL by randomly DROPPING
real notes (prob d) before decoding, compute h from the corrupted context, and train the head to predict the FULL
real onsets. d=0 -> M1a (teacher-forced, run-coherence); high d -> near-empty context -> the head is FORCED to fire
from the audio-in-h. Sampling d∈[0,dmax] per batch spans the spectrum in one pass (no sequential rollout for TRAIN;
rollout only for the GATE). This is the input-corruption form of scheduled sampling / DAgger's cheap cousin.

GATE (the same drift test as M1b): tau calibrated on teacher-forced h, transferred to FREE-RUN; does density recover
from the 0.000 collapse toward real (~0.27)? Controls reused: TF_rollout (≈real ⇒ harness clean), warm-seed.
READ:
  FREE-run density recovers toward real & TF_rollout still ≈ real -> dropout-SS breaks the collapse -> proceed
    (wire into generate(); maybe refine with true own-context SS).
  FREE-run still collapses / explodes -> dropout-proxy insufficient -> escalate to true own-output rollout SS.

  /home/ybx/miniconda3/envs/stepmania-chart-gen/bin/python experiments/generation_typed/probe_seqonset_ss.py
"""
import warnings, os; warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils.reproducibility import set_seed
from src.generation.typed_model import LayeredTypedChartGenerator
from probe_seqcontext_frozenh import AD, NP, DMODEL, CKPT, HReadConv, decode_hidden, precompute_h, load_or_extract
from probe_seqonset_rollout import rollout, _runlen


def train_ss(model, head, train, device, epochs, bs, lr, dmax, pw):
    """Note-dropout scheduled sampling: per batch drop real notes d~U(0,dmax), decode h from the corrupted context
    (frozen decoder, no-grad), train the head to predict the FULL real onsets (grad through the head only)."""
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for ep in range(epochs):
        head.train(); tot = 0.0; nb = 0
        rng2 = np.random.default_rng(ep)
        idx = np.arange(len(train)); rng2.shuffle(idx)
        for i in range(0, len(idx), bs):
            chunk = [train[j] for j in idx[i:i + bs]]; T = max(s['T'] for s in chunk); B = len(chunk)
            X = np.zeros((B, T, AD), np.float32); St = np.zeros((B, T, NP), np.int64); M = np.zeros((B, T), bool)
            for b, s in enumerate(chunk):
                X[b, :s['T']] = s['audio']; St[b, :s['T']] = s['typed']; M[b, :s['T']] = True
            states = torch.from_numpy(St).to(device); mask = torch.from_numpy(M).to(device)
            onset = (states != 0).any(-1)
            d = float(rng2.uniform(0, dmax))
            keep = (torch.rand(B, T, device=device) >= d) | ~onset
            corrupt = states * keep.unsqueeze(-1).long()
            diff = torch.tensor([s['diff'] for s in chunk], device=device)
            h = decode_hidden(model, torch.from_numpy(X).to(device), corrupt, diff, mask)   # (B,T,d), no-grad (frozen)
            logit = head(h.detach())                                          # grad through head only
            loss = nn.functional.binary_cross_entropy_with_logits(logit[mask], onset.float()[mask], pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
        print(f"    epoch {ep}: BCE {tot/max(nb,1):.4f}", flush=True)
    head.eval(); return head


@torch.no_grad()
def drift_gate(model, head, val, device, cap, n_eval, seed=32):
    precompute_h(model, val, device, cap)
    rows = []
    for s in val[:n_eval]:
        T = min(s['T'], cap); real_on = (s['typed'][:T] != 0).any(-1)
        real_d = float(real_on.mean()); real_run = _runlen(real_on)
        p_tf = torch.sigmoid(head(torch.from_numpy(s['h'][:T]).unsqueeze(0).to(device))[0]).cpu().numpy()
        tau = float(np.quantile(p_tf, 1 - real_d)) if real_d > 0 else 0.5
        tfr = float(rollout(model, head, s['audio'][:T], s['diff'], T, tau, device,
                            tf_states=s['typed'][:T].astype(np.int64)).mean())
        on = rollout(model, head, s['audio'][:T], s['diff'], T, tau, device)
        free_d = float(on.mean()); free_run = _runlen(on)
        sd = float(rollout(model, head, s['audio'][:T], s['diff'], T, tau, device,
                           tf_states=s['typed'][:T].astype(np.int64), seed_frames=seed)[seed:].mean()) if T > seed else 0.0
        rows.append((real_d, tfr, free_d, sd, real_run, free_run))
    return np.array(rows)


@torch.no_grad()
def threshold_sweep(model, head, val, device, cap, n_eval, taus):
    """Decouple CALIBRATION from DRIFT: free-run at a range of ABSOLUTE sigmoid thresholds. If some tau gives
    density≈real with run≈real, the head RANKS onsets fine from its own context and the tau-transfer collapse was a
    calibration artifact; if density only ever jumps 0->explosion with no real-like middle, it's genuine drift."""
    out = []
    for tau in taus:
        ds, runs = [], []
        for s in val[:n_eval]:
            T = min(s['T'], cap)
            on = rollout(model, head, s['audio'][:T], s['diff'], T, tau, device)
            ds.append(on.mean()); runs.append(_runlen(on))
        out.append((tau, float(np.mean(ds)), float(np.mean(runs))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=10); ap.add_argument('--bs', type=int, default=12)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--dmax', type=float, default=0.9)
    ap.add_argument('--cap', type=int, default=512); ap.add_argument('--n_eval', type=int, default=12)
    ap.add_argument('--load_head', action='store_true', help='skip training; load cache/seqonset_ss_head.pt')
    ap.add_argument('--taus', type=str, default='0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1', help='comma threshold sweep')
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tc = PROJECT_ROOT / "cache/seqctx_frozenh_train.npz"; vc = PROJECT_ROOT / "cache/seqctx_frozenh_val.npz"
    assert tc.exists() and vc.exists(), "run probe_seqcontext_frozenh.py first"
    train = load_or_extract(None, 800, args.cap, tc, hard_only=False)
    val = load_or_extract(None, 400, args.cap, vc, hard_only=True)
    model = LayeredTypedChartGenerator(audio_dim=AD, d_model=DMODEL, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device)['model_state_dict'], strict=False); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    posrate = np.mean([(s['typed'] != 0).any(-1).mean() for s in train]); pw = torch.tensor((1 - posrate) / posrate, device=device)
    head_path = PROJECT_ROOT / "cache/seqonset_ss_head.pt"
    set_seed(42); head = HReadConv(DMODEL).to(device)
    if args.load_head and head_path.exists():
        head.load_state_dict(torch.load(head_path, map_location=device)); head.eval()
        print(f"loaded SS head from {head_path}", flush=True)
    else:
        print(f"NOTE-DROPOUT scheduled sampling: d~U(0,{args.dmax}), {args.epochs} epochs, {len(train)} train songs", flush=True)
        head = train_ss(model, head, train, device, args.epochs, args.bs, args.lr, args.dmax, pw)
        torch.save(head.state_dict(), head_path); print(f"saved SS head -> {head_path}", flush=True)

    print("\nDRIFT GATE (post-SS): tau calibrated on teacher-forced h -> transferred to FREE-run", flush=True)
    R = drift_gate(model, head, val, device, args.cap, args.n_eval)
    rd, tfr, fd, sd, rr, fr = R.mean(0)
    print(f"  {'real_d':>8} {'TFroll_d':>9} {'FREE_d':>8} {'seed_d':>7} {'real_run':>9} {'FREE_run':>9}", flush=True)
    print(f"  {rd:>8.3f} {tfr:>9.3f} {fd:>8.3f} {sd:>7.3f} {rr:>9.2f} {fr:>9.2f}", flush=True)
    print(f"\n  M1b baseline (teacher-forced head): FREE_d 0.000 (COLLAPSE). Post-SS FREE_d = {fd:.3f}.", flush=True)
    if abs(tfr - rd) > 0.4 * rd:
        print(f"  !! TF_rollout {tfr:.3f} far from real {rd:.3f} -> harness/SS broke the control; debug before interpreting.", flush=True)
    else:
        ratio = fd / max(rd, 1e-6)
        verdict = ("STILL COLLAPSES" if fd < 0.5 * rd else
                   "EXPLODES" if ratio > 1.3 or fr > 1.6 * rr else "RECOVERED (stable)")
        print(f"  CONTROL ok (TF_rollout {tfr:.3f}≈real {rd:.3f}). FREE-run = {ratio:.2f}× real, run {fr:.2f} vs {rr:.2f} -> **{verdict}**", flush=True)
        if verdict.startswith("RECOVERED"):
            print(f"  => dropout-SS BREAKS the collapse -> the head fires from audio-in-h under its own sparse context.", flush=True)
            print(f"     NEXT: wire into generate() (opt-in) + by-ear; optionally refine with true own-context SS.", flush=True)
        else:
            print(f"  => dropout-proxy insufficient ({verdict}) -> escalate to true own-output rollout scheduled sampling.", flush=True)

    # CALIBRATION vs DRIFT: sweep absolute thresholds (the tau above is teacher-forced-calibrated -> may be too high)
    print(f"\n  THRESHOLD SWEEP (free-run at absolute sigmoid thresholds; real_d≈{rd:.2f}, real_run≈{rr:.2f}):", flush=True)
    print(f"  {'tau':>6} {'free_d':>8} {'free_run':>9}", flush=True)
    sweep = threshold_sweep(model, head, val, device, args.cap, args.n_eval,
                            [float(x) for x in args.taus.split(',')])
    for tau, d, run in sweep:
        print(f"  {tau:>6.2f} {d:>8.3f} {run:>9.2f}", flush=True)
    near = [(t, d, run) for t, d, run in sweep if 0.5 * rd <= d <= 1.5 * rd]
    if near:
        t, d, run = near[len(near) // 2]
        print(f"  => a threshold (tau={t:.2f}) gives near-real density {d:.3f} (run {run:.2f}) -> the head RANKS onsets from", flush=True)
        print(f"     its OWN context; the tau-transfer collapse was a CALIBRATION artifact -> fix = free-run/self tau, not more SS.", flush=True)
    else:
        print(f"  => NO threshold gives real-like density (only empty or explosion) -> genuine DRIFT (no stable point) ->", flush=True)
        print(f"     escalate to true own-output rollout scheduled sampling.", flush=True)
    print(f"\n  BOUNDARY: density/run-length drift only (governors off, tap-only greedy). Placement quality = by-ear later.", flush=True)


if __name__ == '__main__':
    main()

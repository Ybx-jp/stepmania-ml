#!/usr/bin/env python3
"""FAIR-TEST (the one M1b-4/5/6 SKIPPED): does COHERENT MANIFOLD groove conditioning tame the seq head's 16th-FLOOD?
(2026-06-29). User critique (correct): M1b-4/5/6 measured ONE under-tuned config (native radar=None, dmax=1.0 head,
global tau) and banked the build — but the 16th-flood (backbone collapse 19/19/62 vs real 64/32/4) is the KNOWN
chaos-smear failure, and in the deployed model coherent MANIFOLD conditioning fixes it (conditioning-mechanics §2;
experiment-design Evidence: the --match_radar fair test overturned "the model has no backbone", survival 0.83). So
condition the seq head's rollout on a backbone-heavy manifold groove and re-measure the phase distribution.

The decoder's `h` (what the seq head reads) is conditioned by `_cond(diff, radar, ...)`; native uses null_radar.
This sweeps radar = manifold targets built via `build_target` (conditional-fill, NOT mean-pin). Backbone-heavy =
LOW chaos (chaos drives off-grid; real charts keep a quarter/8th backbone even on chaotic songs).
⚠️ OOD CAVEAT (Rule 3/11): the SS head was TRAINED on native `h`; conditioning shifts `h` → the head is mildly OOD.
So READ asymmetrically: backbone RECOVERS under conditioning -> the bank was PREMATURE, a conditioned retrain is the
path (strong even through OOD); backbone does NOT recover -> INCONCLUSIVE (OOD may mask it), needs a conditioned
retrain to settle — NOT proof the build is dead.

  /home/ybx/miniconda3/envs/stepmania-chart-gen/bin/python experiments/generation_typed/probe_seqonset_cond.py
"""
import warnings, os; warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, sys
from pathlib import Path
import numpy as np, torch
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils.reproducibility import set_seed
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.radar_manifold import RadarManifold
from probe_seqcontext_frozenh import AD, DMODEL, CKPT, HReadConv, load_or_extract
from probe_seqonset_rollout import rollout, _runlen
from probe_seqonset_phase import phase_shares


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8); ap.add_argument('--cap', type=int, default=512)
    ap.add_argument('--tau', type=float, default=0.55)
    ap.add_argument('--specs', type=str, default='native|mean|chaos=low|chaos=low,stream=high',
                    help='pipe-separated manifold specs; "native"=radar None, "mean"=all-free (mu)')
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
    manifold = RadarManifold.load(PROJECT_ROOT / "cache/radar_manifold.npz")

    specs = args.specs.split('|')
    # precompute the manifold radar vec per spec (Hard=3); native -> None
    radar_of = {}
    for sp in specs:
        if sp == 'native':
            radar_of[sp] = None
        else:
            vec, _ = manifold.build_target('' if sp == 'mean' else sp, 3)
            radar_of[sp] = torch.from_numpy(vec.astype(np.float32)).unsqueeze(0).to(device)
            print(f"  spec {sp:<22} -> radar [stream,voltage,air,freeze,chaos] = "
                  f"{np.array2string(vec, precision=2, floatmode='fixed')}", flush=True)

    real_ph = np.mean([phase_shares((s['typed'][:min(s['T'], args.cap)] != 0).any(-1)) for s in val], 0)
    real_d = np.mean([float((s['typed'][:min(s['T'], args.cap)] != 0).any(-1).mean()) for s in val])
    print(f"\n  phase shares (% of onsets) — REAL Hard ref {real_ph[0]:.0f}/{real_ph[1]:.0f}/{real_ph[2]:.0f}, density {real_d:.3f}\n", flush=True)
    print(f"  {'spec':<24} {'quarter':>8} {'8th':>6} {'16th':>6} {'density':>8} {'run':>6}", flush=True)
    rows = {}
    for sp in specs:
        phs, ds, runs = [], [], []
        for s in val:
            T = min(s['T'], args.cap)
            on = rollout(model, head, s['audio'][:T].astype(np.float32), s['diff'], T, args.tau, device,
                         radar=radar_of[sp])
            phs.append(phase_shares(on)); ds.append(float(on.mean())); runs.append(_runlen(on))
        q, e, s16 = np.mean(phs, 0); rows[sp] = (q, e, s16, np.mean(ds), np.mean(runs))
        print(f"  {sp:<24} {q:>8.1f} {e:>6.1f} {s16:>6.1f} {np.mean(ds):>8.3f} {np.mean(runs):>6.2f}", flush=True)

    nat16 = rows['native'][2]
    best = min((v[2], k) for k, v in rows.items() if k != 'native') if len(rows) > 1 else (nat16, 'native')
    print(f"\n  native 16th-share {nat16:.0f}% (the flood). Best conditioned: {best[1]} -> {best[0]:.0f}%  "
          f"(real {real_ph[2]:.0f}%)", flush=True)
    drop = nat16 - best[0]
    if best[0] <= 1.6 * max(real_ph[2], 1) or drop >= 20:
        print(f"  => MANIFOLD conditioning RECOVERS the backbone (16th {nat16:.0f}%→{best[0]:.0f}%) -> the M1b-4/5/6 BANK", flush=True)
        print(f"     was PREMATURE (one under-tuned config); a conditioned retrain is the live path. Re-open fork (A).", flush=True)
    elif drop >= 8:
        print(f"  => PARTIAL recovery ({drop:.0f}pt) under OOD conditioning -> promising; a conditioned RETRAIN (in-dist) likely", flush=True)
        print(f"     does more. The bank was premature; scope the retrain (the inference-time signal points the right way).", flush=True)
    else:
        print(f"  => conditioning barely moves the flood ({drop:.0f}pt) — but OOD (head trained native) so INCONCLUSIVE, NOT", flush=True)
        print(f"     'dead'. A conditioned RETRAIN is the settling test before any bank (Rule 9/11). /autotune first.", flush=True)
    print(f"\n  BOUNDARY: inference-time conditioning of a NATIVE-trained head (OOD); positive = strong, null = inconclusive.", flush=True)


if __name__ == '__main__':
    main()

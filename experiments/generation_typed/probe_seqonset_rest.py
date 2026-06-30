#!/usr/bin/env python3
"""Does the REST VALVE (+ self-cal tau + inverted phase lever) make the seq head PAUSE? (2026-06-29)
User by-ear: the seq head "never pauses — just continues once started." That's the audio head's natural energy-
silences + stamina, which the seq head lacks and the A/B had OFF. This builds the seq-appropriate surface
(seqonset_decode.py) and measures REST structure (longest silence, # rests) vs real / audio / the flood, before the
expensive export. Cheapest decisive check (exp-design Rule 6)."""
import warnings, os; warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, sys
from pathlib import Path
import numpy as np, torch
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils.reproducibility import set_seed
from src.generation.typed_model import LayeredTypedChartGenerator
from probe_seqcontext_frozenh import AD, DMODEL, CKPT, HReadConv, load_or_extract
from probe_seqonset_rollout import rollout
from probe_seqonset_phase import phase_shares
from probe_seqonset_critic import audio_onset
from seqonset_decode import build_rest_env, selfcal_tau


def rest_stats(onset, R=8):
    """longest empty run + # of rests (empty runs >= R frames, R=8 = a half-beat). Real charts have real rests;
    the flood has ~none ('never pauses')."""
    runs, c = [], 0
    for v in onset:
        if not v:
            c += 1
        elif c:
            runs.append(c); c = 0
    if c:
        runs.append(c)
    maxrun = max(runs) if runs else 0
    nrest = sum(1 for r in runs if r >= R)
    return maxrun, nrest


def summarize(label, ons, reals, T_tot):
    phs = np.mean([phase_shares(o) for o in ons], 0)
    d = np.mean([o.mean() for o in ons])
    mx = np.mean([rest_stats(o)[0] for o in ons])
    nr = sum(rest_stats(o)[1] for o in ons) / T_tot * 1000
    print(f"  {label:<28} {phs[0]:>6.0f} {phs[1]:>5.0f} {phs[2]:>5.0f} {d:>7.3f} {mx:>9.1f} {nr:>9.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8); ap.add_argument('--cap', type=int, default=512)
    ap.add_argument('--gains', type=str, default='0,2,3,4'); ap.add_argument('--b16', type=float, default=1.0)
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

    reals = [(s['typed'][:min(s['T'], args.cap)] != 0).any(-1) for s in val]
    T_tot = sum(min(s['T'], args.cap) for s in val)
    real_d = np.mean([r.mean() for r in reals])
    print(f"\n  rest structure — phase q/8/16, density, longest-silence (frames), #rests(>=8f)/1k frames\n", flush=True)
    print(f"  {'arm':<28} {'q':>6} {'8':>5} {'16':>5} {'dens':>7} {'maxSil':>9} {'rests/1k':>9}", flush=True)
    summarize('REAL', reals, reals, T_tot)
    # deployed audio head @ real density (the reference that DOES pause)
    aud = []
    for s in val:
        T = min(s['T'], args.cap)
        A42 = torch.from_numpy(s['audio'][:T].astype(np.float32)).unsqueeze(0).to(device)
        aud.append(audio_onset(model, A42, torch.tensor([s['diff']], device=device), T, real_d, device).cpu().numpy())
    summarize('AUDIO head @ real density', aud, reals, T_tot)

    for g in [float(x) for x in args.gains.split(',')]:
        ons = []
        for s in val:
            T = min(s['T'], args.cap); a = s['audio'][:T].astype(np.float32)
            env = build_rest_env(model, a, s['diff'], T, device) if g > 0 else None
            target_d = float((s['typed'][:T] != 0).any(-1).mean())
            tau = selfcal_tau(model, head, a, s['diff'], T, device, target_d, rest_env=env, rest_gain=g,
                              phase_pen=(0.0, args.b16))
            on = rollout(model, head, a, s['diff'], T, tau, device, rest_env=env, rest_gain=g, phase_pen=(0.0, args.b16))
            ons.append(on)
        summarize(f'SEQ valve g={g:g} b16={args.b16:g}', ons, reals, T_tot)
    print(f"\n  READ: does a rest_gain bring SEQ's maxSil + rests/1k toward REAL (away from the flood's ~0)?", flush=True)
    print(f"  Self-cal tau holds density ~real; the valve should OPEN silences in low-audio-energy sections.", flush=True)


if __name__ == '__main__':
    main()

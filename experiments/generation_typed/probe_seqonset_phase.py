#!/usr/bin/env python3
"""Phase distribution of the SS head's FREE-RUN onsets vs real vs the audio baseline (2026-06-29).
User by-ear: the seq-onset charts read "bland, only 1/16s" — the chaos-OOD smear signature, but NO chaos conditioning
was applied (radar=None everywhere). Hypothesis: the seq head produces a chaos-LIKE 16th-flood on its OWN. MEASURE
the 16th-grid phase shares (quarter t%4==0 / 8th t%4==2 / 16th-offbeat t%4∈{1,3}) of the onsets. Real Hard ≈
70/25/4. A 16th-flood (backbone gone) = the smear. Uses the M1a val cache (fast, no dataset re-parse)."""
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
from probe_seqonset_critic import audio_onset


def phase_shares(onset):
    """% of onsets on quarter / 8th / 16th-offbeat frames (16th grid). Real Hard ≈ 70 / 25 / 4."""
    t = np.arange(len(onset)); n = max(int(onset.sum()), 1)
    return (100 * onset[t % 4 == 0].sum() / n, 100 * onset[t % 4 == 2].sum() / n,
            100 * onset[(t % 4 == 1) | (t % 4 == 3)].sum() / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8); ap.add_argument('--cap', type=int, default=512)
    ap.add_argument('--tau', type=float, default=0.55)
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

    R = {'real': [], 'audio_raw': [], 'audio_unlock': [], 'seq': []}
    dens = {'real': [], 'seq': []}
    for s in val:
        T = min(s['T'], args.cap); a = s['audio'][:T].astype(np.float32)
        A42 = torch.from_numpy(a).unsqueeze(0).to(device); diff = torch.tensor([s['diff']], device=device)
        real_on = (s['typed'][:T] != 0).any(-1)
        seq_on = rollout(model, head, a, s['diff'], T, args.tau, device)
        d_seq = float(seq_on.mean()); dens['real'].append(float(real_on.mean())); dens['seq'].append(d_seq)
        # audio baselines at the seq density: raw (no calib) vs the deployed 16th-unlock onset_phase_calib=(0,1.0)
        with torch.no_grad():
            ol = model.onset_logits(model.encode_audio(A42), diff)[0]
            ph = torch.arange(T, device=device) % 4
            p_raw = torch.sigmoid(ol).cpu().numpy()
        a_raw = p_raw > float(np.quantile(p_raw, 1 - d_seq))
        a_unlock = audio_onset(model, A42, diff, T, d_seq, device).cpu().numpy()
        R['real'].append(phase_shares(real_on)); R['seq'].append(phase_shares(seq_on))
        R['audio_raw'].append(phase_shares(a_raw)); R['audio_unlock'].append(phase_shares(a_unlock))

    print(f"\n  phase shares (% of onsets), mean over {len(val)} Hard val songs   [real Hard ≈ 70 / 25 / 4]", flush=True)
    print(f"  {'arm':<24} {'quarter':>8} {'8th':>6} {'16th':>6}   density", flush=True)
    for k, label in [('real', 'REAL'), ('audio_raw', 'audio head (raw)'),
                     ('audio_unlock', 'audio + 16th-unlock'), ('seq', 'SEQ head free-run')]:
        q, e, s = np.mean(R[k], 0)
        d = f"{np.mean(dens['seq']):.3f}" if k == 'seq' else (f"{np.mean(dens['real']):.3f}" if k == 'real' else "(=d_seq)")
        print(f"  {label:<24} {q:>8.1f} {e:>6.1f} {s:>6.1f}   {d}", flush=True)
    qs, es, ss = np.mean(R['seq'], 0); qr, er, sr = np.mean(R['real'], 0)
    print(f"\n  SEQ 16th-share {ss:.0f}% vs real {sr:.0f}%  ->  "
          f"{'16th-FLOOD (chaos-like smear, backbone gone) — confirms the by-ear read' if ss > 2.5 * max(sr,1) else 'not a flood'}", flush=True)


if __name__ == '__main__':
    main()

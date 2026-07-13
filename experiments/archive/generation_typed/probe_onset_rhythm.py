#!/usr/bin/env python3
"""PROBE 2 of the jack-heaviness investigation: the ONSET head's contribution.

Probe 1 (probe_jack_temp.py) showed the PATTERN head's temperature is the direct jack lever. This
isolates the ONSET head. Two parts:

  A. NO-OP CHECK: onset_logit_scale (the natural "onset temperature") is applied as p=sigmoid(scale*ol),
     which is MONOTONIC -> it preserves the frame RANKING -> under quantile thresholding (deployed path:
     onset = top-density frames) the selected onset set is IDENTICAL for any scale. So there is no onset
     "temperature" that changes which onsets fire in deployment. We confirm it empirically (frames that
     differ from scale=1.0 across scales -> should be 0).

  B. RHYTHM REALISM: what the onset head CAN contribute to jacks is its SPACING -- if it deterministically
     places more tight 8th/16th clusters than real, that's more jack OPPORTUNITY the pattern head fills
     (prior: the long jacks were 8th-spaced, foot_exertion_findings.md). We compare the model's native
     onset inter-onset-interval (IOI) distribution to REAL's, density HELD (same onset count), so a
     difference is pure CLUSTERING, not count (experiment-design Rules 5/11). Phase grid (16th frames):
     gap 1 = 16th-adjacent, gap 2 = 8th, gap 3-4 = ~quarter, gap >4 = wider.

  python experiments/generation_typed/probe_onset_rhythm.py [--songs 16] [--scales 0.5,1,2]
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.playability_metrics import ACTIVE_SYMBOLS
from compare_foot_physics import load_songs, DIFF_NAMES


def _floats(s):
    return [float(x) for x in s.split(",") if x.strip() != ""]


def ioi_buckets(mask):
    """Inter-onset-interval bucket SHARES for a boolean onset mask, + max consecutive-onset run (gap<=4)."""
    idx = np.flatnonzero(mask)
    if idx.size < 2:
        return None
    g = np.diff(idx)
    n = g.size
    # max run of consecutive onsets within gap<=4 (the jack-OPPORTUNITY length, panel-agnostic)
    runs, cur = [], 1
    for gg in g:
        if gg <= 4:
            cur += 1
        else:
            runs.append(cur); cur = 1
    runs.append(cur)
    return {"g1": float((g == 1).mean()), "g2": float((g == 2).mean()),
            "g34": float(((g == 3) | (g == 4)).mean()), "gwide": float((g > 4).mean()),
            "n_gaps": int(n), "max_onset_run": int(max(runs)), "mean_ioi": float(g.mean())}


def onset_mask_at_scale(model, song, scale, device):
    """The model's NATIVE onset set at onset_logit_scale=scale, density matched to REAL (tau recomputed
    on the scaled probs, as deployment would)."""
    real = np.asarray(song['real'])
    tgt = float(np.isin(real, ACTIVE_SYMBOLS).any(1).mean())
    audio = torch.from_numpy(song['audio']).unsqueeze(0).to(device)
    dt = torch.tensor([song['diff']], device=device)
    radar_t = torch.from_numpy(song['radar']).unsqueeze(0).to(device)
    with torch.no_grad():
        ol = model.onset_logits(model.encode_audio(audio), dt, radar=radar_t)[0].cpu().numpy()
    p = 1.0 / (1.0 + np.exp(-scale * ol))                         # sigmoid(scale*ol)
    tau = float(np.quantile(p, 1 - tgt)) if tgt > 0 else 0.5
    return p > tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/gen_motif_full_fixed/best_val.pt")
    ap.add_argument("--songs", type=int, default=16)
    ap.add_argument("--by", default="rich")
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--scales", type=_floats, default=[0.5, 1.0, 2.0])
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    songs = load_songs(args, device)
    audio_dim = songs[0]['audio'].shape[1]
    model = LayeredTypedChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)['model_state_dict']); model.eval()

    # ---- PART A: onset_logit_scale no-op confirm -------------------------------
    print(f"\nPART A -- onset_logit_scale NO-OP CHECK (density held). frames differing from scale=1.0:")
    base = [onset_mask_at_scale(model, s, 1.0, device) for s in songs]
    for sc in args.scales:
        if sc == 1.0:
            continue
        diff = sum(int((onset_mask_at_scale(model, s, sc, device) != b).sum()) for s, b in zip(songs, base))
        print(f"  scale={sc:>4}: {diff} frame(s) differ across all {len(songs)} songs")
    print("  -> 0 confirms: onset 'temperature' (logit_scale) does NOT change which onsets fire under "
          "thresholding (monotonic ranking). No onset-temperature lever for jacks in deployment.\n")

    # ---- PART B: onset RHYTHM (model native vs REAL), density held -------------
    print("PART B -- ONSET-RHYTHM realism: model NATIVE onsets vs REAL, density matched, by difficulty.")
    print("Does the onset head over-produce tight 8th/16th spacings (= jack opportunity the pattern head fills)?")
    print("IOI gap: g1=16th-adjacent  g2=8th  g34=~quarter  gwide=>quarter.\n")
    by_diff = defaultdict(lambda: {"real": [], "model": []})
    for s in songs:
        real = np.asarray(s['real'])
        om_real = np.isin(real, ACTIVE_SYMBOLS).any(1)
        om_model = onset_mask_at_scale(model, s, 1.0, device)
        rb, mb = ioi_buckets(om_real), ioi_buckets(om_model)
        if rb and mb:
            by_diff[s['diff']]["real"].append(rb); by_diff[s['diff']]["model"].append(mb)

    def avg(rows, k):
        return float(np.mean([r[k] for r in rows])) if rows else float('nan')

    for d in sorted(by_diff):
        rows = by_diff[d]
        if not rows["real"]:
            continue
        print(f"=== {DIFF_NAMES[d]} (n={len(rows['real'])}) ===")
        print(f"  {'src':>7} {'g1(16th)%':>10} {'g2(8th)%':>9} {'g34(qtr)%':>10} {'gwide%':>7} {'meanIOI':>8} {'maxOnsetRun':>12}")
        for src in ("real", "model"):
            r = rows[src]
            print(f"  {src:>7} {100*avg(r,'g1'):>10.1f} {100*avg(r,'g2'):>9.1f} {100*avg(r,'g34'):>10.1f} "
                  f"{100*avg(r,'gwide'):>7.1f} {avg(r,'mean_ioi'):>8.2f} {avg(r,'max_onset_run'):>12.1f}")
        print()

    print("READ: if the model's g1/g2 (tight) shares EXCEED real's at matched density, the onset head clusters "
          "more tightly -> more jack opportunity (a real onset-head contribution, fixable at the onset head). "
          "If the model's IOI mix MATCHES real, the onset head is NOT implicated and the jack-heaviness is "
          "purely the pattern head (Probe 1).")


if __name__ == "__main__":
    main()

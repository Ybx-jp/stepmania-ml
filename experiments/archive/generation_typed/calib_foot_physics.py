#!/usr/bin/env python3
"""Calibrate FootPhysicsBaseline (beta, jump_bias) on the rich set.

WHY (experiment-design Rule 15): the head-vs-physics comparison must baseline against the
STRONGEST physics policy, not a default-knob strawman. This sweeps foot_phys's two knobs and
reports the footwork distance-to-real (same metric as compare_foot_physics: mJumpStrm + len2
+ len3 shares, jump%/jacks excluded -- G2/G6), so the comparison can use the BEST foot_phys.

foot_phys is numpy-only, so this sweep does NOT touch the learned model -- it's fast.

FAIRNESS NOTE: we tune foot_phys to MINIMIZE its own dist->real on the eval songs. That is
deliberately OPTIMISTIC for the baseline (it gets to peek at the target). If the learned head
still beats the so-tuned foot_phys (compare_foot_physics --beta <b> --jump_bias <j>), the
verdict is conservative -- the baseline had every advantage.

  python experiments/generation_typed/calib_foot_physics.py [--songs 16] [--by rich]
         [--betas 0.5,1,2,4] [--jump_biases -3,-2,-1,0]
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
from src.generation.baselines import FootPhysicsBaseline
from src.generation.playability_metrics import chart_metrics, same_panel_run_lengths, run_length_shares, ACTIVE_SYMBOLS
# reuse the harness's loader + distance so calibration and comparison can't drift (Rule 14)
from compare_foot_physics import load_songs, distance_to_real, summarize, DIFF_NAMES


def _floats(s):
    return [float(x) for x in s.split(",") if x.strip() != ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", type=int, default=16)
    ap.add_argument("--by", default="rich")
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--betas", type=_floats, default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--jump_biases", type=_floats, default=[-3.0, -2.0, -1.0, 0.0])
    args = ap.parse_args()
    set_seed(42); device = torch.device('cpu')   # foot_phys is numpy; no GPU needed

    songs = load_songs(args, device)
    # precompute the per-song REAL reference ONCE (it doesn't change across the grid)
    refs = []
    for s in songs:
        real = np.asarray(s['real'])
        om = np.isin(real, ACTIVE_SYMBOLS).any(1)                 # press mask (G1)
        refs.append({'diff': s['diff'], 'bpm': s['bpm'], 'om': om,
                     'real_m': chart_metrics(real), 'real_runs': same_panel_run_lengths(real)})

    diffs = sorted({r['diff'] for r in refs})
    results = {}   # (beta, jb) -> {'overall': x, 'per_diff': {d: x}}
    for beta in args.betas:
        for jb in args.jump_biases:
            by_diff = defaultdict(lambda: {'real_m': [], 'foot_m': [], 'real_runs': [], 'foot_runs': []})
            for r in refs:
                rng = np.random.default_rng(0)                    # same draws per cell -> deterministic
                fp = FootPhysicsBaseline(beta=beta, jump_bias=jb).generate(r['om'], r['diff'], r['bpm'], rng=rng)
                d = r['diff']
                by_diff[d]['real_m'].append(r['real_m']); by_diff[d]['foot_m'].append(chart_metrics(fp))
                by_diff[d]['real_runs'].extend(r['real_runs']); by_diff[d]['foot_runs'].extend(same_panel_run_lengths(fp))
            per_diff = {}
            for d in diffs:
                b = by_diff[d]
                if b['real_m']:
                    per_diff[d] = distance_to_real(b['foot_m'], b['real_m'], b['foot_runs'], b['real_runs'])
            results[(beta, jb)] = {'overall': float(np.mean(list(per_diff.values()))), 'per_diff': per_diff,
                                   'by_diff': by_diff}

    # ---- grid (overall dist->real; lower = better foot_phys) ---------------------
    print(f"\nFOOT_PHYS CALIBRATION  {len(songs)} songs (by={args.by})  dist->real (mJumpStrm,len2,len3); lower=better")
    print("rows = beta (greediness), cols = jump_bias (jump log-bias)\n")
    header = "  beta\\jb " + " ".join(f"{jb:>7.1f}" for jb in args.jump_biases)
    print(header)
    best = min(results, key=lambda k: results[k]['overall'])
    for beta in args.betas:
        cells = []
        for jb in args.jump_biases:
            v = results[(beta, jb)]['overall']
            mark = "*" if (beta, jb) == best else " "
            cells.append(f"{v:>6.3f}{mark}")
        print(f"  {beta:>6.2f}  " + " ".join(cells))
    bb, bjb = best
    print(f"\nBEST: beta={bb}, jump_bias={bjb}  ->  overall dist->real = {results[best]['overall']:.3f}  (* above)")

    # ---- the best cell, per difficulty + run-length shape vs real ----------------
    print(f"\nBEST CELL detail (beta={bb}, jump_bias={bjb}) -- foot_phys vs real, by difficulty:")
    bd = results[best]['by_diff']
    for d in diffs:
        b = bd[d]
        if not b['real_m']:
            continue
        fs, rs = summarize(b['foot_m']), summarize(b['real_m'])
        fsh, rsh = run_length_shares(b['foot_runs']), run_length_shares(b['real_runs'])
        print(f"  --- {DIFF_NAMES[d]} (n={len(b['real_m'])}) ---   dist->real = {results[best]['per_diff'][d]:.3f}")
        print(f"      {'':>10} {'jump%':>6} {'mJumpStrm':>10} {'len2%':>6} {'len3%':>6} {'>=4%':>6}")
        print(f"      {'foot_phys':>10} {100*fs['jump_rate']:>6.1f} {fs['max_jump_stream']:>10.1f} "
              f"{100*fsh['len2_share']:>6.1f} {100*fsh['len3_share']:>6.1f} {100*fsh['ge4_share']:>6.1f}")
        print(f"      {'real':>10} {100*rs['jump_rate']:>6.1f} {rs['max_jump_stream']:>10.1f} "
              f"{100*rsh['len2_share']:>6.1f} {100*rsh['len3_share']:>6.1f} {100*rsh['ge4_share']:>6.1f}")

    print(f"\nNEXT: re-run the head-to-head with the calibrated baseline:")
    print(f"  python experiments/generation_typed/compare_foot_physics.py --songs {args.songs} "
          f"--beta {bb} --jump_bias {bjb} --export_sm 2")
    print("If model_raw STILL beats this foot_phys, the verdict holds against a baseline tuned to its best.")


if __name__ == "__main__":
    main()

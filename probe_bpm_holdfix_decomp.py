#!/usr/bin/env python3
"""Attribute the BPM->quality slope-flattening to the RIGHT shipped knob.

The full 2026-07-02 ship = {hold_stream_penalty=8, floor=0.45} + {footswitch=False}. Rerunning
probe_quality_variance.py under those defaults flattened the BPM->quality slope (-0.68 -> +0.11) AND lifted
ALL songs (+3.5 margin, 30/30). Two knobs shipped together, so which does what is confounded. This isolates
each by a ONE-VARIABLE override on the canonical path (experiment-design Rule 11/16 — credit the right variable):

  arm 'hs'  : hold_stream ON (canonical 8/0.45) + footswitch=True  (revert footswitch)  -> hold_stream ALONE
  arm 'fs'  : hold_stream_penalty=0             + footswitch=False (canonical)          -> footswitch ALONE

Same songs/seed/critic/K as the full run so the slopes are directly comparable to baseline (-0.68) and
full-fix (+0.11). Graded critic margin, 8-gen mean (0.95 reliable). Reuses the canonical helpers.
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, csv, sys
from pathlib import Path
import numpy as np, torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.models import LateFusionClassifier
from probe_quality_features import (load_val_dataset, build_songs, canonical_gen_typed, to_binary,
                                    critic_score, spearman, load_generator, DEPLOYED_CHECKPOINT)

ARMS = {
    'hs': {'footswitch': True},                       # hold_stream ON (default), footswitch reverted -> hold_stream alone
    'fs': {'hold_stream_penalty': 0.0},               # footswitch OFF (default), hold_stream disabled -> footswitch alone
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--arm', required=True, choices=list(ARMS))
    p.add_argument('--seed', type=int, default=42); p.add_argument('--difficulty', type=int, default=3)
    p.add_argument('--n', type=int, default=30); p.add_argument('--k', type=int, default=8)
    p.add_argument('--max_len', type=int, default=768)
    p.add_argument('--checkpoint', default=DEPLOYED_CHECKPOINT)
    p.add_argument('--critic', default='checkpoints/realism_critic_graded/best_val.pt')
    p.add_argument('--out', default=None)
    return p.parse_args()


def main():
    args = parse_args(); set_seed(args.seed)
    ov = ARMS[args.arm]; out = args.out or f'cache/quality_variance_hard_{args.arm}only.csv'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device} | arm={args.arm} override={ov} | n={args.n} x k={args.k}")
    val_ds = load_val_dataset(args.data_dir, args.audio_dir, args.seed)
    songs = build_songs(val_ds, args.n, args.difficulty, args.max_len); print(f"songs={len(songs)}")
    model = load_generator(args.checkpoint, 42, device)
    ck = torch.load(args.critic, map_location=device, weights_only=False)
    critic = LateFusionClassifier(ck['config']).to(device); critic.load_state_dict(ck['model_state_dict']); critic.eval()

    rows = []
    for n, s in enumerate(songs, 1):
        a23 = s['audio'][:, :23]
        margins = np.array([critic_score(critic, a23, to_binary(canonical_gen_typed(model, s, device, ov)), device)[1]
                            for _ in range(args.k)])
        rows.append({'title': s['title'], 'bpm': s['bpm'], 'real_density': s['real_density'],
                     'm_gen_mean': float(margins.mean()), 'm_gen_sd': float(margins.std(ddof=1)),
                     **{f'g{j}': float(margins[j]) for j in range(args.k)}})
        print(f"  [{n}/{len(songs)}] {s['title'][:24]:24s} m_gen={margins.mean():+.2f}±{margins.std(ddof=1):.2f}")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    bpm = np.array([r['bpm'] for r in rows]); mg = np.array([r['m_gen_mean'] for r in rows])
    print(f"\nwrote {out}")
    print(f"  arm={args.arm}: spearman(bpm, m_gen)={spearman(bpm, mg):+.3f}  mean m_gen={mg.mean():+.3f}")
    print(f"  (baseline -0.682/-2.065 ; full-fix +0.111/+1.478)")


if __name__ == '__main__':
    main()

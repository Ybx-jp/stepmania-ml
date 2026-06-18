#!/usr/bin/env python3
"""
Eval-only: load the trained Stage 2 checkpoint and compare decoding strategies.

Greedy argmax collapses to the dominant empty state; this checks whether
temperature/top-k sampling unlocks the learned distribution (onset placement).
No training. Usage mirrors train_transformer.py data flags.
"""

import warnings, os
warnings.filterwarnings('ignore')
os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'

import argparse, glob, sys
from pathlib import Path
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.generation.transformer import ChartGenerator
from src.generation.evaluation import onset_density_metrics, DifficultyCritic

DEFAULT_BPM = 150.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint', default='checkpoints/gen_transformer/best_val_ce.pt')
    p.add_argument('--eval_songs', type=int, default=64)
    p.add_argument('--max_gen_len', type=int, default=768)
    p.add_argument('--num_layers', type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    chart_files = glob.glob(f"{args.data_dir}/**/*.sm", recursive=True)
    chart_files += glob.glob(f"{args.data_dir}/**/*.ssc", recursive=True)
    _, val_files, _ = create_data_splits(chart_files, random_state=args.seed)
    with open(PROJECT_ROOT / "config/model_config.yaml") as f:
        max_seq_len = yaml.safe_load(f)['classifier']['max_sequence_length']
    _, val_ds, _ = create_datasets(train_files=[], val_files=val_files, test_files=[],
                                   audio_dir=args.audio_dir, max_sequence_length=max_seq_len,
                                   cache_dir='cache/samples')
    val_ds.warm_cache(show_progress=True)
    val = []
    for i in range(min(args.eval_songs, len(val_ds))):
        s = val_ds[i]; T = int(s['mask'].sum().item())
        val.append({'chart': s['chart'][:T].numpy().astype(np.float32),
                    'audio': s['audio'][:T].numpy().astype(np.float32),
                    'difficulty': int(s['difficulty'])})
    audio_dim = val[0]['audio'].shape[1]

    model = ChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=args.num_layers).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model_state_dict'])
    model.eval()
    critic = DifficultyCritic(device=device)

    strategies = [
        ('greedy', dict(greedy=True)),
        ('temp=1.0', dict(greedy=False, temperature=1.0)),
        ('temp=0.9 top_k=4', dict(greedy=False, temperature=0.9, top_k=4)),
        ('temp=0.7 top_k=2', dict(greedy=False, temperature=0.7, top_k=2)),
    ]

    print(f"\nReal val density (eval set): "
          f"{np.mean([(s['chart'].sum(1) > 0).mean() for s in val]):.3f}")
    print("=" * 84)
    print(f"  {'decoding':<20} {'onset_F1':>9} {'onset_P':>8} {'onset_R':>8} "
          f"{'gen_dens':>9} {'crit_adj':>9} {'crit_mae':>9}")
    print("-" * 84)

    for name, kw in strategies:
        set_seed(args.seed)
        f1s, ps, rs, dens, gens, tg = [], [], [], [], [], []
        for i in range(0, len(val), 16):
            batch = val[i:i + 16]
            L = min(args.max_gen_len, max(len(s['chart']) for s in batch))
            B = len(batch)
            audio = torch.zeros(B, L, audio_dim); lengths = torch.zeros(B, dtype=torch.long)
            diff = torch.zeros(B, dtype=torch.long)
            for b, s in enumerate(batch):
                t = min(len(s['chart']), L)
                audio[b, :t] = torch.from_numpy(s['audio'][:t]); lengths[b] = t; diff[b] = s['difficulty']
            gen = model.generate(audio.to(device), diff.to(device),
                                 lengths=lengths.to(device), **kw).cpu().numpy()
            for b, s in enumerate(batch):
                t = int(lengths[b]); g = gen[b, :t]
                m = onset_density_metrics(g, reference=s['chart'][:t])
                f1s.append(m['onset_f1']); ps.append(m.get('onset_precision', 0))
                rs.append(m.get('onset_recall', 0)); dens.append(m['gen_density'])
                gens.append(g); tg.append(s['difficulty'])
        preds = np.array([critic.predict(g, s['audio'][:len(g)], bpm=DEFAULT_BPM)['class']
                          for g, s in zip(gens, val)])
        d = np.abs(preds - np.array(tg))
        print(f"  {name:<20} {np.mean(f1s):>9.3f} {np.mean(ps):>8.3f} {np.mean(rs):>8.3f} "
              f"{np.mean(dens):>9.3f} {np.mean(d <= 1):>9.3f} {np.mean(d):>9.3f}")
    print("=" * 84)
    print("  Floor: per-frame MLP onset_F1=0.053; n-gram crit_adj=0.977")


if __name__ == '__main__':
    main()

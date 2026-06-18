#!/usr/bin/env python3
"""
Per-difficulty onset threshold calibration (no retraining).

The factorized model's difficulty fidelity (crit_adj 0.734) lagged because a single
GLOBAL onset threshold ignores that each difficulty has its own density — it
over-places on easy charts and under-places on hard ones, smearing the signal the
critic reads. Fix: calibrate a separate threshold per difficulty so each class's
generated density matches that class's real density. Also tries Bernoulli-sampled
onsets (no threshold at all) as a calibration-free alternative.

Compares three decode strategies on the trained checkpoint:
  - global threshold (the Stage 3 baseline)
  - per-difficulty thresholds
  - per-frame Bernoulli sampling from the difficulty-conditioned onset head

Usage:
    python experiments/generation_factorized/eval_per_difficulty.py \
        --data_dir data/ --audio_dir data/
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
from src.generation.factorized import FactorizedChartGenerator
from src.generation.evaluation import onset_density_metrics, DifficultyCritic

DEFAULT_BPM = 150.0
DIFF_NAMES = ['Beginner', 'Easy', 'Medium', 'Hard']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint', default='checkpoints/gen_factorized/best_val.pt')
    p.add_argument('--eval_songs', type=int, default=96)
    p.add_argument('--max_gen_len', type=int, default=768)
    p.add_argument('--num_layers', type=int, default=4)
    p.add_argument('--onset_layers', type=int, default=2)
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
        s = val_ds[i]; T = min(int(s['mask'].sum().item()), args.max_gen_len)
        val.append({'chart': s['chart'][:T].numpy().astype(np.float32),
                    'audio': s['audio'][:T].numpy().astype(np.float32),
                    'difficulty': int(s['difficulty'])})
    audio_dim = val[0]['audio'].shape[1]

    model = FactorizedChartGenerator(audio_dim=audio_dim, d_model=128,
                                     num_layers=args.num_layers, onset_layers=args.onset_layers).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model_state_dict'])
    model.eval()
    critic = DifficultyCritic(device=device)

    # onset posteriors per song (one non-AR pass) + per-difficulty target densities
    post, real_dens = {}, {}
    with torch.no_grad():
        for s in val:
            d = s['difficulty']
            audio = torch.from_numpy(s['audio']).unsqueeze(0).to(device)
            diff = torch.tensor([d], device=device)
            p = torch.sigmoid(model.onset_logits(model.encode_audio(audio), diff))[0].cpu().numpy()
            post.setdefault(d, []).append(p)
            real_dens.setdefault(d, []).append((s['chart'].sum(1) > 0).mean())

    global_density = float(np.mean([(s['chart'].sum(1) > 0).mean() for s in val]))
    all_post = np.concatenate([p for ps in post.values() for p in ps])
    tau_global = float(np.quantile(all_post, 1 - global_density))
    tau_by_diff = {}
    for d in post:
        pooled = np.concatenate(post[d]); td = float(np.mean(real_dens[d]))
        tau_by_diff[d] = float(np.quantile(pooled, 1 - td))
    print("Per-difficulty target density / threshold:")
    for d in sorted(post):
        print(f"  {DIFF_NAMES[d]:<9} density={np.mean(real_dens[d]):.3f}  tau={tau_by_diff[d]:.3f}  "
              f"(global tau={tau_global:.3f})")

    by_d = {}
    for s in val:
        by_d.setdefault(s['difficulty'], []).append(s)

    def run(strategy):
        f1s, ps, rs, dens = [], [], [], []
        preds, tgts = [], []
        for d, songs in by_d.items():
            for i in range(0, len(songs), 16):
                batch = songs[i:i + 16]
                L = max(len(s['chart']) for s in batch); B = len(batch)
                audio = torch.zeros(B, L, audio_dim); lengths = torch.zeros(B, dtype=torch.long)
                diff = torch.full((B,), d, dtype=torch.long)
                for b, s in enumerate(batch):
                    t = len(s['chart']); audio[b, :t] = torch.from_numpy(s['audio']); lengths[b] = t
                kw = dict(panel_greedy=True)
                if strategy == 'global':
                    kw['onset_threshold'] = tau_global
                elif strategy == 'per_diff':
                    kw['onset_threshold'] = tau_by_diff[d]
                elif strategy == 'bernoulli':
                    kw['onset_sample'] = True
                gen = model.generate(audio.to(device), diff.to(device), lengths=lengths.to(device), **kw).cpu().numpy()
                for b, s in enumerate(batch):
                    t = int(lengths[b]); g = gen[b, :t]
                    m = onset_density_metrics(g, reference=s['chart'][:t])
                    f1s.append(m['onset_f1']); ps.append(m.get('onset_precision', 0)); rs.append(m.get('onset_recall', 0))
                    dens.append(m['gen_density'])
                    preds.append(critic.predict(g, s['audio'][:t], bpm=DEFAULT_BPM)['class']); tgts.append(d)
        dd = np.abs(np.array(preds) - np.array(tgts))
        return dict(onset_f1=np.mean(f1s), prec=np.mean(ps), rec=np.mean(rs), density=np.mean(dens),
                    exact=np.mean(dd == 0), adj=np.mean(dd <= 1), mae=np.mean(dd))

    results = {s: run(s) for s in ['global', 'per_diff', 'bernoulli']}

    print("\n" + "=" * 80)
    print("  PER-DIFFICULTY THRESHOLD  vs  GLOBAL  vs  BERNOULLI")
    print("=" * 80)
    print(f"  {'strategy':<14} {'onset_F1':>9} {'prec':>7} {'rec':>7} {'density':>8} "
          f"{'crit_exact':>11} {'crit_adj':>9} {'crit_mae':>9}")
    print("-" * 80)
    for name in ['global', 'per_diff', 'bernoulli']:
        r = results[name]
        print(f"  {name:<14} {r['onset_f1']:>9.3f} {r['prec']:>7.3f} {r['rec']:>7.3f} {r['density']:>8.3f} "
              f"{r['exact']:>11.3f} {r['adj']:>9.3f} {r['mae']:>9.3f}")
    print("-" * 80)
    print(f"  real density {global_density:.3f}. Stage 3 global baseline crit_adj was 0.734.")
    print("=" * 80)


if __name__ == '__main__':
    main()

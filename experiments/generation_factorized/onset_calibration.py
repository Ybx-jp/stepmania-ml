#!/usr/bin/env python3
"""
Onset-head recalibration (no retraining).

The factorized onset head is over-confident (a pos_weight side effect), so
Bernoulli-sampled decoding over-places. Fix it post-hoc with per-difficulty Platt
scaling — fit p = sigmoid(a*logit + c) per difficulty so each class's calibrated
probabilities are honest and their mean matches that class's real density. Then
per-difficulty Bernoulli should hit correct density AND keep high difficulty
fidelity (best-of-both vs the F1-optimal threshold decode).

Compares: per-difficulty threshold (F1-optimal), raw Bernoulli (over-places),
calibrated per-difficulty Bernoulli (the fix), calibrated per-difficulty threshold.

Usage:
    python experiments/generation_factorized/onset_calibration.py \
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
from sklearn.linear_model import LogisticRegression

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


def ece(probs, labels, n_bins=10):
    """Expected calibration error: avg |confidence - accuracy| over probability bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs >= lo) & (probs < hi)
        if m.sum() > 0:
            e += (m.mean()) * abs(probs[m].mean() - labels[m].mean())
    return float(e)


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

    # raw onset logits + labels per difficulty (one non-AR pass per song)
    logits_d, labels_d, dens_d = {}, {}, {}
    with torch.no_grad():
        for s in val:
            d = s['difficulty']
            audio = torch.from_numpy(s['audio']).unsqueeze(0).to(device)
            diff = torch.tensor([d], device=device)
            lg = model.onset_logits(model.encode_audio(audio), diff)[0].cpu().numpy()
            logits_d.setdefault(d, []).append(lg)
            labels_d.setdefault(d, []).append((s['chart'].sum(1) > 0).astype(np.float32))
            dens_d.setdefault(d, []).append((s['chart'].sum(1) > 0).mean())

    # fit per-difficulty Platt scaling: p = sigmoid(a*logit + c)
    platt, tau_raw, tau_cal = {}, {}, {}
    print("Calibration (per difficulty):")
    print(f"  {'diff':<9} {'real_dens':>9} {'raw_mean_p':>10} {'cal_mean_p':>10} {'raw_ECE':>8} {'cal_ECE':>8}  a/c")
    for d in sorted(logits_d):
        lg = np.concatenate(logits_d[d]); lab = np.concatenate(labels_d[d])
        lr = LogisticRegression(C=1e6, solver='lbfgs').fit(lg.reshape(-1, 1), lab.astype(int))
        a = float(lr.coef_[0, 0]); c = float(lr.intercept_[0]); platt[d] = (a, c)
        raw_p = 1 / (1 + np.exp(-lg)); cal_p = 1 / (1 + np.exp(-(a * lg + c)))
        td = float(np.mean(dens_d[d]))
        tau_raw[d] = float(np.quantile(raw_p, 1 - td))
        tau_cal[d] = float(np.quantile(cal_p, 1 - td))
        print(f"  {DIFF_NAMES[d]:<9} {td:>9.3f} {raw_p.mean():>10.3f} {cal_p.mean():>10.3f} "
              f"{ece(raw_p, lab):>8.3f} {ece(cal_p, lab):>8.3f}  a={a:.2f} c={c:.2f}")

    by_d = {}
    for s in val:
        by_d.setdefault(s['difficulty'], []).append(s)

    def run(strategy):
        f1s, ps, rs, dens, preds, tgts = [], [], [], [], [], []
        for d, songs in by_d.items():
            a, c = platt[d]
            for i in range(0, len(songs), 16):
                batch = songs[i:i + 16]; L = max(len(s['chart']) for s in batch); B = len(batch)
                audio = torch.zeros(B, L, audio_dim); lengths = torch.zeros(B, dtype=torch.long)
                diff = torch.full((B,), d, dtype=torch.long)
                for b, s in enumerate(batch):
                    t = len(s['chart']); audio[b, :t] = torch.from_numpy(s['audio']); lengths[b] = t
                kw = dict(panel_greedy=True)
                if strategy == 'perdiff_thresh':
                    kw['onset_threshold'] = tau_raw[d]
                elif strategy == 'bernoulli_raw':
                    kw['onset_sample'] = True
                elif strategy == 'bernoulli_cal':
                    kw.update(onset_sample=True, onset_logit_scale=a, onset_logit_bias=c)
                elif strategy == 'perdiff_thresh_cal':
                    kw.update(onset_threshold=tau_cal[d], onset_logit_scale=a, onset_logit_bias=c)
                gen = model.generate(audio.to(device), diff.to(device), lengths=lengths.to(device), **kw).cpu().numpy()
                for b, s in enumerate(batch):
                    t = int(lengths[b]); g = gen[b, :t]; m = onset_density_metrics(g, reference=s['chart'][:t])
                    f1s.append(m['onset_f1']); ps.append(m.get('onset_precision', 0)); rs.append(m.get('onset_recall', 0))
                    dens.append(m['gen_density'])
                    preds.append(critic.predict(g, s['audio'][:t], bpm=DEFAULT_BPM)['class']); tgts.append(d)
        dd = np.abs(np.array(preds) - np.array(tgts))
        return dict(f1=np.mean(f1s), prec=np.mean(ps), rec=np.mean(rs), density=np.mean(dens),
                    exact=np.mean(dd == 0), adj=np.mean(dd <= 1), mae=np.mean(dd))

    strategies = ['perdiff_thresh', 'bernoulli_raw', 'bernoulli_cal', 'perdiff_thresh_cal']
    res = {s: run(s) for s in strategies}
    real_density = float(np.mean([(s['chart'].sum(1) > 0).mean() for s in val]))

    print("\n" + "=" * 84)
    print("  ONSET RECALIBRATION  (real density {:.3f})".format(real_density))
    print("=" * 84)
    print(f"  {'strategy':<20} {'onset_F1':>9} {'prec':>7} {'rec':>7} {'density':>8} "
          f"{'crit_exact':>11} {'crit_adj':>9} {'crit_mae':>9}")
    print("-" * 84)
    for s in strategies:
        r = res[s]
        print(f"  {s:<20} {r['f1']:>9.3f} {r['prec']:>7.3f} {r['rec']:>7.3f} {r['density']:>8.3f} "
              f"{r['exact']:>11.3f} {r['adj']:>9.3f} {r['mae']:>9.3f}")
    print("=" * 84)


if __name__ == '__main__':
    main()

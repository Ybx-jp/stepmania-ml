#!/usr/bin/env python3
"""
Hybrid onset decode: deterministic-confident + sample-uncertain.

The onset-calibration run showed an F1-vs-fidelity frontier: threshold maximizes
onset_F1 (deterministic top frames), Bernoulli maximizes difficulty fidelity
(stochastic, preserves per-difficulty character). Hybrid interpolates: using the
CALIBRATED per-difficulty onset probabilities and the density-matching threshold
tau_d, for each frame
    p > tau_d + m  -> onset      (deterministic, high confidence)
    p < tau_d - m  -> no onset   (deterministic, low confidence)
    else           -> Bernoulli(p) (sample the uncertain middle band)
The margin m sweeps m=0 (pure threshold, max F1) -> large m (pure Bernoulli, max
fidelity). No retraining; onset decision injected via generate(onset_override=...).

Usage:
    python experiments/generation_factorized/hybrid_decode.py --data_dir data/ --audio_dir data/
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
    p.add_argument('--margins', type=str, default='0.0,0.05,0.1,0.2,0.5')
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

    # raw onset logits per song; fit per-difficulty Platt; store calibrated p per song
    logits_d, labels_d, dens_d = {}, {}, {}
    with torch.no_grad():
        for s in val:
            d = s['difficulty']
            audio = torch.from_numpy(s['audio']).unsqueeze(0).to(device)
            lg = model.onset_logits(model.encode_audio(audio), torch.tensor([d], device=device))[0].cpu().numpy()
            s['logit'] = lg
            logits_d.setdefault(d, []).append(lg)
            labels_d.setdefault(d, []).append((s['chart'].sum(1) > 0).astype(np.float32))
            dens_d.setdefault(d, []).append((s['chart'].sum(1) > 0).mean())

    platt, tau_d = {}, {}
    for d in logits_d:
        lg = np.concatenate(logits_d[d]); lab = np.concatenate(labels_d[d]).astype(int)
        lr = LogisticRegression(C=1e6, solver='lbfgs').fit(lg.reshape(-1, 1), lab)
        a, c = float(lr.coef_[0, 0]), float(lr.intercept_[0]); platt[d] = (a, c)
        td = float(np.mean(dens_d[d]))
        cal_pool = 1 / (1 + np.exp(-(a * lg + c)))
        tau_d[d] = float(np.quantile(cal_pool, 1 - td))
    for s in val:
        a, c = platt[s['difficulty']]
        s['cal_p'] = 1 / (1 + np.exp(-(a * s['logit'] + c)))

    by_d = {}
    for s in val:
        by_d.setdefault(s['difficulty'], []).append(s)

    def run(m, rng):
        f1s, ps, rs, dens, preds, tgts = [], [], [], [], [], []
        for d, songs in by_d.items():
            tau = tau_d[d]
            for i in range(0, len(songs), 16):
                batch = songs[i:i + 16]; L = max(len(s['chart']) for s in batch); B = len(batch)
                audio = torch.zeros(B, L, audio_dim); lengths = torch.zeros(B, dtype=torch.long)
                override = torch.zeros(B, L, dtype=torch.bool)
                diff = torch.full((B,), d, dtype=torch.long)
                for b, s in enumerate(batch):
                    t = len(s['chart']); audio[b, :t] = torch.from_numpy(s['audio']); lengths[b] = t
                    p = s['cal_p']
                    dec = np.zeros(t, dtype=bool)
                    dec[p > tau + m] = True
                    band = (p >= tau - m) & (p <= tau + m)
                    dec[band] = rng.random(band.sum()) < p[band]
                    override[b, :t] = torch.from_numpy(dec)
                gen = model.generate(audio.to(device), diff.to(device), lengths=lengths.to(device),
                                     onset_override=override.to(device), panel_greedy=True).cpu().numpy()
                for b, s in enumerate(batch):
                    t = int(lengths[b]); g = gen[b, :t]; mm = onset_density_metrics(g, reference=s['chart'][:t])
                    f1s.append(mm['onset_f1']); ps.append(mm.get('onset_precision', 0)); rs.append(mm.get('onset_recall', 0))
                    dens.append(mm['gen_density'])
                    preds.append(critic.predict(g, s['audio'][:t], bpm=DEFAULT_BPM)['class']); tgts.append(d)
        dd = np.abs(np.array(preds) - np.array(tgts))
        return dict(f1=np.mean(f1s), prec=np.mean(ps), rec=np.mean(rs), density=np.mean(dens),
                    exact=np.mean(dd == 0), adj=np.mean(dd <= 1), mae=np.mean(dd))

    margins = [float(x) for x in args.margins.split(',')]
    real_density = float(np.mean([(s['chart'].sum(1) > 0).mean() for s in val]))

    print("\n" + "=" * 88)
    print(f"  HYBRID ONSET DECODE  (calibrated; real density {real_density:.3f})")
    print("=" * 88)
    print(f"  {'margin m':<12} {'onset_F1':>9} {'prec':>7} {'rec':>7} {'density':>8} "
          f"{'crit_exact':>11} {'crit_adj':>9} {'crit_mae':>9}")
    print("-" * 88)
    for m in margins:
        set_seed(args.seed); rng = np.random.default_rng(args.seed)
        r = run(m, rng)
        tag = f"{m:.2f}" + (" (=thresh)" if m == 0 else (" (~bern)" if m >= 0.5 else ""))
        print(f"  {tag:<12} {r['f1']:>9.3f} {r['prec']:>7.3f} {r['rec']:>7.3f} {r['density']:>8.3f} "
              f"{r['exact']:>11.3f} {r['adj']:>9.3f} {r['mae']:>9.3f}")
    print("=" * 88)
    print("  m=0 -> pure per-diff threshold (F1-optimal); large m -> calibrated Bernoulli (fidelity).")


if __name__ == '__main__':
    main()

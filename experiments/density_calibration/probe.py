#!/usr/bin/env python3
"""
Density-calibration probe (no retraining).

Decides whether the Stage 2 generator's over-placement is a calibration/decoding
problem (-> build the factorized onset-then-panel head) or a localization problem
(-> need capacity/training). On the trained checkpoint:

  1. Teacher-forced onset posteriors -> ROC-AUC / PR-AUC: "does it know where?"
  2. Teacher-forced, density-matched threshold -> onset_F1: clean-context ceiling.
  3. Free-running, density-matched threshold -> onset_F1: honest generation number.
  4. Compare to Stage 2 temperature baselines.

Usage:
    python experiments/density_calibration/probe.py --data_dir data/ --audio_dir data/
"""

import warnings, os
warnings.filterwarnings('ignore')
os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'

import argparse, glob, sys
from pathlib import Path
import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.generation.transformer import ChartGenerator
from src.generation.tokenizer import ChartTokenizer, BOS_TOKEN, NUM_PANEL_STATES
from src.generation.evaluation import onset_density_metrics, DifficultyCritic

DEFAULT_BPM = 150.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint', default='checkpoints/gen_transformer/best_val_ce.pt')
    p.add_argument('--eval_songs', type=int, default=64)
    p.add_argument('--max_len', type=int, default=768)
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
        s = val_ds[i]; T = min(int(s['mask'].sum().item()), args.max_len)
        val.append({'chart': s['chart'][:T].numpy().astype(np.float32),
                    'audio': s['audio'][:T].numpy().astype(np.float32),
                    'difficulty': int(s['difficulty'])})
    audio_dim = val[0]['audio'].shape[1]

    model = ChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=args.num_layers).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model_state_dict'])
    model.eval()
    critic = DifficultyCritic(device=device)

    target_density = float(np.mean([(s['chart'].sum(1) > 0).mean() for s in val]))

    # ---- 1. teacher-forced onset posteriors (one forward per song) ----
    all_p, all_y = [], []
    tf_panel, tf_ponset, refs = [], [], []
    with torch.no_grad():
        for s in val:
            T = len(s['chart'])
            audio = torch.from_numpy(s['audio']).unsqueeze(0).to(device)
            states = ChartTokenizer.encode(s['chart'])  # (T,)
            in_tok = torch.full((1, T), BOS_TOKEN, dtype=torch.long)
            in_tok[0, 1:] = states[:-1]
            diff = torch.tensor([s['difficulty']], device=device)
            logits = model.forward(audio, in_tok.to(device), diff)  # (1,T,VOCAB)
            probs = torch.softmax(logits[0, :, :NUM_PANEL_STATES], dim=-1)  # (T,16)
            p_onset = (1 - probs[:, 0]).cpu().numpy()
            panel = (probs[:, 1:].argmax(-1) + 1).cpu().numpy()  # best non-empty state
            true_onset = (s['chart'].sum(1) > 0).astype(int)
            all_p.append(p_onset); all_y.append(true_onset)
            tf_ponset.append(p_onset); tf_panel.append(panel); refs.append(s['chart'])

    flat_p = np.concatenate(all_p); flat_y = np.concatenate(all_y)
    roc = roc_auc_score(flat_y, flat_p)
    pr = average_precision_score(flat_y, flat_p)

    # threshold to match target density on pooled teacher-forced posteriors
    tau = float(np.quantile(flat_p, 1 - target_density))

    # ---- 2. teacher-forced thresholded decode ----
    def decode_from(ponset_list, panel_list):
        f1s, ps, rs, dens = [], [], [], []
        for ponset, panel, ref in zip(ponset_list, panel_list, refs):
            states = np.where(ponset > tau, panel, 0).astype(np.int64)
            chart = ((states[:, None] >> np.arange(4)) & 1).astype(np.float32)
            m = onset_density_metrics(chart, reference=ref)
            f1s.append(m['onset_f1']); ps.append(m.get('onset_precision', 0))
            rs.append(m.get('onset_recall', 0)); dens.append(m['gen_density'])
        return np.mean(f1s), np.mean(ps), np.mean(rs), np.mean(dens)

    tf_f1, tf_p, tf_r, tf_d = decode_from(tf_ponset, tf_panel)

    # ---- 3. free-running thresholded generation ----
    fr_f1s, fr_ps, fr_rs, fr_dens, gens, tgts = [], [], [], [], [], []
    for i in range(0, len(val), 16):
        batch = val[i:i + 16]
        L = max(len(s['chart']) for s in batch); B = len(batch)
        audio = torch.zeros(B, L, audio_dim); lengths = torch.zeros(B, dtype=torch.long)
        diff = torch.zeros(B, dtype=torch.long)
        for b, s in enumerate(batch):
            t = len(s['chart']); audio[b, :t] = torch.from_numpy(s['audio'])
            lengths[b] = t; diff[b] = s['difficulty']
        gen = model.generate(audio.to(device), diff.to(device), lengths=lengths.to(device),
                             onset_threshold=tau).cpu().numpy()
        for b, s in enumerate(batch):
            t = int(lengths[b]); g = gen[b, :t]
            m = onset_density_metrics(g, reference=s['chart'])
            fr_f1s.append(m['onset_f1']); fr_ps.append(m.get('onset_precision', 0))
            fr_rs.append(m.get('onset_recall', 0)); fr_dens.append(m['gen_density'])
            gens.append(g); tgts.append(s['difficulty'])
    preds = np.array([critic.predict(g, s['audio'][:len(g)], bpm=DEFAULT_BPM)['class']
                      for g, s in zip(gens, val)])
    d = np.abs(preds - np.array(tgts))

    print("\n" + "=" * 78)
    print("  DENSITY-CALIBRATION PROBE")
    print("=" * 78)
    print(f"  target (real) density : {target_density:.3f}   threshold tau={tau:.3f}")
    print(f"  onset ROC-AUC         : {roc:.3f}   (0.5 = chance)")
    print(f"  onset PR-AUC          : {pr:.3f}   (base rate {target_density:.3f})")
    print("-" * 78)
    print(f"  {'decode':<28} {'onset_F1':>9} {'onset_P':>8} {'onset_R':>8} {'density':>8}")
    print("-" * 78)
    print(f"  {'teacher-forced @ tau':<28} {tf_f1:>9.3f} {tf_p:>8.3f} {tf_r:>8.3f} {tf_d:>8.3f}")
    print(f"  {'free-running @ tau':<28} {np.mean(fr_f1s):>9.3f} {np.mean(fr_ps):>8.3f} "
          f"{np.mean(fr_rs):>8.3f} {np.mean(fr_dens):>8.3f}")
    print("-" * 78)
    print("  Stage 2 temperature baselines (for reference):")
    print(f"  {'temp 0.7 top_k 2':<28} {0.245:>9.3f} {0.210:>8.3f} {0.332:>8.3f} {0.308:>8.3f}")
    print(f"  {'temp 1.0':<28} {0.300:>9.3f} {0.210:>8.3f} {0.577:>8.3f} {0.536:>8.3f}")
    print("-" * 78)
    print(f"  free-running @ tau critic: exact={np.mean(d==0):.3f} adjacent={np.mean(d<=1):.3f} "
          f"mae={np.mean(d):.3f}")
    print("=" * 78)
    print("  Verdict: onset_F1(free @ tau) vs temp -> calibration win if higher at matched density.")
    print("           low ROC-AUC -> model can't localize (need capacity, not just a head).")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Diagnose + fix the "always Left / repeats the previous note" bias (playtest feedback).

The pattern (which-panels) head was decoded GREEDY -> always the single most-likely
pattern -> Left-bias + jacks (repeating the previous note). Compare decode strategies
on the trained checkpoint (no retrain) against REAL chart statistics:
  panel balance (L/D/U/R), repeat/jack rate, pattern entropy, plus onset_F1 / crit_adj.

Usage:
    python experiments/generation_typed/eval_style.py --data_dir data/ --audio_dir data/
"""

import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, sys
from pathlib import Path
import numpy as np, torch, yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.typed import NUM_PANELS, pair_holds
from src.generation.evaluation import onset_density_metrics, DifficultyCritic

DEFAULT_BPM = 150.0
PANELS = ['L', 'D', 'U', 'R']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint', default='checkpoints/gen_layered/best_val.pt')
    p.add_argument('--eval_songs', type=int, default=48); p.add_argument('--max_gen_len', type=int, default=768)
    return p.parse_args()


def typed_binary(t):
    t = np.asarray(t); return ((t == 1) | (t == 2) | (t == 4)).astype(np.float32)


def style_stats(chart):
    """panel fractions (L/D/U/R), repeat/jack rate, pattern entropy over note-frames."""
    arr = np.asarray(chart)
    active = (arr != 0)                                  # (T,4) which panels have a note
    note_frames = active.any(1)
    rows = active[note_frames]                           # (N,4) active-panel patterns per note
    if len(rows) == 0:
        return np.zeros(4), 0.0, 0.0
    panel_frac = rows.sum(0) / max(rows.sum(), 1)
    # repeat/jack rate: consecutive note-frames with identical active-panel set
    rep = np.mean([np.array_equal(rows[i], rows[i - 1]) for i in range(1, len(rows))]) if len(rows) > 1 else 0.0
    # pattern entropy over the 15 patterns
    idx = (rows * (1 << np.arange(4))).sum(1)
    _, counts = np.unique(idx, return_counts=True)
    p = counts / counts.sum()
    ent = float(-(p * np.log2(p)).sum())
    return panel_frac, float(rep), ent


def main():
    args = parse_args(); set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob(f"{args.data_dir}/**/*.sm", recursive=True) + glob.glob(f"{args.data_dir}/**/*.ssc", recursive=True)
    _, val_files, _ = create_data_splits(cf, random_state=args.seed)
    with open(PROJECT_ROOT / "config/model_config.yaml") as f:
        msl = yaml.safe_load(f)['classifier']['max_sequence_length']
    _, val_ds, _ = create_datasets(train_files=[], val_files=val_files, test_files=[], audio_dir=args.audio_dir,
                                   max_sequence_length=msl, cache_dir='cache/samples')
    val_ds.warm_cache(show_progress=True)
    val = []
    for i in range(len(val_ds)):
        if len(val) >= args.eval_songs: break
        s = val_ds[i]; meta = val_ds.valid_samples[i]; T = int(s['mask'].sum().item())
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        tf = val_ds.parser.convert_to_tensor_typed(meta['chart'], nd); T = min(T, args.max_gen_len, tf.shape[0])
        val.append({'typed': tf[:T].astype(np.int64), 'audio': s['audio'][:T].numpy().astype(np.float32),
                    'difficulty': int(meta['difficulty_class'])})
    audio_dim = val[0]['audio'].shape[1]
    target_density = float(np.mean([(s['typed'] != 0).any(1).mean() for s in val]))

    model = LayeredTypedChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model_state_dict']); model.eval()
    critic = DifficultyCritic(device=device)

    logits = []
    with torch.no_grad():
        for s in val:
            a = torch.from_numpy(s['audio']).unsqueeze(0).to(device); d = torch.tensor([s['difficulty']], device=device)
            logits.append(torch.sigmoid(model.onset_logits(model.encode_audio(a), d))[0].cpu().numpy())
    tau = float(np.quantile(np.concatenate(logits), 1 - target_density))

    # real-chart reference stats
    rp, rr, re = [], [], []
    for s in val:
        pf, rep, ent = style_stats(s['typed']); rp.append(pf); rr.append(rep); re.append(ent)
    rp = np.mean(rp, 0)

    strategies = {
        'greedy (current)': dict(),
        'p-sample t1.0': dict(pattern_sample=True, pattern_temperature=1.0),
        'p-sample t1.0 +rep3': dict(pattern_sample=True, pattern_temperature=1.0, repetition_penalty=3.0),
        'p-sample t1.2 k6 +rep4': dict(pattern_sample=True, pattern_temperature=1.2, pattern_top_k=6, repetition_penalty=4.0),
    }

    print(f"\nREAL: panels L/D/U/R={rp.round(2)} repeat={np.mean(rr):.2f} entropy={np.mean(re):.2f}")
    print("=" * 100)
    print(f"  {'strategy':<24} {'L':>5} {'D':>5} {'U':>5} {'R':>5} {'repeat':>7} {'entropy':>8} {'onset_F1':>9} {'crit_adj':>9}")
    print("-" * 100)
    for name, kw in strategies.items():
        set_seed(args.seed)
        pf_all, rep_all, ent_all, f1s, preds, tgts = [], [], [], [], [], []
        for i in range(0, len(val), 8):
            batch = val[i:i + 8]; L = min(args.max_gen_len, max(len(s['typed']) for s in batch)); B = len(batch)
            audio = torch.zeros(B, L, audio_dim); lengths = torch.zeros(B, dtype=torch.long); diff = torch.zeros(B, dtype=torch.long)
            for b, s in enumerate(batch):
                t = min(len(s['typed']), L); audio[b, :t] = torch.from_numpy(s['audio'][:t]); lengths[b] = t; diff[b] = s['difficulty']
            gen = model.generate(audio.to(device), diff.to(device), lengths=lengths.to(device),
                                 onset_threshold=tau, type_sample=True, type_temperature=0.4,
                                 hold_aware=True, **kw).cpu().numpy()
            for b, s in enumerate(batch):
                t = int(lengths[b]); g = pair_holds(gen[b, :t])
                pf, rep, ent = style_stats(g); pf_all.append(pf); rep_all.append(rep); ent_all.append(ent)
                m = onset_density_metrics((g != 0).astype(np.float32), reference=(s['typed'][:t] != 0).astype(np.float32))
                f1s.append(m['onset_f1'])
                preds.append(critic.predict(typed_binary(g), s['audio'][:t], bpm=DEFAULT_BPM)['class']); tgts.append(s['difficulty'])
        pf = np.mean(pf_all, 0); dd = np.abs(np.array(preds) - np.array(tgts))
        print(f"  {name:<24} {pf[0]:>5.2f} {pf[1]:>5.2f} {pf[2]:>5.2f} {pf[3]:>5.2f} "
              f"{np.mean(rep_all):>7.2f} {np.mean(ent_all):>8.2f} {np.mean(f1s):>9.3f} {np.mean(dd<=1):>9.3f}")
    print("=" * 100)
    print("  repeat = consecutive notes with identical panel set (jacks); higher entropy = more varied.")


if __name__ == '__main__':
    main()

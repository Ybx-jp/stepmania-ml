#!/usr/bin/env python3
"""
Demonstrate decode-time pattern-preference + no-crossover controls on real val audio.

Shows graded control over jump rate (pattern_bias jump), crossovers (no_crossovers),
and panel balance (pattern_bias panel_prefs) -- all decode-time, no retraining -- while
onset_F1 and the difficulty critic stay intact.
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
from src.generation.typed import pair_holds, make_pattern_bias, count_crossovers
from src.generation.evaluation import onset_density_metrics, DifficultyCritic

DEFAULT_BPM = 150.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint', default='checkpoints/gen_layered/best_val.pt')
    p.add_argument('--eval_songs', type=int, default=32); p.add_argument('--max_gen_len', type=int, default=768)
    return p.parse_args()


def typed_binary(t):
    t = np.asarray(t); return ((t == 1) | (t == 2) | (t == 4)).astype(np.float32)


def chart_stats(g):
    act = (g != 0); notes = act[act.any(1)]
    jump = float((notes.sum(1) >= 2).mean()) if len(notes) else 0.0
    cr, ct = count_crossovers(g)
    pf = notes.mean(0) if len(notes) else np.zeros(4)
    return jump, (cr / max(ct, 1)), pf


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

    rj, rx = [], []
    for s in val:
        j, x, _ = chart_stats(s['typed']); rj.append(j); rx.append(x)

    base = dict(pattern_sample=True, pattern_temperature=1.0, type_sample=True, type_temperature=0.4, hold_aware=True)
    settings = {
        'baseline': dict(),
        'jump +1.5': dict(pattern_bias=make_pattern_bias(jump=1.5)),
        'jump -1.5': dict(pattern_bias=make_pattern_bias(jump=-1.5)),
        'no_crossovers': dict(no_crossovers=True),
        'prefer R': dict(pattern_bias=make_pattern_bias(panel_prefs=[0, 0, 0, 1.5])),
    }

    print(f"\nREAL: jump_rate={np.mean(rj):.2f} crossover_rate={np.mean(rx):.2f}")
    print("=" * 92)
    print(f"  {'setting':<16} {'jump':>6} {'crossover':>10} {'L':>5} {'D':>5} {'U':>5} {'R':>5} {'onset_F1':>9} {'crit_adj':>9}")
    print("-" * 92)
    for name, extra in settings.items():
        set_seed(args.seed)
        js, xs, pfs, f1s, preds, tgts = [], [], [], [], [], []
        for i in range(0, len(val), 8):
            batch = val[i:i + 8]; L = min(args.max_gen_len, max(len(s['typed']) for s in batch)); B = len(batch)
            audio = torch.zeros(B, L, audio_dim); lengths = torch.zeros(B, dtype=torch.long); diff = torch.zeros(B, dtype=torch.long)
            for b, s in enumerate(batch):
                t = min(len(s['typed']), L); audio[b, :t] = torch.from_numpy(s['audio'][:t]); lengths[b] = t; diff[b] = s['difficulty']
            gen = model.generate(audio.to(device), diff.to(device), lengths=lengths.to(device),
                                 onset_threshold=tau, **base, **extra).cpu().numpy()
            for b, s in enumerate(batch):
                t = int(lengths[b]); g = pair_holds(gen[b, :t])
                j, x, pf = chart_stats(g); js.append(j); xs.append(x); pfs.append(pf)
                m = onset_density_metrics((g != 0).astype(np.float32), reference=(s['typed'][:t] != 0).astype(np.float32))
                f1s.append(m['onset_f1'])
                preds.append(critic.predict(typed_binary(g), s['audio'][:t], bpm=DEFAULT_BPM)['class']); tgts.append(s['difficulty'])
        pf = np.mean(pfs, 0); dd = np.abs(np.array(preds) - np.array(tgts))
        print(f"  {name:<16} {np.mean(js):>6.2f} {np.mean(xs):>10.2f} {pf[0]:>5.2f} {pf[1]:>5.2f} {pf[2]:>5.2f} {pf[3]:>5.2f} "
              f"{np.mean(f1s):>9.3f} {np.mean(dd<=1):>9.3f}")
    print("=" * 92)


if __name__ == '__main__':
    main()

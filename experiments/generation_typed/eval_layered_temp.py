#!/usr/bin/env python3
"""
Tune the type-sampling temperature on the trained layered checkpoint (no retraining).

no-weight + type-sampling gives crit_adj 0.844 but over-generates holds (tap:hold 3.5:1
vs real ~19:1) because sampling spreads hold probability across many active panels. Lower
type_temperature sharpens the type distribution -> fewer (more confident) holds. Sweep it.
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
from src.generation.typed import NUM_PANELS, SYMBOL_NAMES, symbol_histogram, pair_holds
from src.generation.evaluation import onset_density_metrics, DifficultyCritic

DEFAULT_BPM = 150.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint', default='checkpoints/gen_layered/best_val.pt')
    p.add_argument('--eval_songs', type=int, default=64); p.add_argument('--max_gen_len', type=int, default=768)
    p.add_argument('--temps', type=str, default='1.0,0.7,0.5,0.35')
    return p.parse_args()


def typed_binary(t):
    t = np.asarray(t); return ((t == 1) | (t == 2) | (t == 4)).astype(np.float32)


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
    real_tap = sum(symbol_histogram(s['typed'])['tap'] for s in val)
    real_hold = sum(symbol_histogram(s['typed'])['hold_head'] for s in val)

    print(f"\nreal tap:hold = {real_tap/max(real_hold,1):.1f}:1   tau={tau:.3f}")
    print("=" * 72)
    print(f"  {'type_temp':<10} {'onset_F1':>9} {'density':>8} {'tap:hold':>9} {'crit_adj':>9} {'crit_mae':>9}")
    print("-" * 72)
    for temp in [float(x) for x in args.temps.split(',')]:
        set_seed(args.seed)
        f1s, dens, syms, preds, tgts = [], [], {k: 0 for k in SYMBOL_NAMES}, [], []
        for i in range(0, len(val), 8):
            batch = val[i:i + 8]; L = min(args.max_gen_len, max(len(s['typed']) for s in batch)); B = len(batch)
            audio = torch.zeros(B, L, audio_dim); lengths = torch.zeros(B, dtype=torch.long); diff = torch.zeros(B, dtype=torch.long)
            for b, s in enumerate(batch):
                t = min(len(s['typed']), L); audio[b, :t] = torch.from_numpy(s['audio'][:t]); lengths[b] = t; diff[b] = s['difficulty']
            gen = model.generate(audio.to(device), diff.to(device), lengths=lengths.to(device),
                                 onset_threshold=tau, greedy=True, type_sample=True, type_temperature=temp).cpu().numpy()
            for b, s in enumerate(batch):
                t = int(lengths[b]); g = pair_holds(gen[b, :t])
                m = onset_density_metrics((g != 0).astype(np.float32), reference=(s['typed'][:t] != 0).astype(np.float32))
                f1s.append(m['onset_f1']); dens.append((g != 0).any(1).mean())
                for k, v in symbol_histogram(g).items(): syms[k] += v
                preds.append(critic.predict(typed_binary(g), s['audio'][:t], bpm=DEFAULT_BPM)['class']); tgts.append(s['difficulty'])
        dd = np.abs(np.array(preds) - np.array(tgts))
        ratio = syms['tap'] / max(syms['hold_head'], 1)
        print(f"  {temp:<10} {np.mean(f1s):>9.3f} {np.mean(dens):>8.3f} {ratio:>8.1f}:1 {np.mean(dd<=1):>9.3f} {np.mean(dd):>9.3f}")
    print("=" * 72)
    print("  lower type_temp -> sharper type dist -> fewer holds (toward real ratio).")


if __name__ == '__main__':
    main()

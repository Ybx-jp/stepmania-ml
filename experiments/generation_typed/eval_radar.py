#!/usr/bin/env python3
"""
Radar-conditioning controllability (Step 2). Does the trained model OBEY a target
groove-radar profile? For each radar dim, generate with it set low vs high (others at
the dataset mean) over fixed val audio, and measure the interpretable proxy:
  stream/voltage -> onset density,  air -> jump rate,  freeze -> hold rate.
A working knob shows the proxy rising from low->high on its own dim.

Onsets are Bernoulli-sampled (no fixed threshold) so density is free to respond.

Usage:
    python experiments/generation_typed/eval_radar.py --data_dir data/ --audio_dir data/
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
from src.generation.typed import NUM_PANELS, pair_holds, symbol_histogram

RADAR = ['stream', 'voltage', 'air', 'freeze', 'chaos']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint', default='checkpoints/gen_radar/best_val.pt')
    p.add_argument('--eval_songs', type=int, default=24); p.add_argument('--max_gen_len', type=int, default=512)
    p.add_argument('--low', type=float, default=0.1); p.add_argument('--high', type=float, default=0.9)
    return p.parse_args()


def proxies(g):
    act = (g != 0); notes = act[act.any(1)]
    density = float(act.any(1).mean())
    jump = float((notes.sum(1) >= 2).mean()) if len(notes) else 0.0
    h = symbol_histogram(g); hold_rate = h['hold_head'] / max((g != 0).sum(), 1)
    return density, jump, hold_rate


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
    val, radars = [], []
    for i in range(len(val_ds)):
        if len(val) >= args.eval_songs: break
        s = val_ds[i]; meta = val_ds.valid_samples[i]; T = min(int(s['mask'].sum().item()), args.max_gen_len)
        val.append({'audio': s['audio'][:T].numpy().astype(np.float32), 'difficulty': int(meta['difficulty_class'])})
        radars.append(meta['groove_radar'].to_vector().astype(np.float32))
    audio_dim = val[0]['audio'].shape[1]
    mean_radar = np.mean(radars, 0)

    model = LayeredTypedChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model_state_dict']); model.eval()

    def run(radar_vec):
        ds, js, hs = [], [], []
        for i in range(0, len(val), 8):
            batch = val[i:i + 8]; L = min(args.max_gen_len, max(len(s['audio']) for s in batch)); B = len(batch)
            audio = torch.zeros(B, L, audio_dim); lengths = torch.zeros(B, dtype=torch.long); diff = torch.zeros(B, dtype=torch.long)
            for b, s in enumerate(batch):
                t = len(s['audio']); audio[b, :t] = torch.from_numpy(s['audio']); lengths[b] = t; diff[b] = s['difficulty']
            radar = torch.tensor(radar_vec, device=device).unsqueeze(0).expand(B, -1)
            gen = model.generate(audio.to(device), diff.to(device), lengths=lengths.to(device),
                                 onset_sample=True, type_sample=True, type_temperature=0.4, hold_aware=True,
                                 pattern_sample=True, pattern_temperature=1.0, radar=radar).cpu().numpy()
            for b in range(B):
                d, j, h = proxies(pair_holds(gen[b, :int(lengths[b])])); ds.append(d); js.append(j); hs.append(h)
        return np.mean(ds), np.mean(js), np.mean(hs)

    print(f"\nmean radar={mean_radar.round(2)}; vary each dim low={args.low} -> high={args.high} (others at mean)")
    print("=" * 84)
    print(f"  {'dim varied':<12} {'density(lo->hi)':>22} {'jump(lo->hi)':>20} {'hold(lo->hi)':>20}")
    print("-" * 84)
    for d in range(5):
        lo = mean_radar.copy(); lo[d] = args.low
        hi = mean_radar.copy(); hi[d] = args.high
        set_seed(args.seed); dl, jl, hl = run(lo)
        set_seed(args.seed); dh, jh, hh = run(hi)
        print(f"  {RADAR[d]:<12} {dl:>9.3f} -> {dh:<9.3f} {jl:>8.3f} -> {jh:<8.3f} {hl:>8.3f} -> {hh:<8.3f}")
    print("=" * 84)
    print("  expect: stream/voltage move density; air moves jump; freeze moves hold.")


if __name__ == '__main__':
    main()

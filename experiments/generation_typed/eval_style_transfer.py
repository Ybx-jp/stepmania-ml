#!/usr/bin/env python3
"""
Reference-chart style-transfer controllability (Step 3). Does the model OBEY a reference
chart's feel when generating for *different* audio? We pick two reference archetypes from
the val set -- a SPARSE chart and a DENSE chart -- then over a fixed set of (other-song)
audio, generate conditioned on each reference and measure the proxies:
  density   (sparse-ref should yield fewer notes than dense-ref)
  jump rate (a jump-heavy reference should pull jumps up).
A working knob shows the proxy tracking the reference, and the gap widening with guidance.

Usage:
    python experiments/generation_typed/eval_style_transfer.py --data_dir data/ --audio_dir data/
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint', default='checkpoints/gen_style/best_val.pt')
    p.add_argument('--eval_songs', type=int, default=24); p.add_argument('--max_gen_len', type=int, default=512)
    p.add_argument('--guidance', type=float, nargs='+', default=[1.0, 2.0, 3.0],
                   help='classifier-free guidance scales to sweep (>1 amplifies style)')
    return p.parse_args()


def proxies(g):
    act = (g != 0); notes = act[act.any(1)]
    density = float(act.any(1).mean())
    jump = float((notes.sum(1) >= 2).mean()) if len(notes) else 0.0
    return density, jump


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

    songs = []
    for i in range(len(val_ds)):
        s = val_ds[i]; meta = val_ds.valid_samples[i]; T = min(int(s['mask'].sum().item()), args.max_gen_len)
        nd = next((n for n in meta['chart'].note_data
                   if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None:
            continue
        typed_full = val_ds.parser.convert_to_tensor_typed(meta['chart'], nd)
        T = min(T, typed_full.shape[0])
        songs.append({'audio': s['audio'][:T].numpy().astype(np.float32),
                      'difficulty': int(meta['difficulty_class']),
                      'typed': typed_full[:T].astype(np.int64)})
    audio_dim = songs[0]['audio'].shape[1]

    # pick reference archetypes by density (jumps are a secondary readout)
    dens = np.array([(s['typed'] != 0).any(1).mean() for s in songs])
    sparse_ref = songs[int(dens.argmin())]['typed']
    dense_ref = songs[int(dens.argmax())]['typed']
    print(f"sparse reference density={dens.min():.3f}   dense reference density={dens.max():.3f}")

    audio_pool = songs[:args.eval_songs]

    model = LayeredTypedChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model_state_dict']); model.eval()

    def run(ref_typed, guidance):
        rt = torch.from_numpy(ref_typed).to(device)
        ds, js = [], []
        for i in range(0, len(audio_pool), 8):
            batch = audio_pool[i:i + 8]; L = min(args.max_gen_len, max(len(s['audio']) for s in batch)); B = len(batch)
            audio = torch.zeros(B, L, audio_dim); lengths = torch.zeros(B, dtype=torch.long); diff = torch.zeros(B, dtype=torch.long)
            for b, s in enumerate(batch):
                t = len(s['audio']); audio[b, :t] = torch.from_numpy(s['audio']); lengths[b] = t; diff[b] = s['difficulty']
            # broadcast the single reference chart across the batch (clip/pad to L)
            rl = min(len(ref_typed), L)
            ref = torch.zeros(B, L, NUM_PANELS, dtype=torch.long, device=device)
            ref_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
            ref[:, :rl] = rt[:rl]; ref_mask[:, :rl] = True
            gen = model.generate(audio.to(device), diff.to(device), lengths=lengths.to(device),
                                 onset_sample=True, type_sample=True, type_temperature=0.4, hold_aware=True,
                                 pattern_sample=True, pattern_temperature=1.0,
                                 reference=ref, reference_mask=ref_mask, guidance_scale=guidance).cpu().numpy()
            for b in range(B):
                d, j = proxies(pair_holds(gen[b, :int(lengths[b])])); ds.append(d); js.append(j)
        return np.mean(ds), np.mean(js)

    print("\nstyle transfer: same audio, conditioned on a SPARSE vs DENSE reference chart")
    print("=" * 72)
    print(f"  {'guidance':<10} {'density sparse->dense':>26} {'jump sparse->dense':>24}")
    print("-" * 72)
    for g in args.guidance:
        set_seed(args.seed); ds_s, js_s = run(sparse_ref, g)
        set_seed(args.seed); ds_d, js_d = run(dense_ref, g)
        print(f"  g={g:<8.1f} {ds_s:>11.3f} -> {ds_d:<11.3f} {js_s:>10.3f} -> {js_d:<10.3f}")
    print("=" * 72)
    print("  expect: dense-ref density > sparse-ref density, gap widening with guidance.")


if __name__ == '__main__':
    main()

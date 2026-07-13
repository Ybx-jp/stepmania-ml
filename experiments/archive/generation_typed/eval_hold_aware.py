#!/usr/bin/env python3
"""
Compare stateless vs hold-state-aware decoding on the trained layered checkpoint
(no retraining). Hold-aware runs a per-panel automaton: a head opens a hold, the
panel is occupied, and the hold closes (tail) at the next note the model places on
that panel -> coherent, audio-aligned spans, no orphans.

Reports orphan rate, hold-length distribution (gen vs real), tap:hold, onset_F1,
density, crit_adj for both decoders.
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
    p.add_argument('--type_temperature', type=float, default=0.4)
    return p.parse_args()


def typed_binary(t):
    t = np.asarray(t); return ((t == 1) | (t == 2) | (t == 4)).astype(np.float32)


def hold_stats(chart):
    """Per-panel: count raw orphan heads/tails and the lengths (frames) of valid holds."""
    arr = np.asarray(chart); orphans = 0; lengths = []
    for p in range(NUM_PANELS):
        col = arr[:, p]; open_t = -1
        for t in range(len(col)):
            s = col[t]
            if s in (2, 4):
                if open_t >= 0: orphans += 1
                open_t = t
            elif s == 3:
                if open_t >= 0: lengths.append(t - open_t); open_t = -1
                else: orphans += 1
        if open_t >= 0: orphans += 1
    return orphans, lengths


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

    # real hold-length distribution for reference
    real_lengths = []
    for s in val:
        _, L = hold_stats(s['typed']); real_lengths += L
    rl = np.array(real_lengths) if real_lengths else np.array([0])

    logits = []
    with torch.no_grad():
        for s in val:
            a = torch.from_numpy(s['audio']).unsqueeze(0).to(device); d = torch.tensor([s['difficulty']], device=device)
            logits.append(torch.sigmoid(model.onset_logits(model.encode_audio(a), d))[0].cpu().numpy())
    tau = float(np.quantile(np.concatenate(logits), 1 - target_density))

    print(f"\nreal holds: n={len(real_lengths)} mean_len={rl.mean():.1f} median={np.median(rl):.0f} frames; tau={tau:.3f}")
    print("=" * 96)
    print(f"  {'decoder':<14} {'onset_F1':>9} {'density':>8} {'tap:hold':>9} {'orphan%':>8} "
          f"{'holds':>7} {'mean_len':>9} {'median':>7} {'crit_adj':>9}")
    print("-" * 96)
    for ha in [False, True]:
        set_seed(args.seed)
        f1s, dens, syms, preds, tgts = [], [], {k: 0 for k in SYMBOL_NAMES}, [], []
        tot_orphan = tot_head = 0; gen_lengths = []
        for i in range(0, len(val), 8):
            batch = val[i:i + 8]; L = min(args.max_gen_len, max(len(s['typed']) for s in batch)); B = len(batch)
            audio = torch.zeros(B, L, audio_dim); lengths = torch.zeros(B, dtype=torch.long); diff = torch.zeros(B, dtype=torch.long)
            for b, s in enumerate(batch):
                t = min(len(s['typed']), L); audio[b, :t] = torch.from_numpy(s['audio'][:t]); lengths[b] = t; diff[b] = s['difficulty']
            gen = model.generate(audio.to(device), diff.to(device), lengths=lengths.to(device),
                                 onset_threshold=tau, greedy=True, type_sample=True,
                                 type_temperature=args.type_temperature, hold_aware=ha).cpu().numpy()
            for b, s in enumerate(batch):
                t = int(lengths[b]); g_raw = gen[b, :t]
                o, gl = hold_stats(g_raw); tot_orphan += o; tot_head += int(((g_raw == 2) | (g_raw == 4)).sum()); gen_lengths += gl
                g = pair_holds(g_raw)
                m = onset_density_metrics((g != 0).astype(np.float32), reference=(s['typed'][:t] != 0).astype(np.float32))
                f1s.append(m['onset_f1']); dens.append((g != 0).any(1).mean())
                for k, v in symbol_histogram(g).items(): syms[k] += v
                preds.append(critic.predict(typed_binary(g), s['audio'][:t], bpm=DEFAULT_BPM)['class']); tgts.append(s['difficulty'])
        dd = np.abs(np.array(preds) - np.array(tgts))
        gl = np.array(gen_lengths) if gen_lengths else np.array([0])
        ratio = syms['tap'] / max(syms['hold_head'], 1)
        orphan_pct = 100 * tot_orphan / max(tot_head, 1)
        print(f"  {'hold-aware' if ha else 'stateless':<14} {np.mean(f1s):>9.3f} {np.mean(dens):>8.3f} {ratio:>8.1f}:1 "
              f"{orphan_pct:>7.0f}% {len(gen_lengths):>7} {gl.mean():>9.1f} {np.median(gl):>7.0f} {np.mean(dd<=1):>9.3f}")
    print("=" * 96)
    print("  orphan% = raw orphan heads/tails before pair_holds; lengths in 16th-note frames.")


if __name__ == '__main__':
    main()

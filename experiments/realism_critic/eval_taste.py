#!/usr/bin/env python3
"""
Stage 2a validation — does the realism critic's P(real) work as a TASTE METRIC?

For N val songs, score P(real) (critic) of:
  - REAL human chart            (the ceiling)
  - BASE generation             (gen_stage1, density-matched, recommended decode)
  - CHAOS generation            (gen_stage1, radar chaos=0.9, guidance 2.0)
Expect REAL > BASE > CHAOS — matching the playtest (user judged base "more musical", chaos "no taste").
If the ranking holds, we finally have a number that tracks musical play-feel.

Usage:
    python experiments/realism_critic/eval_taste.py --data_dir data/ --audio_dir data/
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
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
from src.models import LateFusionClassifier
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.typed import pair_holds

GEN_CKPT = "checkpoints/gen_stage1/best_val.pt"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--critic', default='checkpoints/realism_critic/best_val.pt')
    p.add_argument('--eval_songs', type=int, default=64); p.add_argument('--max_len', type=int, default=768)
    p.add_argument('--cache_dir', default='cache/samples_v2')
    return p.parse_args()


def to_binary(t):
    t = np.asarray(t); return ((t == 1) | (t == 2) | (t == 4)).astype(np.float32)


@torch.no_grad()
def score(critic, audio23, chart, device):
    a = torch.from_numpy(audio23).unsqueeze(0).to(device); c = torch.from_numpy(chart).unsqueeze(0).to(device)
    m = torch.ones(1, a.shape[1], device=device)
    logits = critic(a, c, m)
    if isinstance(logits, dict): logits = logits['logits']
    return float(torch.softmax(logits, 1)[0, 1])


def main():
    args = parse_args(); set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob(f"{args.data_dir}/**/*.sm", recursive=True) + glob.glob(f"{args.data_dir}/**/*.ssc", recursive=True)
    _, val_files, _ = create_data_splits(cf, random_state=args.seed)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True, use_metric_phase=True))
    _, val_ds, _ = create_datasets(train_files=[], val_files=val_files, test_files=[], audio_dir=args.audio_dir,
                                   max_sequence_length=msl, feature_extractor=ext, cache_dir=args.cache_dir)
    val_ds.warm_cache(show_progress=False)

    songs = []
    for i in range(len(val_ds)):
        if len(songs) >= args.eval_songs: break
        s = val_ds[i]; meta = val_ds.valid_samples[i]; T = min(int(s['mask'].sum().item()), args.max_len)
        if T < 64: continue
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        tf = val_ds.parser.convert_to_tensor_typed(meta['chart'], nd)[:T]
        songs.append({'audio41': s['audio'][:T].numpy().astype(np.float32), 'real': to_binary(tf),
                      'difficulty': int(meta['difficulty_class']), 'T': T})
    print(f"eval songs={len(songs)}")

    ck = torch.load(args.critic, map_location=device, weights_only=False)
    critic = LateFusionClassifier(ck['config']).to(device); critic.load_state_dict(ck['model_state_dict']); critic.eval()
    gen = LayeredTypedChartGenerator(audio_dim=41, d_model=128, num_layers=4, onset_layers=2).to(device)
    gen.load_state_dict(torch.load(GEN_CKPT, map_location=device)['model_state_dict']); gen.eval()
    mean_radar = np.mean([m['groove_radar'].to_vector() for m in val_ds.valid_samples if 'groove_radar' in m], 0).astype(np.float32)
    chaos_radar = mean_radar.copy(); chaos_radar[4] = 0.9

    def generate(s, radar=None, guidance=1.0):
        a = torch.from_numpy(s['audio41']).unsqueeze(0).to(device); d = torch.tensor([s['difficulty']], device=device)
        with torch.no_grad():
            p = torch.sigmoid(gen.onset_logits(gen.encode_audio(a), d))[0].cpu().numpy()
            tau = float(np.quantile(p, 1 - max((s['real'] != 0).any(1).mean(), 1e-3)))
            rad = None if radar is None else torch.tensor(radar, device=device).unsqueeze(0)
            g = gen.generate(a, d, lengths=torch.tensor([s['T']], device=device), onset_threshold=tau,
                             type_sample=True, type_temperature=0.4, hold_aware=True, pattern_sample=True,
                             pattern_temperature=0.7, no_jump_during_hold=True, radar=rad, guidance_scale=guidance)[0].cpu().numpy()
        return to_binary(pair_holds(g[:s['T']]))

    real_s, base_s, chaos_s = [], [], []
    for s in songs:
        a23 = s['audio41'][:, :23]
        real_s.append(score(critic, a23, s['real'], device))
        base_s.append(score(critic, a23, generate(s), device))
        chaos_s.append(score(critic, a23, generate(s, radar=chaos_radar, guidance=2.0), device))
    print("\n" + "=" * 56)
    print("  TASTE METRIC — critic P(real), mean over songs")
    print("=" * 56)
    print(f"  REAL human chart : {np.mean(real_s):.3f}")
    print(f"  BASE generation  : {np.mean(base_s):.3f}")
    print(f"  CHAOS generation : {np.mean(chaos_s):.3f}")
    print("=" * 56)
    print(f"  expect REAL > BASE > CHAOS (playtest ranking). "
          f"base>chaos: {'YES' if np.mean(base_s) > np.mean(chaos_s) else 'NO'}")


if __name__ == '__main__':
    main()

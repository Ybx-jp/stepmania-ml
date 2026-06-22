#!/usr/bin/env python3
"""
Which conditioning best matches the SOURCE chart's groove? Compares 3 ways to steer generation toward the
original's profile, on the same songs:
  baseline       : no groove conditioning (the drift the user noticed)
  match_radar    : condition on the source chart's 5-dim groove RADAR (guidance 1.5)
  reference_self : condition on the source CHART via the StyleEncoder latent (guidance 2.0)
For each generated chart we compute its full groove radar and report L1 distance to the source radar
(lower = matches better), per approach and per dim. Then play the exported sets to judge feel.
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, sys
from pathlib import Path
import numpy as np, torch, yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits
from src.data.dataset import StepManiaDataset
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
from src.data.groove_radar import GrooveRadarCalculator
from src.data.song_selection import select_by_groove, RADAR_DIMS
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.typed import pair_holds

GEN_CKPT = "checkpoints/gen_stage1/best_val.pt"
TPB = 4  # timesteps per beat (16th grid)


def hold_info_from_typed(typed):
    T = typed.shape[0]; holds = []
    for p in range(4):
        col = typed[:, p]; t = 0
        while t < T:
            if col[t] in (2, 4):
                tt = t + 1
                while tt < T and col[tt] != 3:
                    tt += 1
                holds.append((p, t / TPB, min(tt, T) / TPB)); t = tt + 1
            else:
                t += 1
    note_beats = []  # (beat_position, panel, note_type) -- the format the chaos calc expects
    for t in range(T):
        for p in range(4):
            s = typed[t, p]
            if s in (1, 4):
                note_beats.append((t / TPB, p, 'tap'))
            elif s == 2:
                note_beats.append((t / TPB, p, 'hold_start'))
    return {'holds': holds, 'total_hold_beats': sum(e - s for _, s, e in holds),
            'note_beats': note_beats, 'song_length_beats': T / TPB}


def radar_of(typed, calc, bpm, song_len, timing):
    typed = np.asarray(typed)
    bin = (typed != 0).astype(np.int64)
    r = calc.calculate(chart_tensor=bin, hold_info=hold_info_from_typed(typed),
                       timing_events=timing, song_length_seconds=song_len, avg_bpm=bpm)
    return r.to_vector()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_songs', type=int, default=6); ap.add_argument('--max_len', type=int, default=1024)
    ap.add_argument('--select', default='freeze'); ap.add_argument('--difficulty', default='Hard')
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    _, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True, use_metric_phase=True))
    ds = StepManiaDataset(chart_files=vf[:args.num_songs * 40], audio_dir="data/", max_sequence_length=msl,
                          feature_extractor=ext, cache_dir='cache/samples_v2')
    gen = LayeredTypedChartGenerator(audio_dim=41, d_model=128, num_layers=4, onset_layers=2).to(device)
    gen.load_state_dict(torch.load(GEN_CKPT, map_location=device)['model_state_dict']); gen.eval()
    calc = GrooveRadarCalculator()
    dcls = ['Beginner', 'Easy', 'Medium', 'Hard'].index(args.difficulty)
    order = select_by_groove(ds, n=args.num_songs, by=args.select, difficulty=dcls)

    decode = dict(type_sample=True, type_temperature=0.4, hold_aware=True, pattern_sample=True,
                  pattern_temperature=0.7, no_jump_during_hold=True, no_cross_during_hold=True)
    approaches = ['baseline', 'match_radar', 'reference_self']
    dist = {a: [] for a in approaches}
    for i in order:
        meta = ds.valid_samples[i]; sample = ds[i]
        T = min(int(sample['mask'].sum().item()), args.max_len)
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        orig = np.asarray(ds.parser.convert_to_tensor_typed(meta['chart'], nd))[:T]
        src_radar = meta['groove_radar'].to_vector()
        bpm = float(meta['chart'].bpm); slen = float(meta['chart'].song_length_seconds); timing = meta['chart'].timing_events
        audio = sample['audio'][:T].unsqueeze(0).to(device); diff = torch.tensor([meta['difficulty_class']], device=device)
        with torch.no_grad():
            p_on = torch.sigmoid(gen.onset_logits(gen.encode_audio(audio), diff))[0].cpu().numpy()
            tau = float(np.quantile(p_on, 1 - float((orig != 0).any(1).mean())))
            sr = torch.from_numpy(src_radar.astype(np.float32)).unsqueeze(0).to(device)
            st = gen.encode_style(torch.from_numpy(orig).long().unsqueeze(0).to(device),
                                  torch.ones(1, T, device=device))
            cfg = {'baseline': dict(radar=None, style=None, guidance_scale=1.0),
                   'match_radar': dict(radar=sr, style=None, guidance_scale=1.5),
                   'reference_self': dict(radar=None, style=st, guidance_scale=2.0)}
            for a in approaches:
                set_seed(42)
                g = gen.generate(audio, diff, lengths=torch.tensor([T], device=device), onset_threshold=tau,
                                 **decode, **cfg[a])[0].cpu().numpy()
                gr = radar_of(pair_holds(g), calc, bpm, slen, timing)
                dist[a].append(np.abs(gr - src_radar))

    print(f"\n=== Conditioning match to source groove ({len(dist['baseline'])} {args.select}-selected {args.difficulty} songs) ===")
    print("mean |generated_radar - source_radar| per dim (lower = matches source better)\n")
    hdr = f"{'approach':<16} " + " ".join(f"{d[:4]:>6}" for d in RADAR_DIMS) + f" {'TOTAL':>7}"
    print(hdr); print("-" * len(hdr))
    for a in approaches:
        m = np.mean(dist[a], 0)
        print(f"{a:<16} " + " ".join(f"{v:>6.2f}" for v in m) + f" {m.sum():>7.2f}")
    print("-" * len(hdr))
    print("Lower TOTAL = best profile match. radar=5-dim summary; reference_self=full-chart style latent.")


if __name__ == '__main__':
    main()

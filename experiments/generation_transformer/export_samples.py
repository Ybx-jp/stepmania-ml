#!/usr/bin/env python3
"""
Stage 3: export generated charts as playable .sm song folders.

Loads the trained Stage 2 checkpoint, generates a chart for each of N val
songs (conditioned on the real audio + a difficulty), and writes a StepMania
song folder per song containing the original audio plus a .sm with TWO charts:
the generated one and the original (for in-game A/B comparison).

Usage:
    python experiments/generation_transformer/export_samples.py \
        --data_dir data/ --audio_dir data/ --out_dir outputs/samples \
        --num_songs 8 --temperature 0.8 --top_k 3
"""

import warnings, os
warnings.filterwarnings('ignore')
os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'

import argparse, glob, re, shutil, sys
from pathlib import Path
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits
from src.data.dataset import StepManiaDataset, DIFFICULTY_NAMES
from src.generation.transformer import ChartGenerator
from src.generation.sm_writer import charts_to_sm
from src.generation.evaluation import onset_density_metrics, DifficultyCritic

DEFAULT_BPM = 150.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--audio_dir', required=True)
    p.add_argument('--out_dir', default='outputs/samples')
    p.add_argument('--checkpoint', default='checkpoints/gen_transformer/best_val_ce.pt')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_songs', type=int, default=8)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top_k', type=int, default=3)
    p.add_argument('--greedy', action='store_true')
    p.add_argument('--max_len', type=int, default=1024)
    p.add_argument('--num_layers', type=int, default=4)
    p.add_argument('--copy_audio', action='store_true', default=True)
    return p.parse_args()


def safe_name(s: str) -> str:
    s = (s or "untitled").strip()
    s = re.sub(r'[^\w\- ]+', '', s, flags=re.UNICODE).strip()
    return (s or "untitled")[:60]


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    chart_files = glob.glob(f"{args.data_dir}/**/*.sm", recursive=True)
    chart_files += glob.glob(f"{args.data_dir}/**/*.ssc", recursive=True)
    _, val_files, _ = create_data_splits(chart_files, random_state=args.seed)
    with open(PROJECT_ROOT / "config/model_config.yaml") as f:
        max_seq_len = yaml.safe_load(f)['classifier']['max_sequence_length']

    # Build a dataset over a slice of val files (enough to yield num_songs samples).
    ds = StepManiaDataset(chart_files=val_files[:args.num_songs * 8],
                          audio_dir=args.audio_dir, max_sequence_length=max_seq_len,
                          cache_dir='cache/samples')

    model = ChartGenerator(audio_dim=23, d_model=128, num_layers=args.num_layers).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model_state_dict'])
    model.eval()
    critic = DifficultyCritic(device=device)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    gen_kw = dict(greedy=True) if args.greedy else dict(
        greedy=False, temperature=args.temperature, top_k=args.top_k)

    print(f"\nDecoding: {'greedy' if args.greedy else gen_kw}")
    print(f"{'song':<34} {'diff':<9} {'onset_F1':>8} {'gen_dens':>8} {'ref_dens':>8} {'critic':>10}")
    print("-" * 84)

    exported = 0
    seen_files = set()
    for i in range(len(ds.valid_samples)):
        if exported >= args.num_songs:
            break
        meta = ds.valid_samples[i]
        if meta['chart_file'] in seen_files:  # one chart per song
            continue

        loaded = ds._load_chart_and_audio(meta)
        if loaded is None:
            continue
        chart_tensor, audio_tensor, T = loaded
        T = min(T, args.max_len)
        chart_tensor = np.asarray(chart_tensor)[:T]
        audio_t = torch.from_numpy(np.asarray(audio_tensor)[:T]).float().unsqueeze(0).to(device)
        diff_idx = meta['difficulty_class']
        diff_t = torch.tensor([diff_idx], device=device)

        gen = model.generate(audio_t, diff_t,
                             lengths=torch.tensor([T], device=device), **gen_kw)[0].cpu().numpy()

        chart_obj = meta['chart']
        bpm = float(chart_obj.bpm)
        audio_path = meta['audio_file']
        music_name = os.path.basename(audio_path)
        title = (chart_obj.title or Path(meta['chart_file']).stem)
        dname = DIFFICULTY_NAMES[diff_idx]  # canonical name (avoids parser colon quirk)
        dval = meta['difficulty_value']

        # Build a song folder with the audio + a .sm holding generated + original.
        folder = out_root / f"{exported:02d}_{safe_name(title)}"
        folder.mkdir(parents=True, exist_ok=True)
        if args.copy_audio and os.path.exists(audio_path):
            try:
                shutil.copy2(audio_path, folder / music_name)
            except Exception:
                pass

        sm = charts_to_sm(
            charts=[
                {"chart": gen, "difficulty_name": "Challenge",
                 "difficulty_value": dval, "author": "generated"},
                {"chart": chart_tensor, "difficulty_name": dname,
                 "difficulty_value": dval, "author": "original"},
            ],
            bpm=bpm, title=f"{title} (gen)", artist=chart_obj.artist or "",
            music=music_name, offset=float(chart_obj.offset),
        )
        (folder / "chart.sm").write_text(sm, encoding="utf-8")

        m = onset_density_metrics(gen, reference=chart_tensor)
        cpred = critic.predict(gen, np.asarray(audio_tensor)[:T], bpm=DEFAULT_BPM)
        print(f"{safe_name(title)[:33]:<34} {['Beg','Easy','Med','Hard'][diff_idx]:<9} "
              f"{m['onset_f1']:>8.3f} {m['gen_density']:>8.3f} {m['ref_density']:>8.3f} "
              f"{cpred['name']:>10}")
        seen_files.add(meta['chart_file'])
        exported += 1

    print("-" * 84)
    print(f"Exported {exported} song folders to {out_root}/")
    print("Each folder: original audio + chart.sm (Challenge=generated, original difficulty=real).")


if __name__ == '__main__':
    main()

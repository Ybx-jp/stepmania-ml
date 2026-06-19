#!/usr/bin/env python3
"""
Export playable .sm song folders from the typed hold-aware generator.

Loads the layered checkpoint and, for each of N val songs, generates a FULL-LENGTH
typed chart (taps + holds, hold-aware decoding, KV-cached) conditioned on the real
audio + difficulty, then writes a StepMania song folder: the original audio + a .sm
holding the generated chart (as "Challenge") and the original chart (for A/B), both
with hold/tail symbols. Drop a folder into StepMania and play it.

Usage:
    python experiments/generation_typed/export_typed_samples.py \
        --data_dir data/ --audio_dir data/ --out_dir outputs/typed_samples \
        --num_songs 8 --type_temperature 0.4
"""

import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, re, shutil, sys
from pathlib import Path
import numpy as np, torch, yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits
from src.data.dataset import StepManiaDataset, DIFFICULTY_NAMES
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.typed import symbol_histogram, pair_holds
from src.generation.sm_writer import charts_to_sm
from src.generation.evaluation import DifficultyCritic

DEFAULT_BPM = 150.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--out_dir', default='outputs/typed_samples')
    p.add_argument('--checkpoint', default='checkpoints/gen_layered/best_val.pt')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_songs', type=int, default=8)
    p.add_argument('--type_temperature', type=float, default=0.4)
    p.add_argument('--pattern_temperature', type=float, default=1.0,  # sample patterns for variety (greedy->Left/jacks)
                   help='which-panels sampling temperature; 1.0 matches real panel balance & jack rate')
    p.add_argument('--repetition_penalty', type=float, default=1.0,
                   help='>1 further discourages repeating the previous note; 1.0 already matches real')
    p.add_argument('--max_len', type=int, default=1440)  # full 2-min songs (KV-cache makes it cheap)
    return p.parse_args()


def safe_name(s):
    s = re.sub(r'[^\w\- ]+', '', (s or 'untitled').strip(), flags=re.UNICODE).strip()
    return (s or 'untitled')[:60]


def typed_binary(t):
    t = np.asarray(t); return ((t == 1) | (t == 2) | (t == 4)).astype(np.float32)


def main():
    args = parse_args(); set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob(f"{args.data_dir}/**/*.sm", recursive=True) + glob.glob(f"{args.data_dir}/**/*.ssc", recursive=True)
    _, val_files, _ = create_data_splits(cf, random_state=args.seed)
    with open(PROJECT_ROOT / "config/model_config.yaml") as f:
        msl = yaml.safe_load(f)['classifier']['max_sequence_length']
    ds = StepManiaDataset(chart_files=val_files[:args.num_songs * 8], audio_dir=args.audio_dir,
                          max_sequence_length=msl, cache_dir='cache/samples')

    model = LayeredTypedChartGenerator(audio_dim=23, d_model=128, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model_state_dict']); model.eval()
    critic = DifficultyCritic(device=device)

    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)
    print(f"\n{'song':<34} {'diff':<8} {'gen_dens':>8} {'ref_dens':>8} {'holds':>6} {'critic':>9}")
    print("-" * 80)

    exported, seen = 0, set()
    for i in range(len(ds.valid_samples)):
        if exported >= args.num_songs:
            break
        meta = ds.valid_samples[i]
        if meta['chart_file'] in seen:
            continue
        sample = ds[i]; T = min(int(sample['mask'].sum().item()), args.max_len)
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None:
            continue
        audio_np = sample['audio'][:T].numpy().astype(np.float32)
        orig_typed = ds.parser.convert_to_tensor_typed(meta['chart'], nd)[:T]
        diff_idx = meta['difficulty_class']

        # onset threshold matched to THIS chart's real density
        audio = torch.from_numpy(audio_np).unsqueeze(0).to(device)
        diff = torch.tensor([diff_idx], device=device)
        with torch.no_grad():
            p_onset = torch.sigmoid(model.onset_logits(model.encode_audio(audio), diff))[0].cpu().numpy()
        real_density = float((orig_typed != 0).any(1).mean())
        tau = float(np.quantile(p_onset, 1 - real_density)) if real_density > 0 else 0.5

        gen = model.generate(audio, diff, lengths=torch.tensor([T], device=device),
                             onset_threshold=tau, type_sample=True,
                             type_temperature=args.type_temperature, hold_aware=True,
                             pattern_sample=True, pattern_temperature=args.pattern_temperature,
                             repetition_penalty=args.repetition_penalty)[0].cpu().numpy()
        gen = pair_holds(gen)

        chart_obj = meta['chart']
        bpm = float(chart_obj.bpm); music = os.path.basename(meta['audio_file'])
        title = (chart_obj.title or Path(meta['chart_file']).stem)
        dname = DIFFICULTY_NAMES[diff_idx]

        folder = out_root / f"{exported:02d}_{safe_name(title)}"
        folder.mkdir(parents=True, exist_ok=True)
        if os.path.exists(meta['audio_file']):
            try:
                shutil.copy2(meta['audio_file'], folder / music)
            except Exception:
                pass
        sm = charts_to_sm(
            charts=[
                {"chart": gen, "difficulty_name": "Challenge", "difficulty_value": nd.difficulty_value, "author": "generated"},
                {"chart": orig_typed, "difficulty_name": dname, "difficulty_value": nd.difficulty_value, "author": "original"},
            ],
            bpm=bpm, title=f"{title} (gen)", artist=chart_obj.artist or "",
            music=music, offset=float(chart_obj.offset), typed=True,
        )
        (folder / "chart.sm").write_text(sm, encoding="utf-8")

        h = symbol_histogram(gen)
        cpred = critic.predict(typed_binary(gen), audio_np, bpm=DEFAULT_BPM)['name']
        gen_d = float((gen != 0).any(1).mean())
        print(f"{safe_name(title)[:33]:<34} {dname:<8} {gen_d:>8.3f} {real_density:>8.3f} "
              f"{h['hold_head']:>6} {cpred:>9}")
        seen.add(meta['chart_file']); exported += 1

    print("-" * 80)
    print(f"Exported {exported} playable folders to {out_root}/ (chart.sm: Challenge=generated, "
          f"+ original; both with holds). Drop a folder into StepMania to play.")


if __name__ == '__main__':
    main()

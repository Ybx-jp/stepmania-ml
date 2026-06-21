#!/usr/bin/env python3
"""
Stage 2b — best-of-N reranking with the realism critic (the cheap, decode-time payoff).

For each song, generate K candidate charts (same recommended decode, different sampling seeds), score
each with the taste critic's P(real), and keep the highest. Exports two playable sets on the SAME songs:
  reranked/  — the best-of-K candidate (highest critic P(real))
  first/     — the first (unranked) candidate (the N=1 baseline) for A/B
Also prints the critic-score lift (best vs first vs mean) so the reranking effect is visible offline.

If best > first by the critic AND it feels better to play, the taste metric earns its keep.

Usage:
    python experiments/realism_critic/export_reranked.py --data_dir data/ --audio_dir data/ \
        --num_songs 6 --n_candidates 8
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
from src.data.dataset import StepManiaDataset, DIFFICULTY_NAMES, DIFFICULTY_NAME_TO_IDX
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
from src.models import LateFusionClassifier
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.typed import pair_holds
from src.generation.sm_writer import charts_to_sm

GEN_CKPT = "checkpoints/gen_stage1/best_val.pt"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--out_dir', default='outputs/reranked_samples')
    p.add_argument('--critic', default='checkpoints/realism_critic/best_val.pt')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_songs', type=int, default=6)
    p.add_argument('--n_candidates', type=int, default=8)
    p.add_argument('--max_len', type=int, default=1440)
    p.add_argument('--difficulty', default=None,
                   help="Only export songs whose selected chart is this difficulty "
                        "(Beginner/Easy/Medium/Hard). Keeps conditioning, density target, and A/B "
                        "original all at the same difficulty. Default: each song's own difficulty.")
    p.add_argument('--install', action='store_true',
                   help="After exporting, copy the sets into the StepMania songs dir (no sudo).")
    p.add_argument('--songs_dir', default=None,
                   help="Destination for --install (default: $SM_SONGS_DIR or ~/sm-generated).")
    return p.parse_args()


def safe_name(s):
    s = re.sub(r'[^\w\- ]+', '', (s or 'untitled').strip(), flags=re.UNICODE).strip()
    return (s or 'untitled')[:60]


def to_binary(t):
    t = np.asarray(t); return ((t == 1) | (t == 2) | (t == 4)).astype(np.float32)


@torch.no_grad()
def p_real(critic, audio23, chart_bin, device):
    a = torch.from_numpy(audio23).unsqueeze(0).to(device); c = torch.from_numpy(chart_bin).unsqueeze(0).to(device)
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
    target_class = DIFFICULTY_NAME_TO_IDX[args.difficulty] if args.difficulty else None
    # With a difficulty filter, most songs are dropped — widen the pool so we still find num_songs.
    pool = args.num_songs * (40 if target_class is not None else 8)
    ds = StepManiaDataset(chart_files=val_files[:pool], audio_dir=args.audio_dir,
                          max_sequence_length=msl, feature_extractor=ext, cache_dir='cache/samples_v2')

    gen = LayeredTypedChartGenerator(audio_dim=41, d_model=128, num_layers=4, onset_layers=2).to(device)
    gen.load_state_dict(torch.load(GEN_CKPT, map_location=device)['model_state_dict']); gen.eval()
    ck = torch.load(args.critic, map_location=device, weights_only=False)
    critic = LateFusionClassifier(ck['config']).to(device); critic.load_state_dict(ck['model_state_dict']); critic.eval()

    best_root = Path(args.out_dir) / "best"; first_root = Path(args.out_dir) / "first"
    best_root.mkdir(parents=True, exist_ok=True); first_root.mkdir(parents=True, exist_ok=True)
    print(f"\n{'song':<30} {'diff':<8} {'first':>7} {'best':>7} {'mean':>7} {'lift':>7}")
    print("-" * 72)

    exported, seen = 0, set()
    lifts = []
    for i in range(len(ds.valid_samples)):
        if exported >= args.num_songs: break
        meta = ds.valid_samples[i]
        if meta['chart_file'] in seen: continue
        if target_class is not None and meta['difficulty_class'] != target_class: continue
        sample = ds[i]; T = min(int(sample['mask'].sum().item()), args.max_len)
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        audio41 = sample['audio'][:T].numpy().astype(np.float32)
        orig_typed = ds.parser.convert_to_tensor_typed(meta['chart'], nd)[:T]
        diff_idx = meta['difficulty_class']
        audio = torch.from_numpy(audio41).unsqueeze(0).to(device); diff = torch.tensor([diff_idx], device=device)
        with torch.no_grad():
            p_on = torch.sigmoid(gen.onset_logits(gen.encode_audio(audio), diff))[0].cpu().numpy()
        real_density = float((orig_typed != 0).any(1).mean())
        tau = float(np.quantile(p_on, 1 - real_density)) if real_density > 0 else 0.5
        a23 = audio41[:, :23]

        cands = []
        for k in range(args.n_candidates):
            set_seed(args.seed + 1 + k)  # different sampling draw per candidate
            with torch.no_grad():
                g = gen.generate(audio, diff, lengths=torch.tensor([T], device=device), onset_threshold=tau,
                                 type_sample=True, type_temperature=0.4, hold_aware=True, pattern_sample=True,
                                 pattern_temperature=0.7, no_jump_during_hold=True)[0].cpu().numpy()
            g = pair_holds(g)
            cands.append((g, p_real(critic, a23, to_binary(g), device)))
        scores = np.array([s for _, s in cands])
        best_idx = int(scores.argmax())
        lift = scores[best_idx] - scores[0]; lifts.append(lift)

        chart_obj = meta['chart']; bpm = float(chart_obj.bpm); music = os.path.basename(meta['audio_file'])
        title = chart_obj.title or Path(meta['chart_file']).stem; dname = DIFFICULTY_NAMES[diff_idx]
        for tag, root, g in [("first", first_root, cands[0][0]), ("best", best_root, cands[best_idx][0])]:
            folder = root / f"{exported:02d}_{safe_name(title)}"; folder.mkdir(parents=True, exist_ok=True)
            if os.path.exists(meta['audio_file']):
                try: shutil.copy2(meta['audio_file'], folder / music)
                except Exception: pass
            sm = charts_to_sm(charts=[
                {"chart": g, "difficulty_name": "Challenge", "difficulty_value": nd.difficulty_value, "author": f"gen-{tag}"},
                {"chart": orig_typed, "difficulty_name": dname, "difficulty_value": nd.difficulty_value, "author": "original"},
            ], bpm=bpm, title=f"{title} ({tag})", artist=chart_obj.artist or "", music=music, offset=float(chart_obj.offset), typed=True)
            (folder / "chart.sm").write_text(sm, encoding="utf-8")

        print(f"{safe_name(title)[:29]:<30} {dname:<8} {scores[0]:>7.3f} {scores[best_idx]:>7.3f} "
              f"{scores.mean():>7.3f} {lift:>+7.3f}")
        seen.add(meta['chart_file']); exported += 1

    print("-" * 72)
    print(f"exported {exported} songs x2 (best/ and first/) to {args.out_dir}/  "
          f"mean critic lift best-vs-first = {np.mean(lifts):+.3f}  (N={args.n_candidates})")
    print("Play best/ vs first/ on the same song — does the higher-taste pick feel more musical?")

    if args.install:
        from src.utils.sm_install import install_to_stepmania
        dests = install_to_stepmania(args.out_dir, args.songs_dir)
        print("\nInstalled to StepMania (no sudo):")
        for d in dests:
            print(f"  {d}")


if __name__ == '__main__':
    main()

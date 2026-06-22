#!/usr/bin/env python3
"""
v6 fair eval (Step 2 of notes/chaos_retrain_scope.md), ADDITIVE framing (user 06-22): real chaos is
ADDITIVE — keep quarters AND 8ths, ADD 16ths on top (density rises). NOT a trade. So success is NOT
"maximize 16ths / minimize 8ths"; it is **reproduce real's full per-phase rhythm**: quarters preserved,
8ths preserved AT REAL'S LEVEL, 16ths added. We compare per-phase NOTE RATES (notes/frame), not just
shares, so an 8th COLLAPSE (stripping the groove to make room) is visible.

Coherent conditioning (each song's own real radar) at the song's real density threshold; genuinely chaotic
songs (real 16th-share >= --min16). v4 vs v6 vs REAL. Onset-threshold proxy (which frames clear tau = the
generated chart's onset phases; pattern/type don't change which frames fire).

Reads:
  v6 16th-rate rises toward REAL, AND 8th-rate & quarter-rate stay ~REAL (not below) -> ADDITIVE fix (win).
  v6 16th up but 8th-rate collapses below REAL -> it traded the groove away (the framing the user warned
  against) -> NOT a win even if 16th-share matches.

  python experiments/generation_typed/eval_v6_additive.py --num_songs 40 --min16 0.05
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
from src.generation.typed_model import LayeredTypedChartGenerator

MODELS = {'gen_highres_v4': "checkpoints/gen_highres_v4/best_val.pt",
          'gen_highres_v6': "checkpoints/gen_highres_v6/best_val.pt",
          'gen_highres_v7': "checkpoints/gen_highres_v7/best_val.pt"}
AD = 42


def rates(onset, T):
    """per-phase note RATE (notes/frame): quarter, 8th, 16th."""
    t = np.arange(T)
    return (onset[t % 4 == 0].sum() / T, onset[t % 4 == 2].sum() / T,
            onset[(t % 4 == 1) | (t % 4 == 3)].sum() / T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_songs', type=int, default=40)
    ap.add_argument('--max_len', type=int, default=1024)
    ap.add_argument('--min_difficulty', type=int, default=2)
    ap.add_argument('--min16', type=float, default=0.05, help='only songs whose REAL 16th-share >= this (chaotic)')
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    _, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                   use_metric_phase=True, use_highres_onset=True))
    ds = StepManiaDataset(chart_files=vf[:args.num_songs * 6], audio_dir="data/", max_sequence_length=msl,
                          feature_extractor=ext, cache_dir='cache/samples_v3')
    models = {}
    for name, ckpt in MODELS.items():
        if not os.path.exists(ckpt):
            print(f"(skip {name}: no checkpoint)"); continue
        m = LayeredTypedChartGenerator(audio_dim=AD, d_model=128, num_layers=4, onset_layers=2).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device)['model_state_dict']); m.eval()
        models[name] = m

    real_r = []; gen_r = {k: [] for k in models}
    used = 0
    for i in range(len(ds.valid_samples)):
        if used >= args.num_songs:
            break
        meta = ds.valid_samples[i]
        if meta['difficulty_class'] < args.min_difficulty:
            continue
        sample = ds[i]; T = min(int(sample['mask'].sum().item()), args.max_len)
        if T < 256:
            continue
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None:
            continue
        orig = np.asarray(ds.parser.convert_to_tensor_typed(meta['chart'], nd))[:T]
        note = (orig != 0).any(1); t = np.arange(T)
        n = int(note.sum())
        if n < 32:
            continue
        s16 = (note & ((t % 4 == 1) | (t % 4 == 3))).sum() / max(n, 1)
        if s16 < args.min16:                      # only genuinely chaotic songs
            continue
        rd = float(note.mean())
        real_r.append(rates(note, T))
        radar = torch.from_numpy(meta['groove_radar'].to_vector().astype(np.float32)).unsqueeze(0).to(device)
        diff = torch.tensor([meta['difficulty_class']], device=device)
        audio = sample['audio'][:T, :AD].unsqueeze(0).to(device)
        for name, m in models.items():
            with torch.no_grad():  # coherent: each song's own real radar
                p = torch.sigmoid(m.onset_logits(m.encode_audio(audio), diff, radar=radar))[0].cpu().numpy()
            placed = p > float(np.quantile(p, 1 - rd))
            gen_r[name].append(rates(placed, T))
        used += 1

    rr = np.mean(real_r, 0)
    def share(r):  # phase shares from rates
        s = sum(r); return tuple(100 * x / s for x in r)
    print(f"\n=== v6 ADDITIVE eval ({used} chaotic songs, real 16th-share >= {args.min16}, coherent conditioning) ===")
    print(f"  per-phase NOTE RATE (notes/frame) — additive view; 8th-rate must NOT collapse below REAL")
    print(f"  {'source':<16} {'q-rate':>8} {'8th-rate':>9} {'16th-rate':>10} {'density':>8}   | shares q/8th/16th")
    def row(name, r):
        sh = share(r)
        print(f"  {name:<16} {r[0]:>8.3f} {r[1]:>9.3f} {r[2]:>10.3f} {sum(r):>8.3f}   | {sh[0]:.0f}/{sh[1]:.0f}/{sh[2]:.0f}%")
    row('REAL', rr)
    for name in models:
        row(name, np.mean(gen_r[name], 0))
    print()
    if 'gen_highres_v6' in models:
        v6 = np.mean(gen_r['gen_highres_v6'], 0); v4 = np.mean(gen_r.get('gen_highres_v4', [rr]), 0)
        d16 = "UP" if v6[2] > v4[2] else "flat/down"
        e8 = "preserved (>=80% of REAL)" if v6[1] >= 0.8 * rr[1] else "COLLAPSED below REAL (groove stripped!)"
        print(f"  v6 vs v4: 16th-rate {v4[2]:.3f}->{v6[2]:.3f} ({d16} toward REAL {rr[2]:.3f}); "
              f"v6 8th-rate {v6[1]:.3f} vs REAL {rr[1]:.3f} -> {e8}")
        print(f"  ADDITIVE WIN if 16th-rate UP toward REAL and 8th-rate preserved; TRADE (bad) if 8th collapsed.")


if __name__ == '__main__':
    main()

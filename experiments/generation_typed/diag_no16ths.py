#!/usr/bin/env python3
"""
Localize WHY the generator produces no 16ths (=> chaos radar 0; see playtest_log 06-21). Metric phase on
the 16th grid: quarter t%4==0, 8th t%4==2, 16th t%4 in {1,3}.

For real vs gen_stage1 (41-dim) vs gen_highres (42-dim, the high-res-onset model -- tests if a
16th-resolved feature helps), report:
  - note-fraction by phase (does the chart HAVE 16ths?)
  - mean p_on by phase (does the model's onset posterior ever WANT a 16th?)
  - frac of 16th frames with p_on > tau (would the density threshold ever SELECT one?)

Reads:
  p_on at 16ths ~ 0 / never > tau  -> the onset head can't REPRESENT 16ths (feature resolution, H4) -> need
    a 16th-resolved feature actually engaged (or architecture).
  p_on at 16ths sometimes > tau but gen still ~0%  -> a DECODE issue (threshold/density skips them).
  gen_highres places more 16ths than gen_stage1 -> the high-res feature is the lever (engage it properly).
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
from src.generation.typed import pair_holds

MODELS = {'gen_stage1': ("checkpoints/gen_stage1/best_val.pt", 41),
          'gen_highres': ("checkpoints/gen_highres/best_val.pt", 42),
          'gen_highres_v3': ("checkpoints/gen_highres_v3/best_val.pt", 42)}
DECODE = dict(type_sample=True, type_temperature=0.4, hold_aware=True, pattern_sample=True,
              pattern_temperature=0.7, no_jump_during_hold=True)


def phase_frac(note):
    t = np.arange(len(note)); n = max(int(note.sum()), 1)
    return (100 * note[t % 4 == 0].sum() / n, 100 * note[t % 4 == 2].sum() / n,
            100 * note[(t % 4 == 1) | (t % 4 == 3)].sum() / n)   # quarter, 8th, 16th


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--num_songs', type=int, default=30)
    ap.add_argument('--max_len', type=int, default=1024); ap.add_argument('--min_difficulty', type=int, default=2)
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    _, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                   use_metric_phase=True, use_highres_onset=True))  # 42-dim
    ds = StepManiaDataset(chart_files=vf[:args.num_songs * 4], audio_dir="data/", max_sequence_length=msl,
                          feature_extractor=ext, cache_dir='cache/samples_v3')
    models = {}
    for name, (ckpt, ad) in MODELS.items():
        if not os.path.exists(ckpt):
            print(f"(skip {name}: no checkpoint)"); continue
        m = LayeredTypedChartGenerator(audio_dim=ad, d_model=128, num_layers=4, onset_layers=2).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device)['model_state_dict']); m.eval()
        models[name] = (m, ad)

    real_ph = []
    gen_ph = {k: [] for k in models}; pon_ph = {k: [] for k in models}; sel16 = {k: [] for k in models}
    used = 0
    for i in range(len(ds.valid_samples)):
        if used >= args.num_songs: break
        meta = ds.valid_samples[i]
        if meta['difficulty_class'] < args.min_difficulty: continue
        sample = ds[i]; T = min(int(sample['mask'].sum().item()), args.max_len)
        if T < 256: continue
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        orig = np.asarray(ds.parser.convert_to_tensor_typed(meta['chart'], nd))[:T]
        note_r = (orig != 0).any(1)
        if note_r.sum() < 32: continue
        real_ph.append(phase_frac(note_r)); rd = float(note_r.mean()); t = np.arange(T)
        diff = torch.tensor([meta['difficulty_class']], device=device)
        for name, (m, ad) in models.items():
            audio = sample['audio'][:T, :ad].unsqueeze(0).to(device)
            with torch.no_grad():
                p_on = torch.sigmoid(m.onset_logits(m.encode_audio(audio), diff))[0].cpu().numpy()
                tau = float(np.quantile(p_on, 1 - rd))
                g = m.generate(audio, diff, lengths=torch.tensor([T], device=device), onset_threshold=tau, **DECODE)[0].cpu().numpy()
            gen_ph[name].append(phase_frac((pair_holds(g) != 0).any(1)))
            s16 = (t % 4 == 1) | (t % 4 == 3)
            pon_ph[name].append((p_on[t % 4 == 0].mean(), p_on[t % 4 == 2].mean(), p_on[s16].mean()))
            sel16[name].append(100 * (p_on[s16] > tau).mean())   # % of 16th frames the threshold would pick
        used += 1

    print(f"\n=== Why no 16ths? ({used} songs) ===")
    print("note-fraction by phase (% of notes):")
    print(f"  {'source':<14} {'quarter':>8} {'8th':>8} {'16th':>8}")
    print(f"  {'REAL':<14} " + " ".join(f"{v:>7.1f}%" for v in np.mean(real_ph, 0)))
    for k in models:
        print(f"  {k:<14} " + " ".join(f"{v:>7.1f}%" for v in np.mean(gen_ph[k], 0)))
    print("\nmean p_on by phase (does the onset posterior want a 16th?) + 16th-selectable rate:")
    print(f"  {'model':<14} {'quarter':>8} {'8th':>8} {'16th':>8} {'16th>tau%':>10}")
    for k in models:
        pp = np.mean(pon_ph[k], 0)
        print(f"  {k:<14} {pp[0]:>8.3f} {pp[1]:>8.3f} {pp[2]:>8.3f} {np.mean(sel16[k]):>9.1f}%")
    print("\nREAL has 16ths, gen ~0 => the gap. p_on@16th low & 16th>tau ~0 => posterior can't represent")
    print("16ths (H4/resolution). gen_highres > gen_stage1 at 16ths => the high-res feature is the lever.")


if __name__ == '__main__':
    main()

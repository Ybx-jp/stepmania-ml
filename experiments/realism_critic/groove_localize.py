#!/usr/bin/env python3
"""
Localize WHERE groove (off-beat beat-recurrence, ac_off; see groove_periodicity_findings.md) is lost
between the real target (ac_off 0.187) and the decoded output (0.109). Stages, per song:

  (1) REAL target                      ac_off 0.187 (baseline)
  (2) p_on continuous (off-beat)       does the onset POSTERIOR carry periodicity at all?
  (3) raw thresholded onset (p_on>tau)  onset head + density threshold, NO pattern/hold decode
  (4) full decode (pattern+hold_aware)  = 0.109

Reads:
  - (3) ~ (4): pattern/hold decode is neutral; loss is upstream in the onset head.
  - (2) high but (3) low: thresholding destroys periodicity the posterior had (a DECODE-threshold issue,
    potentially cheap to fix).
  - (2) and (3) both low (~0.109): the non-causal audio-only onset head doesn't REPRESENT groove
    (ARCHITECTURAL — it has no memory to repeat a figure; neither expert data nor decode fixes it).
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

GEN_CKPT = "checkpoints/gen_stage1/best_val.pt"


def autocorr(x, lag):
    x = x.astype(np.float64) - x.mean()
    denom = (x * x).sum()
    if denom < 1e-9 or lag >= len(x):
        return 0.0
    return float((x[lag:] * x[:-lag]).sum() / denom)


def ac_off(sig):
    """lag-4 autocorr of the OFF-beat-only signal (on-beat frames zeroed)."""
    t = np.arange(len(sig))
    s = sig.copy().astype(np.float64); s[t % 4 == 0] = 0.0
    return autocorr(s, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_songs', type=int, default=40); ap.add_argument('--max_len', type=int, default=1024)
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    _, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True, use_metric_phase=True))
    ds = StepManiaDataset(chart_files=vf[:args.num_songs * 4], audio_dir="data/", max_sequence_length=msl,
                          feature_extractor=ext, cache_dir='cache/samples_v2')
    gen = LayeredTypedChartGenerator(audio_dim=41, d_model=128, num_layers=4, onset_layers=2).to(device)
    gen.load_state_dict(torch.load(GEN_CKPT, map_location=device)['model_state_dict']); gen.eval()

    acc = {k: [] for k in ['1 REAL', '2 p_on continuous', '3 raw threshold', '4 full decode']}
    seen, used = set(), 0
    for i in range(len(ds.valid_samples)):
        if used >= args.num_songs: break
        meta = ds.valid_samples[i]
        if meta['chart_file'] in seen: continue
        sample = ds[i]; T = min(int(sample['mask'].sum().item()), args.max_len)
        if T < 128: continue
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        orig = np.asarray(ds.parser.convert_to_tensor_typed(meta['chart'], nd))[:T]
        onset_r = (orig != 0).any(1).astype(np.float64)
        if onset_r.sum() < 16: continue
        real_density = float(onset_r.mean())
        audio = sample['audio'][:T].unsqueeze(0).to(device); diff = torch.tensor([meta['difficulty_class']], device=device)
        with torch.no_grad():
            p_on = torch.sigmoid(gen.onset_logits(gen.encode_audio(audio), diff))[0].cpu().numpy()
            tau = float(np.quantile(p_on, 1 - real_density)) if real_density > 0 else 0.5
            raw_onset = (p_on > tau).astype(np.float64)
            g = gen.generate(audio, diff, lengths=torch.tensor([T], device=device), onset_threshold=tau,
                             type_sample=True, type_temperature=0.4, hold_aware=True, pattern_sample=True,
                             pattern_temperature=0.7, no_jump_during_hold=True)[0].cpu().numpy()
        full = (pair_holds(g) != 0).any(1).astype(np.float64)
        acc['1 REAL'].append(ac_off(onset_r))
        acc['2 p_on continuous'].append(ac_off(p_on))      # continuous posterior
        acc['3 raw threshold'].append(ac_off(raw_onset))   # onset head + threshold, no pattern/hold
        acc['4 full decode'].append(ac_off(full))
        seen.add(meta['chart_file']); used += 1

    print(f"\n=== Groove localization ({used} songs) — ac_off by stage ===\n")
    print(f"{'stage':<22} {'ac_off':>8}")
    print("-" * 32)
    for k in ['1 REAL', '2 p_on continuous', '3 raw threshold', '4 full decode']:
        print(f"{k:<22} {np.mean(acc[k]):>8.3f}")
    print("-" * 32)
    print("(3)~(4): pattern/hold decode neutral -> loss is in the onset head.")
    print("(2) high & (3) low: threshold destroys posterior periodicity -> decode-threshold issue.")
    print("(2)&(3) low ~0.11: non-causal audio-only onset head can't represent groove -> ARCHITECTURAL.")


if __name__ == '__main__':
    main()

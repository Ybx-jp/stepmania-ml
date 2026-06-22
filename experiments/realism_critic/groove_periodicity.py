#!/usr/bin/env python3
"""
Groove / rhythmic-periodicity metric — operationalizes H10 (musical syncopation = REPEATED off-beat
figures = a groove, vs the chaos-gate's pointillist scatter). The on-beat% metric measures syncopation
QUANTITY; this measures STRUCTURE: do off-beat notes recur at the same beat-phase across beats?

Metric (16th grid: 4 sixteenths/beat, 16/measure). For onset indicator o[t] in {0,1}:
  autocorr(x, L) = demeaned Pearson autocorrelation at lag L.
  - ac_beat     = autocorr(o, 4)          : does the whole rhythm repeat each beat?
  - ac_measure  = autocorr(o, 16)         : ... each measure?
  - ac_off_beat = autocorr(o_off, 4)      : do OFF-beat notes recur beat-to-beat?  <-- the H10 discriminator
    (o_off = o but zeroed on on-beat frames t%4==0)

Validation: compare REAL vs a SHUFFLED null (off-beats randomly relocated to other off-beat frames,
preserving on-beat backbone + off-beat count) — destroys periodicity, keeps density/on-beat%. If
REAL ac_off_beat >> shuffled, the metric captures groove (not density). Then: where does the base
generator (gen_stage1, plain) fall — does it produce groove or scatter?
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


def metrics(onset):
    t = np.arange(len(onset))
    o_off = onset.copy().astype(np.float64); o_off[t % 4 == 0] = 0.0
    ob = float(onset[t % 4 == 0].sum()) / max(int(onset.sum()), 1)
    return dict(onbeat=ob, ac_beat=autocorr(onset, 4), ac_meas=autocorr(onset, 16),
                ac_off=autocorr(o_off, 4))


def shuffle_offbeats(onset, rng):
    """Keep on-beat onsets; relocate the off-beat onsets to random off-beat frames (same count)."""
    t = np.arange(len(onset))
    on_mask = (t % 4 == 0) & (onset > 0)
    off_idx = np.where(t % 4 != 0)[0]
    n_off = int(onset[t % 4 != 0].sum())
    out = np.zeros_like(onset)
    out[on_mask] = 1
    if n_off > 0 and len(off_idx) >= n_off:
        out[rng.choice(off_idx, size=n_off, replace=False)] = 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_songs', type=int, default=40); ap.add_argument('--max_len', type=int, default=1024)
    args = ap.parse_args()
    set_seed(42); rng = np.random.default_rng(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    _, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True, use_metric_phase=True))
    ds = StepManiaDataset(chart_files=vf[:args.num_songs * 4], audio_dir="data/", max_sequence_length=msl,
                          feature_extractor=ext, cache_dir='cache/samples_v2')
    gen = LayeredTypedChartGenerator(audio_dim=41, d_model=128, num_layers=4, onset_layers=2).to(device)
    gen.load_state_dict(torch.load(GEN_CKPT, map_location=device)['model_state_dict']); gen.eval()

    rows = {'REAL': [], 'REAL-shuffled (null)': [], 'gen_stage1 (plain)': []}
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
            g = gen.generate(audio, diff, lengths=torch.tensor([T], device=device), onset_threshold=tau,
                             type_sample=True, type_temperature=0.4, hold_aware=True, pattern_sample=True,
                             pattern_temperature=0.7, no_jump_during_hold=True)[0].cpu().numpy()
        onset_g = (pair_holds(g) != 0).any(1).astype(np.float64)
        rows['REAL'].append(metrics(onset_r))
        rows['REAL-shuffled (null)'].append(metrics(shuffle_offbeats(onset_r, rng)))
        rows['gen_stage1 (plain)'].append(metrics(onset_g))
        seen.add(meta['chart_file']); used += 1

    print(f"\n=== Groove periodicity ({used} songs) ===")
    print("ac_off (off-beat beat-recurrence) is the H10 discriminator: groove >> scatter.\n")
    hdr = f"{'source':<24} {'onbeat%':>8} {'ac_beat':>8} {'ac_meas':>8} {'ac_off':>8}"
    print(hdr); print("-" * len(hdr))
    for name, ms in rows.items():
        if not ms: continue
        ob = 100 * np.mean([m['onbeat'] for m in ms]); ab = np.mean([m['ac_beat'] for m in ms])
        am = np.mean([m['ac_meas'] for m in ms]); ao = np.mean([m['ac_off'] for m in ms])
        print(f"{name:<24} {ob:>7.1f}% {ab:>8.3f} {am:>8.3f} {ao:>8.3f}")
    print("-" * len(hdr))
    print("REAL ac_off >> shuffled-null ac_off => metric captures GROOVE (not density). Then read where")
    print("gen_stage1 falls: near REAL = it has groove; near null = it scatters off-beats like the gate.")


if __name__ == '__main__':
    main()

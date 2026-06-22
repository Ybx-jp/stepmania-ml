#!/usr/bin/env python3
"""
Prototype a PHASE-AWARE onset threshold (see notes/chaos_mechanism_plan.md; diag_no16ths localization).

Why: the model HAS 16th onset confidence (p_on@16th ~0.38-0.42) but a single density-quantile threshold
buries it -- 16th frames are out-budgeted by the more-confident 8ths (~0.55), so 16th>tau ~ 0.3-0.5% and
only ~0.4-0.8% of placed notes land on 16ths (real: 4.1%). Training longer made it WORSE (8ths sharpen,
16ths fall further below the cut). So the lever is decode, not epochs.

Phase-stratified allocation: keep the SAME note budget N the global threshold gives (density held constant
for a fair A/B), but split N across the three phase bands by a target distribution (default real's
70.7/25.2/4.1), and pick the top-p_on frames WITHIN each band. Each band thus gets its own implicit
threshold -> the model's own 16th ranking chooses which 16ths, instead of 16ths losing globally to 8ths.

We need NO model change: generate() already takes onset_override, so we compute the phase-aware onset mask
here and pass it in, then measure the realized chart (phase fractions, density, critic-adjacency).

  python experiments/generation_typed/diag_phase_threshold.py --num_songs 20 --shares 0.707,0.252,0.041
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
from src.generation.evaluation import DifficultyCritic

MODELS = {'gen_highres_v4': ("checkpoints/gen_highres_v4/best_val.pt", 42),
          'gen_highres_v5': ("checkpoints/gen_highres_v5/best_val.pt", 42)}
DECODE = dict(type_sample=True, type_temperature=0.4, hold_aware=True, pattern_sample=True,
              pattern_temperature=0.7, no_jump_during_hold=True)
DEFAULT_BPM = 150.0


def phase_frac(note):
    t = np.arange(len(note)); n = max(int(note.sum()), 1)
    return (100 * note[t % 4 == 0].sum() / n, 100 * note[t % 4 == 2].sum() / n,
            100 * note[(t % 4 == 1) | (t % 4 == 3)].sum() / n)   # quarter, 8th, 16th


def phase_alloc_mask(p, tau, shares):
    """Phase-stratified onset mask: keep budget N = (p>tau).sum(), split by `shares` across phase bands,
    top-p_on within each band. Returns bool (T,)."""
    T = len(p); t = np.arange(T)
    bands = [t % 4 == 0, t % 4 == 2, (t % 4 == 1) | (t % 4 == 3)]   # quarter, 8th, 16th
    N = int((p > tau).sum())
    onset = np.zeros(T, dtype=bool)
    for share, band in zip(shares, bands):
        idx = np.where(band)[0]
        if len(idx) == 0:
            continue
        nk = min(int(round(N * share)), len(idx))
        if nk <= 0:
            continue
        top = idx[np.argsort(p[idx])[-nk:]]      # highest-p_on frames in this band
        onset[top] = True
    return onset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_songs', type=int, default=20)
    ap.add_argument('--max_len', type=int, default=768)
    ap.add_argument('--min_difficulty', type=int, default=2)
    ap.add_argument('--shares', default='0.707,0.252,0.041', help='target note shares quarter,8th,16th')
    args = ap.parse_args()
    shares = np.array([float(x) for x in args.shares.split(',')]); shares /= shares.sum()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    _, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                   use_metric_phase=True, use_highres_onset=True))
    ds = StepManiaDataset(chart_files=vf[:args.num_songs * 4], audio_dir="data/", max_sequence_length=msl,
                          feature_extractor=ext, cache_dir='cache/samples_v3')
    critic = DifficultyCritic(device=device)
    models = {}
    for name, (ckpt, ad) in MODELS.items():
        if not os.path.exists(ckpt):
            print(f"(skip {name}: no checkpoint)"); continue
        m = LayeredTypedChartGenerator(audio_dim=ad, d_model=128, num_layers=4, onset_layers=2).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device)['model_state_dict']); m.eval()
        models[name] = (m, ad)

    # per-model, per-mode: phase fractions, density, |pred-target| for critic adjacency
    res = {name: {'global': {'ph': [], 'dens': [], 'dd': []}, 'alloc': {'ph': [], 'dens': [], 'dd': []}}
           for name in models}
    real_ph = []; used = 0
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
        note_r = (orig != 0).any(1)
        if note_r.sum() < 32:
            continue
        real_ph.append(phase_frac(note_r)); rd = float(note_r.mean())
        diff = torch.tensor([meta['difficulty_class']], device=device)
        for name, (m, ad) in models.items():
            audio = sample['audio'][:T, :ad].unsqueeze(0).to(device)
            with torch.no_grad():
                p_on = torch.sigmoid(m.onset_logits(m.encode_audio(audio), diff))[0].cpu().numpy()
            tau = float(np.quantile(p_on, 1 - rd))
            masks = {'global': (p_on > tau), 'alloc': phase_alloc_mask(p_on, tau, shares)}
            for mode, onset in masks.items():
                ov = torch.from_numpy(onset).bool().unsqueeze(0).to(device)
                with torch.no_grad():
                    g = m.generate(audio, diff, lengths=torch.tensor([T], device=device),
                                   onset_override=ov, **DECODE)[0].cpu().numpy()
                gnote = (pair_holds(g) != 0).any(1)
                res[name][mode]['ph'].append(phase_frac(gnote))
                res[name][mode]['dens'].append(float(gnote.mean()))
                gb = ((g == 1) | (g == 2) | (g == 4)).astype(np.float32)
                pred = critic.predict(gb, sample['audio'][:T, :23].numpy(), bpm=DEFAULT_BPM)['class']
                res[name][mode]['dd'].append(abs(pred - meta['difficulty_class']))
        used += 1

    rp = np.mean(real_ph, 0)
    print(f"\n=== Phase-aware threshold prototype ({used} songs, shares={shares.round(3).tolist()}) ===")
    print(f"  {'source / mode':<26} {'quarter':>8} {'8th':>8} {'16th':>8} {'density':>9} {'crit_adj':>9}")
    print(f"  {'REAL':<26} {rp[0]:>7.1f}% {rp[1]:>7.1f}% {rp[2]:>7.1f}%")
    for name in models:
        for mode in ('global', 'alloc'):
            ph = np.mean(res[name][mode]['ph'], 0)
            dens = np.mean(res[name][mode]['dens']); adj = np.mean(np.array(res[name][mode]['dd']) <= 1)
            print(f"  {name + ' / ' + mode:<26} {ph[0]:>7.1f}% {ph[1]:>7.1f}% {ph[2]:>7.1f}% {dens:>8.3f} {adj:>8.3f}")
    print("\nalloc 16th -> ~real & crit_adj holds => phase-aware threshold unlocks the model's own 16ths.")
    print("alloc 16th high but crit_adj drops => 16th ranking is poor (placed in wrong spots).")


if __name__ == '__main__':
    main()

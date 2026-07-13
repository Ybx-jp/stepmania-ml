#!/usr/bin/env python3
"""
Does the chaos-radar CONDITIONING actually dial 16th density up/down -- NOW that the high-res feature is
engaged? (selfsim_chaos_findings.md conclusion: chaos can't be inferred from audio -> it must be
CONDITIONED. H4/H6 found chaos conditioning smeared, but that was PRE-high-res, when the model had no local
16th cue to attach the conditioning to. Re-test with gen_highres_v4 + AUC-0.742 frame placement.)

Cheap offline probe, NO full generation: the onset head takes the radar (onset_logits(..., radar=...)), so
sweep ONLY the chaos dim (hold the song's other 4 radar dims at their real values) and read the posterior.

Per chaos level c in the sweep, aggregate over songs:
  p_on by phase (quarter/8th/16th)  -- does raising chaos raise 16th confidence SPECIFICALLY?
  realized share by phase at the per-song density threshold -- does it translate to PLACED 16ths?
  (optional CFG: amplify the conditioning, ol = ol_uncond + g*(ol_cond - ol_uncond))

Reads:
  p16 & realized-16th-share rise monotonically with chaos, quarter share falls -> conditioning WORKS and is
    SPECIFIC (the win): chaos is a real knob; combine with calib for placement.
  p16 rises but realized-16th-share flat -> conditioning works but the threshold buries it (use calib).
  everything rises together (q,8th,16th) -> SMEAR, not specific (the H4/H6 failure persists).
  flat across c -> conditioning IGNORED.

  python experiments/generation_typed/diag_chaos_condition.py --num_songs 50 --guidance 1.0
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

CKPT, AD = "checkpoints/gen_highres_v4/best_val.pt", 42
CHAOS_IDX = 4   # groove radar = [stream, voltage, air, freeze, chaos]


def phase_means(p, T):
    t = np.arange(T)
    return p[t % 4 == 0].mean(), p[t % 4 == 2].mean(), p[(t % 4 == 1) | (t % 4 == 3)].mean()


def phase_share(onset, T):
    t = np.arange(T); n = max(int(onset.sum()), 1)
    return (onset[t % 4 == 0].sum() / n, onset[t % 4 == 2].sum() / n,
            onset[(t % 4 == 1) | (t % 4 == 3)].sum() / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_songs', type=int, default=50)
    ap.add_argument('--max_len', type=int, default=1024)
    ap.add_argument('--min_difficulty', type=int, default=2)
    ap.add_argument('--guidance', type=float, default=1.0, help='CFG scale to amplify the conditioning (1=off)')
    ap.add_argument('--chaos_grid', default='0.0,0.25,0.5,0.75,1.0')
    args = ap.parse_args()
    grid = [float(x) for x in args.chaos_grid.split(',')]
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    _, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                   use_metric_phase=True, use_highres_onset=True))
    ds = StepManiaDataset(chart_files=vf[:args.num_songs * 4], audio_dir="data/", max_sequence_length=msl,
                          feature_extractor=ext, cache_dir='cache/samples_v3')
    m = LayeredTypedChartGenerator(audio_dim=AD, d_model=128, num_layers=4, onset_layers=2).to(device)
    m.load_state_dict(torch.load(CKPT, map_location=device)['model_state_dict']); m.eval()

    # accumulate per-chaos-level: phase p_on means and realized phase shares
    pm = {c: [] for c in grid}; sh = {c: [] for c in grid}
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
        note = (orig != 0).any(1)
        if note.sum() < 32:
            continue
        rd = float(note.mean())
        base_radar = meta['groove_radar'].to_vector().astype(np.float32)  # the song's real radar
        diff = torch.tensor([meta['difficulty_class']], device=device)
        audio = torch.from_numpy(sample['audio'][:T, :AD].numpy()).unsqueeze(0).to(device)
        with torch.no_grad():
            memory = m.encode_audio(audio)
            ol_unc = m.onset_logits(memory, diff, radar=None)[0].cpu().numpy() if args.guidance != 1.0 else None
            for c in grid:
                r = base_radar.copy(); r[CHAOS_IDX] = c
                rt = torch.from_numpy(r).unsqueeze(0).to(device)
                ol = m.onset_logits(memory, diff, radar=rt)[0].cpu().numpy()
                if args.guidance != 1.0:
                    ol = ol_unc + args.guidance * (ol - ol_unc)
                p = 1 / (1 + np.exp(-ol))
                pm[c].append(phase_means(p, T))
                tau = np.quantile(p, 1 - rd)
                sh[c].append(phase_share(p > tau, T))
        used += 1

    print(f"\n=== Chaos conditioning sweep ({used} songs, guidance={args.guidance}) ===")
    print(f"  chaos -> mean p_on by phase (does raising chaos raise 16th confidence specifically?)")
    print(f"  {'chaos':>6} {'p_quarter':>10} {'p_8th':>8} {'p_16th':>8}")
    for c in grid:
        a = np.mean(pm[c], 0)
        print(f"  {c:>6.2f} {a[0]:>10.3f} {a[1]:>8.3f} {a[2]:>8.3f}")
    print(f"\n  chaos -> realized note share by phase at per-song density (placed notes)")
    print(f"  {'chaos':>6} {'quarter%':>9} {'8th%':>8} {'16th%':>8}")
    for c in grid:
        s = np.mean(sh[c], 0) * 100
        print(f"  {c:>6.2f} {s[0]:>8.1f}% {s[1]:>7.1f}% {s[2]:>7.1f}%")
    p16 = [np.mean(pm[c], 0)[2] for c in grid]; s16 = [np.mean(sh[c], 0)[2] for c in grid]
    print(f"\n  16th p_on: {grid[0]:.2f}->{p16[0]:.3f}  {grid[-1]:.2f}->{p16[-1]:.3f}  "
          f"(delta {p16[-1]-p16[0]:+.3f});  realized 16th share {s16[0]*100:.1f}%->{s16[-1]*100:.1f}%")
    print("  p16 & 16th-share rise w/ chaos + quarter falls => conditioning works & specific. p16 rises but")
    print("  share flat => threshold buries (use calib). all phases rise => smear. flat => ignored.")


if __name__ == '__main__':
    main()

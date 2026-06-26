#!/usr/bin/env python3
"""
Investigate the STAGE-3 breathing ENERGY signal (playtest H-arc-energy): the breathing ceiling uses the onset
head's `p_onset` as "energy", but a playtest found it BIASES toward percussive sections and IGNORES high-pitch
melodic energy (a manic piano solo reads as low energy → breathing rests it). The highres audio features already
carry the harmonic half — HPSS `harm_onset` (col 36) + `perc_onset` (col 35) + 12-dim `chroma` (cols 23:35, tonal
energy) + the raw full-mix `onset_env` (col 13). This probe asks: is p_onset percussion-biased, and which available
signal is a more BALANCED energy (catches the melodic sections p_onset misses)?

  python experiments/generation_typed/diag_breathe_energy.py [--songs 8]

READ: if corr(p_onset, perc) >> corr(p_onset, harm), p_onset is percussion-biased. The "melodic-miss" column =
mean z(p_onset) on frames that are harmonically loud but percussively quiet (z_harm>0.5 & z_perc<-0.5 = piano-solo-
like); strongly negative = p_onset rests them. A balanced signal (onset_env or perc+harm) should NOT.
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, sys
from pathlib import Path
import numpy as np, torch, yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
from src.generation.typed_model import LayeredTypedChartGenerator

# highres 42-dim feature column offsets (see src/data/audio_features.py to_vector order)
COL = dict(onset_env=13, chroma=slice(23, 35), perc=35, harm=36, highres=41)


def smooth_z(x, w=96):
    import torch.nn.functional as F
    s = F.avg_pool1d(torch.as_tensor(x, dtype=torch.float32).view(1, 1, -1), 2 * w + 1, 1, w,
                     count_include_pad=False).view(-1).numpy()
    return (s - s.mean()) / (s.std() + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/gen_motif_full_fixed/best_val.pt")
    ap.add_argument("--songs", type=int, default=8); ap.add_argument("--max_len", type=int, default=1440)
    ap.add_argument("--match", type=str, default=None, help="comma-sep title substrings to select (e.g. the playtest songs)")
    args = ap.parse_args()
    want = [m.strip().lower() for m in args.match.split(',')] if args.match else None
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    tf, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                   use_metric_phase=True, use_highres_onset=True))
    _, vds, _ = create_datasets(train_files=tf, val_files=vf, test_files=[], audio_dir="data/",
                                max_sequence_length=msl, feature_extractor=ext, cache_dir='cache/samples_v3')
    vds.warm_cache(show_progress=False)
    songs = []
    for i in range(len(vds)):
        if len(songs) >= args.songs:
            break
        s = vds[i]; m = vds.valid_samples[i]; T = min(int(s['mask'].sum().item()), args.max_len)
        if int(m['difficulty_class']) != 3:
            continue
        if want is not None and not any(w in m['chart'].title.lower() for w in want):
            continue
        songs.append({'audio': s['audio'][:T].numpy().astype(np.float32), 'len': T,
                      'diff': int(m['difficulty_class']), 'title': m['chart'].title[:20]})
    audio_dim = songs[0]['audio'].shape[1]
    assert audio_dim == 42, f"this probe assumes the 42-dim highres layout, got {audio_dim}"
    model = LayeredTypedChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)['model_state_dict']); model.eval()

    print(f"\nBREATHING ENERGY signal probe  {len(songs)} Hard songs (full length)")
    print(f"  corr(p_onset, X) for each candidate energy X; melodic-miss = mean z(X) on harm-loud/perc-quiet frames")
    print(f"  {'song':>20} {'c_perc':>7} {'c_harm':>7} {'c_onsEnv':>8} {'c_chroma':>8} | {'miss:p_onset':>12} {'onsEnv':>7} {'perc+harm':>9}")
    print("  " + "-" * 92)
    agg = {k: [] for k in ['c_perc', 'c_harm', 'c_onsenv', 'c_chroma', 'miss_p', 'miss_oe', 'miss_ph']}
    for s in songs:
        T = s['len']; audio = torch.from_numpy(s['audio']).unsqueeze(0).to(device)
        diff = torch.tensor([s['diff']], device=device)
        with torch.no_grad():
            p = torch.sigmoid(model.onset_logits(model.encode_audio(audio), diff)[0]).cpu().numpy()
        a = s['audio']
        zp = smooth_z(p, 96)
        z_perc = smooth_z(a[:, COL['perc']]); z_harm = smooth_z(a[:, COL['harm']])
        z_oe = smooth_z(a[:, COL['onset_env']]); z_chr = smooth_z(a[:, COL['chroma']].mean(1))
        z_ph = smooth_z(a[:, COL['perc']] + a[:, COL['harm']])  # balanced perc+harm
        c = lambda x: float(np.corrcoef(zp, x)[0, 1])
        cp, ch, coe, cch = c(z_perc), c(z_harm), c(z_oe), c(z_chr)
        melodic = (z_harm > 0.5) & (z_perc < -0.5)            # harmonically loud, percussively quiet
        mp = float(zp[melodic].mean()) if melodic.any() else float('nan')
        moe = float(z_oe[melodic].mean()) if melodic.any() else float('nan')
        mph = float(z_ph[melodic].mean()) if melodic.any() else float('nan')
        print(f"  {s['title']:>20} {cp:>7.2f} {ch:>7.2f} {coe:>8.2f} {cch:>8.2f} | {mp:>12.2f} {moe:>7.2f} {mph:>9.2f}"
              f"  (n_mel={int(melodic.sum())})")
        for k, v in zip(agg, [cp, ch, coe, cch, mp, moe, mph]):
            if not np.isnan(v):
                agg[k].append(v)
    mean = {k: np.mean(v) if v else float('nan') for k, v in agg.items()}
    print("  " + "-" * 92)
    print(f"  {'MEAN':>20} {mean['c_perc']:>7.2f} {mean['c_harm']:>7.2f} {mean['c_onsenv']:>8.2f} {mean['c_chroma']:>8.2f}"
          f" | {mean['miss_p']:>12.2f} {mean['miss_oe']:>7.2f} {mean['miss_ph']:>9.2f}")
    print("\nREAD: c_perc > c_harm => p_onset is percussion-biased. On melodic-miss frames, p_onset z should be NEGATIVE "
          "(it rests the piano solo); onsEnv / perc+harm should be LESS negative (more balanced) = better breathing "
          "energy. Pick the signal with the least-negative miss that still tracks overall structure.")


if __name__ == "__main__":
    main()

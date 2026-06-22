#!/usr/bin/env python3
"""
Would a global-structure (self-similarity) feature raise the SONG/SECTION-level chaos signal? (the ceiling
from diag_song_chaos.py: the model is strong frame-locally -- 16th AUC 0.742 -- but only ~0.4 correlated at
the song level, i.e. it doesn't know WHICH sections deserve chaos. Root suspected = frame-local features /
shallow receptive field, = [[h5]] global structure.)

Cheap offline feature-informativeness probe (NO retrain). At the window/section level (sliding windows
across songs), does each candidate feature predict the window's real 16th density, and -- crucially -- does
it ADD predictive power over the model's existing signal? Three candidates, so the result says WHICH fix:
  model_p16 : mean model p_on at 16th frames in the window      (the current ~0.4 signal, narrow context)
  busy_wide : mean high-res onset over a 3x-WIDER context        (existing feature, just integrated wider
              -> if this adds R^2, the fix is RECEPTIVE FIELD, not a new feature)
  ssm_homog : within-window timbre/chroma self-similarity         (is this a homogeneous/sustained section)
  ssm_distinct: window-vs-song dissimilarity                      (is this section unusual/intense)
              -> if SSM adds R^2 over model_p16 (and over busy_wide), a self-similarity feature helps.

Reads (incremental R^2 of a linear fit, target = real 16th density per window):
  ssm adds R^2 over model_p16+busy_wide  -> self-similarity carries chaos info the model can't see -> add it.
  busy_wide adds but ssm doesn't          -> it's a context/receptive-field problem, not structure.
  nothing beats model_p16                 -> section chaos isn't predictable from these audio features.

  python experiments/generation_typed/diag_selfsim_chaos.py --num_songs 60 --win 96 --stride 48
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
TIMBRE_COLS = list(range(13)) + list(range(23, 35))   # MFCC (0:13) + chroma (23:35)
HIGHRES_COL = 41


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0 or len(a) < 3:
        return float('nan')
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def r2(X, y):
    """R^2 of OLS y ~ [1, X] (X: (n,k))."""
    X = np.atleast_2d(X);
    if X.shape[0] != len(y):
        X = X.T
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ beta
    ss_res = ((y - pred) ** 2).sum(); ss_tot = ((y - y.mean()) ** 2).sum()
    return 1 - ss_res / max(ss_tot, 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_songs', type=int, default=60)
    ap.add_argument('--max_len', type=int, default=1024)
    ap.add_argument('--min_difficulty', type=int, default=2)
    ap.add_argument('--win', type=int, default=96, help='window length in 16th frames (~6 measures)')
    ap.add_argument('--stride', type=int, default=48)
    args = ap.parse_args()
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

    rows = []  # per-window feature dict
    used = 0
    for i in range(len(ds.valid_samples)):
        if used >= args.num_songs:
            break
        meta = ds.valid_samples[i]
        if meta['difficulty_class'] < args.min_difficulty:
            continue
        sample = ds[i]; T = min(int(sample['mask'].sum().item()), args.max_len)
        if T < args.win + args.stride:
            continue
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None:
            continue
        orig = np.asarray(ds.parser.convert_to_tensor_typed(meta['chart'], nd))[:T]
        note = (orig != 0).any(1)
        if note.sum() < 32:
            continue
        audio = sample['audio'][:T, :AD].numpy()
        diff = torch.tensor([meta['difficulty_class']], device=device)
        with torch.no_grad():
            p_on = torch.sigmoid(m.onset_logits(m.encode_audio(
                torch.from_numpy(audio).unsqueeze(0).to(device)), diff))[0].cpu().numpy()
        t = np.arange(T); is16 = (t % 4 == 1) | (t % 4 == 3)
        # normalized timbre frames (for self-similarity) and song-mean timbre
        tim = audio[:, TIMBRE_COLS]
        tim = tim / (np.linalg.norm(tim, axis=1, keepdims=True) + 1e-8)
        song_mean = tim.mean(0); song_mean /= (np.linalg.norm(song_mean) + 1e-8)
        hr = audio[:, HIGHRES_COL]
        for w in range(0, T - args.win + 1, args.stride):
            sl = slice(w, w + args.win); ww = note[sl]
            if ww.sum() < 4:
                continue
            real16 = (note[sl] & is16[sl]).sum() / args.win               # target: local chaos density
            model_p16 = p_on[sl][is16[sl]].mean() if is16[sl].any() else 0.0
            busy_local = hr[sl].mean()
            wlo, whi = max(0, w - args.win), min(T, w + 2 * args.win)      # 3x-wider context
            busy_wide = hr[wlo:whi].mean()
            sub = tim[sl]
            sim = sub @ sub.T
            homog = (sim.sum() - np.trace(sim)) / max(len(sub) * (len(sub) - 1), 1)  # within-window self-sim
            distinct = 1.0 - float(sub.mean(0) @ song_mean)               # window-vs-song dissimilarity
            rows.append((real16, model_p16, busy_local, busy_wide, homog, distinct))
        used += 1

    R = np.array(rows)
    y = R[:, 0]
    feats = {'model_p16': R[:, 1], 'busy_local': R[:, 2], 'busy_wide': R[:, 3],
             'ssm_homog': R[:, 4], 'ssm_distinct': R[:, 5]}
    print(f"\n=== Self-similarity probe: predict per-window 16th density ({used} songs, {len(R)} windows, "
          f"win={args.win} stride={args.stride}) ===")
    print(f"  target real 16th density: mean {y.mean()*100:.2f}%  std {y.std()*100:.2f}%")
    print(f"\n  feature          Spearman(.,real16)")
    for k, v in feats.items():
        print(f"  {k:<14} {spearman(v, y):>10.3f}")
    base = ['model_p16']
    print(f"\n  incremental R^2 (linear fit, target=real16 density):")
    print(f"  {'model_p16':<34} {r2(np.column_stack([feats[c] for c in base]), y):>7.3f}")
    for add in ('busy_wide', 'ssm_homog', 'ssm_distinct'):
        cols = base + [add]
        print(f"  {'model_p16 + ' + add:<34} {r2(np.column_stack([feats[c] for c in cols]), y):>7.3f}")
    full = ['model_p16', 'busy_wide', 'ssm_homog', 'ssm_distinct']
    print(f"  {'model_p16 + busy_wide':<34} {r2(np.column_stack([feats['model_p16'], feats['busy_wide']]), y):>7.3f}")
    print(f"  {'+ ssm_homog + ssm_distinct (all)':<34} {r2(np.column_stack([feats[c] for c in full]), y):>7.3f}")
    print("\n  ssm adds R^2 over model_p16+busy_wide -> structure feature helps. busy_wide adds but ssm")
    print("  doesn't -> it's a receptive-field problem (widen context). nothing beats model_p16 -> section")
    print("  chaos not predictable from these features.")


if __name__ == '__main__':
    main()

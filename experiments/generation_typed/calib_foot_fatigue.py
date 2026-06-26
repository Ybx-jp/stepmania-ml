#!/usr/bin/env python3
"""
CALIBRATE the per-foot fatigue governor on the EGREGIOUS rich-Hard set (where jump/jack streams actually blow
up — the 8-song val set is too mild). Targets the REAL source charts of the SAME songs (the human reference),
sweeping fatigue_penalty (lambda) x fatigue_free. Reports each config's stream/jump/jack/density vs REAL + a
distance-to-real, and flags the closest config.

  python experiments/generation_typed/calib_foot_fatigue.py [--songs 8] [--lams 0 2 3 4] [--frees 4 6 8]
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
from src.data.song_selection import select_by_groove
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.typed import pair_holds
from src.generation.radar_manifold import RadarManifold
from src.generation.playtest_export import enforce_playability


def metrics(typed):
    """(jump%, max-jump-stream, max-jack-run, jack>=4 share)."""
    onset = [(t, [k for k in range(4) if typed[t, k] in (1, 2, 4)]) for t in range(typed.shape[0])]
    onset = [(t, a) for t, a in onset if a]
    n = max(len(onset), 1); jumps = [(t, a) for t, a in onset if len(a) >= 2]
    js = sorted(t for t, a in jumps); mjs = run = 0
    for i in range(len(js)):
        run = run + 1 if i and js[i] - js[i - 1] <= 4 else 1; mjs = max(mjs, run)
    sg = [(t, a[0]) for t, a in onset if len(a) == 1]; runs = []; i = 0
    while i < len(sg):
        j = i
        while j + 1 < len(sg) and sg[j + 1][1] == sg[i][1] and sg[j + 1][0] - sg[j][0] <= 4:
            j += 1
        runs.append(j - i + 1); i = j + 1
    mjk = max(runs, default=0); ge4 = sum(1 for r in runs if r >= 4) / max(len(runs), 1)
    return len(jumps) / n, mjs, mjk, ge4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/gen_motif_full_fixed/best_val.pt")
    ap.add_argument("--songs", type=int, default=8); ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 2.0, 3.0, 4.0])
    ap.add_argument("--frees", type=float, nargs="+", default=[4.0, 6.0, 8.0])
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    tf, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                   use_metric_phase=True, use_highres_onset=True))
    _, vds, _ = create_datasets(train_files=tf, val_files=vf, test_files=[], audio_dir="data/",
                                max_sequence_length=msl, feature_extractor=ext, cache_dir='cache/samples_v3')
    vds.warm_cache(show_progress=False)
    order = select_by_groove(vds, n=args.songs, by='rich', difficulty=3)   # rich, Hard = the egregious set
    songs = []
    for i in order[:args.songs]:
        s = vds[i]; m = vds.valid_samples[i]; T = min(int(s['mask'].sum().item()), args.max_len)
        nd = next((nn for nn in m['chart'].note_data if nn.difficulty_name == m['difficulty_name']), None)
        real = vds.parser.convert_to_tensor_typed(m['chart'], nd)[:T] if nd is not None else None
        songs.append({'audio': s['audio'][:T].numpy().astype(np.float32), 'len': T,
                      'diff': int(m['difficulty_class']), 'bpm': float(m['chart'].bpm), 'real': real})
    audio_dim = songs[0]['audio'].shape[1]
    model = LayeredTypedChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)['model_state_dict']); model.eval()
    manifold = RadarManifold.load(PROJECT_ROOT / "cache/radar_manifold.npz")
    base_density = {d: manifold.target_density(manifold._bucket(d).mean(0), d) for d in range(4)}

    # REAL reference (the human source charts of these songs)
    rj, rmjs, rmjk, rge4 = np.mean([[*metrics(s['real'])] for s in songs if s['real'] is not None], 0)
    rdens = np.mean([float((s['real'] != 0).any(1).mean()) for s in songs if s['real'] is not None])
    print(f"\nFATIGUE CALIBRATION  [{args.ckpt}]  {len(songs)} rich-Hard songs (the egregious set)")
    print(f"REAL  target:  jump% {100*rj:5.1f}  maxJumpStream {rmjs:4.1f}  maxJackRun {rmjk:4.1f}  jack>=4 {100*rge4:4.1f}%  dens {rdens:.3f}")
    print(f"  {'lam':>4} {'free':>4} {'jump%':>6} {'mJumpStrm':>10} {'mJackRun':>9} {'jack>=4%':>9} {'dens':>6} {'dist→real':>10}")
    print("  " + "-" * 70)

    def gen_cfg(lam, free):
        set_seed(42)  # SAME random draws for every config -> the only difference is the knob (comparable)
        J, MJS, MJK, GE4, D = [], [], [], [], []
        for s in songs:
            T = s['len']; audio = torch.from_numpy(s['audio']).unsqueeze(0).to(device)
            diff = torch.tensor([s['diff']], device=device)
            with torch.no_grad():
                p = torch.sigmoid(model.onset_logits(model.encode_audio(audio), diff)[0]).cpu().numpy()
            tgt = base_density[s['diff']]; tau = float(np.quantile(p, 1 - tgt)) if tgt and tgt > 0 else 0.5
            gk = dict(onset_threshold=tau, type_sample=True, type_temperature=0.4, pattern_sample=True,
                      pattern_temperature=0.7, max_jack_run=2, bpm=s['bpm'],
                      fatigue_penalty=(lam if lam > 0 else None), fatigue_free=free)
            enforce_playability(gk)
            with torch.no_grad():
                g = pair_holds(model.generate(audio, diff, lengths=torch.tensor([T], device=device), **gk)[0].cpu().numpy())
            j, a, b, e = metrics(g); J.append(j); MJS.append(a); MJK.append(b); GE4.append(e)
            D.append(float((g != 0).any(1).mean()))
        return np.mean(J), np.mean(MJS), np.mean(MJK), np.mean(GE4), np.mean(D)

    def row(lam, free):
        j, mjs, mjk, ge4, d = gen_cfg(lam, free)
        dist = abs(j - rj) / max(rj, .05) + abs(mjs - rmjs) / max(rmjs, 1) + abs(mjk - rmjk) / max(rmjk, 1) + abs(ge4 - rge4)
        print(f"  {('OFF' if lam==0 else f'{lam:.1f}'):>4} {free:>4.0f} {100*j:>6.1f} {mjs:>10.1f} {mjk:>9.1f} "
              f"{100*ge4:>9.1f} {d:>6.3f} {dist:>10.2f}")
        return dist
    row(0.0, args.frees[0])                         # OFF once (free has no effect with fatigue off)
    best = None
    for free in args.frees:
        for lam in [l for l in args.lams if l > 0]:
            dist = row(lam, free)
            if best is None or dist < best[0]:
                best = (dist, lam, free)
    if best:
        print(f"\nCLOSEST to real (jump%+streams+jacks): fatigue_penalty={best[1]} fatigue_free={best[2]}  (dist {best[0]:.2f})")
    print("READ: pick the config matching REAL jump%/streams with density held; that's the calibration default to A/B.")


if __name__ == "__main__":
    main()

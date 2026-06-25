#!/usr/bin/env python3
"""
Validate the PER-FOOT FATIGUE governor (typed_model.generate fatigue_penalty) on the real model: does it reduce
BOTH long jack runs AND consecutive-jump streams WITHOUT exploding jump rate (the load-sharing-bias risk: a jump
spreads cost over two feet, so a max-gated per-foot penalty could prefer jumps)? Sweeps fatigue_penalty.

  python experiments/generation_typed/diag_foot_fatigue.py [--songs 8] [--lams 0 2 4]
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
from src.generation.typed import pair_holds
from src.generation.radar_manifold import RadarManifold
from src.generation.playtest_export import enforce_playability


def metrics(typed):
    onset = [(t, [k for k in range(4) if typed[t, k] in (1, 2, 4)]) for t in range(typed.shape[0])]
    onset = [(t, a) for t, a in onset if a]
    n = len(onset); jumps = [(t, a) for t, a in onset if len(a) >= 2]
    # longest consecutive-jump stream (spacing<=4)
    js = sorted(t for t, a in jumps); mjs = run = 0
    for i in range(len(js)):
        run = run + 1 if i and js[i] - js[i - 1] <= 4 else 1
        mjs = max(mjs, run)
    # longest same-panel single jack run (<=4 spacing)
    sg = [(t, a[0]) for t, a in onset if len(a) == 1]; mjk = run = 0; i = 0
    while i < len(sg):
        j = i
        while j + 1 < len(sg) and sg[j + 1][1] == sg[i][1] and sg[j + 1][0] - sg[j][0] <= 4:
            j += 1
        mjk = max(mjk, j - i + 1); i = j + 1
    return n, len(jumps), mjs, mjk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/gen_motif_full_fixed/best_val.pt")
    ap.add_argument("--songs", type=int, default=8); ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 2.0, 4.0])
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
    songs = []
    for i in range(len(vds)):
        if len(songs) >= args.songs:
            break
        s = vds[i]; m = vds.valid_samples[i]; T = min(int(s['mask'].sum().item()), args.max_len)
        songs.append({'audio': s['audio'][:T].numpy().astype(np.float32), 'len': T,
                      'diff': int(m['difficulty_class']), 'bpm': float(m['chart'].bpm)})
    audio_dim = songs[0]['audio'].shape[1]
    model = LayeredTypedChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)['model_state_dict']); model.eval()
    manifold = RadarManifold.load(PROJECT_ROOT / "cache/radar_manifold.npz")
    base_density = {d: manifold.target_density(manifold._bucket(d).mean(0), d) for d in range(4)}

    print(f"\nPER-FOOT FATIGUE sweep  [{args.ckpt}]  {len(songs)} songs (max_jack_run=2, jack_penalty OFF)")
    print(f"  {'fatigue':>8} {'jump%':>7} {'maxJumpStream':>14} {'maxJackRun':>11} {'density':>8}")
    print("  " + "-" * 56)
    for lam in args.lams:
        N = J = 0; mjs = mjk = 0; dens = []
        for s in songs:
            T = s['len']; audio = torch.from_numpy(s['audio']).unsqueeze(0).to(device)
            diff = torch.tensor([s['diff']], device=device)
            with torch.no_grad():
                p = torch.sigmoid(model.onset_logits(model.encode_audio(audio), diff)[0]).cpu().numpy()
            tgt = base_density[s['diff']]; tau = float(np.quantile(p, 1 - tgt)) if tgt and tgt > 0 else 0.5
            gk = dict(onset_threshold=tau, type_sample=True, type_temperature=0.4, pattern_sample=True,
                      pattern_temperature=0.7, max_jack_run=2, bpm=s['bpm'],
                      fatigue_penalty=(lam if lam > 0 else None))
            enforce_playability(gk)
            with torch.no_grad():
                g = pair_holds(model.generate(audio, diff, lengths=torch.tensor([T], device=device), **gk)[0].cpu().numpy())
            n, j, a, b = metrics(g); N += n; J += j; mjs = max(mjs, a); mjk = max(mjk, b)
            dens.append(float((g != 0).any(1).mean()))
        print(f"  {('OFF' if lam==0 else f'{lam:.1f}'):>8} {100*J/max(N,1):>7.1f} {mjs:>14} {mjk:>11} {np.mean(dens):>8.3f}")
    print("\nREAD: maxJumpStream AND maxJackRun should fall; jump% should NOT explode (if it does -> load-sharing "
          "bias, switch max(E) gating to sum(E)). density held = re-routing not deletion.")


if __name__ == "__main__":
    main()

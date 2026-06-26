#!/usr/bin/env python3
"""
Validate the STAGE-2 STAMINA governor (typed_model.generate stamina_ceiling): a per-region DENSITY governor that
thins UPCOMING onset density only where sustained workload is high -- a CEILING, never a global dent. The per-note
foot governor (fatigue_penalty) can only re-route load across feet; stamina is the layer that removes notes.

The decisive test is the DENSITY PROFILE, not the mean. Stamina should shave the SUSTAINED-DENSE peaks (4-measure
windows in the top decile) while leaving the moderate/rest windows ~unchanged. A governor that dents the rest
windows too is just a global density cut (the Stage-1 mistake) and FAILS.

  python experiments/generation_typed/diag_stamina.py [--songs 8] [--ceilings 200 120 80] [--win 64]

READ: peakΔ should be clearly negative and carry the bulk of the change; restΔ should be ~0. maxJackRun should
not regress. As the ceiling drops, peak thinning should grow first; if rest thins too, the ceiling is too low.
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


def max_jack_run(typed):
    onset = [(t, [k for k in range(4) if typed[t, k] in (1, 2, 4)]) for t in range(typed.shape[0])]
    onset = [(t, a) for t, a in onset if a]
    sg = [(t, a[0]) for t, a in onset if len(a) == 1]; mjk = 0; i = 0
    while i < len(sg):
        j = i
        while j + 1 < len(sg) and sg[j + 1][1] == sg[i][1] and sg[j + 1][0] - sg[j][0] <= 4:
            j += 1
        mjk = max(mjk, j - i + 1); i = j + 1
    return mjk


def window_density(occ, win):
    """occ: (T,) bool of 'frame has a note'. Returns per-window densities over non-overlapping `win`-frame windows."""
    T = len(occ); nwin = T // win
    if nwin == 0:
        return np.array([occ.mean()])
    return occ[:nwin * win].reshape(nwin, win).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/gen_motif_full_fixed/best_val.pt")
    ap.add_argument("--songs", type=int, default=8); ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--ceilings", type=float, nargs="+", default=[200.0, 120.0, 80.0, 50.0])
    ap.add_argument("--win", type=int, default=64, help="density-profile window in 16th frames (64 = 4 measures)")
    ap.add_argument("--fatigue", type=float, default=2.0, help="per-note fatigue_penalty (foot model; stamina needs it)")
    ap.add_argument("--stamina_tau", type=float, default=8.0); ap.add_argument("--stamina_scale", type=float, default=60.0)
    ap.add_argument("--stamina_max_bump", type=float, default=0.45)
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

    def run(ceiling):
        """Returns per-song occupancy arrays + maxJackRun + total onsets (so peak/rest masks can be FIXED
        from the baseline run for a paired before/after comparison)."""
        occs, mjk, N = [], 0, 0
        for s in songs:
            set_seed(42)  # same sampling draws across ceilings -> a clean A/B (only the gate differs)
            T = s['len']; audio = torch.from_numpy(s['audio']).unsqueeze(0).to(device)
            diff = torch.tensor([s['diff']], device=device)
            with torch.no_grad():
                p = torch.sigmoid(model.onset_logits(model.encode_audio(audio), diff)[0]).cpu().numpy()
            tgt = base_density[s['diff']]; tau = float(np.quantile(p, 1 - tgt)) if tgt and tgt > 0 else 0.5
            gk = dict(onset_threshold=tau, type_sample=True, type_temperature=0.4, pattern_sample=True,
                      pattern_temperature=0.7, max_jack_run=2, bpm=s['bpm'], fatigue_penalty=args.fatigue,
                      stamina_ceiling=(ceiling if ceiling is not None else None), stamina_tau=args.stamina_tau,
                      stamina_scale=args.stamina_scale, stamina_max_bump=args.stamina_max_bump)
            enforce_playability(gk)
            with torch.no_grad():
                g = pair_holds(model.generate(audio, diff, lengths=torch.tensor([T], device=device), **gk)[0].cpu().numpy())
            occs.append((g != 0).any(1)); mjk = max(mjk, max_jack_run(g)); N += int((g != 0).any(1).sum())
        return occs, mjk, N

    # baseline (stamina OFF) defines the FIXED peak/rest window identity per song
    base_occ, mjk0, N0 = run(None)
    base_wd = [window_density(o, args.win) for o in base_occ]
    peak_mask = [wd >= np.quantile(wd, 0.9) for wd in base_wd]          # which 4-measure windows are sustained-dense
    rest_mask = [wd <= np.median(wd) for wd in base_wd]                 # which are moderate/sparse
    o0 = float(np.mean([o.mean() for o in base_occ]))
    pk0 = float(np.mean([wd[m].mean() for wd, m in zip(base_wd, peak_mask)]))
    rt0 = float(np.mean([wd[m].mean() for wd, m in zip(base_wd, rest_mask)]))

    print(f"\nSTAGE-2 STAMINA sweep  [{args.ckpt}]  {len(songs)} songs  win={args.win}f (~{args.win//16} measures)")
    print(f"  fatigue={args.fatigue}  tau={args.stamina_tau}b  scale={args.stamina_scale}  max_bump={args.stamina_max_bump}")
    print(f"  peak/rest windows FIXED from baseline (paired before/after on the SAME time-windows)")
    print(f"  {'ceiling':>8} {'overall':>8} {'peakWin':>8} {'restWin':>8} {'maxJack':>8} {'onsets':>7}  {'peakΔ':>7} {'restΔ':>7}")
    print("  " + "-" * 78)
    print(f"  {'OFF':>8} {o0:>8.3f} {pk0:>8.3f} {rt0:>8.3f} {mjk0:>8} {N0:>7}  {'--':>7} {'--':>7}")
    for c in args.ceilings:
        occs, mjk, N = run(c)
        wds = [window_density(o, args.win) for o in occs]
        o = float(np.mean([oc.mean() for oc in occs]))
        pk = float(np.mean([wd[m].mean() for wd, m in zip(wds, peak_mask)]))   # same windows that were peaks at baseline
        rt = float(np.mean([wd[m].mean() for wd, m in zip(wds, rest_mask)]))
        print(f"  {c:>8.0f} {o:>8.3f} {pk:>8.3f} {rt:>8.3f} {mjk:>8} {N:>7}  {pk-pk0:>+7.3f} {rt-rt0:>+7.3f}")
    print("\nREAD: a working CEILING shaves peakWin (peakΔ << 0) while restWin holds (restΔ ~ 0). If restΔ is also "
          "strongly negative, the ceiling is too low / scale too aggressive -> it's a global cut, not a governor.")


if __name__ == "__main__":
    main()

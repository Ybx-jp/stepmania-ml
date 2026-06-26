#!/usr/bin/env python3
"""
Validate the STAGE-3 ARC (typed_model.generate stamina_breathe): the stamina ceiling BREATHES with a phrase-smoothed
audio-energy envelope (high at climaxes, low in verses), so the onset thins the verses MORE than the climaxes -> a
difficulty ARC (the H5 fix: generated density is otherwise structurally flat). Ceiling-only, no lower bound.

DECISIVE TEST: does the per-window output density track the audio-energy envelope MORE under breathing than under a
flat ceiling (or OFF)? Measured as corr(window_density, window_energy) and the climax-vs-verse density CONTRAST
(top-tercile-energy windows minus bottom-tercile). Overall density should stay ~constant (REDISTRIBUTION, not a
global cut) — the arc is in the SHAPE, not the mean.

  python experiments/generation_typed/diag_stamina_arc.py [--songs 10] [--ceiling 25] [--breathe 0.7]

READ: breathe should RAISE corr + contrast vs flat (the density develops an arc) while overall density holds.
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


def win_mean(x, win):
    T = len(x); nwin = T // win
    if nwin == 0:
        return np.array([x.mean()])
    return x[:nwin * win].reshape(nwin, win).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/gen_motif_full_fixed/best_val.pt")
    ap.add_argument("--songs", type=int, default=10); ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--ceiling", type=float, default=25.0); ap.add_argument("--breathe", type=float, default=1.2)
    ap.add_argument("--breathe_win", type=int, default=96); ap.add_argument("--win", type=int, default=48)
    ap.add_argument("--fatigue", type=float, default=2.0)
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
        if int(m['difficulty_class']) != 3:        # Hard only (richest structure)
            continue
        songs.append({'audio': s['audio'][:T].numpy().astype(np.float32), 'len': T,
                      'diff': int(m['difficulty_class']), 'bpm': float(m['chart'].bpm)})
    audio_dim = songs[0]['audio'].shape[1]
    model = LayeredTypedChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)['model_state_dict']); model.eval()
    manifold = RadarManifold.load(PROJECT_ROOT / "cache/radar_manifold.npz")
    base_density = {d: manifold.target_density(manifold._bucket(d).mean(0), d) for d in range(4)}

    def energy_env(p, T):
        """phrase-smoothed, z-normalized p_onset over valid T -- the SAME signal the breathing ceiling uses."""
        w = args.breathe_win
        env = torch.nn.functional.avg_pool1d(torch.from_numpy(p[:T]).view(1, 1, -1), 2 * w + 1, 1, w,
                                             count_include_pad=False).view(-1).numpy()
        return (env - env.mean()) / (env.std() + 1e-6)

    def run(ceiling, breathe, floor=0.4):
        corrs, contrasts, dens, tails = [], [], [], []
        for s in songs:
            set_seed(42)
            T = s['len']; audio = torch.from_numpy(s['audio']).unsqueeze(0).to(device)
            diff = torch.tensor([s['diff']], device=device)
            with torch.no_grad():
                p = torch.sigmoid(model.onset_logits(model.encode_audio(audio), diff)[0]).cpu().numpy()
            tgt = base_density[s['diff']]; tau = float(np.quantile(p, 1 - tgt)) if tgt and tgt > 0 else 0.5
            gk = dict(onset_threshold=tau, type_sample=True, type_temperature=0.4, pattern_sample=True,
                      pattern_temperature=0.7, max_jack_run=2, bpm=s['bpm'], fatigue_penalty=args.fatigue,
                      stamina_ceiling=(ceiling if ceiling is not None else None),
                      stamina_breathe=breathe, stamina_breathe_win=args.breathe_win, stamina_breathe_floor=floor)
            enforce_playability(gk)
            with torch.no_grad():
                g = pair_holds(model.generate(audio, diff, lengths=torch.tensor([T], device=device), **gk)[0].cpu().numpy())
            occ = (g != 0).any(1).astype(float)
            env = energy_env(p, T)
            wd = win_mean(occ, args.win); we = win_mean(env, args.win)
            n = min(len(wd), len(we)); wd, we = wd[:n], we[:n]
            if n >= 4 and wd.std() > 1e-6:
                corrs.append(float(np.corrcoef(wd, we)[0, 1]))
                hi = wd[we >= np.quantile(we, 2 / 3)].mean(); lo = wd[we <= np.quantile(we, 1 / 3)].mean()
                contrasts.append(hi - lo)
            dens.append(float(occ.mean()))
            tails.append(occ[-32:].mean() / max(occ.mean(), 1e-6))   # last-2-measures density / song mean (END-FADE)
        return float(np.mean(corrs)), float(np.mean(contrasts)), float(np.mean(dens)), float(np.mean(tails))

    print(f"\nSTAGE-3 ARC  [{args.ckpt}]  {len(songs)} Hard songs  win={args.win}f  breathe_win={args.breathe_win}f")
    print(f"  energy = phrase-smoothed z(p_onset); arc = density tracking energy; tail = last-2-meas dens / song mean")
    print(f"  {'condition':>26} {'corr(dens,energy)':>18} {'climax-verse Δ':>15} {'overall_dens':>13} {'tail/mean':>10}")
    print("  " + "-" * 86)
    conds = [("OFF (no stamina)", None, 0.0, 0.4), (f"flat ceiling={args.ceiling:.0f}", args.ceiling, 0.0, 0.4),
             (f"breathe={args.breathe:.1f} floor=0.0", args.ceiling, args.breathe, 0.0),   # the OLD (no-floor) bug
             (f"breathe={args.breathe:.1f} floor=0.4", args.ceiling, args.breathe, 0.4)]   # the FIX
    for name, ceil, br, fl in conds:
        c, ct, d, tl = run(ceil, br, fl)
        print(f"  {name:>26} {c:>18.3f} {ct:>+15.3f} {d:>13.3f} {tl:>10.2f}")
    print("\nREAD: breathe should RAISE corr + climax-verse Δ vs flat (the ARC), overall_dens ~held (redistribution). "
          "tail/mean ~1 = ending intact; << 1 = EMPTY tail (abrupt ending). floor=0.0 should show the tail bug; "
          "floor=0.4 should fix the tail while keeping most of the arc.")


if __name__ == "__main__":
    main()

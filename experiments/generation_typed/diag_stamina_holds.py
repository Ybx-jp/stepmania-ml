#!/usr/bin/env python3
"""
HOLDS-SPECIFIC test of the Stage-2 STAMINA governor. The ORIGINAL motivation for stamina (notes/
foot_fatigue_design.md) was holds-blindness: a sustained one-foot grind during a hold (the other foot pinned) is
brutal, and the per-note FOOTING governor can only re-route load across feet -- it can never REMOVE notes, so
hold-streams collapse to jacks. Stamina is supposed to be the relief valve (remove notes).

DECISIVE QUESTION (controls for density): among 4-measure windows of SIMILAR baseline density, does stamina thin
the HOLD-OPEN windows MORE than the NO-HOLD windows?
  - MORE  -> genuinely holds-aware relief (the motivation is met).
  - EQUAL -> density-general relief only (still useful, but the foot model not pinning the held foot means stamina
             responds to density, not to the pin -- holds-blindness only partly addressed).
CAVEAT baked in: the foot model does NOT pin the held foot (hold-pinning was reverted), so E_slow during a hold is
the unconstrained 2-foot cost, not the real one-foot-grind cost. This diag measures what actually happens.

  python experiments/generation_typed/diag_stamina_holds.py [--songs 12] [--ceiling 25]

READ: holdΔ (dense hold windows) vs noholdΔ (dense no-hold windows). holdΔ noticeably more negative = holds-aware.
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


def hold_open_mask(g):
    """(T,) bool: a panel is pinned by an OPEN hold (strictly between head 2/4 and tail 3) at frame t."""
    T, P = g.shape; pinned = np.zeros(T, dtype=bool)
    for p in range(P):
        open_head = -1
        for t in range(T):
            s = g[t, p]
            if s in (2, 4):
                open_head = t
            elif s == 3:
                if open_head >= 0:
                    pinned[open_head + 1:t] = True      # frames between head and tail = foot is pinned
                open_head = -1
    return pinned


def max_jack_run_within(g, frame_mask):
    """longest fast (<=4-spacing) same-single-panel run restricted to frames where frame_mask is True."""
    onset = [(t, [k for k in range(4) if g[t, k] in (1, 2, 4)]) for t in range(g.shape[0])]
    sg = [(t, a[0]) for t, a in onset if len(a) == 1 and frame_mask[t]]
    mjk = 0; i = 0
    while i < len(sg):
        j = i
        while j + 1 < len(sg) and sg[j + 1][1] == sg[i][1] and sg[j + 1][0] - sg[j][0] <= 4:
            j += 1
        mjk = max(mjk, j - i + 1); i = j + 1
    return mjk


def win_mean(x, win):
    T = len(x); nwin = T // win
    if nwin == 0:
        return np.array([x.mean()])
    return x[:nwin * win].reshape(nwin, win).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/gen_motif_full_fixed/best_val.pt")
    ap.add_argument("--songs", type=int, default=12); ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--ceiling", type=float, default=25.0); ap.add_argument("--win", type=int, default=64)
    ap.add_argument("--hold_frac", type=float, default=0.15, help="window is 'hold' if >=this frac of frames pinned")
    ap.add_argument("--fatigue", type=float, default=2.0)
    ap.add_argument("--stamina_tau", type=float, default=8.0); ap.add_argument("--stamina_scale", type=float, default=15.0)
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
        out = []
        for s in songs:
            set_seed(42)
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
            press = np.isin(g, (1, 2, 4)).any(1)               # onset density
            pin = hold_open_mask(g)
            out.append({'press': press, 'pin': pin, 'g': g})
        return out

    base = run(None); on = run(args.ceiling)
    win = args.win
    # classify baseline windows by density (median split) and hold-openness
    bd = [win_mean(b['press'], win) for b in base]; bh = [win_mean(b['pin'], win) for b in base]
    od = [win_mean(o['press'], win) for o in on]
    med = np.median(np.concatenate(bd))
    hold_dense, nohold_dense = [], []                          # (baseline_dens, on_dens) pairs for dense windows
    for dB, hB, dO in zip(bd, bh, od):
        n = min(len(dB), len(dO))
        for w in range(n):
            if dB[w] < med:                                    # control: only dense windows
                continue
            (hold_dense if hB[w] >= args.hold_frac else nohold_dense).append((dB[w], dO[w]))
    def summarize(pairs):
        if not pairs:
            return (0, float('nan'), float('nan'), float('nan'))
        a = np.array(pairs); return (len(a), a[:, 0].mean(), a[:, 1].mean(), (a[:, 1] - a[:, 0]).mean())

    nh, h0, h1, hd = summarize(hold_dense)
    nn, n0, n1, nd = summarize(nohold_dense)
    # secondary: during-hold one-foot-grind jacking, and press density on pinned frames, OFF vs ON
    pin_frames = sum(int(b['pin'].sum()) for b in base)
    mjk_hold_off = max(max_jack_run_within(b['g'], b['pin']) for b in base)
    mjk_hold_on = max(max_jack_run_within(o['g'], o['pin']) for o in on)
    def pin_press_density(runs):
        num = sum(int((np.isin(r['g'], (1, 2, 4)).any(1) & r['pin']).sum()) for r in runs)
        den = sum(int(r['pin'].sum()) for r in runs); return num / max(den, 1)

    print(f"\nSTAGE-2 STAMINA — HOLDS-SPECIFIC test  [{args.ckpt}]  {len(songs)} songs  ceiling={args.ceiling}")
    print(f"  win={win}f  hold-window if >= {args.hold_frac:.0%} frames pinned  | total pinned frames (OFF) = {pin_frames}")
    print(f"  DENSE windows only (baseline density >= median {med:.3f}); paired OFF->ON density on SAME windows")
    print(f"  {'group':>14} {'nwin':>5} {'densOFF':>8} {'densON':>8} {'Δ':>8}")
    print("  " + "-" * 50)
    print(f"  {'HOLD-open':>14} {nh:>5} {h0:>8.3f} {h1:>8.3f} {hd:>+8.3f}")
    print(f"  {'no-hold':>14} {nn:>5} {n0:>8.3f} {n1:>8.3f} {nd:>+8.3f}")
    print(f"\n  during-hold (pinned frames):  press-density OFF {pin_press_density(base):.3f} -> ON {pin_press_density(on):.3f}"
          f"   | maxJackRun-in-holds OFF {mjk_hold_off} -> ON {mjk_hold_on}")
    print("\nREAD: holdΔ noticeably more negative than noholdΔ => stamina specifically relieves hold-grind sections "
          "(motivation met). Roughly equal => density-general relief only (foot model not pinning the held foot; "
          "holds-blindness still needs pin-aware cost). during-hold press-density drop + jack-in-holds drop = bonus.")


if __name__ == "__main__":
    main()

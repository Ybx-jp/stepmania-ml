#!/usr/bin/env python3
"""
H15 LOCAL-motif VARYING-SCHEDULE eval — the per-section-control payoff (notes/h15_local_motif_plan.md #3).

The constant-schedule steerability eval (eval_motif.py) sets ONE motif target for the whole chart, so it can't
tell global-style conditioning apart from genuine LOCAL control. This eval sets a knob HIGH in some sections and
LOW in others WITHIN ONE chart (alternating per 64-frame section = the training section), generates, then
RE-ENCODES each section and checks whether the realized knob TRACKS the schedule section-by-section. This is the
thing global conditioning fundamentally CANNOT do.

Per knob/guidance, on the SAME songs+decode:
  - global_Δ  = realized-k(constant +z) - realized-k(constant -z)   [whole-chart steer ceiling]
  - local_Δ   = mean realized-k over +z SECTIONS - mean over -z SECTIONS  [the per-section contrast]
  - track_r   = corr( per-section realized-k , per-section target sign ) [does it follow the schedule?]
local_Δ > 0 and a healthy fraction of global_Δ  =>  local control is real.

  /home/ybx/miniconda3/envs/stepmania-chart-gen/bin/python experiments/generation_typed/eval_motif_schedule.py \
      --ckpt checkpoints/gen_motif_local2/best_val.pt --highres --knobs 3 10 0 --guidance 1 3 --songs 12
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
from src.generation.typed_model import LayeredTypedChartGenerator, MOTIF_DIM
from src.generation.typed import NUM_PANELS, pair_holds
from src.generation.evaluation import onset_density_metrics
from src.generation.motif_codebook import MotifBasis


def collect(ds, cap, n):
    out = []
    for i in range(len(ds)):
        if len(out) >= n:
            break
        sample = ds[i]; meta = ds.valid_samples[i]
        nd = next((nn for nn in meta['chart'].note_data if nn.difficulty_name == meta['difficulty_name']), None)
        if nd is None:
            continue
        typed = ds.parser.convert_to_tensor_typed(meta['chart'], nd)
        T = min(int(sample['mask'].sum().item()), cap, typed.shape[0])
        out.append({'audio': sample['audio'][:T].numpy().astype(np.float32), 'len': T,
                    'difficulty': int(meta['difficulty_class']),
                    'radar': meta['groove_radar'].to_vector().astype(np.float32),
                    'real': typed[:T].astype(np.int64)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/gen_motif_local2/best_val.pt")
    ap.add_argument("--highres", action="store_true")
    ap.add_argument("--knobs", type=int, nargs="+", default=[3, 10, 0])
    ap.add_argument("--z", type=float, default=3.0)
    ap.add_argument("--section", type=int, default=64)
    ap.add_argument("--guidance", type=float, nargs="+", default=[1.0, 3.0])
    ap.add_argument("--songs", type=int, default=12)
    ap.add_argument("--max_len", type=int, default=640)
    ap.add_argument("--pattern_temp", type=float, default=1.0)
    args = ap.parse_args()
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    basis = MotifBasis.load(PROJECT_ROOT / "cache/motif_basis.npz")
    S = args.section

    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    tf, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = (AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                    use_metric_phase=True, use_highres_onset=True))
           if args.highres else None)
    cache_dir = 'cache/samples_v3' if args.highres else 'cache/samples'
    _, val_ds, _ = create_datasets(train_files=tf, val_files=vf, test_files=[], audio_dir="data/",
                                   max_sequence_length=msl, feature_extractor=ext, cache_dir=cache_dir)
    val_ds.warm_cache(show_progress=False)
    songs = collect(val_ds, args.max_len, args.songs)
    audio_dim = songs[0]['audio'].shape[1]
    target_density = float(np.mean([(s['real'] != 0).any(1).mean() for s in songs]))

    model = LayeredTypedChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)['model_state_dict']); model.eval()

    def schedule(L, k, mode):
        """(L,K) per-frame schedule. mode 'plus'/'minus' = constant +z/-z on knob k; 'alt' = +z on even
        sections, -z on odd. Returns also per-section target sign for 'alt'."""
        sch = np.zeros((L, MOTIF_DIM), np.float32)
        n_sec = (L + S - 1) // S
        signs = np.zeros(n_sec, np.float32)
        for s in range(n_sec):
            if mode == 'plus':
                sgn = +1.0
            elif mode == 'minus':
                sgn = -1.0
            else:
                sgn = +1.0 if s % 2 == 0 else -1.0
            sch[s * S:(s + 1) * S, k] = sgn * args.z
            signs[s] = sgn
        return sch, signs

    def gen_one(s, k, mode, g):
        L = s['len']
        audio = torch.from_numpy(s['audio'][:L]).unsqueeze(0).to(device)
        diff = torch.tensor([s['difficulty']], device=device)
        radar = torch.from_numpy(s['radar']).unsqueeze(0).to(device)
        sch, signs = schedule(L, k, mode)
        mot = torch.from_numpy(sch).unsqueeze(0).to(device)
        lengths = torch.tensor([L], device=device)
        with torch.no_grad():
            ol = torch.sigmoid(model.onset_logits(model.encode_audio(audio), diff, radar=radar))  # onset: no motif (v2)
        tau = float(np.quantile(ol.cpu().numpy(), 1 - target_density))
        gen = model.generate(audio, diff, lengths=lengths, onset_threshold=tau, radar=radar, motif=mot,
                             guidance_scale=g, type_sample=True, type_temperature=0.4, hold_aware=True,
                             pattern_sample=True, pattern_temperature=args.pattern_temp, max_jack_run=1)[0].cpu().numpy()
        ch = pair_holds(gen[:L])
        # per-section realized knob-k
        n_sec = (L + S - 1) // S
        realized = np.array([basis.encode_chart(ch[i * S:(i + 1) * S], s['radar'])[k] for i in range(n_sec)])
        f1 = onset_density_metrics((ch != 0).astype(np.float32), reference=(s['real'][:L] != 0).astype(np.float32))['onset_f1']
        dens = (ch != 0).any(1).mean()
        return realized, signs, f1, dens

    print(f"varying-schedule control [{args.ckpt}] audio_dim={audio_dim}  section={S}  "
          f"{len(songs)} songs  real density {target_density:.3f}\n")
    for g in args.guidance:
        print("=" * 88); print(f"GUIDANCE g={g}"); print("=" * 88)
        for k in args.knobs:
            gpl = np.concatenate([gen_one(s, k, 'plus', g)[0] for s in songs])
            gmi = np.concatenate([gen_one(s, k, 'minus', g)[0] for s in songs])
            global_d = gpl.mean() - gmi.mean()
            # alternating: gather per-section realized + target sign
            pr, mr, rs, ts, f1s, dns = [], [], [], [], [], []
            for s in songs:
                realized, signs, f1, dens = gen_one(s, k, 'alt', g)
                pr += list(realized[signs > 0]); mr += list(realized[signs < 0])
                rs += list(realized); ts += list(signs); f1s.append(f1); dns.append(dens)
            local_d = np.mean(pr) - np.mean(mr)
            track_r = np.corrcoef(rs, ts)[0, 1] if len(set(ts)) > 1 else float('nan')
            frac = local_d / global_d if abs(global_d) > 1e-6 else float('nan')
            print(f"\nKNOB {k} '{basis.axis_info[k]['label']}'")
            print(f"  global_Δ (whole-chart +z vs -z): {global_d:+.2f}   "
                  f"local_Δ (per-section +z vs -z): {local_d:+.2f}   ({frac*100:.0f}% of global)")
            print(f"  per-section track corr(realized, target sign): {track_r:+.2f}   "
                  f"alt quality: onset_F1 {np.mean(f1s):.2f}  density {np.mean(dns):.3f} (real {target_density:.3f})")
    print("\nREAD: local_Δ > 0 and a healthy fraction of global_Δ, track_r > 0 => the model steers DIFFERENT "
          "SECTIONS differently (genuine local control, not just a global style shift).")


if __name__ == "__main__":
    main()

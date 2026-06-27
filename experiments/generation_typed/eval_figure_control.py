#!/usr/bin/env python3
"""
H15 HIERARCHICAL figure-token control eval — does setting a DISCRETE figure token make the model REALIZE that
figure? (Especially SWEEP, the axis the continuous knob couldn't isolate; notes/h15_local_motif_plan.md.)

Metric sees the PROPERTY (experiment-design Rule 1): set figure=token (constant), generate, RE-LABEL each
generated section's figure family (motif_codebook.figure_token), and read the FRACTION of sections that match
the target. Compare to the figure=null baseline (the model's natural rate) and to REAL charts (gate: sweep
~15.5%). A positive lift on the target figure = the discrete token steers that figure. motif=null so figure is
isolated; radar=real for density realism (onset is decoupled from figure, so density stays put).

  python experiments/generation_typed/eval_figure_control.py \
      --ckpt checkpoints/gen_motif_figure/best_val.pt --highres --figures 2 1 4 3 --guidance 1 3 --songs 12
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
from src.generation.evaluation import onset_density_metrics
from src.generation.motif_codebook import figure_token, FIGURE_CLASSES


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
    ap.add_argument("--ckpt", default="checkpoints/gen_motif_figure/best_val.pt")
    ap.add_argument("--highres", action="store_true")
    ap.add_argument("--figures", type=int, nargs="+", default=[2, 1, 4, 3])  # sweep, jack, candle, trill
    ap.add_argument("--section", type=int, default=64)
    ap.add_argument("--guidance", type=float, nargs="+", default=[1.0, 3.0])
    ap.add_argument("--songs", type=int, default=12)
    ap.add_argument("--max_len", type=int, default=640)
    args = ap.parse_args()
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    real_fracs = section_fracs([s['real'] for s in songs], S)

    model = LayeredTypedChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)['model_state_dict']); model.eval()

    def gen(s, fig_tok, g):
        L = s['len']
        audio = torch.from_numpy(s['audio'][:L]).unsqueeze(0).to(device)
        diff = torch.tensor([s['difficulty']], device=device)
        radar = torch.from_numpy(s['radar']).unsqueeze(0).to(device)
        fig = None if fig_tok is None else torch.full((1, L), fig_tok, dtype=torch.long, device=device)
        with torch.no_grad():
            ol = torch.sigmoid(model.onset_logits(model.encode_audio(audio), diff, radar=radar))
        tau = float(np.quantile(ol.cpu().numpy(), 1 - target_density))
        out = model.generate(audio, diff, lengths=torch.tensor([L], device=device), onset_threshold=tau,
                             radar=radar, figure=fig, guidance_scale=g, type_sample=True, type_temperature=0.4,
                             hold_aware=True, pattern_sample=True, pattern_temperature=1.0, max_jack_run=1)[0].cpu().numpy()
        ch = pair_holds(out[:L])
        f1 = onset_density_metrics((ch != 0).astype(np.float32), reference=(s['real'][:L] != 0).astype(np.float32))['onset_f1']
        return ch, f1, (ch != 0).any(1).mean()

    print(f"figure-token control [{args.ckpt}] section={S}  {len(songs)} songs  real density {target_density:.3f}")
    print(f"REAL section-figure fractions: " + "  ".join(f"{FIGURE_CLASSES[i]}={real_fracs[i]:.2f}" for i in range(len(FIGURE_CLASSES))) + "\n")
    for g in args.guidance:
        print("=" * 90); print(f"GUIDANCE g={g}"); print("=" * 90)
        base_charts = [gen(s, None, g) for s in songs]            # figure=null baseline
        base_fracs = section_fracs([c for c, _, _ in base_charts], S)
        print(f"baseline (figure=null): " + "  ".join(f"{FIGURE_CLASSES[i]}={base_fracs[i]:.2f}" for i in args.figures)
              + f"   F1 {np.mean([f for _, f, _ in base_charts]):.2f}  dens {np.mean([d for _, _, d in base_charts]):.3f}")
        for tok in args.figures:
            charts = [gen(s, tok, g) for s in songs]
            fr = section_fracs([c for c, _, _ in charts], S)
            lift = fr[tok] - base_fracs[tok]
            print(f"  figure={FIGURE_CLASSES[tok]:16s} -> realized {FIGURE_CLASSES[tok]} frac {fr[tok]:.2f} "
                  f"(baseline {base_fracs[tok]:.2f}, REAL {real_fracs[tok]:.2f}, lift {lift:+.2f})   "
                  f"F1 {np.mean([f for _, f, _ in charts]):.2f}  dens {np.mean([d for _, _, d in charts]):.3f}")
    print("\nREAD: lift > 0 (esp. SWEEP) = the discrete figure token steers the realized figure where the "
          "continuous knob couldn't. Quality (F1/density) should hold.")


def section_fracs(charts, S):
    """Mean fraction of sections of each figure class across charts."""
    from collections import Counter
    acc = np.zeros(len(FIGURE_CLASSES))
    n = 0
    for ch in charts:
        T = ch.shape[0]; cnt = Counter()
        for i in range(0, T, S):
            cnt[figure_token(ch[i:i + S])] += 1
        tot = sum(cnt.values())
        if tot == 0:
            continue
        for k, c in cnt.items():
            acc[k] += c / tot
        n += 1
    return acc / max(1, n)


if __name__ == "__main__":
    main()

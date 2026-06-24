#!/usr/bin/env python3
"""
H15 motif-knob STEERABILITY eval: does pushing a motif knob actually change the realized FIGURES?

For each tested knob k, condition the generator with a target motif vector that is +z on k (others 0) vs -z on
k, generate, then RE-ENCODE the generated chart through the MotifBasis and read the realized knob-k value. If
realized k(+) >> k(-), the knob steers the vocabulary. We also report cross-talk (movement on OTHER knobs),
quality (onset_F1 / density vs real), and the realized dominant figures at each pole (the qualitative check —
do they look like jacks vs sweeps?).

  /home/ybx/miniconda3/envs/stepmania-chart-gen/bin/python \
      experiments/generation_typed/eval_motif.py [--knobs 0 3 10] [--z 3] [--guidance 1 3] [--songs 24]
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
from src.generation.motif_codebook import MotifBasis, motif_str

CKPT = "checkpoints/gen_motif/best_val.pt"
BASIS = PROJECT_ROOT / "cache/motif_basis.npz"


def collect(ds, cap, basis, n):
    out = []
    for i in range(len(ds)):
        if len(out) >= n:
            break
        sample = ds[i]; meta = ds.valid_samples[i]
        nd = next((nn for nn in meta['chart'].note_data
                   if nn.difficulty_name == meta['difficulty_name']), None)
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
    ap.add_argument("--knobs", type=int, nargs="+", default=[0, 3, 10])
    ap.add_argument("--z", type=float, default=3.0, help="knob push magnitude (z units)")
    ap.add_argument("--guidance", type=float, nargs="+", default=[1.0, 3.0])
    ap.add_argument("--songs", type=int, default=24)
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--pattern_temp", type=float, default=1.0, help="lower = less decode noise in the measurement")
    ap.add_argument("--max_jack_run", type=int, default=1, help="set 0/negative to DISABLE the jack cap (fair test for jack-axis knobs)")
    ap.add_argument("--ckpt", default=CKPT, help="generator checkpoint (default gen_motif / Track B)")
    ap.add_argument("--highres", action="store_true", help="Track A: 42-dim high-res features (cache/samples_v3)")
    args = ap.parse_args()
    jack_cap = args.max_jack_run if args.max_jack_run and args.max_jack_run > 0 else None
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    basis = MotifBasis.load(BASIS)

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
    songs = collect(val_ds, args.max_len, basis, args.songs)
    audio_dim = songs[0]['audio'].shape[1]
    target_density = float(np.mean([(s['real'] != 0).any(1).mean() for s in songs]))

    model = LayeredTypedChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)['model_state_dict']); model.eval()

    # per-song onset threshold from a target density (matches export decode)
    def gen_batch(batch, motif_vec, g):
        B = len(batch); L = min(args.max_len, max(s['len'] for s in batch))
        audio = torch.zeros(B, L, audio_dim); lengths = torch.zeros(B, dtype=torch.long); diff = torch.zeros(B, dtype=torch.long)
        radar = torch.zeros(B, 5)
        for b, s in enumerate(batch):
            t = min(s['len'], L); audio[b, :t] = torch.from_numpy(s['audio'][:t]); lengths[b] = t
            diff[b] = s['difficulty']; radar[b] = torch.from_numpy(s['radar'])
        mot = torch.from_numpy(np.tile(motif_vec, (B, 1)).astype(np.float32)).to(device)
        with torch.no_grad():
            ol = torch.sigmoid(model.onset_logits(model.encode_audio(audio.to(device)), diff.to(device),
                                                  radar=radar.to(device), motif=mot))
            tau = float(np.quantile(ol.cpu().numpy(), 1 - target_density))
            gen = model.generate(audio.to(device), diff.to(device), lengths=lengths.to(device),
                                 onset_threshold=tau, radar=radar.to(device), motif=mot, guidance_scale=g,
                                 type_sample=True, type_temperature=0.4, hold_aware=True,
                                 pattern_sample=True, pattern_temperature=args.pattern_temp,
                                 max_jack_run=jack_cap).cpu().numpy()
        return gen, lengths, radar.numpy()

    def realized(batch, motif_vec, g):
        """generate, re-encode through the basis -> mean realized knob vector + quality + top figures."""
        vecs, f1s, dens, hists = [], [], [], []
        for i in range(0, len(batch), 8):
            sub = batch[i:i + 8]
            gen, lengths, radar = gen_batch(sub, motif_vec, g)
            for b, s in enumerate(sub):
                t = int(lengths[b]); ch = pair_holds(gen[b, :t])
                vecs.append(basis.encode_chart(ch, radar[b]))
                hists.append(basis.chart_histogram(ch))
                m = onset_density_metrics((ch != 0).astype(np.float32), reference=(s['real'][:t] != 0).astype(np.float32))
                f1s.append(m['onset_f1']); dens.append((ch != 0).any(1).mean())
        return np.array(vecs).mean(0), float(np.mean(f1s)), float(np.mean(dens)), np.array(hists).mean(0)

    def top_figs(hist, n=3):
        return "  ".join(f"[{motif_str(basis.col_meta[j][1])}]" for j in np.argsort(hist)[::-1][:n] if hist[j] > 0)

    print(f"motif steerability [{args.ckpt}]  audio_dim={audio_dim}  "
          f"({len(songs)} songs, real density {target_density:.3f}, push ±{args.z}z)\n")
    for g in args.guidance:
        print("=" * 94); print(f"GUIDANCE g={g}"); print("=" * 94)
        for k in args.knobs:
            vp = np.zeros(MOTIF_DIM, np.float32); vp[k] = +args.z
            vn = np.zeros(MOTIF_DIM, np.float32); vn[k] = -args.z
            rp, f1p, dp, hp = realized(songs, vp, g)
            rn, f1n, dn, hn = realized(songs, vn, g)
            d_self = rp[k] - rn[k]
            d_others = np.max(np.abs((rp - rn))[[j for j in range(MOTIF_DIM) if j != k]])
            print(f"\nKNOB {k} '{basis.axis_info[k]['label']}'  push ±{args.z}z")
            print(f"  realized knob-{k}:  +push -> {rp[k]:+.2f}z   -push -> {rn[k]:+.2f}z   "
                  f"Δ(self) {d_self:+.2f}   max Δ(other knob) {d_others:.2f}")
            print(f"  quality: onset_F1 +{f1p:.2f}/-{f1n:.2f}  density +{dp:.3f}/-{dn:.3f} (real {target_density:.3f})")
            print(f"  + figures: {top_figs(hp)}")
            print(f"  - figures: {top_figs(hn)}")


if __name__ == "__main__":
    main()

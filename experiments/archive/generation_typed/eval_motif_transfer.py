#!/usr/bin/env python3
"""
H15 motif ON-MANIFOLD transfer eval (the fair / deployment-mode test).

The single-knob eval (eval_motif.py) cranks ONE PCA axis to +-3z with the rest pinned at 0 — a motif
combination real charts never show (off-manifold), which CFG then amplifies incoherently. The real use is to
steer toward a COHERENT motif profile: a real style's full 12-d vector (like the radar manifold gives a
coherent on-manifold target, not one cranked dim). This eval conditions generation on a REAL exemplar chart's
full motif vector and measures how much of the exemplar CONTRAST the output realizes.

For each tested knob k we pick the real val chart that is most +extreme on k (plus_ex) and most -extreme
(minus_ex), take their FULL real motif vectors e+ / e- (on-manifold), condition the SAME generation songs on
each, re-encode the generated charts -> r+ / r-, and report:
  - transfer along the contrast:  ((r+ - r-) . (e+ - e-)) / |e+ - e-|^2     (1.0 = fully realized, 0 = none)
  - realized Δ on knob k itself (directly comparable to the single-knob eval's Δself)
  - quality (onset_F1 / density) and the realized top figures at each pole.

  python \
      experiments/generation_typed/eval_motif_transfer.py [--knobs 0 3 10] [--guidance 1 3] [--songs 20]
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
from src.generation.typed_model import LayeredTypedChartGenerator, MOTIF_DIM
from src.generation.typed import NUM_PANELS, pair_holds
from src.generation.evaluation import onset_density_metrics
from src.generation.motif_codebook import MotifBasis, motif_str

CKPT = "checkpoints/gen_motif/best_val.pt"
BASIS = PROJECT_ROOT / "cache/motif_basis.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knobs", type=int, nargs="+", default=[0, 3, 10])
    ap.add_argument("--guidance", type=float, nargs="+", default=[1.0, 3.0])
    ap.add_argument("--songs", type=int, default=20)
    ap.add_argument("--max_len", type=int, default=640)
    ap.add_argument("--pattern_temp", type=float, default=0.7)
    args = ap.parse_args()
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    basis = MotifBasis.load(BASIS)

    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    tf, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    _, val_ds, _ = create_datasets(train_files=tf, val_files=vf, test_files=[], audio_dir="data/",
                                   max_sequence_length=msl, cache_dir='cache/samples')
    val_ds.warm_cache(show_progress=False)

    # collect songs (audio to generate on) + each chart's real motif vector (exemplar pool)
    songs, pool = [], []
    for i in range(len(val_ds)):
        sample = val_ds[i]; meta = val_ds.valid_samples[i]
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']), None)
        if nd is None:
            continue
        typed = val_ds.parser.convert_to_tensor_typed(meta['chart'], nd)
        T = min(int(sample['mask'].sum().item()), args.max_len, typed.shape[0])
        radar = meta['groove_radar'].to_vector().astype(np.float32)
        vec = basis.encode_chart(typed, radar).astype(np.float32)
        pool.append({'vec': vec, 'title': (meta['chart'].title or Path(meta['chart_file']).stem)[:30]})
        if len(songs) < args.songs:
            songs.append({'audio': sample['audio'][:T].numpy().astype(np.float32), 'len': T,
                          'difficulty': int(meta['difficulty_class']), 'radar': radar,
                          'real': typed[:T].astype(np.int64)})
    P = np.array([p['vec'] for p in pool])
    audio_dim = songs[0]['audio'].shape[1]
    target_density = float(np.mean([(s['real'] != 0).any(1).mean() for s in songs]))

    model = LayeredTypedChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device)['model_state_dict']); model.eval()

    def realized(motif_vec, g):
        vecs, f1s, dens, hists = [], [], [], []
        for i in range(0, len(songs), 8):
            sub = songs[i:i + 8]; B = len(sub); L = min(args.max_len, max(s['len'] for s in sub))
            audio = torch.zeros(B, L, audio_dim); lengths = torch.zeros(B, dtype=torch.long)
            diff = torch.zeros(B, dtype=torch.long); radar = torch.zeros(B, 5)
            for b, s in enumerate(sub):
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
                                     pattern_sample=True, pattern_temperature=args.pattern_temp, max_jack_run=1).cpu().numpy()
            for b, s in enumerate(sub):
                t = int(lengths[b]); ch = pair_holds(gen[b, :t])
                vecs.append(basis.encode_chart(ch, radar[b].numpy())); hists.append(basis.chart_histogram(ch))
                m = onset_density_metrics((ch != 0).astype(np.float32), reference=(s['real'][:t] != 0).astype(np.float32))
                f1s.append(m['onset_f1']); dens.append((ch != 0).any(1).mean())
        return np.array(vecs).mean(0), float(np.mean(f1s)), float(np.mean(dens)), np.array(hists).mean(0)

    def top_figs(hist, n=3):
        return "  ".join(f"[{motif_str(basis.col_meta[j][1])}]" for j in np.argsort(hist)[::-1][:n] if hist[j] > 0)

    print(f"gen_motif ON-MANIFOLD transfer ({len(songs)} songs, real density {target_density:.3f}, "
          f"exemplar pool {len(pool)})\n")
    for g in args.guidance:
        print("=" * 96); print(f"GUIDANCE g={g}"); print("=" * 96)
        for k in args.knobs:
            ip, im = int(P[:, k].argmax()), int(P[:, k].argmin())
            ep, em = P[ip], P[im]                        # full real exemplar vectors (on-manifold)
            rp, f1p, dp, hp = realized(ep, g)
            rn, f1n, dn, hn = realized(em, g)
            contrast = ep - em
            transfer = float((rp - rn) @ contrast / (contrast @ contrast + 1e-9))
            print(f"\nKNOB {k} '{basis.axis_info[k]['label']}'  exemplars: +'{pool[ip]['title']}' (k={ep[k]:+.1f}) "
                  f"vs -'{pool[im]['title']}' (k={em[k]:+.1f})")
            print(f"  TRANSFER along the exemplar contrast: {transfer:+.2f}   (1.0 = output fully realizes it; "
                  f"single-knob eval was ~0 for jacks)")
            print(f"  realized knob-{k}:  +ex -> {rp[k]:+.2f}z   -ex -> {rn[k]:+.2f}z   Δ {rp[k]-rn[k]:+.2f}")
            print(f"  quality: onset_F1 +{f1p:.2f}/-{f1n:.2f}  density +{dp:.3f}/-{dn:.3f} (real {target_density:.3f})")
            print(f"  +ex figures: {top_figs(hp)}")
            print(f"  -ex figures: {top_figs(hn)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
RELEASE GATE — the EASY corner of the feasible region.

We have stress-tested the HARD/chaotic corner exhaustively (chaos2, glitch, elite japa1) but barely the EASY
one. The geometry framing (notes/geometry_feasible_region.md) says a release = "the good region is characterized
to its BOUNDARIES", and low-difficulty / low-density is the boundary we haven't walked. Different failure modes
live there: does the quarter backbone SURVIVE when density is sparse, or does coherence need a minimum note rate?
Does the model KNOW easy = sparse, or does it render easy as "Hard with notes removed"?

Controlled sweep: hold the SONG (audio) fixed, vary only the TARGET-difficulty conditioning Beginner->Hard
(base conditioning — no radar/motif, so we isolate the difficulty axis). Two readouts per difficulty:
  - INTRINSIC density at a fixed threshold (mean p_onset>0.5) — does the model WANT fewer notes when told 'easy'?
  - STRUCTURE at the deployed per-difficulty manifold density — backbone(quarter)/8th/16th-offbeat shares,
    jump frac, and figure-variety entropy. A coherent easy chart sits mostly on quarters with varied figures;
    a "Hard-minus-notes" failure scatters off-grid or collapses to one degenerate figure.

  python experiments/generation_typed/diag_difficulty_corner.py [--songs 6] [--max_len 768]
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, sys
from collections import Counter
from pathlib import Path
import numpy as np, torch, yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
from src.data.dataset import DIFFICULTY_NAMES
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.typed import pair_holds
from src.generation.radar_manifold import RadarManifold
from src.generation.playtest_export import enforce_playability
from src.generation.motif_codebook import figure_token_schedule, FIGURE_CLASSES

CKPT = "checkpoints/gen_motif_full/best_val.pt"


def collect(ds, cap, n):
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
                    'title': (meta['chart'].title or Path(meta['chart_file']).stem)[:22],
                    'src_diff': int(meta['difficulty_class'])})
    return out


def structure(chart, section=64):
    """grid + figure-variety stats of a generated (T,4) chart."""
    active = (chart != 0) & (chart != 3)           # attacks only (H19-correct)
    onset = active.any(1); idx = np.nonzero(onset)[0]
    if len(idx) == 0:
        return dict(density=0.0, quarter=0.0, eighth=0.0, off16=0.0, jump=0.0, fig_entropy=0.0, top_fig="-")
    ph = idx % 4
    n = len(idx)
    quarter = float((ph == 0).mean()); eighth = float((ph == 2).mean())
    off16 = float(((ph == 1) | (ph == 3)).mean())
    jump = float((active[idx].sum(1) >= 2).mean())
    figs = figure_token_schedule(chart, section)[::section]          # one token per section
    figs = figs[figs != 0]                                           # drop 'sparse' sections
    if len(figs):
        c = Counter(int(f) for f in figs); tot = sum(c.values())
        p = np.array([v / tot for v in c.values()])
        ent = float(-(p * np.log2(p)).sum())
        top = FIGURE_CLASSES[c.most_common(1)[0][0]]
    else:
        ent, top = 0.0, "-"
    return dict(density=float(onset.mean()), quarter=quarter, eighth=eighth, off16=off16,
                jump=jump, fig_entropy=ent, top_fig=top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--songs", type=int, default=6)
    ap.add_argument("--max_len", type=int, default=768)
    args = ap.parse_args()
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    tf, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                   use_metric_phase=True, use_highres_onset=True))
    _, val_ds, _ = create_datasets(train_files=tf, val_files=vf, test_files=[], audio_dir="data/",
                                   max_sequence_length=msl, feature_extractor=ext, cache_dir='cache/samples_v3')
    val_ds.warm_cache(show_progress=False)
    songs = collect(val_ds, args.max_len, args.songs)
    audio_dim = songs[0]['audio'].shape[1]

    model = LayeredTypedChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)['model_state_dict']); model.eval()

    manifold = RadarManifold.load(PROJECT_ROOT / "cache/radar_manifold.npz")
    # per-difficulty BASE density = E[density | radar=difficulty-mean] (source-chart-free, the deployed target)
    base_density = {d: manifold.target_density(manifold._bucket(d).mean(0), d) for d in range(4)}

    print(f"\nDIFFICULTY-CORNER feasibility map  [{args.ckpt}]  {len(songs)} songs, base conditioning "
          f"(no radar/motif), playability ON")
    print(f"songs: " + ", ".join(f"{s['title']}({DIFFICULTY_NAMES[s['src_diff']][:3]})" for s in songs))
    print("\n  base conditioning, per TARGET difficulty (audio held fixed):")
    print(f"  {'difficulty':<10} {'nat_dens@0.5':>12} {'tgt_dens':>9} {'gen_dens':>9} "
          f"{'quarter':>8} {'8th':>6} {'off16':>7} {'jump':>6} {'fig_ent':>8}  top_figs")
    print("  " + "-" * 104)

    for d in range(4):
        rows = []
        nat_ds = []
        for s in songs:
            T = s['len']
            audio = torch.from_numpy(s['audio']).unsqueeze(0).to(device)
            diff = torch.tensor([d], device=device)
            with torch.no_grad():
                memory = model.encode_audio(audio)
                p_onset = torch.sigmoid(model.onset_logits(memory, diff)[0]).cpu().numpy()
            nat_ds.append(float((p_onset > 0.5).mean()))     # intrinsic density preference at fixed tau
            tgt = base_density[d]
            tau = float(np.quantile(p_onset, 1 - tgt)) if tgt and tgt > 0 else 0.5
            gk = dict(onset_threshold=tau, type_sample=True, type_temperature=0.4,
                      pattern_sample=True, pattern_temperature=0.7, max_jack_run=1)
            enforce_playability(gk)
            with torch.no_grad():
                gen = model.generate(audio, diff, lengths=torch.tensor([T], device=device), **gk)[0].cpu().numpy()
            rows.append(structure(pair_holds(gen)))
        agg = {k: np.mean([r[k] for r in rows]) for k in
               ('density', 'quarter', 'eighth', 'off16', 'jump', 'fig_entropy')}
        topfigs = Counter(r['top_fig'] for r in rows).most_common(3)
        print(f"  {DIFFICULTY_NAMES[d]:<10} {np.mean(nat_ds):>12.3f} {base_density[d] or 0:>9.3f} "
              f"{agg['density']:>9.3f} {agg['quarter']:>8.2f} {agg['eighth']:>6.2f} {agg['off16']:>7.2f} "
              f"{agg['jump']:>6.2f} {agg['fig_entropy']:>8.2f}  "
              + " ".join(f"{f}×{c}" for f, c in topfigs))
    print("\nREAD: nat_dens@0.5 should DROP toward Beginner (model knows easy=sparse). quarter share should stay "
          "HIGH / rise toward easy (backbone survives); off16 should be near 0 at Beginner. fig_entropy collapsing "
          "to ~0 at easy = degenerate single-figure ('Hard minus notes'); staying >0 = varied & deliberate.")


if __name__ == "__main__":
    main()

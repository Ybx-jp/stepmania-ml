#!/usr/bin/env python3
"""
Validate the FOOT-EXERTION soft jack governor (typed_model.generate jack_penalty): does an escalating,
BPM-aware penalty pull the generated same-panel jack-run distribution toward the HUMAN shape without wrecking
density? Sweeps jack_penalty (lambda) with the relaxed hard cap (max_jack_run=2, so a justified 2-note 16th
survives) and measures same-panel run lengths (8th+16th) vs the real-chart reference.

REAL reference (data/, 8th-spaced same-panel runs): len2 75.8% / len3 12.8% / len>=4 11.4%, max ~8+.

  python experiments/generation_typed/diag_jack_exertion.py [--songs 8] [--lams 0 1.5 3]
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
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.typed import pair_holds
from src.generation.radar_manifold import RadarManifold
from src.generation.playtest_export import enforce_playability


def same_panel_runs(typed):
    """lengths of consecutive SINGLE-panel same-panel runs at <=quarter spacing (the 'jack' streams)."""
    seq = [(t, [k for k in range(4) if typed[t, k] in (1, 2, 4)]) for t in range(typed.shape[0])]
    seq = [(t, a[0]) for t, a in seq if len(a) == 1]
    out, i = [], 0
    while i < len(seq):
        j = i
        while j + 1 < len(seq) and seq[j + 1][1] == seq[i][1] and (seq[j + 1][0] - seq[j][0]) <= 4:
            j += 1
        if j > i:
            out.append(j - i + 1)
        i = j + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/gen_motif_full_fixed/best_val.pt")
    ap.add_argument("--songs", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 1.5, 3.0])
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
        s = vds[i]; m = vds.valid_samples[i]
        T = min(int(s['mask'].sum().item()), args.max_len)
        songs.append({'audio': s['audio'][:T].numpy().astype(np.float32), 'len': T,
                      'diff': int(m['difficulty_class']), 'bpm': float(m['chart'].bpm)})
    audio_dim = songs[0]['audio'].shape[1]
    model = LayeredTypedChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)['model_state_dict']); model.eval()
    manifold = RadarManifold.load(PROJECT_ROOT / "cache/radar_manifold.npz")
    base_density = {d: manifold.target_density(manifold._bucket(d).mean(0), d) for d in range(4)}

    print(f"\nFOOT-EXERTION jack governor sweep  [{args.ckpt}]  {len(songs)} songs, max_jack_run=2 (relaxed)")
    print(f"REAL ref (8th same-panel runs): len2 75.8%  len3 12.8%  len>=4 11.4%")
    print(f"\n  {'jack_penalty':>12} {'runs>=2':>8} {'len2%':>7} {'len3%':>7} {'len>=4%':>8} {'maxlen':>7} {'density':>8}")
    print("  " + "-" * 66)
    for lam in args.lams:
        lens, dens = [], []
        for s in songs:
            T = s['len']; audio = torch.from_numpy(s['audio']).unsqueeze(0).to(device)
            diff = torch.tensor([s['diff']], device=device)
            with torch.no_grad():
                p = torch.sigmoid(model.onset_logits(model.encode_audio(audio), diff)[0]).cpu().numpy()
            tgt = base_density[s['diff']]; tau = float(np.quantile(p, 1 - tgt)) if tgt and tgt > 0 else 0.5
            gk = dict(onset_threshold=tau, type_sample=True, type_temperature=0.4,
                      pattern_sample=True, pattern_temperature=0.7, max_jack_run=2,
                      jack_penalty=(lam if lam > 0 else None), bpm=s['bpm'])
            enforce_playability(gk, override_reason="diag: relaxed jack cap to 2 for the exertion governor study")
            with torch.no_grad():
                g = pair_holds(model.generate(audio, diff, lengths=torch.tensor([T], device=device), **gk)[0].cpu().numpy())
            lens += same_panel_runs(g); dens.append(float((g != 0).any(1).mean()))
        c = Counter(min(L, 4) for L in lens); n = max(len(lens), 1)
        print(f"  {('OFF' if lam==0 else f'{lam:.1f}'):>12} {len(lens):>8} "
              f"{100*c.get(2,0)/n:>7.1f} {100*c.get(3,0)/n:>7.1f} {100*c.get(4,0)/n:>8.1f} "
              f"{max(lens, default=0):>7} {np.mean(dens):>8.3f}")
    print("\nREAD: as jack_penalty rises, len>=4% should fall toward ~11% (real) and maxlen drop, with density "
          "roughly held (the penalty re-routes a long jack to alternation, not to silence).")


if __name__ == "__main__":
    main()

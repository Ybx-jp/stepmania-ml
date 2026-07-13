#!/usr/bin/env python3
"""
Find the CHAOS knee for gen_motif_full: the (chaos, guidance) operating point where 16ths RISE while the
quarter backbone SURVIVES (musical syncopation), vs the H16/H4 smear where guidance dissolves the backbone into
a uniform 16th flood. Gentle manifold (chaos~0.23, g1.8) gave 16th~0; chaos=0.9 g3.0 gave 16th~0.98 (smear).
Cheap sweep (model loaded once, a few songs) to bracket the musical range before exporting playtest sets.

  python experiments/generation_typed/diag_chaos_knee.py [--songs 3]
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
from src.generation.playtest_export import enforce_playability


def phase_fracs(typed):
    onset = (typed != 0).any(1)
    idx = np.nonzero(onset)[0]
    if len(idx) == 0:
        return 0.0, 0.0, 0.0
    q = np.mean(idx % 4 == 0); e = np.mean(idx % 4 == 2); s = np.mean((idx % 4 == 1) | (idx % 4 == 3))
    return float(q), float(e), float(s)   # quarter backbone, 8th, 16th-offbeat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/gen_motif_full/best_val.pt")
    ap.add_argument("--songs", type=int, default=3)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--chaosq", type=float, nargs="+", default=[0.7, 0.8, 0.9, 0.95, 0.99],
                    help="chaos QUANTILE levels fed to the manifold (coherent conditional-fill, like --style)")
    ap.add_argument("--guid", type=float, nargs="+", default=[1.5, 2.0, 2.5, 3.0])
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
    songs = []
    for i in range(len(val_ds)):
        if len(songs) >= args.songs:
            break
        s = val_ds[i]; m = val_ds.valid_samples[i]
        nd = next((n for n in m['chart'].note_data if n.difficulty_name == m['difficulty_name']), None)
        if nd is None:
            continue
        T = min(int(s['mask'].sum().item()), args.max_len)
        typed = val_ds.parser.convert_to_tensor_typed(m['chart'], nd)[:T]
        songs.append({'audio': s['audio'][:T].numpy().astype(np.float32), 'T': T,
                      'diff': int(m['difficulty_class']), 'real': typed})
    from src.generation.radar_manifold import RadarManifold
    mp = Path('cache/radar_manifold.npz')
    manifold = RadarManifold.load(mp) if mp.exists() else RadarManifold.from_loaded_datasets(val_ds)
    print("using the radar MANIFOLD (conditional-fill of free dims via the real Gaussian conditional + project "
          "to the covariance ellipsoid) — coherent on-manifold chaos, like --style. Resolved chaos profile shown.")
    real_q = np.mean([phase_fracs(s['real'])[0] for s in songs])
    real_s = np.mean([phase_fracs(s['real'])[2] for s in songs])
    print(f"{len(songs)} songs. REAL backbone(quarter)={real_q:.2f}  16th={real_s:.2f}  density target per song.\n")

    model = LayeredTypedChartGenerator(audio_dim=42, d_model=128, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)['model_state_dict']); model.eval()

    @torch.no_grad()
    def gen(s, chaosq, g, calib=None):
        audio = torch.from_numpy(s['audio']).unsqueeze(0).to(device)
        diff = torch.tensor([s['diff']], device=device)
        tvec, info = manifold.build_target(f"chaos=q{chaosq}", s['diff'])    # coherent on-manifold fill
        radar = torch.from_numpy(tvec.astype(np.float32)).unsqueeze(0).to(device)
        dens = float(info.get('density') or (s['real'] != 0).any(1).mean())  # manifold E[density|radar,diff]
        ol = model.onset_logits(model.encode_audio(audio), diff, radar=radar)[0]
        olu = model.onset_logits(model.encode_audio(audio), diff, radar=None)[0]
        olc = olu + g * (ol - olu)
        if calib is not None:
            ph = torch.arange(olc.shape[0], device=device) % 4
            olc = olc + torch.where(ph == 2, float(calib[0]), torch.where((ph == 1) | (ph == 3), float(calib[1]), 0.0))
        tau = float(np.quantile(torch.sigmoid(olc).cpu().numpy(), 1 - dens))
        kw = dict(onset_threshold=tau, radar=radar, guidance_scale=g, type_sample=True, type_temperature=0.4,
                  pattern_sample=True, pattern_temperature=0.7, onset_phase_calib=calib)
        enforce_playability(kw)
        out = model.generate(audio, diff, lengths=torch.tensor([s['T']], device=device), **kw)[0].cpu().numpy()
        return pair_holds(out[:s['T']])

    # resolved chaos value per quantile (manifold-filled, difficulty 3=Hard)
    print("\nresolved on-manifold chaos value per quantile (Hard): "
          + "  ".join(f"q{q}->{manifold.build_target(f'chaos=q{q}', 3)[0][4]:.2f}" for q in args.chaosq))
    print("\n16th-offbeat onset fraction (musical knee ~0.15-0.35; ~1.0 = backbone-dissolved smear):")
    print("chaosq\\g " + "  ".join(f"{g:>5.1f}" for g in args.guid))
    for c in args.chaosq:
        row = [f"{np.mean([phase_fracs(gen(s, c, g))[2] for s in songs]):>5.2f}" for g in args.guid]
        print(f"q{c:<6}  " + "  ".join(row))
    print("\n(quarter backbone survival — want it to stay > ~0.15, not collapse):")
    print("chaosq\\g " + "  ".join(f"{g:>5.1f}" for g in args.guid))
    for c in args.chaosq:
        row = [f"{np.mean([phase_fracs(gen(s, c, g))[0] for s in songs]):>5.2f}" for g in args.guid]
        print(f"q{c:<6}  " + "  ".join(row))


if __name__ == "__main__":
    main()

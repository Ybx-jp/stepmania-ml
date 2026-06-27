#!/usr/bin/env python3
"""
ISOLATION: did the CHAOS CONDITIONING get more tasteful, holding the model fixed?

The REAL>BASE>CHAOS re-eval (eval_taste_current.py) showed CHAOS jumped 0.003 -> 0.228 from
old to new machinery -- but the whole stack changed (model, guidance, governor, AND the chaos
mechanism). This script isolates the CONDITIONING by decoding both chaos REQUESTS with the
SAME current model (gen_motif_full_fixed), same guidance, same governor, scored by the same
critic. The ONLY thing that varies is HOW chaos is asked for:

  - MEANPIN  : the OLD request -- global-mean radar with the chaos dim pinned high (OOD;
               others-at-mean), exactly what produced the 0.003 in the original eval_taste.
  - MANIFOLD : the NEW request -- RadarManifold.build_target('chaos=q0.85') (conditional-fill +
               ellipsoid projection -> in-distribution).

If MANIFOLD > MEANPIN with the model held fixed, the tastefulness gain is attributable to the
CONDITIONING redesign, not the model upgrade. (REAL scored as the anchor.)

Usage:
    python experiments/realism_critic/eval_chaos_mechanism.py --data_dir data/ --audio_dir data/ --eval_songs 64
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
from src.models import LateFusionClassifier
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.typed import pair_holds
from src.generation.playtest_export import enforce_playability
from src.generation.radar_manifold import RadarManifold

GEN_CKPT = "checkpoints/gen_motif_full_fixed/best_val.pt"   # held FIXED across both arms
MANIFOLD = "cache/radar_manifold.npz"
CHAOS_IDX = 4   # radar = [stream, voltage, air, freeze, chaos]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--critic', default='checkpoints/realism_critic/best_val.pt')
    p.add_argument('--eval_songs', type=int, default=64); p.add_argument('--max_len', type=int, default=768)
    p.add_argument('--cache_dir', default='cache/samples_v3')
    p.add_argument('--meanpin_chaos', type=float, default=0.9)     # the OLD raw request
    p.add_argument('--manifold_spec', default='chaos=q0.85')
    p.add_argument('--guidance', type=float, default=1.5)          # held FIXED across both arms
    p.add_argument('--fatigue_penalty', type=float, default=2.0)
    return p.parse_args()


def to_binary(t):
    t = np.asarray(t); return ((t == 1) | (t == 2) | (t == 4)).astype(np.float32)


@torch.no_grad()
def score(critic, audio23, chart, device):
    a = torch.from_numpy(audio23).unsqueeze(0).to(device); c = torch.from_numpy(chart).unsqueeze(0).to(device)
    m = torch.ones(1, a.shape[1], device=device)
    logits = critic(a, c, m)
    if isinstance(logits, dict): logits = logits['logits']
    return float(torch.softmax(logits, 1)[0, 1])


def main():
    args = parse_args(); set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob(f"{args.data_dir}/**/*.sm", recursive=True) + glob.glob(f"{args.data_dir}/**/*.ssc", recursive=True)
    _, val_files, _ = create_data_splits(cf, random_state=args.seed)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                   use_metric_phase=True, use_highres_onset=True))
    _, val_ds, _ = create_datasets(train_files=[], val_files=val_files, test_files=[], audio_dir=args.audio_dir,
                                   max_sequence_length=msl, feature_extractor=ext, cache_dir=args.cache_dir)
    val_ds.warm_cache(show_progress=False)

    songs = []
    for i in range(len(val_ds)):
        if len(songs) >= args.eval_songs: break
        s = val_ds[i]; meta = val_ds.valid_samples[i]; T = min(int(s['mask'].sum().item()), args.max_len)
        if T < 64: continue
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        tf = val_ds.parser.convert_to_tensor_typed(meta['chart'], nd)[:T]
        songs.append({'audio': s['audio'][:T].numpy().astype(np.float32), 'real': to_binary(tf),
                      'difficulty': int(meta['difficulty_class']), 'bpm': float(meta['chart'].bpm), 'T': T})
    print(f"eval songs={len(songs)} | audio_dim={songs[0]['audio'].shape[1]}")

    ck = torch.load(args.critic, map_location=device, weights_only=False)
    critic = LateFusionClassifier(ck['config']).to(device); critic.load_state_dict(ck['model_state_dict']); critic.eval()
    gen = LayeredTypedChartGenerator(audio_dim=42, d_model=128, num_layers=4, onset_layers=2).to(device)
    gen.load_state_dict(torch.load(PROJECT_ROOT / GEN_CKPT, map_location=device)['model_state_dict'], strict=False)
    gen.eval()
    manifold = RadarManifold.load(PROJECT_ROOT / MANIFOLD)
    # global mean radar (matches the original eval_taste mean-pin construction)
    mean_radar = np.mean([m['groove_radar'].to_vector() for m in val_ds.valid_samples if 'groove_radar' in m],
                         0).astype(np.float32)

    def decode(s, radar_vec, density):
        """One arm: fixed model + governor + guidance; ONLY radar_vec/density vary."""
        a = torch.from_numpy(s['audio']).unsqueeze(0).to(device); d = torch.tensor([s['difficulty']], device=device)
        radar = torch.from_numpy(np.asarray(radar_vec, np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            memory = gen.encode_audio(a)
            ol = gen.onset_logits(memory, d, radar=radar, style=None)[0]
            if args.guidance != 1.0:
                ol_u = gen.onset_logits(memory, d, radar=None, style=None)[0]
                ol = ol_u + args.guidance * (ol - ol_u)
            p = torch.sigmoid(ol).cpu().numpy()
        tau = float(np.quantile(p, 1 - density)) if density and density > 0 else 0.5
        gk = dict(onset_threshold=tau, type_sample=True, pattern_sample=True, pattern_temperature=0.7,
                  max_jack_run=2, fatigue_penalty=(args.fatigue_penalty if args.fatigue_penalty > 0 else None),
                  bpm=s['bpm'], radar=radar, style=None, guidance_scale=args.guidance)
        enforce_playability(gk, override_reason=None)
        with torch.no_grad():
            g = gen.generate(a, d, lengths=torch.tensor([s['T']], device=device), **gk)[0].cpu().numpy()
        return to_binary(pair_holds(g[:s['T']]))

    real_s, mp_s, mf_s = [], [], []
    mp_chaos, mf_chaos = [], []
    for n, s in enumerate(songs, 1):
        a23 = s['audio'][:, :23]
        # MEANPIN arm: global-mean radar, chaos dim pinned high (OOD); density via manifold coupling on that vec
        mp_vec = mean_radar.copy(); mp_vec[CHAOS_IDX] = args.meanpin_chaos
        mp_dens = manifold.target_density(mp_vec, s['difficulty'])
        # MANIFOLD arm: the in-distribution request
        mf_vec, mf_info = manifold.build_target(args.manifold_spec, s['difficulty'])
        real_s.append(score(critic, a23, s['real'], device))
        mp_s.append(score(critic, a23, decode(s, mp_vec, mp_dens), device))
        mf_s.append(score(critic, a23, decode(s, mf_vec, mf_info['density']), device))
        mp_chaos.append(float(mp_vec[CHAOS_IDX])); mf_chaos.append(float(mf_vec[CHAOS_IDX]))
        print(f"  [{n}/{len(songs)}] diff={s['difficulty']} "
              f"REAL {real_s[-1]:.3f} | MEANPIN {mp_s[-1]:.3f} (chaos={mp_chaos[-1]:.2f}) | "
              f"MANIFOLD {mf_s[-1]:.3f} (chaos={mf_chaos[-1]:.2f})")

    r, mp, mf = map(np.array, (real_s, mp_s, mf_s))
    print("\n" + "=" * 64)
    print("  CHAOS-CONDITIONING ISOLATION — same model, only the request varies")
    print("=" * 64)
    print(f"  REAL human chart       : {r.mean():.3f}")
    print(f"  MEANPIN (OLD request)  : {mp.mean():.3f}   (mean-pin chaos={args.meanpin_chaos}, OOD)")
    print(f"  MANIFOLD (NEW request) : {mf.mean():.3f}   (in-dist '{args.manifold_spec}', "
          f"realized chaos~{np.mean(mf_chaos):.2f})")
    print("=" * 64)
    print(f"  MANIFOLD > MEANPIN aggregate: {'YES' if mf.mean() > mp.mean() else 'NO'} "
          f"(Δ={mf.mean()-mp.mean():+.3f}) | per-song: {(mf > mp).mean()*100:.0f}%")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Re-evaluate the realism critic as a TASTE METRIC against the LATEST decode machinery.

The original eval_taste.py (2026-06-20) scored REAL > BASE > CHAOS using the OLD base
generator (checkpoints/gen_stage1, 41-dim, real-chart density, unconditioned tau, no
governor, OOD mean-pin chaos). That number predates the current decode stack, so it can't
say whether the critic is useful for reranking TODAY's generations.

This harness keeps the same three rungs but produces BASE/CHAOS with the canonical current
decode path (replicating scripts/generate.py exactly):
  - deployed 42-dim model checkpoints/gen_motif_full_fixed
  - manifold source-chart-free density target (RadarManifold.build_target)
  - tau from the SAME conditioned onset logits generate() decodes from
  - the shipped governor default (fatigue_penalty=2) + mandatory playability
  - CHAOS = manifold conditional-fill 'chaos=q0.85' (IN-distribution, guidance 1.5) --
    NOT the old OOD mean-pin chaos=0.9.

It also logs per-chart REALIZED density next to P(real) so we can see whether the critic
still discriminates outside its ~0.2-density training band (the OOD-density worry the v2
note flagged) -- reported as a correlation, not assumed.

Usage:
    python experiments/realism_critic/eval_taste_current.py --data_dir data/ --audio_dir data/ --eval_songs 8
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

GEN_CKPT = "checkpoints/gen_motif_full_fixed/best_val.pt"   # the deployed 42-dim highres model
MANIFOLD = "cache/radar_manifold.npz"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--critic', default='checkpoints/realism_critic/best_val.pt')
    p.add_argument('--eval_songs', type=int, default=64); p.add_argument('--max_len', type=int, default=768)
    p.add_argument('--cache_dir', default='cache/samples_v3')   # 42-dim highres cache
    p.add_argument('--chaos_spec', default='chaos=q0.85')
    p.add_argument('--chaos_guidance', type=float, default=1.5)
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
    assert songs and songs[0]['audio'].shape[1] == 42, f"expected 42-dim audio, got {songs[0]['audio'].shape}"
    print(f"eval songs={len(songs)} | audio_dim={songs[0]['audio'].shape[1]}")

    ck = torch.load(args.critic, map_location=device, weights_only=False)
    critic = LateFusionClassifier(ck['config']).to(device); critic.load_state_dict(ck['model_state_dict']); critic.eval()
    gen = LayeredTypedChartGenerator(audio_dim=42, d_model=128, num_layers=4, onset_layers=2).to(device)
    gen.load_state_dict(torch.load(PROJECT_ROOT / GEN_CKPT, map_location=device)['model_state_dict'], strict=False)
    gen.eval()
    manifold = RadarManifold.load(PROJECT_ROOT / MANIFOLD)

    def decode(s, style, guidance):
        """Replicates scripts/generate.py steps 4-6 for one song. style='' -> BASE; chaos spec -> CHAOS."""
        a = torch.from_numpy(s['audio']).unsqueeze(0).to(device); d = torch.tensor([s['difficulty']], device=device)
        tvec, tinfo = manifold.build_target(style or "", s['difficulty'])
        radar = torch.from_numpy(tvec).unsqueeze(0).to(device)
        gen_density = tinfo['density']
        with torch.no_grad():
            memory = gen.encode_audio(a)
            ol = gen.onset_logits(memory, d, radar=radar, style=None)[0]
            if guidance != 1.0 and style:
                ol_u = gen.onset_logits(memory, d, radar=None, style=None)[0]
                ol = ol_u + guidance * (ol - ol_u)
            p = torch.sigmoid(ol).cpu().numpy()
        tau = float(np.quantile(p, 1 - gen_density)) if gen_density and gen_density > 0 else 0.5
        gk = dict(onset_threshold=tau, type_sample=True, pattern_sample=True, pattern_temperature=0.7,
                  max_jack_run=2,
                  fatigue_penalty=(args.fatigue_penalty if args.fatigue_penalty > 0 else None),
                  bpm=s['bpm'], radar=(radar if style else None), style=None,
                  guidance_scale=(guidance if style else 1.0))
        enforce_playability(gk, override_reason=None)
        with torch.no_grad():
            g = gen.generate(a, d, lengths=torch.tensor([s['T']], device=device), **gk)[0].cpu().numpy()
        return to_binary(pair_holds(g[:s['T']]))

    rows = []  # (rung, P(real), realized_density, difficulty)
    real_s, base_s, chaos_s = [], [], []
    for n, s in enumerate(songs, 1):
        a23 = s['audio'][:, :23]
        real_chart = s['real']; base_chart = decode(s, '', 1.0); chaos_chart = decode(s, args.chaos_spec, args.chaos_guidance)
        for rung, chart, bucket in (('REAL', real_chart, real_s), ('BASE', base_chart, base_s), ('CHAOS', chaos_chart, chaos_s)):
            pr = score(critic, a23, chart, device); dens = float(chart.any(1).mean())
            bucket.append(pr); rows.append((rung, pr, dens, s['difficulty']))
        print(f"  [{n}/{len(songs)}] diff={s['difficulty']} "
              f"REAL {real_s[-1]:.3f} (d={rows[-3][2]:.3f}) | BASE {base_s[-1]:.3f} (d={rows[-2][2]:.3f}) | "
              f"CHAOS {chaos_s[-1]:.3f} (d={rows[-1][2]:.3f})")

    print("\n" + "=" * 60)
    print("  TASTE METRIC — critic P(real), CURRENT decode machinery")
    print("=" * 60)
    print(f"  REAL human chart : {np.mean(real_s):.3f}")
    print(f"  BASE generation  : {np.mean(base_s):.3f}   (governor fp={args.fatigue_penalty})")
    print(f"  CHAOS generation : {np.mean(chaos_s):.3f}   (in-dist '{args.chaos_spec}', g={args.chaos_guidance})")
    print("=" * 60)
    print(f"  expect REAL > BASE > CHAOS.  real>base: {'YES' if np.mean(real_s)>np.mean(base_s) else 'NO'} | "
          f"base>chaos: {'YES' if np.mean(base_s)>np.mean(chaos_s) else 'NO'}")
    # density-range diagnostic: does P(real) just track density (the OOD-band worry)?
    arr = np.array([(r[1], r[2]) for r in rows])
    if arr[:, 1].std() > 1e-6:
        c = np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]
        print(f"  Pearson corr( P(real) , realized_density ) = {c:+.3f}  "
              f"[density range {arr[:,1].min():.3f}-{arr[:,1].max():.3f}]")


if __name__ == '__main__':
    main()

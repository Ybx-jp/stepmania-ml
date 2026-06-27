#!/usr/bin/env python3
"""
Taste-critic interpretability — PHASE B: what does the critic measure on real generations?

Gate (Phase A, critic_saliency.py) is PASSED and selected the trustworthy attribution method: PERTURBATION
(change the arrows at fixed count, watch the logit margin), NOT gradient-IG-from-empty. Phase A also showed the
critic keys on arrow CONFIGURATION, globally. Phase B uses perturbation attribution on the matched quartet from
the chaos isolation (REAL / MEANPIN-chaos / MANIFOLD-chaos / BASE, same model, same songs) to test:

  H1 (off-grid): the critic's "fake" evidence lives on OFF-GRID 16th-offbeat frames (phase grid t%4 in {1,3};
      on-grid = quarter/8th = {0,2}, the backbone — conditioning-mechanics §6). If so, it explains the chaos
      result at the INPUT level: mean-pin scores low BECAUSE the critic penalizes its off-grid flood.

Two perturbations (no clean reference needed, unlike Phase A's repair):
  1. PHASE ABLATION (primary, direct H1 test): remove all OFF-grid notes vs all ON-grid notes, measure how much
     the margin RISES. If off-grid removal recovers "realness" (esp. for MEANPIN) and more than on-grid removal,
     the fake evidence is off-grid. (Removal changes density; the off-vs-on comparison controls for that.)
  2. BLOCK-SCRAMBLE SALIENCY (validated tool, spatial map): per block, scramble its panels (keep count), measure
     margin drop = how much that block's arrow coherence supports "real". Correlate with block off-grid fraction.

Usage:
    python experiments/realism_critic/critic_saliency_phaseB.py --data_dir data/ --audio_dir data/ --eval_songs 12
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

GEN_CKPT = "checkpoints/gen_motif_full_fixed/best_val.pt"
MANIFOLD = "cache/radar_manifold.npz"
CHAOS_IDX = 4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--critic', default='checkpoints/realism_critic/best_val.pt')
    p.add_argument('--cache_dir', default='cache/samples_v3')
    p.add_argument('--eval_songs', type=int, default=12); p.add_argument('--max_len', type=int, default=768)
    p.add_argument('--block', type=int, default=48)
    p.add_argument('--meanpin_chaos', type=float, default=0.9)
    p.add_argument('--manifold_spec', default='chaos=q0.85')
    p.add_argument('--guidance', type=float, default=1.5)
    p.add_argument('--fatigue_penalty', type=float, default=2.0)
    return p.parse_args()


def to_binary(t):
    t = np.asarray(t); return ((t == 1) | (t == 2) | (t == 4)).astype(np.float32)


def margin(critic, audio, chart, mask):
    logits = critic(audio, chart, mask)
    if isinstance(logits, dict): logits = logits['logits']
    return logits[:, 1] - logits[:, 0]


def phase_masks(T):
    """on-grid frames (t%4 in {0,2} = quarter/8th backbone) and off-grid (t%4 in {1,3} = 16th-offbeat)."""
    t = np.arange(T)
    return np.isin(t % 4, [0, 2]), np.isin(t % 4, [1, 3])


def scramble_block(rows, rng):
    out = np.zeros_like(rows)
    for i, row in enumerate(rows):
        k = int(row.sum())
        if k: out[i, rng.choice(4, size=k, replace=False)] = 1.0
    return out


def block_scramble_saliency(critic, audio, chart_np, mask, block, rng, device):
    """Per-block: scramble that block's panels (keep count), measure margin DROP. Validated perturbation tool."""
    T = chart_np.shape[0]
    base = torch.from_numpy(chart_np).unsqueeze(0).to(device)
    with torch.no_grad(): m0 = margin(critic, audio, base, mask).item()
    sal = np.zeros(T)
    for s in range(0, T, block):
        e = min(s + block, T)
        c = chart_np.copy(); c[s:e] = scramble_block(chart_np[s:e], rng)
        with torch.no_grad():
            dm = m0 - margin(critic, audio, torch.from_numpy(c).unsqueeze(0).to(device), mask).item()
        sal[s:e] = dm
    return sal


def main():
    args = parse_args(); set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(args.seed)
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
        if T < 128: continue
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        tf = to_binary(val_ds.parser.convert_to_tensor_typed(meta['chart'], nd)[:T])
        songs.append({'audio': s['audio'][:T].numpy().astype(np.float32), 'real': tf,
                      'difficulty': int(meta['difficulty_class']), 'bpm': float(meta['chart'].bpm), 'T': T})
    print(f"eval songs={len(songs)} | audio_dim={songs[0]['audio'].shape[1]}")

    ck = torch.load(PROJECT_ROOT / args.critic, map_location=device, weights_only=False)
    critic = LateFusionClassifier(ck['config']).to(device); critic.load_state_dict(ck['model_state_dict']); critic.eval()
    gen = LayeredTypedChartGenerator(audio_dim=42, d_model=128, num_layers=4, onset_layers=2).to(device)
    gen.load_state_dict(torch.load(PROJECT_ROOT / GEN_CKPT, map_location=device)['model_state_dict'], strict=False); gen.eval()
    manifold = RadarManifold.load(PROJECT_ROOT / MANIFOLD)
    mean_radar = np.mean([m['groove_radar'].to_vector() for m in val_ds.valid_samples if 'groove_radar' in m],
                         0).astype(np.float32)

    def decode(s, radar_vec, density):
        a = torch.from_numpy(s['audio']).unsqueeze(0).to(device); d = torch.tensor([s['difficulty']], device=device)
        radar = torch.from_numpy(np.asarray(radar_vec, np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            mem = gen.encode_audio(a); ol = gen.onset_logits(mem, d, radar=radar, style=None)[0]
            if args.guidance != 1.0:
                ol_u = gen.onset_logits(mem, d, radar=None, style=None)[0]
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

    def analyze(a23_t, chart_np, mask, T):
        on, off = phase_masks(T)
        notes = chart_np.any(1)                       # (T,) frames with >=1 note
        n_tot = int(notes.sum())
        off_frac = float((notes & off).sum() / max(n_tot, 1))
        with torch.no_grad():
            m = margin(critic, a23_t, torch.from_numpy(chart_np).unsqueeze(0).to(device), mask).item()
            no_off = chart_np.copy(); no_off[off] = 0.0
            no_on = chart_np.copy(); no_on[on] = 0.0
            m_no_off = margin(critic, a23_t, torch.from_numpy(no_off).unsqueeze(0).to(device), mask).item()
            m_no_on = margin(critic, a23_t, torch.from_numpy(no_on).unsqueeze(0).to(device), mask).item()
        return dict(margin=m, off_frac=off_frac, d_off=m_no_off - m, d_on=m_no_on - m, n=n_tot)

    rows = {k: [] for k in ('REAL', 'BASE', 'MEANPIN', 'MANIFOLD')}
    blk_corr = {k: [] for k in rows}        # per-chart corr(block scramble-saliency, block off-grid frac)
    for n, s in enumerate(songs, 1):
        T = s['T']; mask = torch.ones(1, T, device=device); a23 = torch.from_numpy(s['audio'][:, :23]).unsqueeze(0).to(device)
        mp_vec = mean_radar.copy(); mp_vec[CHAOS_IDX] = args.meanpin_chaos
        mf_vec, mf_info = manifold.build_target(args.manifold_spec, s['difficulty'])
        base_vec, base_info = manifold.build_target('', s['difficulty'])
        quartet = {'REAL': s['real'],
                   'BASE': decode(s, base_vec, base_info['density']),
                   'MEANPIN': decode(s, mp_vec, manifold.target_density(mp_vec, s['difficulty'])),
                   'MANIFOLD': decode(s, mf_vec, mf_info['density'])}
        line = f"  [{n}/{len(songs)}] diff={s['difficulty']}"
        for k, chart in quartet.items():
            r = analyze(a23, chart, mask, T); rows[k].append(r)
            # block-scramble saliency map -> correlate with per-block off-grid fraction
            sal = block_scramble_saliency(critic, a23, chart, mask, args.block, rng, device)
            on, off = phase_masks(T); notes = chart.any(1)
            bs, bf = [], []
            for st in range(0, T, args.block):
                e = min(st + args.block, T); nn = notes[st:e].sum()
                if nn > 0: bs.append(sal[st]); bf.append((notes[st:e] & off[st:e]).sum() / nn)
            if len(bs) > 2 and np.std(bf) > 1e-6 and np.std(bs) > 1e-6:
                blk_corr[k].append(float(np.corrcoef(bs, bf)[0, 1]))
            line += f" | {k} m={r['margin']:+.1f} off={r['off_frac']:.2f}"
        print(line)

    print("\n" + "=" * 78)
    print("  PHASE B — H1: does the critic's fake evidence live on OFF-GRID frames?")
    print("=" * 78)
    print(f"  {'rung':9s} {'margin':>8s} {'off_frac':>9s} {'Δm(rm off-grid)':>16s} {'Δm(rm on-grid)':>15s}")
    allm, alloff = [], []
    for k in ('REAL', 'BASE', 'MANIFOLD', 'MEANPIN'):
        R = rows[k]; mm = np.mean([r['margin'] for r in R]); of = np.mean([r['off_frac'] for r in R])
        do = np.mean([r['d_off'] for r in R]); dn = np.mean([r['d_on'] for r in R])
        print(f"  {k:9s} {mm:>+8.2f} {of:>9.2f} {do:>+16.2f} {dn:>+15.2f}")
        allm += [r['margin'] for r in R]; alloff += [r['off_frac'] for r in R]
    allm, alloff = np.array(allm), np.array(alloff)
    print("=" * 78)
    cmo = float(np.corrcoef(allm, alloff)[0, 1])
    print(f"  cross-chart corr(margin, off-grid fraction) = {cmo:+.3f}  (H1 predicts NEGATIVE: more off-grid -> more fake)")
    print(f"  block scramble-saliency vs off-grid fraction, mean corr/chart:  "
          + " | ".join(f"{k} {np.mean(v):+.2f}" if v else f"{k} n/a" for k, v in blk_corr.items()))
    # H1 read: for MEANPIN, removing off-grid should RAISE the margin (positive d_off) and more than on-grid.
    mp = rows['MEANPIN']; mp_doff = np.mean([r['d_off'] for r in mp]); mp_don = np.mean([r['d_on'] for r in mp])
    print(f"\n  H1 verdict signals: corr(margin,off_frac)<0 -> {'YES' if cmo < -0.2 else 'NO'} ({cmo:+.2f}); "
          f"MEANPIN remove-off-grid raises margin more than remove-on-grid -> "
          f"{'YES' if mp_doff > mp_don else 'NO'} ({mp_doff:+.2f} vs {mp_don:+.2f})")


if __name__ == '__main__':
    main()

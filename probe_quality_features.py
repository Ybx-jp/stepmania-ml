#!/usr/bin/env python3
"""Which audio features drive the per-song QUALITY DEFICIT of the canonical-default generator?

quality_deficit[song] = critic P(real)(REAL human chart) - mean_K critic P(real)(canonical generation)
  high  = the canonical generator does MUCH worse than the human chart on this song
  ~0    = the generation is scored ~as human on this song

Generation replicates the DEPLOYED canonical decode EXACTLY via the decode harness + CANONICAL_DECODE
(pattern_temp=1.0, fatigue 2 / free 6, stamina 50 / breathe 1.2, the onset_phase_calib 16th-unlock, tau from
the SAME conditioned+phase-calibrated onset logits). It does NOT copy eval_taste_current.py's block — that one is
STALE (pattern_temperature=0.7, no 16th-unlock) and would measure a different, un-played model
(generation-defaults skill §3). No groove conditioning (radar/style/motif/figure=None, guidance 1.0) = the
canonical BASE the user plays.

Quality = critic P(real). Deficit (vs the song's own real chart) controls for each song's intrinsic critic
baseline and isolates where the GENERATOR underperforms (user's choice, 2026-06-30).

Experiment-design guards:
  Rule 11 — GATE on the deficit's DYNAMIC RANGE before believing any correlation (the taste critic is documented
            near-binary; if the deficit barely varies across songs, no feature can 'explain its variation').
  Rule 12 — report deficit by DIFFICULTY (don't pool a heterogeneous population); report whether difficulty itself
            drives it.
  Rule 1  — the 42-dim MFCC/chroma/spectral-contrast/metric-phase dims are per-song z-scored -> their aggregates
            are ~constant across songs; near-constant regressors are auto-flagged and dropped from the ranking.
            Timbre/harmony descriptors are recomputed from RAW audio (which z-scoring erased).
  Rule 6  — cheapest-first: --n small + --k 1 SMOKE gate the dynamic range before the full run.

Usage:
  SMOKE : python probe_quality_features.py --data_dir data/ --audio_dir data/ --n 8  --k 1
  FULL  : python probe_quality_features.py --data_dir data/ --audio_dir data/ --n 64 --k 3 --out cache/quality_features.csv
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, sys
from pathlib import Path
import numpy as np, torch, yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets, discover_chart_files
from src.models import LateFusionClassifier
from src.generation.typed import pair_holds
from src.generation.playtest_export import enforce_playability
from src.generation.decode_defaults import CANONICAL_DECODE
from src.generation.decode_harness import (
    conditioned_p_onset, compute_tau, make_feature_extractor, load_generator, DEPLOYED_CHECKPOINT)

MANIFOLD = "cache/radar_manifold.npz"  # (unused when no conditioning, but load-checked for parity)
DIFF_NAMES = ['Beginner', 'Easy', 'Medium', 'Hard']

# --- 42-dim highres layout (src/data/audio_features.py get_aligned_features) -----------------------------------
# 0-12 MFCC(z) | 13 onset_env(max-norm) | 14 onset_rate | 15 tempo(BPM/const) | 16-22 spectral_contrast(z)
# 23-34 chroma(z) | 35 perc_onset(max-norm) | 36 harm_onset(max-norm) | 37-40 metric_phase(sin/cos) | 41 highres_onset(max-norm)
# z-scored dims (0-12,16-34,37-40) have per-song mean~=0 std~=1 -> aggregates near-constant across songs.


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--critic', default='checkpoints/realism_critic/best_val.pt')
    p.add_argument('--checkpoint', default=DEPLOYED_CHECKPOINT)
    p.add_argument('--n', type=int, default=64, help='#songs from the val split')
    p.add_argument('--k', type=int, default=3, help='#generation samples per song, critic-averaged (de-noise the target)')
    p.add_argument('--max_len', type=int, default=768)
    p.add_argument('--cache_dir', default='cache/samples_v3')  # 42-dim highres cache
    p.add_argument('--out', default='cache/quality_features.csv')
    p.add_argument('--difficulty', type=int, default=None, help='restrict to one difficulty class (0-3) to avoid pooling')
    return p.parse_args()


def to_binary(t):
    t = np.asarray(t); return ((t == 1) | (t == 2) | (t == 4)).astype(np.float32)


@torch.no_grad()
def critic_score(critic, audio23, chart, device):
    """Returns (P(real), logit_margin). The probability SATURATES on generations (near-binary critic -> ~0 for
    almost all Hard gens), so the pre-sigmoid logit MARGIN (logit_real - logit_fake) is the higher-dynamic-range
    target: it keeps ordering info the saturated probability throws away (0.005 vs 0.003 -> margin -5.3 vs -6.8)."""
    a = torch.from_numpy(audio23).unsqueeze(0).to(device); c = torch.from_numpy(chart).unsqueeze(0).to(device)
    m = torch.ones(1, a.shape[1], device=device)
    logits = critic(a, c, m)
    if isinstance(logits, dict): logits = logits['logits']
    prob = float(torch.softmax(logits, 1)[0, 1])
    margin = float(logits[0, 1] - logits[0, 0])
    return prob, margin


def canonical_gen(model, s, device):
    """Binary onset-grid of a canonical generation (for the critic, which reads presence not hold-type)."""
    return to_binary(canonical_gen_typed(model, s, device))


def canonical_gen_typed(model, s, device):
    """ONE canonical-default generation for song s, TYPED (holds preserved: 1 tap /2 hold-head /3 tail /4 roll).
    Faithful to export_typed_samples.py / generate() (the deployed path): no conditioning, governor on, 16th-unlock,
    tau from the SAME conditioned+phase-calibrated onset logits. Choreography metrics need the typed chart."""
    phase_calib = CANONICAL_DECODE['onset_phase_calib']           # (0.0, 1.0) the 16th-unlock
    audio = torch.from_numpy(s['audio']).unsqueeze(0).to(device)
    diff = torch.tensor([s['difficulty']], device=device)
    with torch.no_grad():
        memory = model.encode_audio(audio)
        # tau source: conditioned (here unconditioned: radar/style None) + phase-calibrated onset logits, EXACTLY
        # as generate() will decode. gen_density = the source chart's own density (canonical BASE: no --style /
        # --target_density, so the manifold branch is skipped and density falls through to the real chart).
        p_onset = conditioned_p_onset(model, memory, diff, radar=None, style=None,
                                      guidance=1.0, phase_calib=phase_calib, extra_offset=None)
    tau = compute_tau(p_onset, s['real_density'])
    gk = dict(onset_threshold=tau, bpm=s['bpm'],
              type_sample=True, pattern_sample=True,
              radar=None, style=None, motif=None, figure=None, guidance_scale=1.0,
              **CANONICAL_DECODE)   # type/pattern temp, fatigue, stamina, onset_phase_calib — the whole palette
    enforce_playability(gk, None)  # FORCES hold_aware + no_jump/cross_during_hold on
    with torch.no_grad():
        g = model.generate(audio, diff, lengths=torch.tensor([s['T']], device=device), **gk)[0].cpu().numpy()
    return pair_holds(g[:s['T']])


def raw_descriptors(audio_file):
    """Timbre/harmony descriptors from RAW audio — the axes the per-song z-scoring in the 42-dim cache erases."""
    import librosa
    try:
        y, sr = librosa.load(audio_file, sr=22050, mono=True)
    except Exception:
        return {}
    if y.size < sr:  # <1s -> unusable
        return {}
    try:
        return _raw_descriptors_impl(y, sr)
    except Exception as e:
        print(f"    (raw_descriptors failed on {os.path.basename(audio_file)}: {e})")
        return {}


def _raw_descriptors_impl(y, sr):
    import librosa
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    flat = librosa.feature.spectral_flatness(y=y)[0]
    rms = librosa.feature.rms(y=y)[0]
    # tuning=0.0 skips estimate_tuning()->piptrack, whose numba gufunc SEGFAULTS here (matches audio_features.py's
    # own chroma_stft call, which passes tuning=0.0 for exactly this reason).
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, tuning=0.0).mean(1)
    cd = chroma / (chroma.sum() + 1e-9)
    chroma_entropy = float(-(cd * np.log(cd + 1e-12)).sum())     # tonal spread / harmonic complexity (0..log12)
    return {
        'raw_spec_centroid_mean': float(cent.mean()), 'raw_spec_centroid_cv': float(cent.std() / (cent.mean() + 1e-9)),
        'raw_spec_flatness_mean': float(flat.mean()),  # noisiness vs tonalness
        'raw_rms_mean': float(rms.mean()), 'raw_rms_cv': float(rms.std() / (rms.mean() + 1e-9)),  # loudness dynamics
        'raw_chroma_entropy': chroma_entropy,
    }


def cache_features(a42):
    """Aggregates of the deployed 42-dim input. Curated interpretable set (only the cross-song-informative dims) +
    the full mean/std block (near-constant z-scored dims get dropped at ranking time)."""
    f = {}
    # curated interpretable (max-normalized envelopes + tempo + onset_rate carry real cross-song variation):
    f['onset_env_mean'] = float(a42[:, 13].mean());  f['onset_env_cv'] = float(a42[:, 13].std() / (a42[:, 13].mean() + 1e-9))
    f['onset_rate_mean'] = float(a42[:, 14].mean())
    f['perc_mean'] = float(a42[:, 35].mean());        f['harm_mean'] = float(a42[:, 36].mean())
    f['perc_harm_ratio'] = float(a42[:, 35].mean() / (a42[:, 36].mean() + 1e-9))  # percussive vs harmonic drive
    f['highres_onset_mean'] = float(a42[:, 41].mean()); f['highres_onset_cv'] = float(a42[:, 41].std() / (a42[:, 41].mean() + 1e-9))
    # full raw 42-dim mean/std block (labeled d##_*); most z-scored ones are near-constant -> auto-dropped later.
    for d in range(a42.shape[1]):
        f[f'd{d:02d}_mean'] = float(a42[:, d].mean()); f[f'd{d:02d}_std'] = float(a42[:, d].std())
    return f


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 or np.std(x[ok]) < 1e-9 or np.std(y[ok]) < 1e-9:
        return np.nan
    rx = np.argsort(np.argsort(x[ok])); ry = np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])


def load_val_dataset(data_dir, audio_dir, seed, cache_dir='cache/samples_v3'):
    """Deterministic val split + 42-dim highres dataset (lazy — NO warm_cache: it eagerly extracts the whole val
    set, ~30min CPU; we only touch `n` songs via val_ds[i], indexed in order so cache aliasing is safe)."""
    cf = discover_chart_files(data_dir)
    _, val_files, _ = create_data_splits(cf, random_state=seed)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    fs = make_feature_extractor('highres')
    _, val_ds, _ = create_datasets(train_files=[], val_files=val_files, test_files=[], audio_dir=audio_dir,
                                   max_sequence_length=msl, feature_extractor=fs.extractor, cache_dir=cache_dir)
    return val_ds


def build_songs(val_ds, n, difficulty=None, max_len=768):
    """First `n` valid songs from val_ds (optionally filtered to one difficulty class). Carries the audio (42-dim),
    the real chart BINARY (critic) and TYPED (choreography/holds), density, bpm, title, audio_file."""
    songs = []
    for i in range(len(val_ds)):
        if len(songs) >= n: break
        s = val_ds[i]; meta = val_ds.valid_samples[i]; T = min(int(s['mask'].sum().item()), max_len)
        if T < 64: continue
        if difficulty is not None and int(meta['difficulty_class']) != difficulty: continue
        nd = next((nn for nn in meta['chart'].note_data if nn.difficulty_name == meta['difficulty_name']
                   and nn.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        tf = np.asarray(val_ds.parser.convert_to_tensor_typed(meta['chart'], nd))[:T]
        songs.append({'audio': s['audio'][:T].numpy().astype(np.float32), 'real': to_binary(tf), 'real_typed': tf,
                      'difficulty': int(meta['difficulty_class']), 'bpm': float(meta['chart'].bpm), 'T': T,
                      'real_density': float((tf != 0).any(1).mean()), 'audio_file': meta['audio_file'],
                      'title': (meta['chart'].title or Path(meta['chart_file']).stem)})
    assert songs and songs[0]['audio'].shape[1] == 42, "expected 42-dim highres audio"
    return songs


def main():
    args = parse_args(); set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device} | checkpoint={args.checkpoint} | k={args.k} samples/song")

    val_ds = load_val_dataset(args.data_dir, args.audio_dir, args.seed, args.cache_dir)
    songs = build_songs(val_ds, args.n, args.difficulty, args.max_len)
    print(f"songs={len(songs)} | dims={songs[0]['audio'].shape[1]}")

    ck = torch.load(args.critic, map_location=device, weights_only=False)
    critic = LateFusionClassifier(ck['config']).to(device); critic.load_state_dict(ck['model_state_dict']); critic.eval()
    model = load_generator(args.checkpoint, 42, device)

    # ---- score each song. Quality signals (per song):
    #   p_gen / m_gen  = generator chart's critic PROB / LOGIT-MARGIN  (the DIRECT generator-quality signal)
    #   deficit        = p_real - p_gen           (user's choice; but PROB saturates -> ~= p_real, contaminated)
    #   mdeficit       = m_real - m_gen           (margin deficit; keeps dynamic range the prob throws away)
    rows = []
    for n, s in enumerate(songs, 1):
        a23 = s['audio'][:, :23]
        p_real, m_real = critic_score(critic, a23, s['real'], device)
        gens = [critic_score(critic, a23, canonical_gen(model, s, device), device) for _ in range(args.k)]
        p_gen = float(np.mean([g[0] for g in gens])); p_gen_sd = float(np.std([g[0] for g in gens]))
        m_gen = float(np.mean([g[1] for g in gens]))
        feats = {'title': s['title'], 'difficulty': s['difficulty'], 'bpm': s['bpm'],
                 'real_density': s['real_density'], 'p_real': p_real, 'p_gen': p_gen, 'p_gen_sd': p_gen_sd,
                 'm_real': m_real, 'm_gen': m_gen, 'deficit': p_real - p_gen, 'mdeficit': m_real - m_gen}
        feats.update(cache_features(s['audio'])); feats.update(raw_descriptors(s['audio_file']))
        rows.append(feats)
        print(f"  [{n}/{len(songs)}] {DIFF_NAMES[s['difficulty']]:8s} {s['title'][:24]:24s} "
              f"P_real={p_real:.3f} P_gen={p_gen:.3f}±{p_gen_sd:.3f} | M_gen={m_gen:+.2f} deficit={p_real-p_gen:+.3f}")

    # ---- write CSV --------------------------------------------------------------------------------------------
    import csv
    keys = list(rows[0].keys())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nwrote {args.out} ({len(rows)} songs x {len(keys)} cols)")

    diffs = np.array([r['difficulty'] for r in rows])
    targets = {  # name -> (array, is_generator_quality_signal)
        'p_gen (generator PROB)':   (np.array([r['p_gen'] for r in rows]), True),
        'm_gen (generator MARGIN)': (np.array([r['m_gen'] for r in rows]), True),
        'deficit (p_real-p_gen)':   (np.array([r['deficit'] for r in rows]), False),
        'mdeficit (m_real-m_gen)':  (np.array([r['mdeficit'] for r in rows]), False),
    }
    p_real = np.array([r['p_real'] for r in rows]); p_gen = targets['p_gen (generator PROB)'][0]

    # ---- Rule 11 GATE: which candidate quality target actually VARIES among GENERATIONS? ----------------------
    print("\n" + "=" * 78)
    print("  DYNAMIC-RANGE GATE (experiment-design Rule 11) — the generator-quality target must vary")
    print("=" * 78)
    print(f"  P(real) human : mean {p_real.mean():.3f} sd {p_real.std():.3f} [{p_real.min():.3f},{p_real.max():.3f}]")
    for name, (arr, isgen) in targets.items():
        print(f"  {name:26s}: mean {arr.mean():+.3f} sd {arr.std():.3f} [{arr.min():+.3f},{arr.max():+.3f}]")
    mid = np.mean((p_gen > 0.1) & (p_gen < 0.9))
    print(f"  gen PROB in discriminating band (0.1-0.9): {mid*100:.0f}%   (0% => the probability is SATURATED)")
    print(f"  corr(deficit, p_real)={np.corrcoef(targets['deficit (p_real-p_gen)'][0], p_real)[0,1]:+.2f} "
          f"(if ~1, 'deficit' is just the human score, NOT generator quality — Rule 11)")
    print(f"  mean within-song sampling sd of P_gen: {np.mean([r['p_gen_sd'] for r in rows]):.3f}")

    # ---- Rule 12: quality by difficulty (only meaningful when pooled) -----------------------------------------
    if len(set(diffs.tolist())) > 1:
        print("\n  p_gen by difficulty (Rule 12 — the coarse axis; generator quality tracks difficulty):")
        for d in sorted(set(diffs.tolist())):
            m = diffs == d
            print(f"    {DIFF_NAMES[d]:8s} n={m.sum():2d}  p_gen {p_gen[m].mean():.3f}  deficit {targets['deficit (p_real-p_gen)'][0][m].mean():+.3f}")

    # ---- feature correlations against EACH generator-quality target -------------------------------------------
    exclude = {'title', 'deficit', 'mdeficit', 'p_real', 'p_gen', 'p_gen_sd', 'm_real', 'm_gen'}
    featkeys = [k for k in keys if k not in exclude]
    def rank_against(tgt):
        out = []
        for k in featkeys:
            vals = np.array([r.get(k, np.nan) for r in rows], float)
            if np.sum(np.isfinite(vals)) < max(6, len(rows) // 2) or np.nanstd(vals) < 1e-6: continue
            r_sp = spearman(vals, tgt)
            if np.isfinite(r_sp): out.append((k, r_sp))
        out.sort(key=lambda x: -abs(x[1])); return out
    tag = lambda k: 'raw ' if k.startswith('raw_') else ('d## ' if (len(k) > 3 and k[1:3].isdigit()) else 'cur ')
    thr = 1.96 / np.sqrt(len(rows))
    for tname in ('m_gen (generator MARGIN)', 'p_gen (generator PROB)'):
        ranked = rank_against(targets[tname][0])
        print("\n" + "=" * 78)
        print(f"  FEATURE -> Spearman(feature, {tname})   [n={len(rows)}; |r|>~{thr:.2f} ~ p<.05 UNCORRECTED]")
        print("=" * 78)
        for k, r_sp in ranked[:18]:
            print(f"    {tag(k)}{k:24s} r={r_sp:+.3f}")
    print(f"\n  ({len(rank_against(p_gen))} non-degenerate features ranked; full table in {args.out})")
    print("  NOTE: small N, many features — a RANKING/hypothesis generator, not confirmatory p-values. If the gate")
    print("  shows the generator target is SATURATED (0% band / near-zero sd), NO feature ranking here is trustworthy.")


if __name__ == '__main__':
    main()

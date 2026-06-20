#!/usr/bin/env python3
"""
Musicality metrics that onset_F1 / crit_adj are blind to (see notes/playtest_log.md). Compares a
generator against the REAL charts on the axes the playtests exposed:

  1. within-beat PHASE distribution (16th grid: on-beat / 8th-off / 16ths) — base over-sits on the
     beat (0.91 vs real 0.80); H1/H4 predict the musical-feature model places offbeats more like a human.
  2. STRUCTURE correlation — corr(generated density-vs-position, real density-vs-position) over deciles,
     plus the end-decile (the climax fade). H5.
  3. arrow-change <-> chroma-change ALIGNMENT — do the chart's pattern changes land where the music
     changes (chroma novelty)? point-biserial corr over note frames; the most direct H1 readout.
  4. regression guards — onset_F1 and critic-adjacency (must not drop).

Loads features from cache/samples_v2 (41-dim) regardless of model: the 41-dim features have the 23 as a
prefix, so the BASE (23-dim) model is fed audio[:, :23] and chroma (dims 23:35) is available for both.

Usage:
    python experiments/generation_typed/eval_musicality.py --features stage1 \
        --checkpoint checkpoints/gen_stage1/best_val.pt --data_dir data/ --audio_dir data/
    python experiments/generation_typed/eval_musicality.py --features base \
        --checkpoint checkpoints/gen_radar/best_val.pt --data_dir data/ --audio_dir data/
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
from src.generation.typed import NUM_PANELS, pair_holds
from src.generation.evaluation import onset_density_metrics, DifficultyCritic

DEFAULT_BPM = 150.0
CHROMA_SLICE = slice(23, 35)  # chroma dims within the 41-dim feature vector


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--features', choices=['base', 'stage1'], required=True,
                   help='base=feed 23 dims (slice); stage1=feed all 41')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval_songs', type=int, default=48); p.add_argument('--max_gen_len', type=int, default=768)
    p.add_argument('--cache_dir', default='cache/samples_v2')
    return p.parse_args()


def phase_dist(typed):
    """within-beat phase histogram over note frames: [on-beat(4th), 16th, 8th-off, 16th]."""
    onset = (typed != 0).any(1)
    ph = np.arange(len(typed))[onset] % 4
    h = np.bincount(ph, minlength=4).astype(float)
    return h / max(h.sum(), 1)


def density_deciles(typed, k=10):
    occ = (typed != 0).any(1).astype(float); T = len(occ)
    idx = (np.arange(T) * k // max(T, 1))
    return np.array([occ[idx == b].mean() if (idx == b).any() else 0.0 for b in range(k)])


def chroma_pattern_alignment(typed, chroma):
    """point-biserial corr between 'the chart changes pattern here' and chroma novelty, over note frames.
    Positive => arrows change when the music changes (musical choreography)."""
    active = (typed != 0)
    onset = active.any(1)
    frames = np.where(onset)[0]
    if len(frames) < 8:
        return np.nan
    nov = np.zeros(len(typed))
    nov[1:] = np.abs(np.diff(chroma, axis=0)).sum(1)  # chroma novelty (L1 of delta)
    # pattern change: active-panel set differs from the previous note's set
    sets = active[frames]
    changed = np.zeros(len(frames))
    changed[1:] = (sets[1:] != sets[:-1]).any(1).astype(float)
    x = nov[frames]
    if changed.std() < 1e-6 or x.std() < 1e-6:
        return np.nan
    return float(np.corrcoef(x, changed)[0, 1])


def main():
    args = parse_args(); set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob(f"{args.data_dir}/**/*.sm", recursive=True) + glob.glob(f"{args.data_dir}/**/*.ssc", recursive=True)
    _, val_files, _ = create_data_splits(cf, random_state=args.seed)
    with open(PROJECT_ROOT / "config/model_config.yaml") as f:
        msl = yaml.safe_load(f)['classifier']['max_sequence_length']
    ext41 = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True, use_metric_phase=True))
    _, val_ds, _ = create_datasets(train_files=[], val_files=val_files, test_files=[], audio_dir=args.audio_dir,
                                   max_sequence_length=msl, feature_extractor=ext41, cache_dir=args.cache_dir)
    val_ds.warm_cache(show_progress=True)

    songs = []
    for i in range(len(val_ds)):
        if len(songs) >= args.eval_songs: break
        s = val_ds[i]; meta = val_ds.valid_samples[i]; T = min(int(s['mask'].sum().item()), args.max_gen_len)
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        typed = meta['chart']; tf = val_ds.parser.convert_to_tensor_typed(meta['chart'], nd)
        T = min(T, tf.shape[0])
        songs.append({'audio41': s['audio'][:T].numpy().astype(np.float32), 'difficulty': int(meta['difficulty_class']),
                      'typed': tf[:T].astype(np.int64)})
    audio_dim = 41 if args.features == 'stage1' else 23
    print(f"features={args.features} (audio_dim={audio_dim})  eval_songs={len(songs)}")
    target_density = float(np.mean([(s['typed'] != 0).any(1).mean() for s in songs]))

    model = LayeredTypedChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model_state_dict']); model.eval()
    critic = DifficultyCritic(device=device)

    def feats(a41):  # (T,41) -> model input on the right dim
        return a41 if audio_dim == 41 else a41[:, :23]

    # density-matched onset threshold from the model's own onset logits
    logits = []
    with torch.no_grad():
        for s in songs:
            a = torch.from_numpy(feats(s['audio41'])).unsqueeze(0).to(device); d = torch.tensor([s['difficulty']], device=device)
            logits.append(torch.sigmoid(model.onset_logits(model.encode_audio(a), d))[0].cpu().numpy())
    tau = float(np.quantile(np.concatenate(logits), 1 - target_density))

    gen_phase, real_phase = [], []
    struct_corr, end_gen, end_real = [], [], []
    align_gen, align_real = [], []
    f1s, preds, tgts = [], [], []
    for s in songs:
        a = torch.from_numpy(feats(s['audio41'])).unsqueeze(0).to(device); d = torch.tensor([s['difficulty']], device=device)
        with torch.no_grad():
            g = model.generate(a, d, lengths=torch.tensor([len(s['typed'])], device=device), onset_threshold=tau,
                               type_sample=True, type_temperature=0.4, hold_aware=True,
                               pattern_sample=True, pattern_temperature=0.7)[0].cpu().numpy()
        g = pair_holds(g); real = s['typed']; chroma = s['audio41'][:, CHROMA_SLICE]
        gen_phase.append(phase_dist(g)); real_phase.append(phase_dist(real))
        gd, rd = density_deciles(g), density_deciles(real)
        if gd.std() > 1e-6 and rd.std() > 1e-6:
            struct_corr.append(float(np.corrcoef(gd, rd)[0, 1]))
        end_gen.append(gd[-1]); end_real.append(rd[-1])
        ag = chroma_pattern_alignment(g, chroma); ar = chroma_pattern_alignment(real, chroma)
        if not np.isnan(ag): align_gen.append(ag)
        if not np.isnan(ar): align_real.append(ar)
        m = onset_density_metrics((g != 0).astype(np.float32), reference=(real != 0).astype(np.float32))
        f1s.append(m['onset_f1'])
        tb = lambda t: ((t == 1) | (t == 2) | (t == 4)).astype(np.float32)
        # the critic is the Phase-1 classifier -> always feed it the original 23-dim features
        preds.append(critic.predict(tb(g), s['audio41'][:, :23], bpm=DEFAULT_BPM)['class'])
        tgts.append(s['difficulty'])

    gp, rp = np.mean(gen_phase, 0), np.mean(real_phase, 0)
    dd = np.abs(np.array(preds) - np.array(tgts))
    print("\n" + "=" * 72)
    print(f"  MUSICALITY EVAL — features={args.features}")
    print("=" * 72)
    print(f"  [1] within-beat phase  on-beat/16th/8th-off/16th")
    print(f"        GEN  {gp.round(3)}")
    print(f"        REAL {rp.round(3)}    L1 dist = {np.abs(gp - rp).sum():.3f}  (lower = more human)")
    print(f"  [2] structure corr (gen vs real density-arc): {np.mean(struct_corr):+.3f}")
    print(f"        end-decile density  GEN {np.mean(end_gen):.3f}  REAL {np.mean(end_real):.3f}")
    print(f"  [3] arrow<->chroma alignment   GEN {np.mean(align_gen):+.3f}   REAL {np.mean(align_real):+.3f}")
    print(f"        (gen approaching real = arrows track musical change; H1)")
    print(f"  [4] guards   onset_F1 {np.mean(f1s):.3f}   crit_adj {np.mean(dd <= 1):.3f}")
    print("=" * 72)


if __name__ == '__main__':
    main()

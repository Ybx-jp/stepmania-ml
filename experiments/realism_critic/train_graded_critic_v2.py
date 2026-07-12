#!/usr/bin/env python3
"""GRADED taste critic on the 48th GRID (v2) -- the R1/R2 prerequisite for taste-aligned best-of-N.

WHY (explore/taste-critic-quality-resolution, the 2026-07-11 diagnostic): the deployed generator ships on the
48th grid (highres_v2, timesteps_per_beat=12), but BOTH existing critics (binary + graded) were trained on the
16th grid (cache/samples). To score an f48 chart a 16th critic must FLOOR it onto the 16th grid, deleting exactly
the triplet/sub-16th placement that IS the f48 quality signal. So a best-of-N loop on a 16th critic is blind to
the axis where f48's variance lives (score_personal.py + offgrid_personal.py). This retrains the graded critic on
the 48th cache so it can SEE f48 output.

Design = train_graded_critic.py's objective (within-song corruption-ladder margin-ranking + end BCE anchor,
warm-started from the binary critic) ported to the v2 dataset (for_v2 parser + highres_v2 features +
cache/samples_v3_48th, sliced [:23] for the 23-dim critic space), PLUS one new corruption axis:

  * JITTER ladder -- displace a FRACTION of on-16th-grid notes by +-1..2 frames onto pure-48th cells ({1,5,7,11}).
    Count/panels/audio held FIXED; only sub-16th PLACEMENT degrades. This is the degradation a 16th critic
    literally could not represent -> if the v2 critic grades it monotonically, R1 (sees f48) is cleared.

The graded-ness GATE (per epoch, on val): each ladder's per-level mean margin must be MONOTONE decreasing +
SPREAD (real >> full-corrupt), especially on the JITTER axis.

Usage:
    OMP_NUM_THREADS=4 python experiments/realism_critic/train_graded_critic_v2.py \
        --data_dir data --audio_dir data --max_train_songs 1200 --epochs 12
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.data.stepmania_parser import StepManiaParser
from src.generation.decode_harness import make_feature_extractor
from src.models import LateFusionClassifier

BINARY_CRITIC = "checkpoints/realism_critic/best_val.pt"     # 16th warm-start: knows the taste DIRECTION
V2_MSL = 5400                                                # dataset truncation (train_motif_figure_v2 parity)
SUBDIV = 12
PANEL_LADDER = [0.0, 0.2, 0.45, 0.7, 1.0]                    # fraction of note-frames panel-scrambled
SHIFT_LADDER = [0, 6, 18, 48]                                # audio-shift frames (48th grid: [0,.125,.375,1] beat)
JITTER_LADDER = [0.0, 0.15, 0.35, 0.6]                       # NEW: fraction of on-16th notes jittered off-grid
SIXTEENTH = {0, 3, 6, 9}                                     # 16th-aligned cells @ subdiv=12


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=12); p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4); p.add_argument('--max_len', type=int, default=2304)
    p.add_argument('--rank_margin', type=float, default=0.75)
    p.add_argument('--anchor_w', type=float, default=0.3)
    p.add_argument('--max_train_songs', type=int, default=1200); p.add_argument('--max_val_songs', type=int, default=250)
    p.add_argument('--cache_dir', default='cache/samples_v3_48th')
    p.add_argument('--checkpoint_dir', default='checkpoints/realism_critic_graded_v2')
    p.add_argument('--no_warmstart', action='store_true')
    return p.parse_args()


def to_binary(typed):
    t = np.asarray(typed); return ((t == 1) | (t == 2) | (t == 4)).astype(np.float32)


def corrupt_panels_frac(chart, frac, rng):
    if frac <= 0: return chart.copy()
    out = chart.copy(); rows = np.where(chart.any(1))[0]
    pick = rng.random(len(rows)) < frac
    for t, hit in zip(rows, pick):
        if not hit: continue
        k = int(chart[t].sum()); out[t] = 0.0
        out[t, rng.choice(4, size=k, replace=False)] = 1.0
    return out


def corrupt_shift(chart, off):
    if off <= 0 or len(chart) < 32: return chart.copy()
    return np.roll(chart, int(off), axis=0)


def corrupt_jitter(chart, frac, rng):
    """Displace a FRACTION of ON-16th-grid note-frames by +-1 frame -> a pure-48th cell (the sub-16th degradation
    a 16th critic can't see). Count/panels preserved; only placement moves. Skips a move that would collide."""
    if frac <= 0 or len(chart) < 4: return chart.copy()
    out = chart.copy()
    rows = [t for t in np.where(chart.any(1))[0] if (t % SUBDIV) in SIXTEENTH]
    for t in rows:
        if rng.random() >= frac: continue
        d = rng.choice([-1, 1]); nt = t + d
        if nt < 0 or nt >= len(out): continue
        if out[nt].any(): continue          # don't stack onto an existing note
        out[nt] = out[t]; out[t] = 0.0
    return out


def collect(ds, cap, max_len):
    out = []
    for i in range(len(ds)):
        if len(out) >= cap: break
        s = ds[i]
        if s is None: continue
        meta = ds.valid_samples[i]; T = min(int(s['mask'].sum().item()), max_len)
        if T < 64: continue
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        tf = ds.parser.convert_to_tensor_typed(meta['chart'], nd)[:T]
        real = to_binary(tf)
        if not real.any(): continue
        out.append({'audio': s['audio'][:T].numpy().astype(np.float32)[:, :23], 'real': real, 'T': T})
    return out


def score_batch(critic, charts, audio_np, T, device):
    B = len(charts); a = torch.zeros(B, T, 23); c = torch.zeros(B, T, 4); m = torch.ones(B, T)
    for b, ch in enumerate(charts):
        a[b] = torch.from_numpy(audio_np); c[b] = torch.from_numpy(ch)
    logits = critic(a.to(device), c.to(device), m.to(device))
    if isinstance(logits, dict): logits = logits['logits']
    return logits[:, 1] - logits[:, 0]      # margin = the graded score


def ladders(song, rng):
    real = song['real']
    return {'panel': [corrupt_panels_frac(real, f, rng) for f in PANEL_LADDER],
            'shift': [corrupt_shift(real, o) for o in SHIFT_LADDER],
            'jitter': [corrupt_jitter(real, f, rng) for f in JITTER_LADDER]}


def rank_and_anchor(margins, args):
    loss = margins.new_zeros(())
    for i in range(len(margins) - 1):
        loss = loss + torch.relu(args.rank_margin - (margins[i] - margins[i + 1]))
    bce = nn.functional.binary_cross_entropy_with_logits
    loss = loss + args.anchor_w * (bce(margins[0], margins.new_ones(())) + bce(margins[-1], margins.new_zeros(())))
    return loss


@torch.no_grad()
def evaluate(critic, val, device, rng):
    critic.eval()
    acc = {'panel': [[] for _ in PANEL_LADDER], 'shift': [[] for _ in SHIFT_LADDER],
           'jitter': [[] for _ in JITTER_LADDER]}
    mono = {'panel': 0, 'shift': 0, 'jitter': 0}; n = 0
    for song in val:
        lad = ladders(song, rng)
        for key, charts in lad.items():
            m = score_batch(critic, charts, song['audio'], song['T'], device).cpu().numpy()
            for j, v in enumerate(m): acc[key][j].append(v)
            mono[key] += int(np.all(np.diff(m) <= 1e-6))
        n += 1
    means = {k: [float(np.mean(x)) for x in v] for k, v in acc.items()}
    monor = {k: mono[k] / max(n, 1) for k in mono}
    return means, monor


def main():
    args = parse_args(); set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob(f"{args.data_dir}/**/*.sm", recursive=True) + glob.glob(f"{args.data_dir}/**/*.ssc", recursive=True)
    train_files, val_files, _ = create_data_splits(cf, random_state=args.seed)   # seed 42 == v2-train split
    spec = make_feature_extractor("highres_v2")                                   # 48th grid, 42-dim
    train_ds, val_ds, _ = create_datasets(train_files=train_files, val_files=val_files, test_files=[],
                                          audio_dir=args.audio_dir, max_sequence_length=V2_MSL,
                                          feature_extractor=spec.extractor, cache_dir=args.cache_dir,
                                          parser=StepManiaParser.for_v2())        # == the cached grid (index-keyed)
    # NO warm_cache: the 48th cache is already built (5.2G); collect() reads only the capped songs via ds[i].
    train = collect(train_ds, args.max_train_songs, args.max_len)
    val = collect(val_ds, args.max_val_songs, args.max_len)
    print(f"train songs={len(train)} val songs={len(val)}  max_len={args.max_len} (48th grid)")
    print(f"ladders: panel{PANEL_LADDER} shift{SHIFT_LADDER} jitter{JITTER_LADDER}")

    cfg = dict(yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier'])
    cfg['num_classes'] = 2; cfg['head_type'] = 'classification'
    cfg['use_groove_radar'] = False; cfg['use_projection_head'] = False
    critic = LateFusionClassifier(cfg).to(device)
    if not args.no_warmstart and Path(PROJECT_ROOT / BINARY_CRITIC).exists():
        sd = torch.load(PROJECT_ROOT / BINARY_CRITIC, map_location=device)['model_state_dict']
        missing, unexpected = critic.load_state_dict(sd, strict=False)
        print(f"warm-started from {BINARY_CRITIC} (16th binary critic); missing={len(missing)} unexpected={len(unexpected)}")
    opt = torch.optim.AdamW(critic.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    Path(PROJECT_ROOT / args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_spread = -1e9; best_path = Path(PROJECT_ROOT / args.checkpoint_dir) / "best_val.pt"

    for epoch in range(args.epochs):
        critic.train(); order = list(range(len(train))); rng.shuffle(order); tot = 0.0; nb = 0
        for k in range(0, len(order), args.batch_size):
            opt.zero_grad(); batch_loss = torch.zeros((), device=device)
            for idx in order[k:k + args.batch_size]:
                song = train[idx]
                for charts in ladders(song, rng).values():
                    m = score_batch(critic, charts, song['audio'], song['T'], device)
                    batch_loss = batch_loss + rank_and_anchor(m, args)
            batch_loss = batch_loss / max(len(order[k:k + args.batch_size]), 1)
            batch_loss.backward(); torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0); opt.step()
            tot += float(batch_loss); nb += 1
        means, monor = evaluate(critic, val, device, rng)
        # spread = sum of real-vs-fullycorrupt gaps across the 3 axes (jitter is the R1-critical one)
        spread = sum(means[k][0] - means[k][-1] for k in means)
        def fmt(k): return "[" + " ".join(f"{v:+.2f}" for v in means[k]) + f"] m={monor[k]:.2f}"
        star = "  *" if spread > best_spread else ""
        print(f"ep {epoch+1}/{args.epochs} loss={tot/max(nb,1):.3f} | panel {fmt('panel')} | "
              f"shift {fmt('shift')} | jitter {fmt('jitter')}{star}")
        if spread > best_spread:
            best_spread = spread
            torch.save({'model_state_dict': critic.state_dict(), 'config': cfg, 'epoch': epoch,
                        'grid': '48th', 'subdiv': SUBDIV, 'max_len': args.max_len,
                        'panel_ladder': PANEL_LADDER, 'shift_ladder': SHIFT_LADDER, 'jitter_ladder': JITTER_LADDER,
                        'panel_means': means['panel'], 'jitter_means': means['jitter'],
                        'val_mono': monor}, best_path)
    print(f"\nbest summed real-vs-corrupt spread={best_spread:.2f} -> {best_path}")
    print("GATE: jitter ladder must be MONOTONE + SPREAD (real >> off-grid) -> R1 (critic sees f48) cleared.")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Stage 2a (v2) — train a REALISM/TASTE CRITIC with CORRUPTED-REAL negatives.
See notes/stage2a_critic_findings.md (v1 failed: with generated negatives the critic learned the
generator FINGERPRINT, not taste, and scored backwards vs the playtest).

Fix: negatives are REAL charts perturbed at FIXED density and timing, so the only cue separating
positive from negative is arrow-choice *taste* (not density, not timing, not a generator artifact):
  positive       real chart                                   label 1
  neg-panels     per note-frame, reassign which panels are active (keep the count)   label 0
                 -> destroys arrow-choice coherence (alternation, melody-following, motifs)
  neg-shift      roll the chart vs its audio by a random offset                       label 0
                 -> destroys audio alignment (taste = notes land on musical events)

Both preserve overall density/onset-structure, forcing the critic onto musical judgment. No generator
is involved in training (fast). Validate with eval_taste.py (expect REAL > BASE > CHAOS on generations).

Usage:
    python experiments/realism_critic/train_critic.py --data_dir data/ --audio_dir data/ \
        --max_train_songs 1500 --epochs 12
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
from src.models import LateFusionClassifier

CLS_CKPT = "checkpoints/ordinal_exp/standard_ordinal_multi/best_val_loss.pt"  # Phase-1 warm-start


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--max_len', type=int, default=768)
    p.add_argument('--max_train_songs', type=int, default=1500)
    p.add_argument('--max_val_songs', type=int, default=300)
    p.add_argument('--cache_dir', default='cache/samples')   # 23-dim features (critic doesn't need chroma)
    p.add_argument('--checkpoint_dir', default='checkpoints/realism_critic')
    p.add_argument('--no_warmstart', action='store_true')
    return p.parse_args()


def to_binary(typed):
    t = np.asarray(typed); return ((t == 1) | (t == 2) | (t == 4)).astype(np.float32)


def corrupt_panels(chart, rng):
    """Per note-frame, reassign which panels are active (keep the per-frame count). Destroys
    arrow-choice coherence/taste while preserving density, jump-rate and timing exactly."""
    out = np.zeros_like(chart)
    rows = np.where(chart.any(1))[0]
    for t in rows:
        k = int(chart[t].sum())
        panels = rng.choice(4, size=k, replace=False)
        out[t, panels] = 1.0
    return out


def corrupt_shift(chart, rng):
    """Roll the chart vs its audio by a random offset -> notes no longer land on the audio events.
    Destroys audio alignment while keeping the chart's internal structure & density."""
    T = len(chart)
    if T < 32:
        return chart.copy()
    off = int(rng.integers(8, max(9, T - 8)))
    return np.roll(chart, off, axis=0)


def collect(ds, cap, max_len):
    out = []
    for i in range(len(ds)):
        if len(out) >= cap: break
        s = ds[i]; meta = ds.valid_samples[i]; T = min(int(s['mask'].sum().item()), max_len)
        if T < 64: continue
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        tf = ds.parser.convert_to_tensor_typed(meta['chart'], nd)[:T]
        out.append({'audio': s['audio'][:T].numpy().astype(np.float32)[:, :23], 'real': to_binary(tf), 'T': T})
    return out


def build_examples(n):
    ex = []
    for i in range(n):
        ex.append((i, 'real')); ex.append((i, 'panels')); ex.append((i, 'shift'))
    return ex


def make_batch(songs, eb, device, rng):
    L = max(songs[i]['T'] for i, _ in eb); B = len(eb)
    audio = torch.zeros(B, L, 23); chart = torch.zeros(B, L, 4); mask = torch.zeros(B, L); y = torch.zeros(B, dtype=torch.long)
    for b, (i, kind) in enumerate(eb):
        s = songs[i]; T = s['T']
        audio[b, :T] = torch.from_numpy(s['audio'])
        if kind == 'real':
            c = s['real']; y[b] = 1
        elif kind == 'panels':
            c = corrupt_panels(s['real'], rng); y[b] = 0
        else:
            c = corrupt_shift(s['real'], rng); y[b] = 0
        chart[b, :T] = torch.from_numpy(c); mask[b, :T] = 1.0
    return audio.to(device), chart.to(device), mask.to(device), y.to(device)


def roc_auc(scores, labels):
    s = np.asarray(scores); y = np.asarray(labels)
    n_pos = y.sum(); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0: return float('nan')
    ranks = np.argsort(np.argsort(s)) + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    args = parse_args(); set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob(f"{args.data_dir}/**/*.sm", recursive=True) + glob.glob(f"{args.data_dir}/**/*.ssc", recursive=True)
    train_files, val_files, _ = create_data_splits(cf, random_state=args.seed)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    train_ds, val_ds, _ = create_datasets(train_files=train_files, val_files=val_files, test_files=[],
                                          audio_dir=args.audio_dir, max_sequence_length=msl, cache_dir=args.cache_dir)
    train_ds.warm_cache(show_progress=False); val_ds.warm_cache(show_progress=False)
    print("collecting real samples...")
    train = collect(train_ds, args.max_train_songs, args.max_len)
    val = collect(val_ds, args.max_val_songs, args.max_len)
    print(f"train songs={len(train)} val songs={len(val)}  (negatives = corrupted-real: panels + shift)")

    cfg = dict(yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier'])
    cfg['num_classes'] = 2; cfg['head_type'] = 'classification'
    cfg['use_groove_radar'] = False; cfg['use_projection_head'] = False
    critic = (LateFusionClassifier(cfg).to(device) if args.no_warmstart
              else LateFusionClassifier.from_pretrained(CLS_CKPT, cfg, device=str(device)))
    print(f"critic params: {sum(p.numel() for p in critic.parameters()):,}")
    opt = torch.optim.AdamW(critic.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(); rng = np.random.default_rng(args.seed)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_auc = -1.0; best_path = Path(args.checkpoint_dir) / "best_val.pt"
    train_ex = build_examples(len(train)); val_ex = build_examples(len(val))

    for epoch in range(args.epochs):
        critic.train(); rng.shuffle(train_ex); tot = 0.0; nb = 0
        for k in range(0, len(train_ex), args.batch_size):
            audio, chart, mask, y = make_batch(train, train_ex[k:k + args.batch_size], device, rng)
            opt.zero_grad(); loss = crit(critic(audio, chart, mask), y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        critic.eval(); sc, lb = [], []; cat = {'real': [], 'panels': [], 'shift': []}
        with torch.no_grad():
            for k in range(0, len(val_ex), args.batch_size):
                eb = val_ex[k:k + args.batch_size]
                audio, chart, mask, y = make_batch(val, eb, device, rng)
                logits = critic(audio, chart, mask)
                pr = torch.softmax(logits, 1)[:, 1].cpu().numpy()
                for (i, kind), p, yy in zip(eb, pr, y.cpu().numpy()):
                    sc.append(p); lb.append(int(yy)); cat[kind].append(p)
        auc = roc_auc(sc, lb)
        print(f"epoch {epoch+1}/{args.epochs} loss={tot/max(nb,1):.3f}  AUC={auc:.3f}  "
              f"P(real): real={np.mean(cat['real']):.3f} panels={np.mean(cat['panels']):.3f} shift={np.mean(cat['shift']):.3f}"
              + ("  *" if auc > best_auc else ""))
        if auc > best_auc:
            best_auc = auc
            torch.save({'model_state_dict': critic.state_dict(), 'config': cfg, 'epoch': epoch, 'val_auc': auc}, best_path)

    print(f"\nbest val AUC={best_auc:.3f} -> {best_path}")
    print("Next: eval_taste.py — expect REAL > BASE > CHAOS if the critic now scores taste.")


if __name__ == '__main__':
    main()

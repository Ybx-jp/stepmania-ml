#!/usr/bin/env python3
"""GRADED realism/taste critic — the non-saturating successor to the near-binary v2 critic.

WHY (notes/quality_feature_attribution_findings.md): the deployed critic (train_critic.py) is trained with BINARY
cross-entropy on SEVERE corrupted-real negatives (full panel-scramble / full audio-shift = 0) vs real = 1. That
objective bakes in SATURATION — it learns an easy binary boundary, so anything "somewhat wrong" (a real generation)
rails to ~0 (~94% of canonical Hard gens, 0% mid-band). A monotonic rescale (temperature/Platt) CANNOT fix a
rank-based use (identical ranks); the fix must change the OBJECTIVE.

FIX (keeps the v2 anti-fingerprint win — NO generator in training, per stage2a_critic_findings.md):
  - GRADED corrupted-real: corrupt a FRACTION of note-frames (a ladder from real -> fully corrupt), NOT all-or-none.
  - WITHIN-SONG MARGIN-RANKING loss: the score must DECREASE monotonically along each song's corruption ladder
    (adjacent-pair margin ranking). Within-song pairs hold density/timing/audio FIXED -> the only cue is taste,
    exactly the v2 property, now graded. + a light BCE anchor on the ladder ENDS (real high / full-corrupt low) to
    pin the absolute scale (ranking alone is shift-invariant).
  - Score = the logit MARGIN (logit_real - logit_fake) of the SAME LateFusionClassifier arch (minimal change);
    warm-started from the existing binary critic (it already knows the taste DIRECTION; we teach it to SPREAD).

Two corruption axes (each a ladder from real): PANELS (arrow-choice taste) and SHIFT (audio-alignment taste).

Validate: the score must be MONOTONE + GRADED across ladder levels (not saturated), and later spread on real
generations. Usage:
    python experiments/realism_critic/train_graded_critic.py --data_dir data/ --audio_dir data/ \
        --max_train_songs 1200 --epochs 10
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

BINARY_CRITIC = "checkpoints/realism_critic/best_val.pt"          # warm-start: knows taste direction, saturated
PANEL_LADDER = [0.0, 0.2, 0.45, 0.7, 1.0]                          # fraction of note-frames panel-scrambled
SHIFT_LADDER = [0, 2, 6, 16]                                       # audio-shift offset (frames)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=10); p.add_argument('--batch_size', type=int, default=6)
    p.add_argument('--lr', type=float, default=1e-4); p.add_argument('--max_len', type=int, default=768)
    p.add_argument('--rank_margin', type=float, default=0.75)     # min logit-margin gap between adjacent levels
    p.add_argument('--anchor_w', type=float, default=0.3)         # weight on the end BCE anchor
    p.add_argument('--max_train_songs', type=int, default=1200); p.add_argument('--max_val_songs', type=int, default=250)
    p.add_argument('--cache_dir', default='cache/samples')        # 23-dim (critic space, matches deployed critic)
    p.add_argument('--checkpoint_dir', default='checkpoints/realism_critic_graded')
    p.add_argument('--no_warmstart', action='store_true')
    return p.parse_args()


def to_binary(typed):
    t = np.asarray(typed); return ((t == 1) | (t == 2) | (t == 4)).astype(np.float32)


def corrupt_panels_frac(chart, frac, rng):
    """Panel-scramble a FRACTION of note-frames (reassign active panels, keep per-frame count); rest stay REAL.
    frac=0 -> real, frac=1 -> full scramble. Density/timing/jump-rate preserved exactly (the v2 taste isolation)."""
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


def score_batch(critic, charts, audio_np, T, device):
    """Score a list of (T,4) charts sharing one song's audio -> logit-margin tensor (len(charts),)."""
    B = len(charts); a = torch.zeros(B, T, 23); c = torch.zeros(B, T, 4); m = torch.ones(B, T)
    for b, ch in enumerate(charts):
        a[b] = torch.from_numpy(audio_np); c[b] = torch.from_numpy(ch)
    logits = critic(a.to(device), c.to(device), m.to(device))
    if isinstance(logits, dict): logits = logits['logits']
    return logits[:, 1] - logits[:, 0]     # margin as the graded score


def ladder_charts(song, rng):
    """The two per-song ladders (real -> fully corrupt), each list monotonically WORSE."""
    real = song['real']
    panels = [corrupt_panels_frac(real, f, rng) for f in PANEL_LADDER]
    shift = [corrupt_shift(real, o) for o in SHIFT_LADDER]
    return panels, shift


def rank_and_anchor(margins, args):
    """margins: score per ladder level, level 0 = best (real). Monotone-decreasing margin-ranking + end anchor."""
    loss = margins.new_zeros(())
    for i in range(len(margins) - 1):                # adjacent pairs: better - worse >= rank_margin
        loss = loss + torch.relu(args.rank_margin - (margins[i] - margins[i + 1]))
    # anchor the ends to pin absolute scale: real (level 0) high, fully-corrupt (last) low
    bce = nn.functional.binary_cross_entropy_with_logits
    loss = loss + args.anchor_w * (bce(margins[0], margins.new_ones(())) + bce(margins[-1], margins.new_zeros(())))
    return loss


@torch.no_grad()
def evaluate(critic, val, device, rng):
    """Mean score per ladder LEVEL (pooled over songs) + monotonicity rate — the graded-ness gate."""
    critic.eval()
    pan = [[] for _ in PANEL_LADDER]; shf = [[] for _ in SHIFT_LADDER]; mono = 0; ntot = 0
    for song in val:
        panels, shift = ladder_charts(song, rng)
        mp = score_batch(critic, panels, song['audio'], song['T'], device).cpu().numpy()
        ms = score_batch(critic, shift, song['audio'], song['T'], device).cpu().numpy()
        for j, v in enumerate(mp): pan[j].append(v)
        for j, v in enumerate(ms): shf[j].append(v)
        mono += int(np.all(np.diff(mp) <= 0)); ntot += 1
    pan_m = [float(np.mean(x)) for x in pan]; shf_m = [float(np.mean(x)) for x in shf]
    return pan_m, shf_m, mono / max(ntot, 1)


def main():
    args = parse_args(); set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob(f"{args.data_dir}/**/*.sm", recursive=True) + glob.glob(f"{args.data_dir}/**/*.ssc", recursive=True)
    train_files, val_files, _ = create_data_splits(cf, random_state=args.seed)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    train_ds, val_ds, _ = create_datasets(train_files=train_files, val_files=val_files, test_files=[],
                                          audio_dir=args.audio_dir, max_sequence_length=msl, cache_dir=args.cache_dir)
    # NO warm_cache (it eagerly extracts the whole ~4452-file train set = ~30min CPU); collect() reads only the
    # capped songs ONCE into memory via lazy ds[i], so every epoch trains on the in-memory copy.
    train = collect(train_ds, args.max_train_songs, args.max_len)
    val = collect(val_ds, args.max_val_songs, args.max_len)
    print(f"train songs={len(train)} val songs={len(val)}  ladders: panels{PANEL_LADDER} shift{SHIFT_LADDER}")

    cfg = dict(yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier'])
    cfg['num_classes'] = 2; cfg['head_type'] = 'classification'
    cfg['use_groove_radar'] = False; cfg['use_projection_head'] = False
    critic = LateFusionClassifier(cfg).to(device)
    if not args.no_warmstart and Path(PROJECT_ROOT / BINARY_CRITIC).exists():
        sd = torch.load(PROJECT_ROOT / BINARY_CRITIC, map_location=device)['model_state_dict']
        critic.load_state_dict(sd); print(f"warm-started from {BINARY_CRITIC} (the saturated binary critic)")
    opt = torch.optim.AdamW(critic.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_spread = -1.0; best_path = Path(args.checkpoint_dir) / "best_val.pt"

    for epoch in range(args.epochs):
        critic.train(); order = list(range(len(train))); rng.shuffle(order); tot = 0.0; nb = 0
        for k in range(0, len(order), args.batch_size):
            opt.zero_grad(); batch_loss = torch.zeros((), device=device)
            for idx in order[k:k + args.batch_size]:
                song = train[idx]; panels, shift = ladder_charts(song, rng)
                mp = score_batch(critic, panels, song['audio'], song['T'], device)
                ms = score_batch(critic, shift, song['audio'], song['T'], device)
                batch_loss = batch_loss + rank_and_anchor(mp, args) + rank_and_anchor(ms, args)
            batch_loss = batch_loss / len(order[k:k + args.batch_size])
            batch_loss.backward(); torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0); opt.step()
            tot += float(batch_loss); nb += 1
        pan_m, shf_m, mono = evaluate(critic, val, device, rng)
        spread = pan_m[0] - pan_m[-1]                    # real vs fully-corrupt margin gap (want LARGE + graded)
        graded = " GRADED" if all(pan_m[i] > pan_m[i+1] for i in range(len(pan_m)-1)) else " (not monotone!)"
        print(f"epoch {epoch+1}/{args.epochs} loss={tot/max(nb,1):.3f} | panels[" +
              " ".join(f"{v:+.2f}" for v in pan_m) + f"] mono={mono:.2f}{graded} | shift[" +
              " ".join(f"{v:+.2f}" for v in shf_m) + "]" + ("  *" if spread > best_spread else ""))
        if spread > best_spread:
            best_spread = spread
            torch.save({'model_state_dict': critic.state_dict(), 'config': cfg, 'epoch': epoch,
                        'panel_ladder': PANEL_LADDER, 'panel_means': pan_m, 'val_mono': mono}, best_path)

    print(f"\nbest real-vs-corrupt spread={best_spread:.2f} -> {best_path}")
    print("Next: re-run probe_quality_features.py with this critic (margin score) — the target should now be "
          "NON-saturating across canonical generations; then the feature attribution is meaningful.")


if __name__ == '__main__':
    main()

"""critic-v3 trainer (2026-07-13, taste-critic arc) — the FULL rebuild the user chose.

WindowedLocalCritic on the 48th-grid v2 cache, from scratch (warm-start broken by the
42-dim audio + typed chart change). Closes gaps (i) chart-typed, (ii) 42-dim audio,
(iv) length/locality. The preference objective (E2) rides on top LATER once E0.1 labels exist.

Objective (pre-labels): graded corruption LADDER (rank + anchor, like graded_v2) on the
SOFT-MIN song score, PLUS a LOCALITY term — a tail-only jitter must drop the TAIL windows
(not the body), training the per-window head to localize (the thing the old global-pool
critic structurally couldn't do).

Run:  python experiments/realism_critic/train_critic_v3.py --epochs 12 --max_train_songs 1200
Smoke: python experiments/realism_critic/train_critic_v3.py --epochs 1 --max_train_songs 40 --max_val_songs 20
"""
import warnings, os, argparse, sys, glob
from pathlib import Path
import numpy as np, torch, torch.nn as nn

ROOT = Path('/home/ybx/code/stepmania-chart-generator'); sys.path.insert(0, str(ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.data.stepmania_parser import StepManiaParser
from src.generation.decode_harness import make_feature_extractor
from experiments.realism_critic.windowed_critic import WindowedLocalCritic

SUBDIV = 12; SIXTEENTH = {0, 3, 6, 9}; V2_MSL = 5400   # v2 model-sequence-length (48th grid)
JITTER_LADDER = [0.0, 0.25, 0.5, 0.75, 1.0]
PANEL_LADDER = [0.0, 0.25, 0.5, 1.0]
SHIFT_LADDER = [0, 1, 3, 6]
bce = nn.functional.binary_cross_entropy_with_logits
relu = torch.relu


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='checkpoints/realism_critic_v3')
    p.add_argument('--data_dir', default='data')
    p.add_argument('--audio_dir', default='data')
    p.add_argument('--cache_dir', default='cache/samples_v3_48th')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch_songs', type=int, default=4)      # songs accumulated per optim step
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--max_len', type=int, default=5400)       # SAFETY cap only (windowing is length-agnostic)
    p.add_argument('--max_train_songs', type=int, default=1200)
    p.add_argument('--max_val_songs', type=int, default=200)
    p.add_argument('--rank_margin', type=float, default=0.75)
    p.add_argument('--anchor_w', type=float, default=0.5)
    p.add_argument('--loc_w', type=float, default=1.0)        # locality term weight
    p.add_argument('--loc_margin', type=float, default=0.5)   # tail(real) must beat tail(corrupt) by this
    p.add_argument('--body_w', type=float, default=0.5)       # penalize body-window drift under a tail corruption
    p.add_argument('--tail_frac', type=float, default=0.30)   # last 30% = the "tail" region
    p.add_argument('--softmin_beta', type=float, default=1.0)
    p.add_argument('--patience', type=int, default=3)
    return p.parse_args()


# ---- typed-grid corruptions (operate on (T,4) int codes {0..4}; NOT binarized) ----
def _note_rows(typed):
    return np.where((typed != 0).any(1))[0]


def corrupt_panels_typed(typed, frac, rng):
    if frac <= 0: return typed.copy()
    out = typed.copy()
    rows = _note_rows(typed)
    for t, h in zip(rows, rng.random(len(rows)) < frac):
        if not h: continue
        syms = out[t][out[t] != 0]
        out[t] = 0
        out[t, rng.choice(4, size=len(syms), replace=False)] = syms   # same symbols, shuffled panels
    return out


def corrupt_shift_typed(typed, off):
    if off <= 0 or len(typed) < 32: return typed.copy()
    return np.roll(typed, int(off), axis=0)


def corrupt_jitter_typed(typed, frac, rng, rows_filter=None):
    """Displace on-16th note-rows +-1 frame to a pure-48th cell (the sub-16th degradation)."""
    if frac <= 0 or len(typed) < 4: return typed.copy()
    out = typed.copy()
    rows = [t for t in _note_rows(typed) if (t % SUBDIV) in SIXTEENTH]
    if rows_filter is not None:
        rows = [t for t in rows if rows_filter(t)]
    for t in rows:
        if rng.random() >= frac: continue
        d = int(rng.choice([-1, 1])); nt = t + d
        if nt < 0 or nt >= len(out) or (out[nt] != 0).any(): continue
        out[nt] = out[t]; out[t] = 0
    return out


def ladders(typed, rng):
    return {'jitter': [corrupt_jitter_typed(typed, f, rng) for f in JITTER_LADDER],
            'panel':  [corrupt_panels_typed(typed, f, rng) for f in PANEL_LADDER],
            'shift':  [corrupt_shift_typed(typed, o) for o in SHIFT_LADDER]}


# ---- data ----
def collect(ds, cap, max_len):
    out = []
    for i in range(len(ds)):
        if len(out) >= cap: break
        s = ds[i]
        if s is None: continue
        meta = ds.valid_samples[i]; T = min(int(s['mask'].sum().item()), max_len)
        if T < 128: continue
        nd = next((n for n in meta['chart'].note_data
                   if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        typed = ds.parser.convert_to_tensor_typed(meta['chart'], nd)[:T].astype(np.int64)
        if not (typed != 0).any(): continue
        out.append({'audio': s['audio'][:T].numpy().astype(np.float32), 'typed': typed, 'T': T})
    return out


def to_t(typed, dev):
    return torch.from_numpy(typed).long().to(dev)


def score_ladder(model, audio_t, typed_list, dev):
    ch = torch.stack([to_t(c, dev) for c in typed_list])          # (B,T,4)
    a = audio_t.expand(ch.shape[0], -1, -1)                       # (B,T,42)
    return model(a, ch)                                           # (B,)


def rank_and_anchor(margins, args):
    loss = margins.new_zeros(())
    for i in range(len(margins) - 1):
        loss = loss + relu(args.rank_margin - (margins[i] - margins[i + 1]))
    loss = loss + args.anchor_w * (bce(margins[0], margins.new_ones(())) +
                                   bce(margins[-1], margins.new_zeros(())))
    return loss


def locality_term(model, audio_t, typed, rng, args, dev):
    T = typed.shape[0]; tail_start = int(T * (1 - args.tail_frac))
    corrupt = corrupt_jitter_typed(typed, 1.0, rng, rows_filter=lambda t: t >= tail_start)
    ch = torch.stack([to_t(typed, dev), to_t(corrupt, dev)])     # (2,T,4)
    _, allm, layout = model(audio_t.expand(2, -1, -1), ch, return_windows=True)
    real_w, corr_w = allm[0], allm[1]
    overlaps_tail = torch.tensor([e > tail_start for (s, e, W) in layout], device=dev)
    loss = relu(args.loc_margin - (real_w[overlaps_tail] - corr_w[overlaps_tail])).mean()
    if (~overlaps_tail).any():
        loss = loss + args.body_w * (real_w[~overlaps_tail] - corr_w[~overlaps_tail]).abs().mean()
    return loss


@torch.no_grad()
def evaluate(model, val, dev, rng, args):
    model.eval()
    mono = {k: 0 for k in ('jitter', 'panel', 'shift')}
    means = {k: [] for k in mono}
    tail_drop, body_drift = [], []
    for song in val:
        at = torch.from_numpy(song['audio']).float().to(dev)[None]
        for k, charts in ladders(song['typed'], rng).items():
            m = score_ladder(model, at, charts, dev).cpu().numpy()
            mono[k] += int(np.all(np.diff(m) <= 1e-6))
            means[k].append(m)
        # length/locality gate
        T = song['typed'].shape[0]; ts = int(T * (1 - args.tail_frac))
        corrupt = corrupt_jitter_typed(song['typed'], 1.0, rng, rows_filter=lambda t: t >= ts)
        ch = torch.stack([to_t(song['typed'], dev), to_t(corrupt, dev)])
        _, allm, layout = model(at.expand(2, -1, -1), ch, return_windows=True)
        ot = torch.tensor([e > ts for (s, e, W) in layout], device=dev)
        tail_drop.append(float((allm[0][ot] - allm[1][ot]).mean()))
        if (~ot).any():
            body_drift.append(float((allm[0][~ot] - allm[1][~ot]).abs().mean()))
    n = max(len(val), 1)
    monor = {k: mono[k] / n for k in mono}
    laddm = {k: np.mean(np.stack(means[k]), 0).tolist() for k in means}
    return monor, laddm, float(np.mean(tail_drop)), float(np.mean(body_drift) if body_drift else 0.0)


def main():
    args = parse()
    set_seed(args.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    warnings.filterwarnings('ignore')

    cf = (glob.glob(f"{args.data_dir}/**/*.sm", recursive=True) +
          glob.glob(f"{args.data_dir}/**/*.ssc", recursive=True))
    train_files, val_files, _ = create_data_splits(cf, random_state=args.seed)   # seed 42 == v2-train split
    spec = make_feature_extractor("highres_v2")                                   # 48th grid, 42-dim
    train_ds, val_ds, _ = create_datasets(train_files=train_files, val_files=val_files, test_files=[],
                                          audio_dir=args.audio_dir, max_sequence_length=V2_MSL,
                                          feature_extractor=spec.extractor, cache_dir=args.cache_dir,
                                          parser=StepManiaParser.for_v2())        # == the cached grid (index-keyed)
    train = collect(train_ds, args.max_train_songs, args.max_len)
    val = collect(val_ds, args.max_val_songs, args.max_len)
    print(f"train songs={len(train)} val songs={len(val)} (48th grid, full 42-dim audio, typed chart, windowed-local)")

    model = WindowedLocalCritic(audio_dim=42, softmin_beta=args.softmin_beta).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    best, bad = -1e9, 0
    for ep in range(args.epochs):
        model.train()
        rng.shuffle(train)
        running = 0.0; nb = 0
        opt.zero_grad()
        for si, song in enumerate(train):
            at = torch.from_numpy(song['audio']).float().to(dev)[None]
            loss = at.new_zeros(())
            for k, charts in ladders(song['typed'], rng).items():
                loss = loss + rank_and_anchor(score_ladder(model, at, charts, dev), args)
            loss = loss + args.loc_w * locality_term(model, at, song['typed'], rng, args, dev)
            (loss / args.batch_songs).backward()
            running += float(loss); nb += 1
            if (si + 1) % args.batch_songs == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()

        monor, laddm, tail_drop, body_drift = evaluate(model, val, dev, rng, args)
        val_metric = monor['jitter']            # R1 gate = jitter monotonicity (sees f48)
        print(f"[ep {ep}] loss={running/max(nb,1):.3f} | mono jit={monor['jitter']:.2f} "
              f"pan={monor['panel']:.2f} shf={monor['shift']:.2f} | "
              f"tail_drop={tail_drop:+.2f} body_drift={body_drift:.2f}")
        print(f"        jitter ladder means = {[round(x,2) for x in laddm['jitter']]}")
        if val_metric > best:
            best = val_metric; bad = 0
            torch.save({'state_dict': model.state_dict(),
                        'audio_dim': 42, 'grid': '48th', 'subdiv': SUBDIV,
                        'scales': list(model.scales), 'softmin_beta': args.softmin_beta,
                        'val_jitter_mono': best, 'tail_drop': tail_drop},
                       os.path.join(args.out, 'best_val.pt'))
            print(f"        saved best (jitter mono {best:.2f}, tail_drop {tail_drop:+.2f})")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop (no jitter-mono improvement in {args.patience})"); break
    print(f"DONE best jitter-mono={best:.2f} -> {args.out}/best_val.pt")


if __name__ == "__main__":
    main()

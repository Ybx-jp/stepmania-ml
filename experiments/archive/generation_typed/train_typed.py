#!/usr/bin/env python3
"""
Train the typed (multi-step-type) generator.

Same factorization as the focal factorized model, warm-started from its checkpoint,
but the panel head predicts a per-panel symbol {none,tap,hold-head,tail,roll}. Onset
head (binary) unchanged. Panel loss = per-panel CE over 5 symbols on onset frames with
class weighting (holds are ~20x rarer than taps); onset loss = focal.

Key question: does the model produce holds (symbols 2/3) and do they round-trip to
playable .sm? Eval reports onset_F1/density/critic (binarized for the Phase-1 critic),
generated symbol histogram, hold orphan rate, and teacher-forced per-symbol recall.

Usage:
    python experiments/generation_typed/train_typed.py --data_dir data/ --audio_dir data/ \
        --epochs 20 --warmup_freeze 3 --batch_size 8
"""

import warnings, os
warnings.filterwarnings('ignore')
os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'

import argparse, glob, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.generation.typed_model import TypedChartGenerator
from src.generation.typed import NUM_SYMBOLS, NUM_PANELS, SYMBOL_NAMES, symbol_histogram, pair_holds
from src.generation.evaluation import onset_density_metrics, DifficultyCritic

DEFAULT_BPM = 150.0
FOCAL_CKPT = "checkpoints/gen_factorized_focal/best_val.pt"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--max_train_len', type=int, default=1024)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup_freeze', type=int, default=3)
    p.add_argument('--focal_gamma', type=float, default=2.0)
    p.add_argument('--panel_loss', choices=['weighted_ce', 'focal'], default='focal',
                   help='focal = calibration-preserving (recommended); weighted_ce over-corrects')
    p.add_argument('--panel_weight', choices=['inv', 'inv_sqrt', 'none'], default='inv_sqrt',
                   help='rare-symbol weighting scheme; inv_sqrt is milder than inv')
    p.add_argument('--eval_songs', type=int, default=64)
    p.add_argument('--max_gen_len', type=int, default=768)
    p.add_argument('--checkpoint_dir', default='checkpoints/gen_typed')
    return p.parse_args()


def typed_binary(typed: np.ndarray) -> np.ndarray:
    """Binarize a typed chart for the Phase-1 (binary) difficulty critic: tap/hold-head/
    roll-head -> 1, tail/none -> 0 (matches the frozen convert_to_tensor mapping)."""
    t = np.asarray(typed)
    return ((t == 1) | (t == 2) | (t == 4)).astype(np.float32)


def collect_typed(ds, cap):
    """Pair the dataset's cached aligned audio (ds[i]) with typed charts re-derived from
    the same parse (convert_to_tensor_typed). ds[i] hits the warm cache (fast); the typed
    chart and the cached audio share length = chart.timesteps_total, so they align."""
    out = []
    for i in range(len(ds)):
        sample = ds[i]                       # cached: fast, index-aligned (no retry on warm cache)
        meta = ds.valid_samples[i]
        T = int(sample['mask'].sum().item())
        nd = next((n for n in meta['chart'].note_data
                   if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None:
            continue
        typed_full = ds.parser.convert_to_tensor_typed(meta['chart'], nd)
        T = min(T, cap, typed_full.shape[0])
        out.append({'typed': typed_full[:T].astype(np.int64),
                    'audio': sample['audio'][:T].numpy().astype(np.float32),
                    'difficulty': int(meta['difficulty_class'])})
    return out


def focal_bce(logits, targets, gamma):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    return ((1 - p_t) ** gamma * bce).mean()


def focal_ce(logits, targets, gamma, weight=None):
    """Multi-class focal loss: (1-p_t)^gamma * CE. Down-weights easy/confident frames so
    rare classes (holds) get learned without the generation-time over-prediction that heavy
    class weights cause. Optional mild per-class weight."""
    logp = nn.functional.log_softmax(logits, dim=-1)
    logp_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
    p_t = logp_t.exp()
    loss = -((1 - p_t) ** gamma) * logp_t
    if weight is not None:
        loss = loss * weight[targets]
    return loss.mean()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(args.seed)

    chart_files = glob.glob(f"{args.data_dir}/**/*.sm", recursive=True)
    chart_files += glob.glob(f"{args.data_dir}/**/*.ssc", recursive=True)
    train_files, val_files, _ = create_data_splits(chart_files, random_state=args.seed)
    with open(PROJECT_ROOT / "config/model_config.yaml") as f:
        max_seq_len = yaml.safe_load(f)['classifier']['max_sequence_length']
    train_ds, val_ds, _ = create_datasets(train_files=train_files, val_files=val_files,
                                          test_files=[], audio_dir=args.audio_dir,
                                          max_sequence_length=max_seq_len, cache_dir='cache/samples')
    print("Warming caches..."); train_ds.warm_cache(show_progress=True); val_ds.warm_cache(show_progress=True)
    print("Collecting typed samples...")
    train = collect_typed(train_ds, args.max_train_len)
    val = collect_typed(val_ds, args.max_train_len)
    print(f"train={len(train)} val={len(val)}")
    audio_dim = train[0]['audio'].shape[1]

    # report symbol distribution in training data
    tot = {k: 0 for k in SYMBOL_NAMES}
    for s in train:
        h = symbol_histogram(s['typed'])
        for k in tot: tot[k] += h[k]
    print("train symbols:", {k: f"{v}" for k, v in tot.items()})

    # per-panel symbol class weights on onset frames (balanced; roll has ~0 count)
    sym_counts = np.ones(NUM_SYMBOLS)
    for s in train:
        on = (s['typed'] != 0).any(1)
        vals = s['typed'][on].reshape(-1)
        sym_counts += np.bincount(vals, minlength=NUM_SYMBOLS)
    balanced = sym_counts.sum() / (NUM_SYMBOLS * sym_counts)
    if args.panel_weight == 'inv':
        sym_w = np.clip(balanced, 0, 20.0)
    elif args.panel_weight == 'inv_sqrt':
        sym_w = np.clip(np.sqrt(balanced), 0, 5.0)
    else:  # none
        sym_w = np.ones(NUM_SYMBOLS)
    panel_weight = torch.tensor(sym_w, dtype=torch.float32, device=device)
    print(f"panel loss={args.panel_loss} weight={args.panel_weight}:",
          {SYMBOL_NAMES[i]: round(float(sym_w[i]), 2) for i in range(NUM_SYMBOLS)})

    target_density = float(np.mean([(s['typed'] != 0).any(1).mean() for s in val]))

    try:
        import mlflow; mlflow.set_experiment("stepmania-chart-generator"); mlflow_on = True
    except ImportError:
        mlflow_on = False

    model = TypedChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=4, onset_layers=2).to(device)
    ck = torch.load(FOCAL_CKPT, map_location='cpu', weights_only=False)
    print(f"warm-started {model.load_compatible(ck['model_state_dict'])} tensors from focal checkpoint")
    model.freeze_audio_encoder(True)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    ce_crit = nn.CrossEntropyLoss(weight=panel_weight)

    def panel_loss_fn(logits, targets):
        if args.panel_loss == 'focal':
            return focal_ce(logits, targets, args.focal_gamma, weight=panel_weight)
        return ce_crit(logits, targets)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def batches(samples, bs, shuffle):
        idx = np.arange(len(samples))
        if shuffle: rng.shuffle(idx)
        for i in range(0, len(idx), bs):
            yield [samples[j] for j in idx[i:i + bs]]

    def to_tensors(batch):
        T = max(len(s['typed']) for s in batch); B = len(batch)
        audio = torch.zeros(B, T, audio_dim)
        states = torch.zeros(B, T, NUM_PANELS, dtype=torch.long)
        mask = torch.zeros(B, T, dtype=torch.bool)
        diff = torch.zeros(B, dtype=torch.long)
        for b, s in enumerate(batch):
            t = len(s['typed'])
            audio[b, :t] = torch.from_numpy(s['audio'])
            states[b, :t] = torch.from_numpy(s['typed'])
            mask[b, :t] = True; diff[b] = s['difficulty']
        return audio.to(device), states.to(device), mask.to(device), diff.to(device)

    def losses(ol, pl, states, mask):
        onset_t = (states != 0).any(-1).float()
        o_loss = focal_bce(ol[mask], onset_t[mask], args.focal_gamma)
        sel = mask & (onset_t > 0.5)
        if sel.any():
            p_loss = panel_loss_fn(pl[sel].reshape(-1, NUM_SYMBOLS), states[sel].reshape(-1))
        else:
            p_loss = torch.zeros((), device=device)
        return o_loss, p_loss

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best = float('inf'); best_path = Path(args.checkpoint_dir) / "best_val.pt"
    if mlflow_on:
        mlflow.start_run(run_name="gen-typed")
        mlflow.log_params({'epochs': args.epochs, 'lr': args.lr, 'focal_gamma': args.focal_gamma})

    for epoch in range(args.epochs):
        if epoch == args.warmup_freeze:
            model.freeze_audio_encoder(False); print(f"  [epoch {epoch+1}] unfroze audio encoder")
        model.train(); tro = trp = 0.0; nb = 0
        for batch in batches(train, args.batch_size, True):
            audio, states, mask, diff = to_tensors(batch)
            optimizer.zero_grad()
            ol, pl = model(audio, states, diff, mask)
            ol_l, pl_l = losses(ol, pl, states, mask)
            (ol_l + pl_l).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            tro += ol_l.item(); trp += pl_l.item(); nb += 1
        model.eval(); vo = vp = 0.0; vnb = 0
        with torch.no_grad():
            for batch in batches(val, args.batch_size, False):
                audio, states, mask, diff = to_tensors(batch)
                ol, pl = model(audio, states, diff, mask)
                ol_l, pl_l = losses(ol, pl, states, mask)
                vo += ol_l.item(); vp += pl_l.item(); vnb += 1
        vo /= max(1, vnb); vp /= max(1, vnb); vt = vo + vp
        print(f"  epoch {epoch+1}/{args.epochs}  train(o={tro/max(1,nb):.4f} p={trp/max(1,nb):.4f})  "
              f"val(o={vo:.4f} p={vp:.4f} tot={vt:.4f})" + ("  *" if vt < best else ""))
        if mlflow_on: mlflow.log_metrics({'val_onset': vo, 'val_panel': vp, 'val_total': vt}, step=epoch)
        if vt < best:
            best = vt
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'val_total': vt}, best_path)

    # ---- eval ----
    print(f"\nLoading best (val_total={best:.4f}) for eval...")
    model.load_state_dict(torch.load(best_path, map_location=device)['model_state_dict']); model.eval()
    eval_set = val[:args.eval_songs]
    critic = DifficultyCritic(device=device)

    # teacher-forced per-symbol recall (does the panel head learn holds?)
    sym_correct = np.zeros(NUM_SYMBOLS); sym_total = np.zeros(NUM_SYMBOLS)
    with torch.no_grad():
        for batch in batches(eval_set, args.batch_size, False):
            audio, states, mask, diff = to_tensors(batch)
            _, pl = model(audio, states, diff, mask)
            pred = pl.argmax(-1)
            on = (states != 0).any(-1) & mask
            for sym in range(NUM_SYMBOLS):
                m = on.unsqueeze(-1) & (states == sym)
                sym_total[sym] += int(m.sum())
                sym_correct[sym] += int((m & (pred == sym)).sum())
    recall = {SYMBOL_NAMES[i]: (sym_correct[i] / sym_total[i] if sym_total[i] else float('nan'))
              for i in range(NUM_SYMBOLS)}

    # generation: onset threshold to match density, per-panel greedy
    onset_logits_all = []
    with torch.no_grad():
        for s in eval_set:
            a = torch.from_numpy(s['audio']).unsqueeze(0).to(device)
            d = torch.tensor([s['difficulty']], device=device)
            onset_logits_all.append(torch.sigmoid(model.onset_logits(model.encode_audio(a), d))[0].cpu().numpy())
    tau = float(np.quantile(np.concatenate(onset_logits_all), 1 - target_density))

    f1s, dens, gen_syms, preds, tgts, holds, orphans = [], [], {k: 0 for k in SYMBOL_NAMES}, [], [], 0, 0
    for i in range(0, len(eval_set), 8):
        batch = eval_set[i:i + 8]
        L = min(args.max_gen_len, max(len(s['typed']) for s in batch)); B = len(batch)
        audio = torch.zeros(B, L, audio_dim); lengths = torch.zeros(B, dtype=torch.long); diff = torch.zeros(B, dtype=torch.long)
        for b, s in enumerate(batch):
            t = min(len(s['typed']), L); audio[b, :t] = torch.from_numpy(s['audio'][:t]); lengths[b] = t; diff[b] = s['difficulty']
        gen = model.generate(audio.to(device), diff.to(device), lengths=lengths.to(device),
                             onset_threshold=tau).cpu().numpy()
        for b, s in enumerate(batch):
            t = int(lengths[b]); g_raw = gen[b, :t]
            # raw orphan rate (model quality), then pair holds -> always-valid chart
            for pnl in range(NUM_PANELS):
                col = g_raw[:, pnl]; h = int(((col == 2) | (col == 4)).sum()); tl = int((col == 3).sum())
                holds += h; orphans += abs(h - tl)
            g = pair_holds(g_raw)
            m = onset_density_metrics((g != 0).astype(np.float32), reference=(s['typed'][:t] != 0).astype(np.float32))
            f1s.append(m['onset_f1']); dens.append((g != 0).any(1).mean())
            for k, v in symbol_histogram(g).items(): gen_syms[k] += v
            preds.append(critic.predict(typed_binary(g), s['audio'][:t], bpm=DEFAULT_BPM)['class']); tgts.append(s['difficulty'])
    dd = np.abs(np.array(preds) - np.array(tgts))

    print("\n" + "=" * 76)
    print("  TYPED GENERATOR")
    print("=" * 76)
    print(f"  onset_F1 {np.mean(f1s):.3f}  density {np.mean(dens):.3f} (real {target_density:.3f})  "
          f"crit_adj {np.mean(dd<=1):.3f}  crit_mae {np.mean(dd):.3f}")
    print(f"  teacher-forced per-symbol recall: " +
          " ".join(f"{k}={recall[k]:.2f}" for k in SYMBOL_NAMES))
    print(f"  generated symbols: {gen_syms}")
    print(f"  generated holds (heads): {holds}  head/tail mismatch (orphans): {orphans}")
    print("=" * 76)
    if mlflow_on:
        mlflow.log_metrics({'gen_onset_f1': float(np.mean(f1s)), 'gen_density': float(np.mean(dens)),
                            'crit_adj': float(np.mean(dd <= 1)), 'hold_recall': float(recall['hold_head'])})
        mlflow.end_run()


if __name__ == '__main__':
    main()

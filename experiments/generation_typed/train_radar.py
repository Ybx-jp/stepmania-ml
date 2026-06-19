#!/usr/bin/env python3
"""
Train the LAYERED typed generator: onset -> pattern (which panels, 15-way) -> type
(per active panel: tap/hold-head/tail/roll, 4-way). Decouples is-panel-active from
what-type, fixing the none-bias of the flat per-panel 5-way head.

Pattern head warm-starts from the binary factorized panel_head (was crit_adj 0.93);
type head is new and small (only tap-vs-holds imbalance, ~20:1, no none).

Usage:
    python experiments/generation_typed/train_layered.py --data_dir data/ --audio_dir data/ \
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
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.typed import (NUM_PATTERNS, NUM_TYPES, NUM_PANELS, SYMBOL_NAMES,
                                   symbol_histogram, pair_holds, panels_to_pattern)
from src.generation.evaluation import onset_density_metrics, DifficultyCritic

DEFAULT_BPM = 150.0
FOCAL_CKPT = "checkpoints/gen_layered/best_val.pt"  # warm-start from the trained layered model
TYPE_NAMES = ['tap', 'hold_head', 'tail', 'roll_head']


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
    p.add_argument('--type_weight', choices=['none', 'inv_sqrt'], default='none',
                   help='none = calibrated focal (sample types at true rate); inv_sqrt over-inflates holds')
    p.add_argument('--eval_songs', type=int, default=64)
    p.add_argument('--max_gen_len', type=int, default=768)
    p.add_argument('--checkpoint_dir', default='checkpoints/gen_radar')
    p.add_argument('--cfg_drop', type=float, default=0.15, help='classifier-free guidance: prob of dropping radar conditioning per batch')
    return p.parse_args()


def typed_binary(typed):
    t = np.asarray(typed)
    return ((t == 1) | (t == 2) | (t == 4)).astype(np.float32)


def collect_typed(ds, cap):
    out = []
    for i in range(len(ds)):
        sample = ds[i]; meta = ds.valid_samples[i]
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
                    'difficulty': int(meta['difficulty_class']),
                    'radar': meta['groove_radar'].to_vector().astype(np.float32)})  # (5,) normalized target
    return out


def focal_bce(logits, targets, gamma):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits); p_t = p * targets + (1 - p) * (1 - targets)
    return ((1 - p_t) ** gamma * bce).mean()


def focal_ce(logits, targets, gamma, weight=None):
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
    target_density = float(np.mean([(s['typed'] != 0).any(1).mean() for s in val]))

    try:
        import mlflow; mlflow.set_experiment("stepmania-chart-generator"); mlflow_on = True
    except ImportError:
        mlflow_on = False

    model = LayeredTypedChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=4, onset_layers=2).to(device)
    ck = torch.load(FOCAL_CKPT, map_location='cpu', weights_only=False)
    print(f"warm-started {model.load_from_factorized(ck['model_state_dict'])} tensors (incl. pattern_head)")
    model.freeze_audio_encoder(True)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # mild inv_sqrt class weights for the 4 TYPES (tap dominant ~20:1) on active panels
    type_counts = np.ones(NUM_TYPES)
    for s in train:
        act = s['typed'][s['typed'] != 0]  # active-panel symbols 1..4
        type_counts += np.bincount(act - 1, minlength=NUM_TYPES)
    if args.type_weight == 'inv_sqrt':
        type_w = np.clip(np.sqrt(type_counts.sum() / (NUM_TYPES * type_counts)), 0, 5.0)
    else:  # none -> uniform (calibrated focal; rely on sampling for hold rate)
        type_w = np.ones(NUM_TYPES)
    type_weight = torch.tensor(type_w, dtype=torch.float32, device=device)
    print("type class weights:", {TYPE_NAMES[i]: round(float(type_w[i]), 2) for i in range(NUM_TYPES)})

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def batches(samples, bs, shuffle):
        idx = np.arange(len(samples))
        if shuffle: rng.shuffle(idx)
        for i in range(0, len(idx), bs):
            yield [samples[j] for j in idx[i:i + bs]]

    def to_tensors(batch):
        T = max(len(s['typed']) for s in batch); B = len(batch)
        audio = torch.zeros(B, T, audio_dim); states = torch.zeros(B, T, NUM_PANELS, dtype=torch.long)
        mask = torch.zeros(B, T, dtype=torch.bool); diff = torch.zeros(B, dtype=torch.long)
        radar = torch.zeros(B, 5)
        for b, s in enumerate(batch):
            t = len(s['typed'])
            audio[b, :t] = torch.from_numpy(s['audio']); states[b, :t] = torch.from_numpy(s['typed'])
            mask[b, :t] = True; diff[b] = s['difficulty']; radar[b] = torch.from_numpy(s['radar'])
        # pattern target (B,T) and type target (B,T,4)
        active = (states != 0)
        pat = torch.from_numpy(panels_to_pattern(active.numpy())).clamp(min=0)  # (B,T)
        typ = (states - 1).clamp(min=0)  # (B,T,4) symbol->type idx (valid on active)
        return (audio.to(device), states.to(device), mask.to(device), diff.to(device),
                pat.to(device), typ.to(device), active.to(device), radar.to(device))

    def losses(ol, pat_l, typ_l, states, mask, pat_t, typ_t, active):
        onset_t = (states != 0).any(-1).float()
        o = focal_bce(ol[mask], onset_t[mask], args.focal_gamma)
        sel = mask & (onset_t > 0.5)
        p = focal_ce(pat_l[sel], pat_t[sel], args.focal_gamma) if sel.any() else torch.zeros((), device=device)
        act = active & mask.unsqueeze(-1)
        t = focal_ce(typ_l[act], typ_t[act], args.focal_gamma, weight=type_weight) if act.any() else torch.zeros((), device=device)
        return o, p, t

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best = float('inf'); best_path = Path(args.checkpoint_dir) / "best_val.pt"
    if mlflow_on:
        mlflow.start_run(run_name="gen-radar"); mlflow.log_params({'epochs': args.epochs, 'lr': args.lr})

    for epoch in range(args.epochs):
        if epoch == args.warmup_freeze:
            model.freeze_audio_encoder(False); print(f"  [epoch {epoch+1}] unfroze audio encoder")
        model.train(); tr = [0.0, 0.0, 0.0]; nb = 0
        for batch in batches(train, args.batch_size, True):
            audio, states, mask, diff, pat_t, typ_t, active, radar = to_tensors(batch)
            optimizer.zero_grad()
            cond_radar = None if rng.random() < args.cfg_drop else radar  # CFG: drop conditioning sometimes
            ol, pat_l, typ_l = model(audio, states, diff, mask, radar=cond_radar)
            o, p, t = losses(ol, pat_l, typ_l, states, mask, pat_t, typ_t, active)
            (o + p + t).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            tr[0] += o.item(); tr[1] += p.item(); tr[2] += t.item(); nb += 1
        model.eval(); v = [0.0, 0.0, 0.0]; vnb = 0
        with torch.no_grad():
            for batch in batches(val, args.batch_size, False):
                audio, states, mask, diff, pat_t, typ_t, active, radar = to_tensors(batch)
                ol, pat_l, typ_l = model(audio, states, diff, mask, radar=radar)
                o, p, t = losses(ol, pat_l, typ_l, states, mask, pat_t, typ_t, active)
                v[0] += o.item(); v[1] += p.item(); v[2] += t.item(); vnb += 1
        v = [x / max(1, vnb) for x in v]; vt = sum(v)
        print(f"  epoch {epoch+1}/{args.epochs}  train(o={tr[0]/max(1,nb):.3f} pat={tr[1]/max(1,nb):.3f} typ={tr[2]/max(1,nb):.3f})  "
              f"val(o={v[0]:.3f} pat={v[1]:.3f} typ={v[2]:.3f} tot={vt:.3f})" + ("  *" if vt < best else ""))
        if mlflow_on: mlflow.log_metrics({'val_onset': v[0], 'val_pattern': v[1], 'val_type': v[2], 'val_total': vt}, step=epoch)
        if vt < best:
            best = vt
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'val_total': vt}, best_path)

    # ---- eval ----
    print(f"\nLoading best (val_total={best:.4f}) for eval...")
    model.load_state_dict(torch.load(best_path, map_location=device)['model_state_dict']); model.eval()
    eval_set = val[:args.eval_songs]
    critic = DifficultyCritic(device=device)

    # teacher-forced type recall on active panels (does it learn holds without none-bias?)
    tc = np.zeros(NUM_TYPES); tt = np.zeros(NUM_TYPES)
    with torch.no_grad():
        for batch in batches(eval_set, args.batch_size, False):
            audio, states, mask, diff, pat_t, typ_t, active, radar = to_tensors(batch)
            _, _, typ_l = model(audio, states, diff, mask, radar=radar)
            pred = typ_l.argmax(-1)
            for ty in range(NUM_TYPES):
                m = active & mask.unsqueeze(-1) & (typ_t == ty)
                tt[ty] += int(m.sum()); tc[ty] += int((m & (pred == ty)).sum())
    type_recall = {TYPE_NAMES[i]: (tc[i] / tt[i] if tt[i] else float('nan')) for i in range(NUM_TYPES)}

    onset_logits_all = []
    with torch.no_grad():
        for s in eval_set:
            a = torch.from_numpy(s['audio']).unsqueeze(0).to(device); d = torch.tensor([s['difficulty']], device=device)
            rad = torch.from_numpy(s['radar']).unsqueeze(0).to(device)
            onset_logits_all.append(torch.sigmoid(model.onset_logits(model.encode_audio(a), d, radar=rad))[0].cpu().numpy())
    tau = float(np.quantile(np.concatenate(onset_logits_all), 1 - target_density))

    f1s, dens, gen_syms, preds, tgts, holds, orphans = [], [], {k: 0 for k in SYMBOL_NAMES}, [], [], 0, 0
    for i in range(0, len(eval_set), 8):
        batch = eval_set[i:i + 8]
        L = min(args.max_gen_len, max(len(s['typed']) for s in batch)); B = len(batch)
        audio = torch.zeros(B, L, audio_dim); lengths = torch.zeros(B, dtype=torch.long); diff = torch.zeros(B, dtype=torch.long)
        radar = torch.zeros(B, 5)
        for b, s in enumerate(batch):
            t = min(len(s['typed']), L); audio[b, :t] = torch.from_numpy(s['audio'][:t]); lengths[b] = t
            diff[b] = s['difficulty']; radar[b] = torch.from_numpy(s['radar'])
        gen = model.generate(audio.to(device), diff.to(device), lengths=lengths.to(device),
                             onset_threshold=tau, greedy=True, type_sample=True, radar=radar.to(device)).cpu().numpy()
        for b, s in enumerate(batch):
            t = int(lengths[b]); g_raw = gen[b, :t]
            for pnl in range(NUM_PANELS):
                col = g_raw[:, pnl]; h = int(((col == 2) | (col == 4)).sum()); tl = int((col == 3).sum())
                holds += h; orphans += abs(h - tl)
            g = pair_holds(g_raw)
            m = onset_density_metrics((g != 0).astype(np.float32), reference=(s['typed'][:t] != 0).astype(np.float32))
            f1s.append(m['onset_f1']); dens.append((g != 0).any(1).mean())
            for k, val_ in symbol_histogram(g).items(): gen_syms[k] += val_
            preds.append(critic.predict(typed_binary(g), s['audio'][:t], bpm=DEFAULT_BPM)['class']); tgts.append(s['difficulty'])
    dd = np.abs(np.array(preds) - np.array(tgts))
    real_taps = sum(symbol_histogram(s['typed'])['tap'] for s in eval_set)
    real_holds = sum(symbol_histogram(s['typed'])['hold_head'] for s in eval_set)

    print("\n" + "=" * 76)
    print("  LAYERED TYPED GENERATOR")
    print("=" * 76)
    print(f"  onset_F1 {np.mean(f1s):.3f}  density {np.mean(dens):.3f} (real {target_density:.3f})  "
          f"crit_adj {np.mean(dd<=1):.3f}  crit_mae {np.mean(dd):.3f}")
    print(f"  teacher-forced type recall (active panels): " + " ".join(f"{k}={type_recall[k]:.2f}" for k in TYPE_NAMES))
    print(f"  generated symbols: {gen_syms}")
    gh = gen_syms['hold_head']; gt = gen_syms['tap']
    print(f"  generated tap:hold = {gt}:{gh} ({gt/max(gh,1):.1f}:1)   real tap:hold = {real_taps/max(real_holds,1):.1f}:1")
    print(f"  raw hold orphans (pre-pairing): {orphans}/{holds}")
    print("=" * 76)
    if mlflow_on:
        mlflow.log_metrics({'gen_onset_f1': float(np.mean(f1s)), 'gen_density': float(np.mean(dens)),
                            'crit_adj': float(np.mean(dd <= 1)), 'hold_recall': float(type_recall['hold_head'])})
        mlflow.end_run()


if __name__ == '__main__':
    main()

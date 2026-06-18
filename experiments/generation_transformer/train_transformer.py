#!/usr/bin/env python3
"""
Stage 2: train the autoregressive transformer chart generator.

Teacher-forced training with masked class-weighted CE; checkpoint on val CE;
final headline eval = generation onset_F1 + DifficultyCritic agreement on a
capped val subset, compared to the Stage 1 floor (onset_F1 ~= 0.05).

Usage:
    python experiments/generation_transformer/train_transformer.py \
        --data_dir data/ --audio_dir data/ --epochs 20 --warmup_freeze 3
"""

import warnings
import os

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.generation.transformer import ChartGenerator
from src.generation.tokenizer import ChartTokenizer, BOS_TOKEN, VOCAB_SIZE, NUM_PANEL_STATES
from src.generation.baselines import compute_state_class_weights
from src.generation.evaluation import onset_density_metrics, DifficultyCritic

DEFAULT_BPM = 150.0
PHASE1_CKPT = "checkpoints/ordinal_exp/standard_ordinal_multi/best_val_loss.pt"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup_freeze', type=int, default=3,
                   help='epochs to keep the warm-started audio encoder frozen')
    p.add_argument('--num_layers', type=int, default=4)
    p.add_argument('--eval_songs', type=int, default=64)
    p.add_argument('--max_gen_len', type=int, default=768)
    p.add_argument('--checkpoint_dir', default='checkpoints/gen_transformer')
    return p.parse_args()


def setup(data_dir, audio_dir, seed):
    chart_files = glob.glob(f"{data_dir}/**/*.sm", recursive=True)
    chart_files += glob.glob(f"{data_dir}/**/*.ssc", recursive=True)
    print(f"Found {len(chart_files)} chart files")
    train_files, val_files, _ = create_data_splits(chart_files, random_state=seed)
    with open(PROJECT_ROOT / "config/model_config.yaml") as f:
        model_config = yaml.safe_load(f)
    max_seq_len = model_config['classifier']['max_sequence_length']
    train_ds, val_ds, _ = create_datasets(
        train_files=train_files, val_files=val_files, test_files=[],
        audio_dir=audio_dir, max_sequence_length=max_seq_len, cache_dir='cache/samples',
    )
    print("Warming caches...")
    train_ds.warm_cache(show_progress=True)
    val_ds.warm_cache(show_progress=True)
    return train_ds, val_ds


def collect(ds):
    out = []
    for i in range(len(ds)):
        s = ds[i]
        T = int(s['mask'].sum().item())
        out.append({
            'chart': s['chart'][:T].numpy().astype(np.float32),
            'audio': s['audio'][:T].numpy().astype(np.float32),
            'difficulty': int(s['difficulty']),
        })
    return out


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(args.seed)

    train_ds, val_ds = setup(args.data_dir, args.audio_dir, args.seed)
    print("Collecting samples...")
    train, val = collect(train_ds), collect(val_ds)
    print(f"train={len(train)} val={len(val)}")
    audio_dim = train[0]['audio'].shape[1]

    # MLflow (optional)
    try:
        import mlflow
        mlflow.set_experiment("stepmania-chart-generator")
        mlflow_on = True
    except ImportError:
        mlflow_on = False

    model = ChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=args.num_layers).to(device)
    ck = torch.load(PHASE1_CKPT, map_location="cpu", weights_only=False)
    n_warm = model.load_audio_encoder(ck['model_state_dict'])
    print(f"Warm-started {n_warm} audio_encoder tensors from {PHASE1_CKPT}")
    model.freeze_audio_encoder(True)

    nparams = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {nparams:,}")

    # 19-length class weights: panel-states inverse-sqrt, specials = 1.0
    w = torch.ones(VOCAB_SIZE)
    w[:NUM_PANEL_STATES] = compute_state_class_weights([s['chart'] for s in train])
    criterion = nn.CrossEntropyLoss(weight=w.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def batches(samples, bs, shuffle):
        idx = np.arange(len(samples))
        if shuffle:
            rng.shuffle(idx)
        for i in range(0, len(idx), bs):
            yield [samples[j] for j in idx[i:i + bs]]

    def to_tensors(batch):
        T = max(len(s['chart']) for s in batch)
        B = len(batch)
        audio = torch.zeros(B, T, audio_dim)
        in_tok = torch.full((B, T), BOS_TOKEN, dtype=torch.long)
        target = torch.zeros(B, T, dtype=torch.long)
        mask = torch.zeros(B, T, dtype=torch.bool)
        diff = torch.zeros(B, dtype=torch.long)
        for b, s in enumerate(batch):
            t = len(s['chart'])
            states = ChartTokenizer.encode(s['chart'])  # (t,) 0..15
            audio[b, :t] = torch.from_numpy(s['audio'])
            target[b, :t] = states
            in_tok[b, 1:t] = states[:-1]  # [BOS, s_0, ..., s_{t-2}]
            mask[b, :t] = True
            diff[b] = s['difficulty']
        return (audio.to(device), in_tok.to(device), target.to(device),
                mask.to(device), diff.to(device))

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_val = float('inf')
    best_path = Path(args.checkpoint_dir) / "best_val_ce.pt"

    run_ctx = mlflow.start_run(run_name="gen-transformer/stage2") if mlflow_on else None
    if mlflow_on:
        mlflow.log_params({'epochs': args.epochs, 'lr': args.lr, 'num_layers': args.num_layers,
                           'warmup_freeze': args.warmup_freeze, 'batch_size': args.batch_size,
                           'd_model': 128, 'params': nparams})

    for epoch in range(args.epochs):
        if epoch == args.warmup_freeze:
            model.freeze_audio_encoder(False)
            print(f"  [epoch {epoch+1}] unfroze audio encoder")

        model.train()
        tr, nb = 0.0, 0
        for batch in batches(train, args.batch_size, True):
            audio, in_tok, target, mask, diff = to_tensors(batch)
            optimizer.zero_grad()
            logits = model(audio, in_tok, diff, mask)
            loss = criterion(logits[mask], target[mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr += loss.item(); nb += 1

        model.eval()
        vl, vnb = 0.0, 0
        with torch.no_grad():
            for batch in batches(val, args.batch_size, False):
                audio, in_tok, target, mask, diff = to_tensors(batch)
                logits = model(audio, in_tok, diff, mask)
                vl += criterion(logits[mask], target[mask]).item(); vnb += 1
        vl /= max(1, vnb)
        tr /= max(1, nb)
        print(f"  epoch {epoch+1}/{args.epochs}  train_ce={tr:.4f}  val_ce={vl:.4f}"
              + ("  *" if vl < best_val else ""))
        if mlflow_on:
            mlflow.log_metrics({'train_ce': tr, 'val_ce': vl}, step=epoch)
        if vl < best_val:
            best_val = vl
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                        'val_ce': vl, 'args': vars(args)}, best_path)

    # ---- final headline eval from best checkpoint ----
    print(f"\nLoading best checkpoint (val_ce={best_val:.4f}) for generation eval...")
    model.load_state_dict(torch.load(best_path, map_location=device)['model_state_dict'])
    model.eval()

    eval_set = val[:args.eval_songs]
    critic = DifficultyCritic(device=device)
    onset_f1s, gen_dens, ref_dens, gen_charts, tgts = [], [], [], [], []
    for batch in batches(eval_set, args.batch_size, False):
        L = min(args.max_gen_len, max(len(s['chart']) for s in batch))
        B = len(batch)
        audio = torch.zeros(B, L, audio_dim)
        lengths = torch.zeros(B, dtype=torch.long)
        diff = torch.zeros(B, dtype=torch.long)
        for b, s in enumerate(batch):
            t = min(len(s['chart']), L)
            audio[b, :t] = torch.from_numpy(s['audio'][:t])
            lengths[b] = t
            diff[b] = s['difficulty']
        gen = model.generate(audio.to(device), diff.to(device),
                             lengths=lengths.to(device), greedy=True).cpu().numpy()
        for b, s in enumerate(batch):
            t = int(lengths[b])
            g = gen[b, :t]
            ref = s['chart'][:t]
            m = onset_density_metrics(g, reference=ref)
            onset_f1s.append(m['onset_f1']); gen_dens.append(m['gen_density'])
            ref_dens.append(m['ref_density'])
            gen_charts.append(g); tgts.append(s['difficulty'])

    # critic agreement on generated charts
    preds = [critic.predict(g, s['audio'][:len(g)], bpm=DEFAULT_BPM)['class']
             for g, s in zip(gen_charts, eval_set)]
    preds, tg = np.array(preds), np.array(tgts)
    d = np.abs(preds - tg)

    res = {
        'best_val_ce': best_val,
        'onset_f1': float(np.mean(onset_f1s)),
        'gen_density': float(np.mean(gen_dens)),
        'ref_density': float(np.mean(ref_dens)),
        'crit_exact': float(np.mean(d == 0)),
        'crit_adjacent': float(np.mean(d <= 1)),
        'crit_mae': float(np.mean(d)),
        'eval_songs': len(gen_charts),
    }
    if mlflow_on:
        mlflow.log_metrics({k: v for k, v in res.items() if isinstance(v, float)})
        mlflow.end_run()

    print("\n" + "=" * 70)
    print("  STAGE 2 TRANSFORMER vs STAGE 1 FLOOR")
    print("=" * 70)
    print(f"  best_val_ce     : {res['best_val_ce']:.4f}")
    print(f"  onset_F1        : {res['onset_f1']:.3f}   (floor: per-frame MLP 0.053)")
    print(f"  gen_density     : {res['gen_density']:.3f}   (real: {res['ref_density']:.3f})")
    print(f"  critic exact    : {res['crit_exact']:.3f}   (floor: n-gram 0.508)")
    print(f"  critic adjacent : {res['crit_adjacent']:.3f}   (floor: n-gram 0.977)")
    print(f"  critic MAE      : {res['crit_mae']:.3f}")
    print(f"  (eval on {res['eval_songs']} val songs, max_gen_len={args.max_gen_len})")
    print("=" * 70)


if __name__ == '__main__':
    main()

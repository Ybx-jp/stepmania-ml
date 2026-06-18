#!/usr/bin/env python3
"""
Stage 1: fit/train the dumb generative baselines and score them.

Establishes the floor (per-frame NLL, onset F1, density match, critic
difficulty-agreement) that the Stage 2 autoregressive transformer must beat.

Usage:
    python experiments/generation_baselines/run_baselines.py \
        --data_dir data/ --audio_dir data/ --mlp_epochs 8
"""

import warnings
import os

warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')
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
from src.generation.baselines import NGramChartModel, PerFrameMLP, compute_state_class_weights
from src.generation.tokenizer import ChartTokenizer, NUM_PANEL_STATES
from src.generation.evaluation import onset_density_metrics, DifficultyCritic

DEFAULT_BPM = 150.0  # nominal BPM for generated-chart groove-radar (Stage 1 approximation)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--mlp_epochs', type=int, default=8)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--critic_limit', type=int, default=128,
                   help='max val songs to run the difficulty critic on')
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
    return train_ds, val_ds, model_config


def collect_samples(ds):
    """Trim each sample to real length: list of dicts {chart, audio, difficulty}."""
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


# ----- per-frame onset F1 in teacher-forced mode (model defines P(state|context)) -----

def ngram_density(model, samples, rng):
    gen_d, ref_d = [], []
    for s in samples:
        gen = model.sample(len(s['chart']), s['difficulty'], rng)
        gen_d.append((gen.sum(1) > 0).mean())
        ref_d.append((s['chart'].sum(1) > 0).mean())
    return float(np.mean(gen_d)), float(np.mean(ref_d))


def critic_agreement_generated(critic, gen_charts, samples):
    preds, tgts = [], []
    for gen, s in zip(gen_charts, samples):
        out = critic.predict(gen, s['audio'], bpm=DEFAULT_BPM)
        preds.append(out['class'])
        tgts.append(s['difficulty'])
    preds, tgts = np.array(preds), np.array(tgts)
    diff = np.abs(preds - tgts)
    return {
        'exact': float(np.mean(diff == 0)),
        'adjacent': float(np.mean(diff <= 1)),
        'mae': float(np.mean(diff)),
        'n': len(preds),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(args.seed)

    train_ds, val_ds, model_config = setup(args.data_dir, args.audio_dir, args.seed)
    print("Collecting samples...")
    train = collect_samples(train_ds)
    val = collect_samples(val_ds)
    print(f"train={len(train)} val={len(val)}")

    train_charts = [s['chart'] for s in train]
    train_diffs = [s['difficulty'] for s in train]
    val_charts = [s['chart'] for s in val]
    val_diffs = [s['difficulty'] for s in val]

    critic = DifficultyCritic(device=device)
    critic_val = val[:args.critic_limit]

    results = {}

    # ---------------- Baseline 1: N-gram ----------------
    print("\n=== Fitting N-gram bigram ===")
    ngram = NGramChartModel().fit(train_charts, train_diffs)
    nll = ngram.mean_nll(val_charts, val_diffs)
    gen_dens, ref_dens = ngram_density(ngram, val, rng)
    ngram_gen = [ngram.sample(len(s['chart']), s['difficulty'], rng) for s in critic_val]
    ngram_crit = critic_agreement_generated(critic, ngram_gen, critic_val)
    results['ngram'] = {'val_nll': nll, 'gen_density': gen_dens, 'ref_density': ref_dens, **ngram_crit}

    # ---------------- Baseline 2: Per-frame MLP ----------------
    print("\n=== Training Per-frame MLP ===")
    audio_dim = train[0]['audio'].shape[1]
    mlp = PerFrameMLP(audio_dim=audio_dim).to(device)
    class_weights = compute_state_class_weights(train_charts).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)

    def make_batches(samples, bs, shuffle):
        idx = np.arange(len(samples))
        if shuffle:
            rng.shuffle(idx)
        for i in range(0, len(idx), bs):
            yield [samples[j] for j in idx[i:i + bs]]

    def to_tensors(batch):
        T = max(len(s['chart']) for s in batch)
        B = len(batch)
        audio = torch.zeros(B, T, audio_dim)
        tokens = torch.zeros(B, T, dtype=torch.long)
        mask = torch.zeros(B, T, dtype=torch.bool)
        diff = torch.zeros(B, dtype=torch.long)
        for b, s in enumerate(batch):
            t = len(s['chart'])
            audio[b, :t] = torch.from_numpy(s['audio'])
            tokens[b, :t] = ChartTokenizer.encode(s['chart'])
            mask[b, :t] = True
            diff[b] = s['difficulty']
        return audio.to(device), tokens.to(device), mask.to(device), diff.to(device)

    best_val = float('inf')
    for epoch in range(args.mlp_epochs):
        mlp.train()
        tr_loss = 0.0
        nb = 0
        for batch in make_batches(train, args.batch_size, shuffle=True):
            audio, tokens, mask, diff = to_tensors(batch)
            optimizer.zero_grad()
            logits = mlp(audio, diff)  # (B,T,16)
            loss = criterion(logits[mask], tokens[mask])
            loss.backward()
            optimizer.step()
            tr_loss += loss.item(); nb += 1

        mlp.eval()
        v_loss, vnb = 0.0, 0
        with torch.no_grad():
            for batch in make_batches(val, args.batch_size, shuffle=False):
                audio, tokens, mask, diff = to_tensors(batch)
                logits = mlp(audio, diff)
                v_loss += criterion(logits[mask], tokens[mask]).item(); vnb += 1
        v_loss /= max(1, vnb)
        print(f"  epoch {epoch+1}/{args.mlp_epochs}  train_ce={tr_loss/max(1,nb):.4f}  val_ce={v_loss:.4f}")
        if v_loss < best_val:
            best_val = v_loss

    # MLP onset F1 (generation = argmax decode, aligned to real audio) + density + critic
    mlp.eval()
    onset_f1s, gen_dens_mlp, mlp_gen = [], [], []
    with torch.no_grad():
        for s in val:
            audio = torch.from_numpy(s['audio']).unsqueeze(0).to(device)
            diff = torch.tensor([s['difficulty']], device=device)
            gen = mlp.generate(audio, diff).squeeze(0).cpu().numpy()  # (T,4)
            m = onset_density_metrics(gen, reference=s['chart'])
            onset_f1s.append(m['onset_f1'])
            gen_dens_mlp.append(m['gen_density'])
            if len(mlp_gen) < args.critic_limit:
                mlp_gen.append(gen)
    mlp_crit = critic_agreement_generated(critic, mlp_gen, critic_val)
    results['mlp'] = {
        'val_ce': best_val,
        'onset_f1': float(np.mean(onset_f1s)),
        'gen_density': float(np.mean(gen_dens_mlp)),
        'ref_density': ref_dens,
        **mlp_crit,
    }

    # ---------------- Report ----------------
    print("\n" + "=" * 78)
    print("  STAGE 1 BASELINE FLOOR")
    print("=" * 78)
    print(f"  Real val mean density: {ref_dens:.3f}")
    print("-" * 78)
    print(f"  {'Model':<14} {'val_loss':>9} {'onset_F1':>9} {'gen_dens':>9} "
          f"{'crit_exact':>11} {'crit_adj':>9} {'crit_mae':>9}")
    print("-" * 78)
    n = results['ngram']
    print(f"  {'n-gram':<14} {n['val_nll']:>9.4f} {'n/a':>9} {n['gen_density']:>9.3f} "
          f"{n['exact']:>11.3f} {n['adjacent']:>9.3f} {n['mae']:>9.3f}")
    mm = results['mlp']
    print(f"  {'per-frame MLP':<14} {mm['val_ce']:>9.4f} {mm['onset_f1']:>9.3f} {mm['gen_density']:>9.3f} "
          f"{mm['exact']:>11.3f} {mm['adjacent']:>9.3f} {mm['mae']:>9.3f}")
    print("-" * 78)
    print("  (n-gram val_loss = bigram NLL; MLP val_loss = weighted CE — not comparable.)")
    print(f"  Critic run on {n['n']} val songs. onset_F1 vs real charts (aligned).")
    print("=" * 78)


if __name__ == '__main__':
    main()

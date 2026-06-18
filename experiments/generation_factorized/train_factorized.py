#!/usr/bin/env python3
"""
Stage 3: train the factorized onset-then-panel generator.

Two losses: onset BCE (pos_weighted for the empty-frame majority) + panel CE on
real-onset frames. Onset is audio-driven (no token feedback), so density-matched
threshold decoding is stable instead of collapsing. Final eval thresholds the
onset head to match real density, decodes the panel autoregressively, and compares
to the Stage 1 floor / Stage 2 transformer / density probe.

Usage:
    python experiments/generation_factorized/train_factorized.py \
        --data_dir data/ --audio_dir data/ --epochs 20 --warmup_freeze 3
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
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.generation.factorized import FactorizedChartGenerator, NUM_NONEMPTY
from src.generation.tokenizer import ChartTokenizer, BOS_TOKEN
from src.generation.evaluation import onset_density_metrics, DifficultyCritic

DEFAULT_BPM = 150.0
PHASE1_CKPT = "checkpoints/ordinal_exp/standard_ordinal_multi/best_val_loss.pt"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--max_train_len', type=int, default=1024,
                   help='truncate training/val samples to bound the onset encoder O(T^2) memory')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup_freeze', type=int, default=3)
    p.add_argument('--num_layers', type=int, default=4)
    p.add_argument('--onset_layers', type=int, default=2)
    p.add_argument('--panel_loss_weight', type=float, default=1.0)
    p.add_argument('--eval_songs', type=int, default=64)
    p.add_argument('--max_gen_len', type=int, default=768)
    p.add_argument('--checkpoint_dir', default='checkpoints/gen_factorized')
    return p.parse_args()


def setup(data_dir, audio_dir, seed):
    chart_files = glob.glob(f"{data_dir}/**/*.sm", recursive=True)
    chart_files += glob.glob(f"{data_dir}/**/*.ssc", recursive=True)
    print(f"Found {len(chart_files)} chart files")
    train_files, val_files, _ = create_data_splits(chart_files, random_state=seed)
    with open(PROJECT_ROOT / "config/model_config.yaml") as f:
        max_seq_len = yaml.safe_load(f)['classifier']['max_sequence_length']
    train_ds, val_ds, _ = create_datasets(train_files=train_files, val_files=val_files,
                                          test_files=[], audio_dir=audio_dir,
                                          max_sequence_length=max_seq_len, cache_dir='cache/samples')
    print("Warming caches..."); train_ds.warm_cache(show_progress=True); val_ds.warm_cache(show_progress=True)
    return train_ds, val_ds


def collect(ds, cap=None):
    out = []
    for i in range(len(ds)):
        s = ds[i]; T = int(s['mask'].sum().item())
        if cap is not None:
            T = min(T, cap)
        out.append({'chart': s['chart'][:T].numpy().astype(np.float32),
                    'audio': s['audio'][:T].numpy().astype(np.float32),
                    'difficulty': int(s['difficulty'])})
    return out


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(args.seed)

    train_ds, val_ds = setup(args.data_dir, args.audio_dir, args.seed)
    print("Collecting samples..."); train, val = collect(train_ds, args.max_train_len), collect(val_ds, args.max_train_len)
    print(f"train={len(train)} val={len(val)}")
    audio_dim = train[0]['audio'].shape[1]

    # onset pos_weight = (#empty / #onset) over training frames
    onsets = np.concatenate([(s['chart'].sum(1) > 0).astype(np.float32) for s in train])
    pos_rate = float(onsets.mean())
    pos_weight = torch.tensor((1 - pos_rate) / max(pos_rate, 1e-6), device=device)
    target_density = float(np.mean([(s['chart'].sum(1) > 0).mean() for s in val]))
    print(f"train onset rate={pos_rate:.3f}  pos_weight={pos_weight.item():.2f}  val density={target_density:.3f}")

    try:
        import mlflow; mlflow.set_experiment("stepmania-chart-generator"); mlflow_on = True
    except ImportError:
        mlflow_on = False

    model = FactorizedChartGenerator(audio_dim=audio_dim, d_model=128,
                                     num_layers=args.num_layers, onset_layers=args.onset_layers).to(device)
    ck = torch.load(PHASE1_CKPT, map_location="cpu", weights_only=False)
    print(f"Warm-started {model.load_audio_encoder(ck['model_state_dict'])} audio_encoder tensors")
    model.freeze_audio_encoder(True)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    onset_crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    panel_crit = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def batches(samples, bs, shuffle):
        idx = np.arange(len(samples))
        if shuffle: rng.shuffle(idx)
        for i in range(0, len(idx), bs):
            yield [samples[j] for j in idx[i:i + bs]]

    def to_tensors(batch):
        T = max(len(s['chart']) for s in batch); B = len(batch)
        audio = torch.zeros(B, T, audio_dim)
        in_tok = torch.full((B, T), BOS_TOKEN, dtype=torch.long)
        onset_t = torch.zeros(B, T)
        panel_t = torch.zeros(B, T, dtype=torch.long)
        mask = torch.zeros(B, T, dtype=torch.bool)
        diff = torch.zeros(B, dtype=torch.long)
        for b, s in enumerate(batch):
            t = len(s['chart']); states = ChartTokenizer.encode(s['chart'])  # (t,) 0..15
            audio[b, :t] = torch.from_numpy(s['audio'])
            in_tok[b, 1:t] = states[:-1]
            onset_t[b, :t] = (states > 0).float()
            panel_t[b, :t] = (states - 1).clamp(min=0)  # valid only where onset
            mask[b, :t] = True; diff[b] = s['difficulty']
        return (audio.to(device), in_tok.to(device), onset_t.to(device),
                panel_t.to(device), mask.to(device), diff.to(device))

    def compute_losses(onset_logits, panel_logits, onset_t, panel_t, mask):
        ol = onset_crit(onset_logits[mask], onset_t[mask])
        sel = mask & (onset_t > 0.5)
        pl = panel_crit(panel_logits[sel], panel_t[sel]) if sel.any() else torch.zeros((), device=device)
        return ol, pl

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_val = float('inf'); best_path = Path(args.checkpoint_dir) / "best_val.pt"
    if mlflow_on:
        mlflow.start_run(run_name="gen-factorized/stage3")
        mlflow.log_params({'epochs': args.epochs, 'lr': args.lr, 'pos_weight': pos_weight.item(),
                           'num_layers': args.num_layers, 'onset_layers': args.onset_layers})

    for epoch in range(args.epochs):
        if epoch == args.warmup_freeze:
            model.freeze_audio_encoder(False); print(f"  [epoch {epoch+1}] unfroze audio encoder")
        model.train(); tr_o = tr_p = 0.0; nb = 0
        for batch in batches(train, args.batch_size, True):
            audio, in_tok, onset_t, panel_t, mask, diff = to_tensors(batch)
            optimizer.zero_grad()
            ol_logits, pl_logits = model(audio, in_tok, diff, mask)
            o_loss, p_loss = compute_losses(ol_logits, pl_logits, onset_t, panel_t, mask)
            (o_loss + args.panel_loss_weight * p_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            tr_o += o_loss.item(); tr_p += p_loss.item(); nb += 1

        model.eval(); v_o = v_p = 0.0; vnb = 0
        with torch.no_grad():
            for batch in batches(val, args.batch_size, False):
                audio, in_tok, onset_t, panel_t, mask, diff = to_tensors(batch)
                ol_logits, pl_logits = model(audio, in_tok, diff, mask)
                o_loss, p_loss = compute_losses(ol_logits, pl_logits, onset_t, panel_t, mask)
                v_o += o_loss.item(); v_p += p_loss.item(); vnb += 1
        v_o /= max(1, vnb); v_p /= max(1, vnb); v_total = v_o + args.panel_loss_weight * v_p
        print(f"  epoch {epoch+1}/{args.epochs}  train(o={tr_o/max(1,nb):.4f} p={tr_p/max(1,nb):.4f})  "
              f"val(o={v_o:.4f} p={v_p:.4f} tot={v_total:.4f})" + ("  *" if v_total < best_val else ""))
        if mlflow_on:
            mlflow.log_metrics({'val_onset': v_o, 'val_panel': v_p, 'val_total': v_total}, step=epoch)
        if v_total < best_val:
            best_val = v_total
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                        'val_total': v_total, 'args': vars(args)}, best_path)

    # ---- final eval: onset AUC + density-matched threshold decode ----
    print(f"\nLoading best (val_total={best_val:.4f}) for generation eval...")
    model.load_state_dict(torch.load(best_path, map_location=device)['model_state_dict']); model.eval()
    eval_set = val[:args.eval_songs]
    critic = DifficultyCritic(device=device)

    # onset posteriors over eval set (one non-AR pass each) -> AUC + threshold
    all_p, all_y = [], []
    with torch.no_grad():
        for s in eval_set:
            audio = torch.from_numpy(s['audio']).unsqueeze(0).to(device)
            diff = torch.tensor([s['difficulty']], device=device)
            p = torch.sigmoid(model.onset_logits(model.encode_audio(audio), diff))[0].cpu().numpy()
            all_p.append(p); all_y.append((s['chart'].sum(1) > 0).astype(int))
    flat_p, flat_y = np.concatenate(all_p), np.concatenate(all_y)
    roc = roc_auc_score(flat_y, flat_p); pr = average_precision_score(flat_y, flat_p)
    tau = float(np.quantile(flat_p, 1 - target_density))

    f1s, ps, rs, dens, gens, tgts = [], [], [], [], [], []
    for i in range(0, len(eval_set), 16):
        batch = eval_set[i:i + 16]
        L = min(args.max_gen_len, max(len(s['chart']) for s in batch)); B = len(batch)
        audio = torch.zeros(B, L, audio_dim); lengths = torch.zeros(B, dtype=torch.long); diff = torch.zeros(B, dtype=torch.long)
        for b, s in enumerate(batch):
            t = min(len(s['chart']), L); audio[b, :t] = torch.from_numpy(s['audio'][:t]); lengths[b] = t; diff[b] = s['difficulty']
        gen = model.generate(audio.to(device), diff.to(device), lengths=lengths.to(device),
                             onset_threshold=tau, panel_greedy=True).cpu().numpy()
        for b, s in enumerate(batch):
            t = int(lengths[b]); g = gen[b, :t]; m = onset_density_metrics(g, reference=s['chart'][:t])
            f1s.append(m['onset_f1']); ps.append(m.get('onset_precision', 0)); rs.append(m.get('onset_recall', 0))
            dens.append(m['gen_density']); gens.append(g); tgts.append(s['difficulty'])
    preds = np.array([critic.predict(g, s['audio'][:len(g)], bpm=DEFAULT_BPM)['class'] for g, s in zip(gens, eval_set)])
    d = np.abs(preds - np.array(tgts))
    if mlflow_on:
        mlflow.log_metrics({'onset_roc_auc': roc, 'onset_pr_auc': pr, 'gen_onset_f1': float(np.mean(f1s)),
                            'gen_density': float(np.mean(dens)), 'crit_adjacent': float(np.mean(d <= 1))})
        mlflow.end_run()

    print("\n" + "=" * 76)
    print("  STAGE 3 FACTORIZED  vs  prior")
    print("=" * 76)
    print(f"  onset ROC-AUC {roc:.3f}  PR-AUC {pr:.3f}  (probe single-head: 0.813 / 0.469)")
    print(f"  threshold tau={tau:.3f}  target density={target_density:.3f}")
    print("-" * 76)
    print(f"  {'decode':<26} {'onset_F1':>9} {'onset_P':>8} {'onset_R':>8} {'density':>8} {'crit_adj':>9}")
    print("-" * 76)
    print(f"  {'factorized @ tau':<26} {np.mean(f1s):>9.3f} {np.mean(ps):>8.3f} {np.mean(rs):>8.3f} "
          f"{np.mean(dens):>8.3f} {np.mean(d <= 1):>9.3f}")
    print(f"  {'(probe free-run @ tau)':<26} {0.000:>9.3f} {0.000:>8.3f} {0.000:>8.3f} {0.000:>8.3f} {0.469:>9.3f}")
    print(f"  {'(Stage2 temp 1.0)':<26} {0.300:>9.3f} {0.210:>8.3f} {0.577:>8.3f} {0.536:>8.3f} {0.562:>9.3f}")
    print(f"  {'(Stage1 MLP floor)':<26} {0.053:>9.3f} {'-':>8} {'-':>8} {0.014:>8.3f} {'-':>9}")
    print("-" * 76)
    print(f"  factorized critic: exact={np.mean(d==0):.3f} adjacent={np.mean(d<=1):.3f} mae={np.mean(d):.3f}")
    print("=" * 76)


if __name__ == '__main__':
    main()

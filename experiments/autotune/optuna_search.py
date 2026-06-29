"""Optuna hyperparameter search for the factorized generator.

Optimizes the SAME metric the real trainer checkpoints on: val_total =
val_onset + panel_loss_weight * val_panel (minimize). Each trial runs a short
training with the harness, reports val_total per epoch to Optuna, and a
MedianPruner kills clearly-losing trials early so the budget goes to good ones.

Search space targets the generator's actual knobs (lr, batch_size, num_layers,
onset_layers, panel_loss_weight, weight_decay) plus the amp flag, NOT the
classifier/diffusion spaces in config/optuna_config.yaml (those are for other
models). Run on a free GPU.

Usage:
  python experiments/autotune/optuna_search.py --data_dir data/ --audio_dir data/ \
      --n_trials 30 --epochs 6
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import optuna
import torch

from _harness import (autocast_ctx, build_model, to_tensors, iter_batches,
                      make_losses, pos_weight_for, load_split)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.reproducibility import set_seed  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--audio_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_len", type=int, default=1024)
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument("--epochs", type=int, default=6, help="short budget per trial")
    p.add_argument("--warmup_freeze", type=int, default=2)
    p.add_argument("--storage", default="sqlite:///experiments/autotune/optuna_factorized.db")
    p.add_argument("--study_name", default="factorized_generator")
    return p.parse_args()


def train_trial(trial, data, device, args):
    train, val, audio_dim = data
    set_seed(args.seed)  # same init each trial -> HPs are the only varying factor
    rng = np.random.default_rng(args.seed)

    lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
    bs = trial.suggest_categorical("batch_size", [8, 16, 32])
    num_layers = trial.suggest_int("num_layers", 2, 6)
    onset_layers = trial.suggest_int("onset_layers", 1, 3)
    plw = trial.suggest_float("panel_loss_weight", 0.25, 4.0, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
    use_amp = trial.suggest_categorical("amp", [False, True])
    bucketed = trial.suggest_categorical("bucketed", [False, True])

    model = build_model(audio_dim, device, num_layers, onset_layers, frozen=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    compute = make_losses(pos_weight_for(train, device), device)

    best_val = float("inf")
    for epoch in range(args.epochs):
        if epoch == args.warmup_freeze:
            model.freeze_audio_encoder(False)
        model.train()
        for batch in iter_batches(train, bs, rng, shuffle=True, bucketed=bucketed):
            audio, in_tok, onset_t, panel_t, mask, diff = to_tensors(batch, audio_dim, device)
            opt.zero_grad()
            with autocast_ctx(use_amp, device):
                ol, pl = model(audio, in_tok, diff, mask)
                o_loss, p_loss = compute(ol, pl, onset_t, panel_t, mask)
                loss = o_loss + plw * p_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval(); v_o = v_p = 0.0; n = 0
        with torch.no_grad():
            for batch in iter_batches(val, bs, rng, shuffle=False, bucketed=bucketed):
                audio, in_tok, onset_t, panel_t, mask, diff = to_tensors(batch, audio_dim, device)
                with autocast_ctx(use_amp, device):
                    ol, pl = model(audio, in_tok, diff, mask)
                    o_loss, p_loss = compute(ol, pl, onset_t, panel_t, mask)
                v_o += o_loss.item(); v_p += p_loss.item(); n += 1
        v_total = (v_o + plw * v_p) / max(1, n)

        trial.report(v_total, epoch)
        best_val = min(best_val, v_total)
        if trial.should_prune():
            raise optuna.TrialPruned()

    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_val


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        if (total - free) / 2**20 > 500:
            print("WARNING: GPU already in use by another process; trials will be slow "
                  "and may OOM. Free the GPU first.\n")

    train, val = load_split(args.data_dir, args.audio_dir, args.seed, args.max_train_len)
    audio_dim = train[0]["audio"].shape[1]
    data = (train, val, audio_dim)
    print(f"train={len(train)} val={len(val)} audio_dim={audio_dim}\n")

    study = optuna.create_study(
        direction="minimize", study_name=args.study_name, storage=args.storage,
        load_if_exists=True, pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    study.optimize(lambda t: train_trial(t, data, device, args), n_trials=args.n_trials)

    print("\n" + "=" * 60)
    print(f"Best val_total: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    print("Translate to a train_factorized.py run, e.g.:")
    bp = study.best_params
    print(f"  --lr {bp['lr']:.2e} --batch_size {bp['batch_size']} "
          f"--num_layers {bp['num_layers']} --onset_layers {bp['onset_layers']} "
          f"--panel_loss_weight {bp['panel_loss_weight']:.3f}")
    print("Storage:", args.storage, "(resume with the same --study_name to add trials)")


if __name__ == "__main__":
    main()

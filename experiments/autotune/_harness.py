"""Shared training-step harness for autotune scripts.

Reuses the *exact* data pipeline (setup/collect, warm cache) and model from
train_factorized.py, and replicates its batching/tensor/loss logic so the
benchmark and Optuna search exercise the same code path the real trainer does.

bf16 autocast is used for mixed precision (Ampere/RTX 3060 supports it natively;
no GradScaler needed, unlike fp16). See SKILL.md for why AMP is the headline lever.
"""
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "generation_factorized"))

from src.generation.factorized import FactorizedChartGenerator  # noqa: E402
from src.generation.tokenizer import ChartTokenizer, BOS_TOKEN  # noqa: E402
# setup() does data splits + warms the sample cache; collect() materializes samples.
from train_factorized import setup, collect, PHASE1_CKPT  # noqa: E402


def autocast_ctx(use_amp, device):
    if use_amp and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def build_model(audio_dim, device, num_layers=4, onset_layers=2, warm_start=True, frozen=False):
    model = FactorizedChartGenerator(audio_dim=audio_dim, d_model=128,
                                     num_layers=num_layers, onset_layers=onset_layers).to(device)
    if warm_start:
        ck = torch.load(PROJECT_ROOT / PHASE1_CKPT, map_location="cpu", weights_only=False)
        model.load_audio_encoder(ck["model_state_dict"])
    model.freeze_audio_encoder(frozen)
    return model


def to_tensors(batch, audio_dim, device):
    """Pad a list of samples to the batch's max length and move to device.

    Identical to train_factorized.to_tensors (kept in sync by hand). Padding is to
    the longest chart in the batch, which is exactly why length bucketing helps:
    the onset encoder is O(T^2), so one long chart inflates the whole batch.
    """
    T = max(len(s["chart"]) for s in batch); B = len(batch)
    audio = torch.zeros(B, T, audio_dim)
    in_tok = torch.full((B, T), BOS_TOKEN, dtype=torch.long)
    onset_t = torch.zeros(B, T)
    panel_t = torch.zeros(B, T, dtype=torch.long)
    mask = torch.zeros(B, T, dtype=torch.bool)
    diff = torch.zeros(B, dtype=torch.long)
    for b, s in enumerate(batch):
        t = len(s["chart"]); states = ChartTokenizer.encode(s["chart"])
        audio[b, :t] = torch.from_numpy(s["audio"])
        in_tok[b, 1:t] = states[:-1]
        onset_t[b, :t] = (states > 0).float()
        panel_t[b, :t] = (states - 1).clamp(min=0)
        mask[b, :t] = True; diff[b] = s["difficulty"]
    return (audio.to(device), in_tok.to(device), onset_t.to(device),
            panel_t.to(device), mask.to(device), diff.to(device))


def iter_batches(samples, bs, rng, shuffle=True, bucketed=False):
    """Yield batches. bucketed=True groups similar-length charts to cut padding waste,
    then shuffles batch *order* so each step still sees a fresh mix of buckets."""
    n = len(samples)
    if bucketed:
        order = np.argsort([len(s["chart"]) for s in samples])  # short -> long
        batch_starts = list(range(0, n, bs))
        if shuffle:
            rng.shuffle(batch_starts)
        for i in batch_starts:
            yield [samples[j] for j in order[i:i + bs]]
    else:
        idx = np.arange(n)
        if shuffle:
            rng.shuffle(idx)
        for i in range(0, n, bs):
            yield [samples[j] for j in idx[i:i + bs]]


def make_losses(pos_weight, device):
    onset_crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    panel_crit = nn.CrossEntropyLoss()

    def compute(onset_logits, panel_logits, onset_t, panel_t, mask):
        ol = onset_crit(onset_logits[mask], onset_t[mask])
        sel = mask & (onset_t > 0.5)
        pl = panel_crit(panel_logits[sel], panel_t[sel]) if sel.any() else torch.zeros((), device=device)
        return ol, pl
    return compute


def pos_weight_for(train, device):
    onsets = np.concatenate([(s["chart"].sum(1) > 0).astype(np.float32) for s in train])
    pr = float(onsets.mean())
    return torch.tensor((1 - pr) / max(pr, 1e-6), device=device)


def load_split(data_dir, audio_dir, seed, cap):
    train_ds, val_ds = setup(data_dir, audio_dir, seed)
    return collect(train_ds, cap), collect(val_ds, cap)

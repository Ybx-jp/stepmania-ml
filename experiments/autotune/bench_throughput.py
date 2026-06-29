"""Throughput / memory sweep for the factorized generator.

For each (amp, bucketing, batch_size) it runs a few warmup steps then times a
window of real train steps (fwd+bwd+opt) and records samples/sec + peak GPU MiB.
OOM is caught and reported, not fatal. Prints a table and recommends the fastest
config that fits, so you can carry it into the *next* train_factorized.py run.

IMPORTANT: run this only when the GPU is otherwise idle. If a training job is
already on the card, timings are meaningless (you'd be measuring contention).

Usage:
  python experiments/autotune/bench_throughput.py --data_dir data/ --audio_dir data/
  python experiments/autotune/bench_throughput.py --data_dir data/ --audio_dir data/ \
      --batch_sizes 8 16 32 48 --amp both --warmup 5 --measure 20
"""
import argparse
import time

import torch

from _harness import (autocast_ctx, build_model, to_tensors, iter_batches,
                      make_losses, pos_weight_for, load_split)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.reproducibility import set_seed  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--audio_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_len", type=int, default=1024)
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[8, 16, 32, 48])
    p.add_argument("--amp", choices=["off", "on", "both"], default="both")
    p.add_argument("--bucketing", choices=["off", "on", "both"], default="both")
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--onset_layers", type=int, default=2)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--measure", type=int, default=20)
    return p.parse_args()


def bench_one(train, audio_dim, device, rng, compute, *, bs, use_amp, bucketed,
              num_layers, onset_layers, warmup, measure):
    """Returns (samples_per_sec, peak_mib) or raises RuntimeError on OOM."""
    model = build_model(audio_dim, device, num_layers, onset_layers, frozen=False)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    if device.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    seen, t0, step = 0, None, 0
    target = warmup + measure
    while step < target:
        for batch in iter_batches(train, bs, rng, shuffle=True, bucketed=bucketed):
            audio, in_tok, onset_t, panel_t, mask, diff = to_tensors(batch, audio_dim, device)
            opt.zero_grad()
            with autocast_ctx(use_amp, device):
                ol, pl = model(audio, in_tok, diff, mask)
                o_loss, p_loss = compute(ol, pl, onset_t, panel_t, mask)
                loss = o_loss + p_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step == warmup:  # start clock after warmup; sync for honest timing
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter(); seen = 0
            elif step > warmup:
                seen += len(batch)
            if step >= target:
                break
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return seen / dt, peak


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = __import__("numpy").random.default_rng(args.seed)

    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        used = (total - free) / 2**20
        if used > 500:
            print(f"WARNING: {used:.0f} MiB already allocated on this GPU by another "
                  f"process. Timings will reflect contention. Free the GPU first.\n")

    train, _ = load_split(args.data_dir, args.audio_dir, args.seed, args.max_train_len)
    audio_dim = train[0]["audio"].shape[1]
    compute = make_losses(pos_weight_for(train, device), device)

    amp_opts = {"off": [False], "on": [True], "both": [False, True]}[args.amp]
    buck_opts = {"off": [False], "on": [True], "both": [False, True]}[args.bucketing]

    print(f"\nDevice: {device}  |  train samples: {len(train)}  |  audio_dim: {audio_dim}")
    print(f"warmup={args.warmup} measure={args.measure} steps per config\n")
    header = f"{'amp':>4} {'bucket':>7} {'batch':>6} {'samples/s':>11} {'peak MiB':>9} {'rel':>6}"
    print(header); print("-" * len(header))

    results = []
    baseline = None
    for use_amp in amp_opts:
        for bucketed in buck_opts:
            for bs in sorted(args.batch_sizes):
                try:
                    sps, peak = bench_one(train, audio_dim, device, rng, compute,
                                          bs=bs, use_amp=use_amp, bucketed=bucketed,
                                          num_layers=args.num_layers, onset_layers=args.onset_layers,
                                          warmup=args.warmup, measure=args.measure)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        print(f"{str(use_amp):>4} {str(bucketed):>7} {bs:>6} {'OOM':>11} {'-':>9} {'-':>6}")
                        continue
                    raise
                if baseline is None:
                    baseline = sps
                rel = sps / baseline
                results.append((use_amp, bucketed, bs, sps, peak))
                print(f"{str(use_amp):>4} {str(bucketed):>7} {bs:>6} {sps:>11.1f} {peak:>9.0f} {rel:>5.2f}x")

    if results:
        best = max(results, key=lambda r: r[3])
        print("\n" + "=" * 60)
        print(f"FASTEST FITTING CONFIG: amp={best[0]} bucketing={best[1]} batch_size={best[2]}")
        print(f"  {best[3]:.1f} samples/s, {best[4]:.0f} MiB peak "
              f"({best[3]/baseline:.2f}x over first config)")
        print("  Carry into the next run, e.g.:")
        print(f"    --batch_size {best[2]}" + ("  --amp" if best[0] else "")
              + ("  --bucketed" if best[1] else ""))
        print("  (train_factorized.py doesn't yet take --amp/--bucketed; see SKILL.md "
              "for the 3-line patch, or run /autotune apply.)")
        print("=" * 60)


if __name__ == "__main__":
    main()

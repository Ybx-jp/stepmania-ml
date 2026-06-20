#!/usr/bin/env python3
"""
Parallel cache warm for the Stage-1 musical features (audio_dim 41: +chroma +HPSS +metric phase).

Populates cache/samples_v2/{train,val} with the 41-dim feature extractor so train_stage1.py can
train without re-extracting. HPSS dominates cost (~4.4s/sample on top of ~0.7s); this parallelizes
across CPU cores (fork-based pool; each worker writes its own sample_NNNNNN.pt, no collision) and
forces single-threaded librosa per worker to avoid oversubscription.

Usage:
    python experiments/generation_typed/warm_cache_v2.py --data_dir data/ --audio_dir data/ --workers 4
"""

import os
# single-threaded numeric libs per worker (set before numpy/librosa import)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("AUDIOREAD_LOG_LEVEL", "ERROR")
import warnings; warnings.filterwarnings("ignore")

import argparse, glob, sys, time
import multiprocessing as mp
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig

_DS = None  # set per-dataset before the pool is forked


def _warm_one(idx):
    try:
        _ = _DS[idx]
        return 1
    except Exception:
        return 0


def warm(ds, workers, label):
    global _DS
    _DS = ds
    n = len(ds)
    done = ok = 0
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for r in pool.imap_unordered(_warm_one, range(n), chunksize=4):
            done += 1; ok += r
            if done % 100 == 0 or done == n:
                el = time.time() - t0
                rate = done / max(el, 1e-9)
                eta = (n - done) / max(rate, 1e-9)
                print(f"[{label}] {done}/{n}  ok={ok}  {rate:.1f}/s  elapsed {el/60:.1f}m  eta {eta/60:.1f}m",
                      flush=True)
    print(f"[{label}] DONE {ok}/{n} in {(time.time()-t0)/60:.1f}m", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True); p.add_argument("--audio_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache_dir", default="cache/samples_v2")
    args = p.parse_args()
    set_seed(args.seed)

    cf = glob.glob(f"{args.data_dir}/**/*.sm", recursive=True) + glob.glob(f"{args.data_dir}/**/*.ssc", recursive=True)
    train_files, val_files, _ = create_data_splits(cf, random_state=args.seed)
    with open(PROJECT_ROOT / "config/model_config.yaml") as f:
        msl = yaml.safe_load(f)["classifier"]["max_sequence_length"]

    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True, use_metric_phase=True))
    train_ds, val_ds, _ = create_datasets(train_files=train_files, val_files=val_files, test_files=[],
                                          audio_dir=args.audio_dir, max_sequence_length=msl,
                                          feature_extractor=ext, cache_dir=args.cache_dir)
    print(f"workers={args.workers}  train={len(train_ds)}  val={len(val_ds)}  cache={args.cache_dir}", flush=True)
    warm(val_ds, args.workers, "val")      # smaller first -> quick confidence
    warm(train_ds, args.workers, "train")
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate deployed-C0 charts for the CURRENT REAL train split — the train-context for the MATCHED own-output
seq-onset refiner gate (M0). Trains a note-context net ON C0 context + evals ON C0 context (vs the 06-28
MISMATCHED train-real/eval-C0 = 0.667), the genuinely-untested arm of the 06-22 wall.

WHY FRESH EXTRACTION (not the existing seqctx_train_cache.npz): that cache is STALE — it holds 800 rows but the
current split has 786 valid samples, so row j no longer maps to valid_samples[j] ([[dataset-cache-footgun]]).
We re-extract audio + real notes + bpm + difficulty from the CURRENT split in one self-consistent pass, then
generate C0 from it — so every arm of the gate trains on one snapshot, no cross-cache index alignment to misget.

CORRECTNESS (experiment-design Rule 2): generation = the UNMODIFIED gen_c0 from probe_seqcontext_c0 (canonical
deployed config), ONE song at a time. NO batched generate() — onset_threshold/bpm are batch-scalars; batching
would mis-apply one song's tau/bpm to all → a confounded C0. Resource use = PROCESS sharding (B=1 AR decode
underutilizes the GPU; 4 procs on 4 cores ≈ near-linear) + parallel DataLoader extraction, NOT batch growth.

Modes (4 cores: extraction and generation are SEPARATE phases to avoid worker oversubscription):
  --extract             : parallel DataLoader extraction of the full current train split ->
                          cache/trainfresh_cache.npz  (audio, real_notes, bpm, diff, gi). CPU-bound, once.
  --shard k --nshards N : generate C0 for this shard's stride from the fresh cache (pure GPU). Writes a shard npz.
  --merge --nshards N   : assemble shards -> cache/seqctx_trainc0_cache.npz (audio, real, c0), aligned to fresh.
"""
import warnings, os, sys, argparse, glob, time
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
from pathlib import Path
import numpy as np, torch, yaml
from torch.utils.data import DataLoader, Subset
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
from src.generation.typed_model import LayeredTypedChartGenerator
from probe_seqcontext_c0 import gen_c0, AD, NP, CKPT

FRESH_CACHE = PROJECT_ROOT / "cache/trainfresh_cache.npz"
OUT_CACHE   = PROJECT_ROOT / "cache/seqctx_trainc0_cache.npz"
def shard_path(k): return PROJECT_ROOT / f"cache/seqctx_trainc0_shard{k}.npz"


def build_dataset():
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    tf, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                   use_metric_phase=True, use_highres_onset=True))
    tr_ds, _, _ = create_datasets(train_files=tf, val_files=vf, test_files=[], audio_dir="data/",
                                  max_sequence_length=msl, feature_extractor=ext, cache_dir=None)
    return tr_ds


def extract(cap_len=1024, workers=4, max_songs=800):
    ds = build_dataset()
    idxs = list(range(min(len(ds.valid_samples), max_songs)))   # match the 06-28 positive-control power (800)
    print(f"extracting {len(idxs)}/{len(ds.valid_samples)} current train charts (workers={workers})...", flush=True)
    loader = DataLoader(Subset(ds, idxs), batch_size=8, num_workers=workers, shuffle=False,
                        collate_fn=lambda b: b)
    out = []
    for bi, batch in enumerate(loader):
        for it, gi in zip(batch, idxs[bi*8:bi*8+len(batch)]):
            T = int(it['mask'].sum().item())
            if T < 128: continue
            T = min(T, cap_len)
            audio = it['audio'][:T, :AD].numpy().astype(np.float32)
            notes = (it['chart'][:T].numpy() != 0).astype(np.float32)
            bpm = float(ds.valid_samples[gi]['chart'].bpm)
            diff = int(it['difficulty'].item())
            out.append((audio, notes, np.float32(bpm), np.int64(diff), np.int64(gi)))
        if bi % 5 == 0: print(f"    {len(out)} songs ({(bi+1)*8} scanned)...", flush=True)
    np.savez(FRESH_CACHE, data=np.array(out, dtype=object))
    diffs = np.array([r[3] for r in out])
    print(f"wrote {FRESH_CACHE}: {len(out)} songs  diff hist={np.bincount(diffs).tolist()}", flush=True)


def run_shard(k, nshards):
    set_seed(42 + k)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = np.load(FRESH_CACHE, allow_pickle=True)['data']; n = len(d)
    model = LayeredTypedChartGenerator(audio_dim=AD, d_model=128, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device)['model_state_dict'], strict=False); model.eval()
    idxs = list(range(k, n, nshards))            # strided -> even length/difficulty mix per shard
    print(f"[shard {k}/{nshards}] {len(idxs)} songs on {device}", flush=True)
    rows = []; t0 = time.time()
    for c, i in enumerate(idxs):
        audio, real, bpm, diff, gi = d[i]; T = audio.shape[0]
        real_d = float((real.sum(-1) > 0).mean())
        at = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(device)
        c0 = gen_c0(model, at, torch.tensor([int(diff)], device=device), T, float(bpm), real_d, device)
        rows.append((i, (c0[:T] != 0).astype(np.float32)))
        if c % 25 == 0:
            el = time.time() - t0
            print(f"[shard {k}] {c+1}/{len(idxs)}  {el:.0f}s  eta {el/(c+1)*(len(idxs)-c-1)/60:.1f}min", flush=True)
    np.savez(shard_path(k), i=np.array([r[0] for r in rows]),
             c0=np.array([r[1] for r in rows], dtype=object))
    print(f"[shard {k}] done in {(time.time()-t0)/60:.1f}min -> {shard_path(k)}", flush=True)


def merge(nshards):
    d = np.load(FRESH_CACHE, allow_pickle=True)['data']; n = len(d)
    c0_by_i = {}
    for k in range(nshards):
        s = np.load(shard_path(k), allow_pickle=True)
        for i, c0 in zip(s['i'], s['c0']): c0_by_i[int(i)] = c0
    missing = [i for i in range(n) if i not in c0_by_i]
    assert not missing, f"missing C0 for {len(missing)} songs: {missing[:10]}"
    out = [(d[i][0], d[i][1], c0_by_i[i]) for i in range(n)]   # (audio, real, c0) aligned to fresh
    np.savez(OUT_CACHE, data=np.array(out, dtype=object))
    dens_real = np.mean([(r.sum(-1) > 0).mean() for _, r, _ in out])
    dens_c0   = np.mean([(c.sum(-1) > 0).mean() for _, _, c in out])
    print(f"merged {n} songs -> {OUT_CACHE}  (mean onset density real={dens_real:.3f} c0={dens_c0:.3f})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--extract', action='store_true')
    ap.add_argument('--max_songs', type=int, default=800)
    ap.add_argument('--shard', type=int, default=None)
    ap.add_argument('--nshards', type=int, default=4)
    ap.add_argument('--merge', action='store_true')
    a = ap.parse_args()
    if a.extract: extract(max_songs=a.max_songs)
    elif a.merge: merge(a.nshards)
    elif a.shard is not None: run_shard(a.shard, a.nshards)
    else: ap.error("one of --extract / --shard / --merge required")

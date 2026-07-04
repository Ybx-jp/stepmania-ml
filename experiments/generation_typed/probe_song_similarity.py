#!/usr/bin/env python3
"""
Find the val songs most similar to a REFERENCE song in AUDIO-FEATURE space.

Motivation (2026-07-04): Grand Chariot produced a "HOLY... LOCK IN" 6.5/5 playtest at
chaos=0.9/g1.5 (notes/playtest_log.md). Question: do the songs whose AUDIO is most like
Grand Chariot's also generate fantastically at the same setting? This probe ranks the
top-k val songs by audio-feature distance so we can generate them and listen.

AUDIO FEATURES = the deployed model's 42-dim highres per-frame vector (dim0 energy,
dim35 perc-onset, dim36 harm-onset, MFCC/spectral/onset bands) -- NOT the groove radar
(which is chart-derived). Per-song fingerprint = mean+std pooling over valid frames
(84-dim). Ranking z-scores the pooled dims across the pool (equal vote per band) and
sorts by Euclidean distance (cosine reported as an agreement check).

PERSISTENT FINGERPRINT CACHE (so "songs like X" is instant on every later run):
  cache/audio_fingerprints_highres.npz  keyed by chart_file (identity-stamped, NOT index
  -- dodges the dataset-cache-footgun). The expensive librosa decode runs ONCE per song,
  ever; subsequent runs (any reference) load the store and only extract songs not yet in
  it. Delete the npz to force a full re-extraction (e.g. if the feature extractor changes).

Imports the decode harness (make_feature_extractor) + data-split helper per project
convention; builds the dataset with cache_dir=None to dodge the index-keyed sample cache,
and extracts features DIRECTLY (not via the dataset's retry-on-failure getitem, which
would silently substitute a different song).

Usage:
    # first run builds the cache (~40 min cold); later runs are seconds
    python experiments/generation_typed/probe_song_similarity.py \
        --data_dir data/ --audio_dir data/ --ref "grand chariot" --topk 20
"""
import warnings, os, sys
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, csv
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import split_chart_files
from src.data.dataset import StepManiaDataset
from src.generation.decode_harness import make_feature_extractor
import yaml

FEATURE_VERSION = "highres42"  # bump if the extractor's dims/meaning change -> invalidates the store


def pool_fingerprint(feats):
    """(T, D) aligned audio features -> (2D,) mean|std over valid frames."""
    feats = np.asarray(feats, dtype=np.float64)
    return np.concatenate([feats.mean(axis=0), feats.std(axis=0)])


def load_store(path):
    """Return {chart_file: (fingerprint, radar)} from the npz store (empty if absent/stale)."""
    if not path.exists():
        return {}
    z = np.load(path, allow_pickle=True)
    if str(z.get('feature_version', '')) != FEATURE_VERSION:
        print(f"store feature_version mismatch -> ignoring {path.name} (will rebuild)", flush=True)
        return {}
    files, fps, rads = z['chart_files'], z['fingerprints'], z['radars']
    return {str(f): (fps[i], rads[i]) for i, f in enumerate(files)}


def save_store(path, store):
    files = list(store)
    np.savez(path,
             feature_version=FEATURE_VERSION,
             chart_files=np.array(files, dtype=object),
             fingerprints=np.stack([store[f][0] for f in files]),
             radars=np.stack([store[f][1] for f in files]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--audio_dir', required=True)
    ap.add_argument('--ref', default='grand chariot', help='case-insensitive substring of the reference song path')
    ap.add_argument('--topk', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--store', default='cache/audio_fingerprints_highres.npz',
                    help='persistent per-song fingerprint cache (identity-stamped by chart_file)')
    ap.add_argument('--out', default='cache/song_similarity_grandchariot.csv')
    args = ap.parse_args()
    set_seed(args.seed)
    store_path = PROJECT_ROOT / args.store

    _, val_files, _ = split_chart_files(root=args.data_dir, random_state=args.seed)
    print(f"val files: {len(val_files)}", flush=True)

    feat_ext, audio_dim, _ = make_feature_extractor('highres')  # 42-dim deployed space
    with open(PROJECT_ROOT / "config/model_config.yaml") as f:
        msl = yaml.safe_load(f)['classifier']['max_sequence_length']
    # cache_dir=None: subset probe, avoid the index-keyed sample cache footgun (memory dataset-cache-footgun)
    ds = StepManiaDataset(chart_files=val_files, audio_dir=args.audio_dir,
                          max_sequence_length=msl, feature_extractor=feat_ext, cache_dir=None)

    # one entry per SONG (dedup by chart_file; audio is difficulty-invariant). Prefer the Hard sample's radar.
    by_song = {}
    for s in ds.valid_samples:
        cf = s['chart_file']
        if cf not in by_song or s['difficulty_name'] == 'Hard':
            by_song[cf] = s
    songs = list(by_song.items())
    print(f"unique songs: {len(songs)}", flush=True)

    store = load_store(store_path)                                   # {chart_file: (fp, radar)}
    todo = [(cf, m) for cf, m in songs if cf not in store]
    print(f"fingerprints cached: {len(store)}   to extract: {len(todo)}", flush=True)

    for i, (cf, meta) in enumerate(todo):
        if i % 25 == 0:
            print(f"  extracting {i}/{len(todo)}...", flush=True)
            if i > 0:
                save_store(store_path, store)                        # checkpoint so a crash never loses the sweep
        try:
            af = feat_ext.extract_from_chart(meta['audio_file'], meta['chart'])
            if af is None:
                continue
            feats = af.get_aligned_features()  # (T, 42)
            if feats is None or feats.shape[0] < 8 or np.any(~np.isfinite(feats)):
                continue
            store[cf] = (pool_fingerprint(feats), np.asarray(meta['groove_radar'].to_vector(), float))
        except Exception as e:
            print(f"  skip {cf}: {e}", flush=True)
            continue
    save_store(store_path, store)
    print(f"store now holds {len(store)} songs -> {store_path}", flush=True)

    # ----- rank the val pool against the reference (cheap; pure numpy on the cached store) -----
    val_songs = [cf for cf, _ in songs if cf in store]
    titles = {cf: (getattr(m['chart'], 'title', None) or Path(cf).parent.name) for cf, m in songs}
    ref_cf = next((cf for cf in val_songs if args.ref.lower() in cf.lower()), None)
    if ref_cf is None:
        raise SystemExit(f"reference {args.ref!r} not among fingerprinted val songs")

    X = np.stack([store[cf][0] for cf in val_songs])                 # (N, 84)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    ref = Z[val_songs.index(ref_cf)]
    eucl = np.linalg.norm(Z - ref, axis=1)
    cos = (Z @ ref) / (np.linalg.norm(Z, axis=1) * np.linalg.norm(ref) + 1e-9)

    order = np.argsort(eucl)
    print(f"\nreference: {titles[ref_cf]}  ({ref_cf})", flush=True)
    print("# rank  eucl   cos    radar[str,vol,air,frz,cha]        title", flush=True)
    out_rows = []
    for rank, j in enumerate(order):
        cf = val_songs[j]
        rad = store[cf][1]
        radstr = "[" + ",".join(f"{v:.2f}" for v in rad) + "]"
        tag = "  <-- REF" if cf == ref_cf else ""
        if rank <= args.topk:
            print(f"{rank:>3}  {eucl[j]:6.2f}  {cos[j]:+.2f}  {radstr:32s}  {titles[cf][:42]}{tag}", flush=True)
        out_rows.append((rank, float(eucl[j]), float(cos[j]), titles[cf], cf, *[float(v) for v in rad]))

    outp = PROJECT_ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rank', 'eucl_z', 'cos_z', 'title', 'chart_file', 'stream', 'voltage', 'air', 'freeze', 'chaos'])
        w.writerows(out_rows)
    print(f"\nwrote {outp}", flush=True)


if __name__ == '__main__':
    main()

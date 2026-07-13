#!/usr/bin/env python3
"""Feature-cache registry + CLI: one place that knows every feature cache and fitted artifact.

The project has FOUR generations of feature caches (all `cache/<name>/{train,val}/sample_NNNNNN.pt`,
built lazily by StepManiaDataset on first use) plus a handful of small FITTED artifacts. Historically
that knowledge lived in skills/notes only; this module makes it executable.

Commands (run from the repo root of the checkout that holds cache/):
    python tools/cache.py status                 # every cache: files, size, manifest freshness
    python tools/cache.py write-manifest NAME    # snapshot a cache's state into cache/<name>/manifest.json
    python tools/cache.py verify NAME            # recount vs manifest (drift check)
    python tools/cache.py verify NAME --deep     # also load one sample per split and report tensor dims

Cache identity note: sample files are INDEX-keyed with an identity stamp (post-d6bde49); subset
probes must still use cache_dir=None (see memory `dataset-cache-footgun`).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------- registry
# description / spec are documentation; dims is the expected feature dim (checked by verify --deep).
FEATURE_CACHES = {
    "samples": dict(
        dims=23, grid="16th (subdiv=4)",
        spec="base AudioFeatureConfig (MFCC/onset/spectral)",
        consumers="legacy classifier + 23-dim gen_style lineage; realism-critic feature space",
    ),
    "samples_v2": dict(
        dims=41, grid="16th (subdiv=4)",
        spec="+chroma +HPSS onsets +metric phase (NO highres onset)",
        consumers="old stage-1 lineage (archived trainers)",
        warm="python experiments/generation_typed/warm_cache_v2.py --data_dir data/ --audio_dir data/ --workers 4",
    ),
    "samples_v3": dict(
        dims=42, grid="16th (subdiv=4)",
        spec="highres: +chroma +HPSS +metric_phase +highres_onset (generation-defaults skill §0)",
        consumers="legacy v1 deployed model gen_motif_full_fixed (--features highres)",
    ),
    "samples_v3_48th": dict(
        dims=42, grid="48th (subdiv=12, beat_sync)",
        spec="highres_v2: highres channels on the 48th beat-synchronous grid (StepManiaParser.for_v2())",
        consumers="DEPLOYED model gen_motif_v2_48th_cont (--features highres_v2); both CLIs' default",
    ),
}

FITTED_ARTIFACTS = {
    "radar_manifold.npz": "groove-radar joint Gaussian (RadarManifold; refit via RadarManifold.from_vectors(...).save) — SHIPPED (tracked in git); refit is RETRAIN-GATED, see notes/manifold_radar_subdiv_findings.md",
    "motif_basis.npz": "12-axis radar-orthogonal motif basis (fit by experiments/generation_typed/fit_motif_basis.py)",
    "song_bpms.npz": "per-song BPM index derived from the training packs",
    "audio_fingerprints_highres.npz": "audio similarity fingerprints (src/data/similarity.py consumers)",
}


def repo_root() -> Path:
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True)
    return Path(out.stdout.strip())


def git_sha(root: Path) -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def scan(cache_dir: Path) -> dict:
    splits = {}
    total_bytes = 0
    for split_dir in sorted(p for p in cache_dir.iterdir() if p.is_dir()):
        files = list(split_dir.glob("sample_*.pt"))
        nbytes = sum(f.stat().st_size for f in files)
        splits[split_dir.name] = dict(count=len(files), bytes=nbytes)
        total_bytes += nbytes
    return dict(splits=splits, total_bytes=total_bytes)


def human(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}TB"


def cmd_status(root: Path) -> int:
    cache = root / "cache"
    if not cache.exists():
        print(f"no cache/ under {root} (feature caches live only in the primary checkout)")
        return 1
    print(f"cache root: {cache}\n")
    for name, meta in FEATURE_CACHES.items():
        d = cache / name
        if not d.exists():
            print(f"  {name:18} MISSING   ({meta['dims']}-dim, {meta['grid']})")
            continue
        state = scan(d)
        counts = ", ".join(f"{k}:{v['count']}" for k, v in state["splits"].items()) or "empty"
        man = d / "manifest.json"
        if man.exists():
            recorded = json.loads(man.read_text())
            drift = "" if recorded.get("splits") == state["splits"] else "  ⚠ manifest STALE (run verify)"
            man_s = f"manifest {recorded.get('written_at', '?')[:10]}{drift}"
        else:
            man_s = "no manifest (run write-manifest)"
        print(f"  {name:18} {human(state['total_bytes']):>8}  [{counts}]  {meta['dims']}-dim {meta['grid']}  — {man_s}")
    print("\nfitted artifacts:")
    for fname, desc in FITTED_ARTIFACTS.items():
        f = cache / fname
        mark = human(f.stat().st_size) if f.exists() else "MISSING"
        print(f"  {fname:32} {mark:>8}  {desc.split(' — ')[0]}")
    return 0


def cmd_write_manifest(root: Path, name: str) -> int:
    d = root / "cache" / name
    if not d.exists():
        print(f"no such cache: {d}")
        return 1
    meta = FEATURE_CACHES[name]
    manifest = dict(
        name=name,
        dims=meta["dims"],
        grid=meta["grid"],
        spec=meta["spec"],
        consumers=meta["consumers"],
        warm=meta.get("warm", "lazy: StepManiaDataset(cache_dir='cache/%s') populates on first use" % name),
        written_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        git_sha=git_sha(root),
        **scan(d),
    )
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {d / 'manifest.json'}  ({manifest['splits']})")
    return 0


def cmd_verify(root: Path, name: str, deep: bool) -> int:
    d = root / "cache" / name
    man = d / "manifest.json"
    if not man.exists():
        print(f"no manifest for {name} — run write-manifest first")
        return 1
    recorded = json.loads(man.read_text())
    state = scan(d)
    ok = recorded["splits"] == state["splits"]
    print(f"{name}: manifest {'MATCHES ✓' if ok else 'DRIFTED ✗'}")
    if not ok:
        print(f"  recorded: {recorded['splits']}\n  actual:   {state['splits']}")
    if deep:
        import torch  # heavy; only on --deep
        expected = FEATURE_CACHES[name]["dims"]
        for split in state["splits"]:
            sample_files = sorted((d / split).glob("sample_*.pt"))
            if not sample_files:
                continue
            obj = torch.load(sample_files[0], map_location="cpu", weights_only=False)
            dims = _find_feature_dim(obj)
            status = "✓" if dims == expected else f"✗ expected {expected}"
            print(f"  {split}/{sample_files[0].name}: feature dim {dims} {status}")
    return 0 if ok else 1


def _find_feature_dim(obj):
    """Cached sample layout (StepManiaDataset, identity-stamped): {'key': str, 'sample': {'audio': (T, C), ...}}."""
    import torch
    if isinstance(obj, dict) and isinstance(obj.get("sample"), dict):
        audio = obj["sample"].get("audio")
        if torch.is_tensor(audio) and audio.dim() == 2:
            return audio.shape[-1]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    for c in ("write-manifest", "verify"):
        p = sub.add_parser(c)
        p.add_argument("name", choices=sorted(FEATURE_CACHES))
        if c == "verify":
            p.add_argument("--deep", action="store_true", help="load one sample per split, check feature dims")
    args = ap.parse_args()
    root = repo_root()
    if args.cmd == "status":
        return cmd_status(root)
    if args.cmd == "write-manifest":
        return cmd_write_manifest(root, args.name)
    return cmd_verify(root, args.name, args.deep)


if __name__ == "__main__":
    sys.exit(main())

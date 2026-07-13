#!/usr/bin/env python3
"""(Re)generate the golden decode fingerprints: tests/golden/decode_fingerprints.json.

Run this ONLY when a decode-output change is INTENDED (a new canonical knob, a deliberate default
change). The resulting .json diff in the PR is the reviewable statement "this change alters decoded
output for these songs, in these ways" — commit it together with the change that caused it.

Runs every case in tools/decode_fingerprint.GOLDEN_CASES through the real exporter CLI
(~2.5 min for v2 + ~1 min for v1 on the RTX 3060 box). Requires: data/, the feature caches,
the deployed + legacy checkpoints, CUDA.
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decode_fingerprint as dfp


def bless() -> int:
    golden = {"_meta": {}, "cases": {}}
    for case in dfp.GOLDEN_CASES:
        with tempfile.TemporaryDirectory(prefix=f"golden_{case}_") as td:
            cmd = dfp.exporter_cmd(case, td)
            print(f"== {case}: {' '.join(cmd[1:])}")
            t0 = time.time()
            r = subprocess.run(cmd, cwd=dfp.REPO_ROOT, capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout[-2000:], r.stderr[-2000:], sep="\n")
                print(f"FAILED: exporter exited {r.returncode} for {case}")
                return 1
            fps = dfp.fingerprint_export_dir(td)
            print(f"   {len(fps)} songs fingerprinted in {time.time() - t0:.0f}s: {sorted(fps)}")
            golden["cases"][case] = fps

    import torch
    golden["_meta"] = {
        "blessed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=dfp.REPO_ROOT,
                                  capture_output=True, text=True).stdout.strip(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "note": "Byte-determinism is per-machine/torch-version; re-bless after an environment change.",
    }
    dfp.GOLDEN_JSON.parent.mkdir(parents=True, exist_ok=True)
    dfp.GOLDEN_JSON.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
    print(f"BLESSED -> {dfp.GOLDEN_JSON.relative_to(dfp.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(bless())

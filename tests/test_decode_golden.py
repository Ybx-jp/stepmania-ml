"""GOLDEN DECODE REGRESSION: the bare canonical export must keep producing the blessed output.

The values<->behavior validator (sibling of tools/check_export_defaults.py [docs<->values] and
tools/check_repo_layout.py [layout<->convention]): runs the REAL exporter CLI on the pinned golden
songs and compares structural fingerprints against tests/golden/decode_fingerprints.json. The
decode stack is deterministic (seed 42 + cudnn.deterministic), so ANY mismatch = the decode
behavior changed. If the change was intended, regenerate + commit the goldens:

    python tools/bless_golden.py

Slow (~4 min total, GPU): deselect with `pytest -m "not golden"` for quick iterations.
Skips (rather than fails) when the machine lacks the artifacts (data/, caches, checkpoints, CUDA)
or when no golden file has been blessed yet.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))
import decode_fingerprint as dfp  # noqa: E402

pytestmark = pytest.mark.golden


def _skip_reason():
    if not dfp.GOLDEN_JSON.exists():
        return "no golden file blessed yet (run tools/bless_golden.py)"
    if not (REPO_ROOT / "data").exists():
        return "training dataset (data/) not on this machine"
    for ckpt in ("checkpoints/gen_motif_v2_48th_cont/best_val.pt",
                 "checkpoints/gen_motif_full_fixed/best_val.pt"):
        if not (REPO_ROOT / ckpt).exists():
            return f"missing {ckpt}"
    import torch
    if not torch.cuda.is_available():
        return "no CUDA (goldens are blessed on the GPU box; CPU decode would differ and crawl)"
    return None


@pytest.fixture(scope="module")
def golden():
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    return json.loads(dfp.GOLDEN_JSON.read_text())


@pytest.mark.parametrize("case", list(dfp.GOLDEN_CASES))
def test_decode_golden(case, golden, tmp_path):
    expected = golden["cases"].get(case)
    if expected is None:
        pytest.skip(f"case {case!r} not in golden file (re-bless to add it)")

    cmd = dfp.exporter_cmd(case, tmp_path)
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, f"exporter failed ({r.returncode}):\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}"

    actual = dfp.fingerprint_export_dir(tmp_path)
    if actual != expected:
        detail = "\n".join(dfp.diff_fingerprints(expected, actual, label=f"[{case}] "))
        pytest.fail(
            "Decoded output DIFFERS from the blessed golden — the decode behavior changed.\n"
            f"{detail}\n"
            "If this change is INTENDED (new canonical knob / deliberate default change), regenerate\n"
            "and commit the goldens alongside it:  python tools/bless_golden.py\n"
            f"(golden blessed at {golden['_meta'].get('blessed_at')} on {golden['_meta'].get('gpu')}, "
            f"git {golden['_meta'].get('git_sha')})"
        )

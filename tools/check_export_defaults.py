#!/usr/bin/env python3
"""VALIDATOR: the documented CANONICAL EXPORT DEFAULTS must match export_typed_samples.py's LIVE argparse defaults.

The exporter's bare defaults ARE the one canonical config (generation-defaults skill). This guards against the
doc drifting from the code — a stale "canonical defaults" description silently mis-guides the next probe/export.
Run as part of the `/refresh` cycle; exit 1 (with the diffs) if HANDOFF.md's canonical block is out of alignment.

Source of truth = export_typed_samples.py argparse. Documented mirror = the block in notes/HANDOFF.md between
`<!-- CANONICAL-EXPORT-DEFAULTS:START -->` and `:END`, as `key = value` lines.

  python tools/check_export_defaults.py        # -> "ALIGNED" + exit 0, or the mismatches + exit 1
"""
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "generation_typed"))

HANDOFF = PROJECT_ROOT / "notes" / "HANDOFF.md"
START, END = "<!-- CANONICAL-EXPORT-DEFAULTS:START", "<!-- CANONICAL-EXPORT-DEFAULTS:END"


def live_defaults():
    """The exporter's argparse defaults (the source of truth)."""
    import export_typed_samples as exp
    argv = sys.argv
    sys.argv = ["x", "--data_dir", "data/", "--audio_dir", "data/"]   # satisfy the two required args
    try:
        ns = exp.parse_args()
    finally:
        sys.argv = argv
    return vars(ns)


def documented_block():
    """The `key = value` lines from the HANDOFF canonical block."""
    if not HANDOFF.exists():
        sys.exit(f"FAIL: {HANDOFF} not found")
    text = HANDOFF.read_text(encoding="utf-8")
    try:
        body = text.split(START, 1)[1].split(END, 1)[0]
    except IndexError:
        sys.exit(f"FAIL: canonical-defaults markers ({START} ... :END) missing from {HANDOFF.name}")
    out = {}
    for line in body.splitlines():
        m = re.match(r"\s*([a-zA-Z_]\w*)\s*=\s*(.+?)\s*$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def matches(doc_val, live_val):
    """Compare a documented string to a live argparse default, tolerant of float formatting / None."""
    if live_val is None:
        return doc_val.strip().lower() in ("none", "off", "")
    try:                                            # numeric compare when both look numeric
        return abs(float(doc_val) - float(live_val)) < 1e-9
    except (TypeError, ValueError):
        return doc_val.strip() == str(live_val).strip()


def main():
    live, doc = live_defaults(), documented_block()
    if not doc:
        sys.exit("FAIL: no `key = value` lines found inside the canonical block")
    bad = []
    for key, dval in doc.items():
        if key not in live:
            bad.append(f"  {key}: documented but NOT an export arg (renamed/removed?)  doc={dval!r}")
        elif not matches(dval, live[key]):
            bad.append(f"  {key}: doc={dval!r}  != live default={live[key]!r}")
    print(f"checked {len(doc)} documented canonical defaults against export_typed_samples.py")
    if bad:
        print("MISALIGNED — update notes/HANDOFF.md (+ the generation-defaults skill) to match the code:")
        print("\n".join(bad))
        sys.exit(1)
    print("ALIGNED ✓  (HANDOFF canonical export defaults == export_typed_samples.py argparse defaults)")


if __name__ == "__main__":
    main()

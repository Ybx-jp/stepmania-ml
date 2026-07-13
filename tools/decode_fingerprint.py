#!/usr/bin/env python3
"""Golden-decode fingerprints: the single source for WHAT the golden regression pins and HOW.

The values↔behavior rung of the validator ladder (check_export_defaults = docs↔values,
check_repo_layout = layout↔convention, THIS = values↔behavior): a bare canonical run of the
exporter on a pinned song set must keep producing byte-identical charts. The full decode stack is
deterministic end-to-end (set_seed 42 + cudnn.deterministic; verified 2026-07-13: two independent
full runs byte-identical), so any fingerprint change means the decode BEHAVIOR changed — either an
intended canonical change (re-bless, and the .json diff documents it in review) or a regression.

Pieces:
  GOLDEN_CASES            the pinned exporter invocations (bare canonical v2 + legacy v1)
  fingerprint_sm()        structural fingerprint of one .sm (per-chart hashes + symbol counts)
  fingerprint_export_dir() fingerprints of every exported song folder
  diff_fingerprints()     human-readable explanation of what moved (for the test's assert message)

Workflow:
  python tools/bless_golden.py          # (re)generate tests/golden/decode_fingerprints.json
  pytest tests/test_decode_golden.py    # enforce it (also in the default suite; -m "not golden" to skip)

The golden songs (val set, chosen to pin distinct decode regimes on the deployed v2 model):
  A Stupid Barber  (T=2624)  short control — onset window is a no-op
  Giudecca         (T=3577)  just UNDER the W3600 universal-window boundary — pins that edge
  Dead Heat        (T=4080)  window + tail hangover FIRE
The v1_legacy case runs the same songs through the legacy 16th-grid stack, pinning every
"v1 / subdiv=4 byte-identical no-op" claim the v2 knobs make.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_JSON = REPO_ROOT / "tests" / "golden" / "decode_fingerprints.json"

_SONG_FILTER = "stupid barber,giudecca,dead heat"

# Case name -> exporter argv tail. The v2 case is DELIBERATELY bare: the whole point is that the
# unadorned canonical run (deployed defaults) is what gets pinned.
GOLDEN_CASES = {
    "v2_deployed": [],
    "v1_legacy": ["--features", "highres",
                  "--checkpoint", "checkpoints/gen_motif_full_fixed/best_val.pt"],
}


def exporter_cmd(case: str, out_dir, python=sys.executable) -> list[str]:
    """The exact pinned invocation for a golden case (paths relative to repo root)."""
    return [
        python, "scripts/export_typed_samples.py",
        "--data_dir", "data/", "--audio_dir", "data/",
        "--num_songs", "3", "--song_filter", _SONG_FILTER,
        "--out_dir", str(out_dir),
        *GOLDEN_CASES[case],
    ]


def _header_field(txt: str, name: str) -> str:
    m = re.search(rf"#{name}:([^;]*);", txt)
    return m.group(1).strip() if m else ""


def fingerprint_sm(path) -> dict:
    """Structural fingerprint of one .sm: alignment headers + per-chart note-data hash & symbol counts."""
    txt = Path(path).read_text(errors="replace")
    charts = {}
    for m in re.finditer(r"#NOTES:(.*?);", txt, re.S):
        parts = m.group(1).split(":")
        if len(parts) < 6:
            continue
        steps_type, author, diff, meter = (p.strip() for p in parts[:4])
        notes = parts[5]
        rows = [r.strip() for meas in notes.split(",") for r in meas.strip().splitlines() if r.strip()]
        norm = ",".join(
            "\n".join(r.strip() for r in meas.strip().splitlines() if r.strip())
            for meas in notes.split(",")
        )
        charts[f"{steps_type}:{diff}:{meter}"] = {
            "notes_sha256": hashlib.sha256(norm.encode()).hexdigest()[:16],
            "n_measures": notes.count(",") + 1,
            "n_rows": len(rows),
            "taps": sum(r.count("1") for r in rows),
            "hold_heads": sum(r.count("2") for r in rows),
            "hold_tails": sum(r.count("3") for r in rows),
            "rolls": sum(r.count("4") for r in rows),
            "author": author,
        }
    return {
        "file_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16],
        "offset": _header_field(txt, "OFFSET"),
        "bpms": _header_field(txt, "BPMS"),
        "charts": charts,
    }


def fingerprint_export_dir(out_dir) -> dict:
    """Fingerprint every exported `<NN_song>/chart.sm` under an exporter --out_dir."""
    out_dir = Path(out_dir)
    fps = {}
    for sm in sorted(out_dir.glob("*/chart.sm")):
        fps[sm.parent.name] = fingerprint_sm(sm)
    return fps


def diff_fingerprints(expected: dict, actual: dict, label: str = "") -> list[str]:
    """Explain a golden mismatch: which songs/charts changed and how (counts, hashes, headers)."""
    lines = []
    for song in sorted(set(expected) | set(actual)):
        if song not in actual:
            lines.append(f"{label}{song}: MISSING from this run")
            continue
        if song not in expected:
            lines.append(f"{label}{song}: NEW song not in golden (selection changed?)")
            continue
        e, a = expected[song], actual[song]
        if e == a:
            continue
        for h in ("offset", "bpms"):
            if e.get(h) != a.get(h):
                lines.append(f"{label}{song}: #{h.upper()} {e.get(h)!r} -> {a.get(h)!r}")
        for chart in sorted(set(e["charts"]) | set(a["charts"])):
            ec, ac = e["charts"].get(chart), a["charts"].get(chart)
            if ec == ac:
                continue
            if ec is None or ac is None:
                lines.append(f"{label}{song} [{chart}]: {'added' if ec is None else 'removed'}")
                continue
            deltas = [f"{k}: {ec[k]} -> {ac[k]}" for k in ec if ec[k] != ac[k]]
            lines.append(f"{label}{song} [{chart}]: " + "; ".join(deltas))
        if e["charts"] == a["charts"] and e["file_sha256"] != a["file_sha256"]:
            lines.append(f"{label}{song}: note data identical but file bytes differ (header/presentation change)")
    return lines


if __name__ == "__main__":
    # Ad-hoc use: fingerprint an export dir (or a single .sm) and print JSON.
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if target is None:
        print(__doc__)
        sys.exit(0)
    fp = fingerprint_sm(target) if target.suffix == ".sm" else fingerprint_export_dir(target)
    print(json.dumps(fp, indent=2))

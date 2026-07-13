#!/usr/bin/env python3
"""
DIVERGENCE check (notes/phase3_generative_design.md, data-gate follow-up): the 484 multi-pack titles are a
potential distribution-supervision pool -- but only if independent human chartings actually DIFFER (and
aren't copies). For each multi-pack title, compare two chartings at a matched difficulty:
  - identical note data -> COPY (not independent).
  - same grid (same length + same BPM) -> compute onset IoU + 16th-frame IoU (placement agreement; LOW =
    humans place notes differently = the AMBIGUITY we're trying to model).
  - different length/BPM -> structurally different (counts as differing, no IoU).

Reads:
  many non-copies with LOW 16th-IoU -> real placement ambiguity + usable distribution data -> Phase-3 viable.
  mostly copies / high IoU -> the 484 aren't independent re-chartings -> distribution-supervision data is thin.

  python experiments/generation_typed/diag_chart_divergence.py --max_titles 300
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, re, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.stepmania_parser import StepManiaParser
from src.data.dataset import get_difficulty_class


def norm_title(p):
    try: txt = open(p, encoding='utf-8', errors='ignore').read(4000)
    except Exception: return None
    m = re.search(r'#TITLE:([^;]*);', txt)
    return re.sub(r'\s+', ' ', m.group(1).strip().lower()) if m else None


def singles(chart):
    """{difficulty_class: NoteData} for dance-single charts (highest value per class)."""
    out = {}
    for nd in chart.note_data:
        c = get_difficulty_class(nd.difficulty_name)
        if c is None: continue
        if c not in out or nd.difficulty_value > out[c].difficulty_value:
            out[c] = nd
    return out


def iou(o1, o2):
    u = (o1 | o2).sum(); return float((o1 & o2).sum() / u) if u else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_titles', type=int, default=300); ap.add_argument('--bpm_tol', type=float, default=0.5)
    args = ap.parse_args()
    p = StepManiaParser()
    files = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    tf = defaultdict(set)
    for f in files:
        t = norm_title(f)
        if t: tf[t].add((os.path.dirname(os.path.dirname(f)), f))
    multi = [list({fp for _, fp in v}) for k, v in tf.items() if len({pk for pk, _ in v}) >= 2]
    print(f"{len(multi)} multi-pack titles; checking up to {args.max_titles}\n", flush=True)

    n_pair = n_copy = n_difflen = n_comparable = 0
    onset_iou, s16_iou = [], []
    checked = 0
    for fps in multi:
        if checked >= args.max_titles: break
        try:
            charts = [p.parse_file(fp) for fp in fps[:2]]
        except Exception:
            continue
        if any(c is None for c in charts): continue
        s = [singles(c) for c in charts]
        common = set(s[0]) & set(s[1])
        if not common: continue
        c = max(common)                                   # highest common difficulty class
        try:
            t1 = np.asarray(p.convert_to_tensor_typed(charts[0], s[0][c]))
            t2 = np.asarray(p.convert_to_tensor_typed(charts[1], s[1][c]))
        except Exception:
            continue
        checked += 1; n_pair += 1
        o1 = (t1 != 0).any(1); o2 = (t2 != 0).any(1)
        if t1.shape == t2.shape and np.array_equal(t1, t2):
            n_copy += 1; continue
        same_grid = (len(o1) == len(o2)) and abs(float(charts[0].bpm) - float(charts[1].bpm)) <= args.bpm_tol
        if not same_grid:
            n_difflen += 1; continue
        n_comparable += 1
        onset_iou.append(iou(o1, o2))
        tt = np.arange(len(o1)); m16 = (tt % 4 == 1) | (tt % 4 == 3)
        s16_iou.append(iou(o1 & m16, o2 & m16))

    print(f"=== DIVERGENCE check ({n_pair} title-pairs compared) ===")
    print(f"  COPIES (identical note data):        {n_copy:>4}  ({100*n_copy/max(n_pair,1):.0f}%)")
    print(f"  different grid (BPM/length differ):  {n_difflen:>4}  ({100*n_difflen/max(n_pair,1):.0f}%) -> differ, no IoU")
    print(f"  comparable (same grid, non-copy):    {n_comparable:>4}  ({100*n_comparable/max(n_pair,1):.0f}%)")
    if onset_iou:
        print(f"\n  for comparable pairs (placement agreement; LOW = humans differ):")
        print(f"    onset IoU   mean {np.mean(onset_iou):.3f}  median {np.median(onset_iou):.3f}")
        print(f"    16th  IoU   mean {np.mean(s16_iou):.3f}  median {np.median(s16_iou):.3f}   "
              f"(<0.5 = substantial 16th-placement divergence)")
    print(f"\n  many non-copies + LOW 16th-IoU -> real placement ambiguity + usable distribution data.")
    print(f"  mostly copies / high IoU -> not independent re-chartings -> thin distribution data.")


if __name__ == '__main__':
    main()

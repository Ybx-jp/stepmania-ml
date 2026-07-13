#!/usr/bin/env python3
"""
Does the max-2-simultaneous filter (validate_pattern_quality, called in process_chart) exclude real
HARD charts disproportionately — i.e. is it a data-layer contributor to "generated Hard feels tame"?

For every difficulty in every chart file (parsed RAW, before the filter), compute max simultaneous
panels and the fraction of note-frames with 3+ (hands) / 4 (quads). Tally, per difficulty class, how
many would be REJECTED by the >2 rule. If Hard is rejected far more than Easy/Medium, the dataset's
"Hard" is a tame subset and the model can't learn hands because it never sees them.
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import sys, glob
from pathlib import Path
from collections import defaultdict
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.stepmania_parser import StepManiaParser
from src.data.dataset import get_difficulty_class, DIFFICULTY_NAMES


def main():
    parser = StepManiaParser()
    files = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    print(f"parsing {len(files)} chart files (raw, no audio)...")

    n_total = defaultdict(int)        # difficulties seen, by class
    n_rejected = defaultdict(int)     # would be dropped by >2 rule
    hands_frac = defaultdict(list)    # fraction of note-frames with 3+ panels (kept+rejected)
    quad_any = defaultdict(int)       # difficulties with at least one 4-panel frame
    done = 0
    for f in files:
        try:
            chart = parser.parse_file(f)
        except Exception:
            continue
        if chart is None:
            continue
        done += 1
        for nd in chart.note_data:
            cls = get_difficulty_class(nd.difficulty_name)
            if cls is None:
                continue
            try:
                bin_t = parser.convert_to_tensor(chart, nd)  # (T,4) binary, what validate uses
            except Exception:
                continue
            persum = bin_t.sum(axis=1)
            note_frames = (persum > 0).sum()
            if note_frames == 0:
                continue
            mx = int(persum.max())
            n_total[cls] += 1
            if mx > 2:
                n_rejected[cls] += 1
            hands_frac[cls].append(float((persum >= 3).sum()) / note_frames)
            if mx >= 4:
                quad_any[cls] += 1

    print(f"parsed {done} files\n")
    hdr = f"{'difficulty':<10} {'#diffs':>7} {'rejected(>2)':>13} {'rej%':>6} {'mean hands-frame%':>18} {'has-quad%':>10}"
    print(hdr); print("-" * len(hdr))
    for c in range(4):
        n = n_total[c]
        if n == 0:
            print(f"{DIFFICULTY_NAMES[c]:<10} {0:>7}"); continue
        rej = n_rejected[c]
        hf = 100 * float(np.mean(hands_frac[c])) if hands_frac[c] else 0.0
        qa = 100 * quad_any[c] / n
        print(f"{DIFFICULTY_NAMES[c]:<10} {n:>7} {rej:>13} {100*rej/n:>5.1f}% {hf:>17.2f}% {qa:>9.1f}%")
    print("-" * len(hdr))
    print("High Hard rej% (vs Easy) => the >2 filter biases the dataset's 'Hard' toward tame, hands-free")
    print("charts; the model literally never sees hands, so it can't generate the intensity real 11s have.")


if __name__ == '__main__':
    main()

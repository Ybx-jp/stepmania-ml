#!/usr/bin/env python3
"""
Chart-representation INTEGRITY audit — before the H19 retrain (which refits cache/motif_basis.npz and the motif
training targets, both built from the typed chart tensor), sweep for OTHER symbol-misinterpretation / lossy-
construction bugs in the {0 none,1 tap,2 hold_head,3 tail,4 roll_head} pipeline. H19 (figure detector counted
tails) was one instance; this looks for siblings + construction loss. Chart-only (parses sampled .sm), fast.

Checks:
  A. MIRROR (motif_codebook._MIRROR) is a valid permutation AND an involution (M(M(x))=x) — else figure
     canonicalization merges the wrong patterns.
  B. ROLL-HEAD ('4') drop: convert_to_tensor / _extended (the BINARY + radar path) only handle '1','2','3' —
     roll heads are INVISIBLE to the groove radar, while convert_to_tensor_typed KEEPS them (symbol 4). Quantify
     roll prevalence + the resulting typed-vs-binary onset-count gap (radar under-counts these songs).
  C. QUANTIZATION COLLISIONS: ts=floor(beat*4) is a fixed 16th grid; sub-16th notes (24th/32nd/48th) collide
     onto an occupied (ts,panel) and the later symbol OVERWRITES the earlier (note loss). Quantify lost notes.
  D. HOLD pairing health: orphan tails (3 with no head) and zero-length holds (head+tail same ts) that pair_holds
     has to demote/drop — a measure of how many holds the grid mangles.

  python experiments/generation_typed/diag_repr_integrity.py [--n 300]
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, sys, random
from collections import Counter
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.stepmania_parser import StepManiaParser
from src.generation.typed import pattern_to_panels, panels_to_pattern, NUM_PATTERNS
from src.generation.motif_codebook import _MIRROR, onset_tokens


def check_mirror():
    perm = _MIRROR
    is_perm = sorted(int(x) for x in perm) == list(range(NUM_PATTERNS))
    involution = all(int(_MIRROR[int(_MIRROR[i])]) == i for i in range(NUM_PATTERNS))
    # spot semantics: L-only(0b0001 -> idx0) must map to R-only(0b1000 -> idx7); U-only must map to itself
    L = int(panels_to_pattern(np.array([1, 0, 0, 0]))); R = int(panels_to_pattern(np.array([0, 0, 0, 1])))
    U = int(panels_to_pattern(np.array([0, 0, 1, 0])))
    lr_ok = int(_MIRROR[L]) == R and int(_MIRROR[R]) == L
    u_ok = int(_MIRROR[U]) == U
    print("A. MIRROR (figure canonicalization):")
    print(f"   valid permutation: {is_perm}   involution M(M(x))=x: {involution}   "
          f"L<->R: {lr_ok}   U fixed: {u_ok}   => {'PASS' if all([is_perm, involution, lr_ok, u_ok]) else 'FAIL'}")


def raw_notes(parser, note_data, total):
    """Re-walk the measures like the converters; return per-(ts,panel) list of chars + off-grid count."""
    tpb = parser.timesteps_per_beat
    cells = {}                      # (ts,panel) -> list of chars landing there (collision if >1 attack)
    offgrid = 0; total_notes = 0
    cur = 0.0
    for measure in note_data.notes.split(','):
        lines = [l.strip() for l in measure.strip().split('\n') if l.strip()]
        if not lines:
            continue
        bpl = 4.0 / len(lines)
        for li, line in enumerate(lines):
            if len(line) < 4:
                continue
            beat = cur + li * bpl
            ts = int(np.floor(beat * tpb))
            if not (0 <= ts < total):
                continue
            on_grid = abs(beat * tpb - round(beat * tpb)) < 1e-6
            for p in range(4):
                c = line[p]
                if c in '1234':
                    total_notes += 1
                    if c in '124' and not on_grid:
                        offgrid += 1
                    cells.setdefault((ts, p), []).append(c)
        cur += 4.0
    return cells, total_notes, offgrid


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=300); ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    parser = StepManiaParser()

    check_mirror()

    files = glob.glob("data/**/*.sm", recursive=True)
    random.shuffle(files); files = files[:args.n]

    sym = Counter()                 # typed symbol histogram across all charts
    charts_with_roll = 0; n_charts = 0
    roll_onsets = 0; total_onsets = 0
    collide_lost = 0; collide_notes = 0; offgrid_total = 0; raw_total = 0
    orphan_tail = 0; zerolen_hold = 0; total_holds = 0
    typed_minus_binary = []         # per-chart onset-count gap (rolls the binary path misses)

    for f in files:
        try:
            chart = parser.parse_file(f)
        except Exception:
            continue                      # missing audio / unparseable -> skip
        if chart is None or not chart.note_data:
            continue
        nd = max(chart.note_data, key=lambda n: n.difficulty_value)   # hardest chart per song
        total = chart.timesteps_total
        if total <= 0:
            continue
        n_charts += 1
        typed = parser.convert_to_tensor_typed(chart, nd)
        h = Counter(int(v) for v in typed.reshape(-1))
        for k, v in h.items():
            sym[k] += v
        has_roll = (typed == 4).any()
        charts_with_roll += int(has_roll)
        # onsets: attacks only (1,2,4) per the H19-correct detector
        toks = onset_tokens(typed); total_onsets += len(toks)
        roll_onsets += int((typed == 4).any(1).sum())   # frames containing a roll head
        # typed vs binary onset gap: binary path keeps only 1,2 -> drops roll heads (and tails, correctly)
        bin_t, hold_info = parser.convert_to_tensor_extended(chart, nd)
        typed_attacks = int(((typed != 0) & (typed != 3)).any(1).sum())
        binary_onsets = int((bin_t != 0).any(1).sum())
        typed_minus_binary.append(typed_attacks - binary_onsets)

        # collisions + off-grid (construction loss)
        cells, tn, og = raw_notes(parser, nd, total)
        raw_total += tn; offgrid_total += og
        for (ts, p), chars in cells.items():
            attacks = [c for c in chars if c in '124']
            if len(attacks) > 1:
                collide_lost += len(attacks) - 1     # all but one overwritten
                collide_notes += len(attacks)
            # zero-length hold: a head and its tail collide on the same cell
            if any(c in '24' for c in chars) and '3' in chars:
                zerolen_hold += 1
        # hold pairing health from hold_info-independent scan of typed: orphan tails / heads
        for p in range(4):
            col = typed[:, p]; open_head = -1
            for s in col:
                if s in (2, 4):
                    if open_head >= 0:
                        pass
                    open_head = 1; total_holds += 1
                elif s == 3:
                    if open_head < 0:
                        orphan_tail += 1
                    else:
                        open_head = -1

    print(f"\n   (sampled {n_charts} charts, hardest per song)")
    tot_sym = sum(sym.values())
    names = {0: 'none', 1: 'tap', 2: 'hold_head', 3: 'tail', 4: 'roll_head'}
    print("\nB. ROLL-HEAD drop (radar/binary blindness):")
    print("   typed symbol shares: " + "  ".join(f"{names[k]} {sym[k]/tot_sym:.4f}" for k in sorted(names)))
    nz = tot_sym - sym[0]
    print(f"   of NON-empty cells: roll_head {sym[4]/max(nz,1):.4f}  (these are dropped by convert_to_tensor*)")
    print(f"   charts containing >=1 roll: {charts_with_roll}/{n_charts} ({charts_with_roll/max(n_charts,1):.1%})")
    tmb = np.array(typed_minus_binary)
    print(f"   typed_attacks - binary_onsets per chart: mean {tmb.mean():+.1f}, max {tmb.max()} "
          f"(>0 = onsets the radar never sees; rolls + any 1-vs-2 handling)")
    print("\nC. QUANTIZATION COLLISIONS (16th grid note loss):")
    print(f"   off-grid (sub-16th) attack notes: {offgrid_total}/{raw_total} ({offgrid_total/max(raw_total,1):.2%})")
    print(f"   attacks LOST to (ts,panel) overwrite: {collide_lost}/{raw_total} ({collide_lost/max(raw_total,1):.2%})")
    print("\nD. HOLD pairing health:")
    print(f"   total holds (heads): {total_holds}   orphan tails (3 w/o head): {orphan_tail}   "
          f"zero-length holds (head+tail same cell): {zerolen_hold}")


if __name__ == "__main__":
    main()

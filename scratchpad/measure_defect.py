#!/usr/bin/env python3
"""measure_defect.py — the canonical metric for DEFECT #3 (free-foot-stream-under-hold).

The felt pathology (footspeed_floor_findings.md §5/§5b): a LONG hold (5-6 beats) pins one foot while the
FREE foot sustains a fast stream (8ths @148bpm) underneath it. That is unplayable-feeling and human charters
release the hold to stream two-footed. Prior levers (hold_stream_penalty head-gate, stamina salience-thinning)
either delete the hold (fights freeze=high) or can't touch it (the stream is loud, survives salience thinning).

METRIC = the RUN-COUNT the §5b A/B is quoted against ("free-foot stream >=4 notes @<=8th UNDER a hold:
holdfix 2, holdbug 4"). NOT the pair-fraction of freefoot_during_hold (that trap is documented: match the
metric to the FELT run, not a convenient aggregate).

Definition (on the assigned-feet sequence, bipedal_metrics.foot_moves):
  For each foot, walk its notes in time order; a "free-foot-stream-under-hold RUN" is a maximal chain of
  consecutive notes where EVERY note has the OTHER foot pinned by an open hold (other_pinned=True) and each
  step gap (t_i+1 - t_i) <= gap_max frames (<=8th). Count runs whose length >= min_run notes.
    gap_max default = subdiv//2  (an 8th note; 6 frames on the 48th grid, 2 on the 16th grid)
    min_run default = 4          (>=4 notes = a sustained stream, per §5b)

Works on either an in-memory typed grid (T,4) {0 none,1 tap,2 hold-head,3 tail,4 roll} OR a parsed .sm.
Anchor (validate the metric): outputs/watchout_holdfix (subdiv-fix ON) -> 2 ; outputs/watchout_holdbug -> 4.

Usage:
  # measure an exported chart (.sm) -- default reads the FIRST/generated #NOTES block
  python scratchpad/measure_defect.py --sm "outputs/watchout_holdfix/00_Watch Out Pt2/chart.sm"
  # detail every run:
  python scratchpad/measure_defect.py --sm <chart.sm> --detail
"""
import argparse
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'experiments' / 'realism_critic'))
from bipedal_metrics import pinned_mask, foot_moves   # noqa: E402
from choreography_metrics import note_starts           # noqa: E402

SYM = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, 'M': 0, 'L': 0, 'F': 0}  # mine/lift/fake -> none


def parse_sm(path, block=0, rows_per_measure=48):
    """Parse a .sm's `block`-th #NOTES block into a typed (T,4) grid, resampling each measure to
    rows_per_measure (48 = the 48th grid). block=0 = the first (generated/Challenge) chart."""
    text = Path(path).read_text(errors='ignore')
    # split into #NOTES sections; each ends at the next '#' directive
    sections = []
    for chunk in text.split('#NOTES:')[1:]:
        end = chunk.find('\n#')
        sections.append(chunk if end < 0 else chunk[:end])
    if block >= len(sections):
        raise ValueError(f"chart has {len(sections)} #NOTES block(s); asked for block {block}")
    body = sections[block]
    # drop the 5 metadata lines (type/desc/diff/meter/radar) -- everything up to & incl. the 5th ':'
    # then read note rows; measures separated by ',' ; ';' ends the chart
    lines = body.splitlines()
    # find where the note grid starts: after the metadata line ending in a colon that holds the radar values
    colon_meta = 0
    measures, cur = [], []
    started = False
    for ln in lines:
        s = ln.strip()
        if not started:
            if s.endswith(':'):
                colon_meta += 1
                if colon_meta >= 5:          # 5th ':' closes the radar-values metadata line
                    started = True
            continue
        if s.startswith(';'):
            break
        if s.startswith(','):
            measures.append(cur); cur = []
            continue
        if s and set(s) <= set('0123456789MLFxyz'):
            cur.append(s[:4])
    if cur:
        measures.append(cur)
    # resample each measure to rows_per_measure and build the grid
    grid = []
    for m in measures:
        R = len(m)
        if R == 0:
            grid.extend([[0, 0, 0, 0]] * rows_per_measure)
            continue
        out = [[0, 0, 0, 0] for _ in range(rows_per_measure)]
        for i, row in enumerate(m):
            dst = int(round(rows_per_measure * i / R))
            if dst >= rows_per_measure:
                dst = rows_per_measure - 1
            for p in range(4):
                c = row[p] if p < len(row) else '0'
                out[dst][p] = SYM.get(c, 0)
        grid.extend(out)
    return np.asarray(grid, dtype=int)


def hold_speed_violations(typed, speed_gap=6):
    """USER-DEFINED defect (2026-07-12): an 8th (speed_gap frames) is the fastest allowable note-speed during a hold.
    A violation = two consecutive onsets at gap < speed_gap (faster than an 8th) where the EARLIER onset is under an
    OPEN hold (pin true at that frame -- INCLUDING the tail/release frame, which the player still reads as inside the
    hold). Counts the release-coincident note that require_persist wrongly discarded. Returns list of (t0,t1,gap)."""
    pin = pinned_mask(typed); ns = note_starts(typed); T = typed.shape[0]
    persist = pin.copy(); persist[:-1] &= pin[1:]                 # a hold that stays pinned PAST this frame
    per = persist.any(1)
    ons = [t for t in range(T) if ns[t].any()]
    out = []
    for k in range(1, len(ons)):
        t0, t1 = ons[k - 1], ons[k]
        # violation: earlier note under a hold that PERSISTS past it (foot truly stuck) + next note faster than an
        # 8th. If the hold RELEASES on t0 (tail there, per the fix), the foot is freed -> the fast t1 is two-foot -> OK.
        if 0 < (t1 - t0) < speed_gap and per[t0]:
            out.append((t0, t1, t1 - t0))
    return out


def freefoot_stream_runs(typed, gap_max, min_run, require_persist=True):
    """Return the list of (foot, [(panel,frame)...]) runs of >=min_run consecutive free-foot notes, each
    step gap<=gap_max, with the OTHER foot pinned throughout. The §5b defect count = len(this list).

    require_persist (DEFAULT): a note only counts as 'under a hold' if the pinning hold persists PAST this
    note (pin[t+1] still held) -- so the ESCAPE note (the free-foot note that coincides with the hold's tail/
    release) is NOT counted. This matches the user's rule (2026-07-12): 3 notes is an acceptable one-foot
    flourish, the 4th note is where the hold RELEASES so its freed foot can take it -> that 4th note is played
    two-foot-free, not a defect. Anchor-neutral (the holdfix/holdbug runs are mid-hold, no release-escape)."""
    moves = foot_moves(typed)
    pin = pinned_mask(typed); T = pin.shape[0]
    pin_persist = pin.copy()
    pin_persist[:-1] &= pin[1:]                       # (T,4) a hold that is still held at t+1
    persists = pin_persist.any(1)                     # (T,) some hold continues past this frame
    runs = []
    for f in (0, 1):
        seq = sorted(moves[f], key=lambda x: x[1])   # (panel, frame, other_pinned)
        pinned = lambda note: note[2] and (not require_persist or persists[note[1]])
        i = 0
        n = len(seq)
        while i < n:
            if not pinned(seq[i]):                    # note must be under a PERSISTING hold
                i += 1
                continue
            run = [seq[i]]
            j = i + 1
            while j < n and pinned(seq[j]) and 0 < (seq[j][1] - run[-1][1]) <= gap_max:
                run.append(seq[j]); j += 1
            if len(run) >= min_run:
                runs.append((f, [(p, t) for (p, t, _) in run]))
            i = j if j > i + 1 else i + 1
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sm', help='path to an exported .sm chart')
    ap.add_argument('--npy', help='alternatively, a saved typed (T,4) grid .npy')
    ap.add_argument('--block', type=int, default=0, help='which #NOTES block (0=first/generated)')
    ap.add_argument('--subdiv', type=int, default=12, help='timesteps per beat (12=48th grid, 4=16th)')
    ap.add_argument('--gap_max', type=int, default=None, help='max step gap in frames (default subdiv//2 = 8th)')
    ap.add_argument('--min_run', type=int, default=4, help='min consecutive free-foot notes for a defect run')
    ap.add_argument('--detail', action='store_true', help='print every defect run')
    args = ap.parse_args()

    gap_max = args.gap_max if args.gap_max is not None else args.subdiv // 2
    if args.sm:
        typed = parse_sm(args.sm, block=args.block, rows_per_measure=args.subdiv * 4)
        src = args.sm
    elif args.npy:
        typed = np.load(args.npy)
        src = args.npy
    else:
        ap.error('need --sm or --npy')

    T = typed.shape[0]
    ns = note_starts(typed)
    pin = pinned_mask(typed)
    n_notes = int(ns.sum())
    n_heads = int(((typed == 2) | (typed == 4)).sum())
    runs = freefoot_stream_runs(typed, gap_max, args.min_run)

    print(f"chart: {src}")
    print(f"  frames={T}  notes={n_notes}  hold/roll-heads={n_heads}  "
          f"frames-under-a-hold={int(pin.any(1).sum())}")
    print(f"  metric: gap_max={gap_max} frames (<=8th)  min_run={args.min_run} notes")
    print(f"  *** DEFECT #3 count (free-foot stream >=%d @<=8th under a hold) = %d ***" % (args.min_run, len(runs)))
    if args.detail or runs:
        for f, notes in runs:
            f0, f1 = notes[0][1], notes[-1][1]
            panels = ''.join('LDUR'[p] for p, _ in notes)
            print(f"    foot {f}  frames {f0}-{f1}  ({len(notes)} notes, ~beat {f0/args.subdiv:.1f}-{f1/args.subdiv:.1f})  {panels}")


if __name__ == '__main__':
    main()

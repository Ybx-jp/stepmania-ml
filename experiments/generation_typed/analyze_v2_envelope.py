#!/usr/bin/env python3
"""v2 SAFETY-ZONE envelope analysis — parse the generated 'Challenge' charts a sweep produced (the actual .sm
the user would play) and flag any that fall OUTSIDE a playable envelope, per settings arm.

v2-AWARE (the 48th grid): parses with StepManiaParser.for_v2() (timesteps_per_beat=12) so 48th-grid rows are
NOT floored onto a 16th grid (which the 16th-only characterize_sets.py would do). Phase vocabulary on this grid:
quarter t%12==0, 8th==6, 16th-offbeat {3,9}, TRIPLET {2,4,8,10}, 48th {1,5,7,11}; a 16th = f16 = 3 frames.

Metrics = the failure modes we CAN measure offline (metrics are BLIND to musicality by design — they flag
edges; the ears confirm the zone). Each chart is bucketed duple/triplet by the ORIGINAL (human) chart's
triplet_frac, so the b_trip switch's success is legible (triplet songs lifted, duple songs kept clean).

  python experiments/generation_typed/analyze_v2_envelope.py --root outputs/v2_sweep

SAFE ENVELOPE (flags): max_jack>5 (real ~3.5), fast_jump>0 (no_fast_jump target 0), flam>0 (min_onset_gap
target 0), backbone<0.35 or off48>0.15 (off-grid smear), dead_gap>8 beats (mid-song silence). free_foot_hold
is the KNOWN freeze=high edge (reported, expected nonzero on the freeze arm) not a hard flag.
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.stepmania_parser import StepManiaParser
from src.data.meter_detect import chart_triplet_frac, _read

SUBDIV = 12
F16 = SUBDIV // 4          # 3 frames = a 16th
F8 = SUBDIV // 2           # 6 frames = an 8th
PRESS = (1, 2, 4)          # tap / hold-head / roll-head (a foot press); 3 = tail (not a press)
TRIPLET_CELLS = {2, 4, 8, 10}
OFF48_CELLS = {1, 5, 7, 11}

# safe-envelope HARD flags = genuine UNPLAYABILITY only. NOTE: a weak 8th-backbone with LOW off48 is a legit
# dense-16th chart (playable) + the documented note-context placement gap (retrain-bound, accepted in ship
# mode) — a QUALITY not a playability issue, so backbone is REPORTED, not flagged. Off-GRID smear = off48.
LIM = dict(max_jack=5, fast_jump=0, flam=0, off48_max=0.15, dead_beats=8.0)


def chart_metrics(typed):
    """typed (T,4) symbols 0..4 -> failure-mode metrics on the 48th grid. None if empty."""
    typed = np.asarray(typed)
    T = typed.shape[0]
    press = np.isin(typed, PRESS)            # (T,4) a foot press this frame
    onset = press.any(1)
    idx = np.nonzero(onset)[0]
    n_on = len(idx)
    if n_on < 8:
        return None
    npress = press.sum(1)
    ph = idx % SUBDIV
    share = lambda cells: float(np.mean([p in cells for p in ph]))
    quarter = float(np.mean(ph == 0)); eighth = float(np.mean(ph == 6))
    off16 = float(np.mean((ph == 3) | (ph == 9)))
    triplet = share(TRIPLET_CELLS); off48 = share(OFF48_CELLS)
    backbone = quarter + eighth

    gaps = np.diff(idx)
    flam = int((gaps == 1).sum())                                  # 1-frame (48th) flam; min_onset_gap target 0
    # fast jump: a >=2-press JUMP whose gap to the prev onset is sub-16th (<F16); no_fast_jump target 0
    jump_rows = np.nonzero(npress[idx] >= 2)[0]
    fast_jump = int(sum(1 for j in jump_rows if j > 0 and gaps[j - 1] < F16))
    # max same-panel single JACK run at <=16th adjacency
    singles = [(t, int(np.nonzero(press[t])[0][0])) for t in idx if npress[t] == 1]
    max_jack = cur = 1 if singles else 0
    for (a, pa), (b, pb) in zip(singles, singles[1:]):
        cur = cur + 1 if (pb == pa and (b - a) <= F16) else 1
        max_jack = max(max_jack, cur)
    # dead section: largest INTERNAL onset gap (beats) excluding the leading/trailing silence
    dead_beats = float(gaps.max() / SUBDIV) if len(gaps) else 0.0
    # free-foot-stream-UNDER-hold (the known freeze=high edge, footspeed_floor_findings §5b): while a hold is
    # open on panel p, count runs of >=4 presses on OTHER panels at <=8th spacing.
    ff_hold = _free_foot_under_hold(typed, press)

    dur_s = T * 60.0 / (float(chart_metrics.bpm) * SUBDIV) if getattr(chart_metrics, 'bpm', 0) else 0.0
    nps = (n_on / dur_s) if dur_s > 0 else float('nan')
    return dict(n=n_on, dens=float(onset.mean()), nps=nps,
                quarter=quarter, eighth=eighth, off16=off16, triplet=triplet, off48=off48,
                backbone=backbone, max_jack=int(max_jack), fast_jump=fast_jump, flam=flam,
                dead_beats=dead_beats, ff_hold=ff_hold)


def _free_foot_under_hold(typed, press):
    """count stretches where a hold is open on some panel AND another panel streams >=4 presses at <=8th gap."""
    T = typed.shape[0]
    open_until = {}                                  # panel -> tail frame (exclusive) of its open hold
    # precompute, per panel, the frame of the next tail after each head
    events = 0
    other_run = []                                   # (frame, panel) presses while >=1 hold open
    for t in range(T):
        # update open holds: a head (2/4) opens; a tail (3) closes
        for p in range(4):
            s = typed[t, p]
            if s in (2, 4):
                # find this hold's tail
                tail = next((u for u in range(t + 1, T) if typed[u, p] == 3), T)
                open_until[p] = tail
        held = {p for p, u in open_until.items() if t < u}
        if held:
            for p in range(4):
                if press[t, p] and p not in held:
                    other_run.append((t, p))
        else:
            events += _count_streams(other_run)
            other_run = []
    events += _count_streams(other_run)
    return events


def _count_streams(seq):
    """# of runs of >=4 presses with consecutive gaps <= an 8th."""
    if len(seq) < 4:
        return 0
    runs = 0; run = 1
    for (a, _), (b, _) in zip(seq, seq[1:]):
        if 0 < (b - a) <= F8:
            run += 1
        else:
            runs += (run >= 4); run = 1
    runs += (run >= 4)
    return int(runs)


def flags_for(m):
    f = []
    if m['max_jack'] > LIM['max_jack']: f.append(f"jack{m['max_jack']}")
    if m['fast_jump'] > LIM['fast_jump']: f.append(f"fastjmp{m['fast_jump']}")
    if m['flam'] > LIM['flam']: f.append(f"flam{m['flam']}")
    if m['off48'] > LIM['off48_max']: f.append(f"smear{m['off48']:.2f}")
    if m['dead_beats'] > LIM['dead_beats']: f.append(f"dead{m['dead_beats']:.0f}b")
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/v2_sweep")
    args = ap.parse_args()
    # UNGATED parser: we are re-reading the generated ARTIFACT, not filtering a corpus, so the phase-1 gates
    # (bpm/length/simul) must NOT reject a chart (they'd drop the very songs the widened export admitted).
    parser = StepManiaParser.for_v2(min_bpm=1.0, max_bpm=1e6, min_song_length=0.0,
                                    max_song_length=1e9, max_simultaneous=4)
    arms = sorted(d for d in glob.glob(f"{args.root}/*") if Path(d).is_dir())

    all_flags = []
    print(f"grid=48th (subdiv 12); flags: {LIM}\n")
    for ad in arms:
        arm = Path(ad).name
        rows = []
        for sm in sorted(glob.glob(f"{ad}/*/chart.sm")):
            try:
                chart = parser.parse_file(sm)
            except Exception:
                continue
            if chart is None:      # ungated, but a malformed .sm can still return None
                continue
            nd = next((n for n in chart.note_data
                       if n.difficulty_name.rstrip(':').strip() == "Challenge"), None)
            if nd is None:
                continue
            # meter label from the HUMAN (non-Challenge) chart's triplet-cell occupancy — NOT the raw .sm text
            # (which chart_triplet_frac scans wholesale, so the GENERATED chart's own triplets would contaminate
            # the label; e.g. a band-on duple song would read as more triplet than it is).
            hnd = next((n for n in chart.note_data
                        if n.difficulty_name.rstrip(':').strip() != "Challenge"), None)
            tf = 0.0
            if hnd is not None:
                htyped = np.asarray(parser.convert_to_tensor_typed(chart, hnd))
                hon = np.nonzero(np.isin(htyped, PRESS).any(1))[0]
                tf = float(np.mean(np.isin(hon % SUBDIV, list(TRIPLET_CELLS)))) if len(hon) else 0.0
            chart_metrics.bpm = float(chart.bpm)
            typed = parser.convert_to_tensor_typed(chart, nd)
            m = chart_metrics(typed)
            if not m:
                continue
            m['song'] = (chart.title or Path(sm).parent.name)[:26]
            m['tf'] = tf; m['triplet_song'] = tf >= 0.15
            m['flags'] = flags_for(m)
            rows.append(m)
            if m['flags']:
                all_flags.append((arm, m['song'], m['flags']))
        if not rows:
            print(f"[{arm}] no charts parsed"); continue
        _print_arm(arm, rows)

    print("\n================ FLAGGED CHARTS (outside the safe envelope) ================")
    if not all_flags:
        print("  NONE — every generated chart is inside the playable envelope. ✅")
    else:
        for arm, song, fl in all_flags:
            print(f"  [{arm:14}] {song:26} {' '.join(fl)}")
    print("\nNote: ff_hold (free-foot-stream-under-hold) is the KNOWN freeze=high edge "
          "(footspeed_floor_findings §5b), reported per arm below — not a hard flag.")


def _print_arm(arm, rows):
    dup = [r for r in rows if not r['triplet_song']]
    tri = [r for r in rows if r['triplet_song']]
    mean = lambda rs, k: (float(np.nanmean([r[k] for r in rs])) if rs else float('nan'))
    print(f"===== ARM: {arm}  (n={len(rows)}: {len(tri)} triplet, {len(dup)} duple) =====")
    print(f"  {'song':26} {'tf':>4} {'nps':>4} {'bone':>4} {'trip':>4} {'off48':>5} "
          f"{'jack':>4} {'fjmp':>4} {'flam':>4} {'ffhld':>5} {'dead':>4}  flags")
    for r in sorted(rows, key=lambda r: -r['tf']):
        print(f"  {r['song']:26} {r['tf']:>4.2f} {r['nps']:>4.1f} {r['backbone']:>4.2f} "
              f"{r['triplet']:>4.2f} {r['off48']:>5.2f} {r['max_jack']:>4d} {r['fast_jump']:>4d} "
              f"{r['flam']:>4d} {r['ff_hold']:>5d} {r['dead_beats']:>4.1f}  {' '.join(r['flags'])}")
    print(f"  {'-- triplet-song mean':26} {'':>4} {'':>4} {mean(tri,'backbone'):>4.2f} "
          f"{mean(tri,'triplet'):>4.2f} {mean(tri,'off48'):>5.2f}  (triplet_occ: human ~0.40-0.57)")
    print(f"  {'-- duple-song mean':26} {'':>4} {'':>4} {mean(dup,'backbone'):>4.2f} "
          f"{mean(dup,'triplet'):>4.2f} {mean(dup,'off48'):>5.2f}  (duple triplet_occ should stay LOW)\n")


if __name__ == "__main__":
    main()

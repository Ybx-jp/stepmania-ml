"""data-layer-v2 phase-2b SIZING: how badly does the single-avg-BPM hop mis-time AUDIO on variable-BPM songs?

Mirrors the 2a displacement probe, for the audio half. Phase 2a fixed note DISPLACEMENT (chart-space, fixed-BPM
triplets). 2b would beat-synchronize the AUDIO (TimingMap.frame_times) for TEMPO-CHANGE songs. Before building a
beat-sync extractor + its own rebuild, size the payload: the constant-avg-hop places beat b's audio at
t_const = b*60/avg_bpm, but the true musical moment is t_true = TimingMap.beat_to_time(b). The offset-removed gap
is the audio-vs-note drift 2b removes. Uses the PARSER's real compute_average_bpm (an earlier hand-rolled avg had
a slope bug that inflated this ~16x — Rule 7).
"""
import glob
import numpy as np
import sys as _sys; from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repo root (src imports)
from src.data.stepmania_parser import StepManiaParser, TimingEvent
from src.data.timing import TimingMap

FILES = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
PARSER = StepManiaParser.for_inference()  # widened gates so variable-BPM songs parse; real compute_average_bpm

rows = []
n_parsed = 0
for f in FILES:
    try:
        chart = PARSER.parse_file(f)
    except Exception:
        continue
    if chart is None:
        continue
    n_parsed += 1
    bpms = [(e.beat, e.value) for e in chart.timing_events if e.event_type == 'bpm' and e.value > 0]
    if len(bpms) < 2 or any(v > 400 or v < 15 for _, v in bpms):  # variable-BPM, non-gimmick
        continue
    ab = PARSER.compute_average_bpm(chart.timing_events, chart.song_length_seconds)  # THE pipeline's avg
    tm = TimingMap([TimingEvent(b, v, 'bpm') for b, v in bpms])
    total_beats = tm.time_to_beat(chart.song_length_seconds)
    if total_beats < 8:
        continue
    beats = np.linspace(0, total_beats, 200)
    t_const = beats * 60.0 / ab
    t_true = tm.beat_to_time(beats)
    d = np.abs((t_true - t_const) - np.median(t_true - t_const))  # offset-removed (constant lag = #OFFSET, not drift)
    span = max(v for _, v in bpms) - min(v for _, v in bpms)
    rows.append((d.mean(), d.max(), span))

n = len(rows)
print(f"parsed charts: {n_parsed} | variable-BPM (>=2 non-gimmick #BPMS): {n} ({100*n/max(n_parsed,1):.1f}% of parsed)")
if n:
    mean_d = np.array([r[0] for r in rows]); max_d = np.array([r[1] for r in rows]); span = np.array([r[2] for r in rows])
    print(f"\n--- AUDIO-vs-note drift under the constant-avg-hop (ms, offset-removed) ---")
    print(f"per-song MEAN drift: median {np.median(mean_d)*1000:.1f} ms   90th {np.percentile(mean_d,90)*1000:.1f} ms   max {mean_d.max()*1000:.0f} ms")
    print(f"per-song MAX  drift: median {np.median(max_d)*1000:.1f} ms   90th {np.percentile(max_d,90)*1000:.1f} ms")
    print(f"BPM span (max-min): median {np.median(span):.1f}   90th {np.percentile(span,90):.1f}  (most variable-BPM = tiny micro-adjustments)")
    for thr in (0.023, 0.050, 0.100):
        k = (mean_d >= thr).sum()
        print(f"  songs with MEAN drift >= {thr*1000:.0f} ms: {k} ({100*k/n_parsed:.2f}% of ALL charts)")
    print(f"\n(23 ms @150BPM ~ one judgment window. 2a triplet fix for comparison: 50.5 -> 0.3 ms on ~7% of songs.)")
    print("DECISION: is the >=23ms-drift population big enough to justify a beat-sync extractor + its own rebuild?")

"""Unit tests for the v2 timing spine (src/data/timing.TimingMap)."""
import numpy as np
import pytest
from src.data.stepmania_parser import TimingEvent
from src.data.timing import TimingMap


def bpm(beat, v):
    return TimingEvent(beat=beat, value=v, event_type="bpm")


def stop(beat, dur):
    return TimingEvent(beat=beat, value=dur, event_type="stop")


# ----------------------------------------------------------------- single BPM
def test_single_bpm_linear_and_inverse():
    tm = TimingMap([bpm(0.0, 120.0)])  # 120 BPM -> 0.5 s/beat
    assert tm.beat_to_time(0.0) == pytest.approx(0.0)
    assert tm.beat_to_time(4.0) == pytest.approx(2.0)
    assert tm.beat_to_time(1.0) == pytest.approx(0.5)
    # inverse + round-trip
    assert tm.time_to_beat(2.0) == pytest.approx(4.0)
    for b in [0.0, 0.333, 1.75, 8.0, 63.9]:
        assert tm.time_to_beat(tm.beat_to_time(b)) == pytest.approx(b, abs=1e-9)


def test_vectorized():
    tm = TimingMap([bpm(0.0, 150.0)])
    beats = np.array([0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(tm.beat_to_time(beats), beats * 60.0 / 150.0)


# ----------------------------------------------------------------- variable BPM
def test_two_segment_hand_computed():
    # 120 BPM on [0,4), then 240 BPM. beat4 -> 2.0s; beat6 (2 beats @ 0.25s) -> 2.5s
    tm = TimingMap([bpm(0.0, 120.0), bpm(4.0, 240.0)])
    assert tm.beat_to_time(4.0) == pytest.approx(2.0)
    assert tm.beat_to_time(6.0) == pytest.approx(2.5)
    assert tm.beat_to_time(5.0) == pytest.approx(2.25)
    # round-trip across the tempo change
    for b in [0.5, 3.9, 4.0, 4.1, 10.0]:
        assert tm.time_to_beat(tm.beat_to_time(b)) == pytest.approx(b, abs=1e-9)


def test_matches_probe_reference_no_stops():
    """Drift-free equivalence to the validated probe core (probe_meter_equivariant_sb.bpm_map/time_to_beat)."""
    segs = [(0.0, 128.0), (32.0, 160.0), (96.5, 90.0), (140.0, 175.0)]
    events = [bpm(b, v) for b, v in segs]
    tm = TimingMap(events)

    # replicate the probe's exact algorithm as the reference
    beats = np.array([s[0] for s in segs]); bpms = np.array([s[1] for s in segs])
    times = np.zeros(len(segs))
    for i in range(1, len(segs)):
        times[i] = times[i - 1] + (beats[i] - beats[i - 1]) * 60.0 / bpms[i - 1]

    def probe_time_to_beat(t):
        seg = np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(beats) - 1)
        return beats[seg] + (t - times[seg]) * bpms[seg] / 60.0

    for t in np.linspace(0, times[-1] + 20, 50):
        assert tm.time_to_beat(t) == pytest.approx(probe_time_to_beat(t), abs=1e-9)


# ----------------------------------------------------------------- stops
def test_stop_inserts_dead_time():
    # 120 BPM (0.5 s/beat) with a 1.0 s stop at beat 4
    tm = TimingMap([bpm(0.0, 120.0), stop(4.0, 1.0)])
    assert tm.beat_to_time(3.999) == pytest.approx(1.9995, abs=1e-6)  # before the stop
    assert tm.beat_to_time(4.0) == pytest.approx(2.0)                 # note ON the stop beat = before pause
    assert tm.beat_to_time(5.0) == pytest.approx(2.0 + 1.0 + 0.5)     # after: +stop +1 beat
    assert tm.total_stop_seconds == pytest.approx(1.0)


def test_multiple_stops_accumulate():
    tm = TimingMap([bpm(0.0, 120.0), stop(2.0, 0.5), stop(6.0, 1.5)])
    # beat 8 = 4.0s tempo + both stops (0.5+1.5)
    assert tm.beat_to_time(8.0) == pytest.approx(4.0 + 2.0)
    # beat 4 is past only the first stop
    assert tm.beat_to_time(4.0) == pytest.approx(2.0 + 0.5)
    assert tm.total_stop_seconds == pytest.approx(2.0)


# ----------------------------------------------------------------- frame grid
def test_frame_grid_48th():
    tm = TimingMap([bpm(0.0, 120.0)])
    fb = tm.frame_beats(total_beats=4.0, subdiv=12)  # 48th grid
    assert fb[0] == 0.0
    assert fb[1] == pytest.approx(1.0 / 12)           # one 48th-note step
    assert np.all(np.diff(fb) > 0)                    # monotonic
    ft = tm.frame_times(total_beats=4.0, subdiv=12)
    assert np.all(np.diff(ft) > 0)
    assert ft[-1] == pytest.approx(tm.beat_to_time(fb[-1]))


def test_bpm_before_beat0_is_clamped():
    # a BPM event starting after 0 still yields a map spanning [0, inf)
    tm = TimingMap([bpm(2.0, 100.0)])
    assert tm.beat_to_time(0.0) == pytest.approx(0.0)
    assert tm.beat_to_time(2.0) == pytest.approx(2.0 * 60.0 / 100.0)


def test_no_bpm_raises():
    with pytest.raises(ValueError):
        TimingMap([stop(0.0, 1.0)])

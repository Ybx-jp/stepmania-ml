"""Unit tests for the audio start-offset detector (src/data/offset_detect.py).

The detector recovers the within-beat downbeat phase from audio given the BPM (the "deaf choreography" beat-anchor
fix). We test it on a synthetic click track at a known BPM + offset. The absolute error carries the fixed ~31 ms
onset-latency constant (calibrated for real music, not clicks), so the absolute check is loose; the RELATIVE check
(two offsets, same track) cancels the constant and pins the tracking tightly.
"""
import numpy as np
import pytest
import soundfile as sf

from src.data.offset_detect import detect_offset_audio, LATENCY_SEC

SR = 22050


def _click_track(tmp_path, bpm, offset, dur=20.0, sr=SR):
    """A click at each beat: offset + k*(60/bpm). Each click = a short decaying transient (mimics a percussive hit)."""
    n = int(dur * sr)
    y = np.zeros(n, dtype=np.float32)
    period = 60.0 / bpm
    click = (np.exp(-np.arange(int(0.01 * sr)) / (0.002 * sr))
             * np.sin(2 * np.pi * 2000 * np.arange(int(0.01 * sr)) / sr)).astype(np.float32)
    t = offset
    while t < dur:
        i = int(t * sr)
        if 0 <= i < n - len(click):
            y[i:i + len(click)] += click
        t += period
    path = tmp_path / f"click_{bpm}_{offset:.3f}.wav"
    sf.write(path, y, sr)
    return str(path)


def _circular_err(a, b, period):
    d = abs(a - b) % period
    return min(d, period - d)


@pytest.mark.parametrize("bpm,offset", [(128.0, 0.20), (150.0, 0.05), (100.0, 0.35)])
def test_detects_offset_within_tolerance(tmp_path, bpm, offset):
    path = _click_track(tmp_path, bpm, offset)
    r = detect_offset_audio(path, bpm)
    assert r is not None
    period = 60.0 / bpm
    # absolute error carries the latency constant (clicks aren't real music); allow it + slack
    err = _circular_err(r["offset_sec"], offset, period)
    assert err < LATENCY_SEC + 0.020, f"offset {offset} @ {bpm}bpm: err {err*1000:.1f}ms"


def test_relative_tracking_cancels_latency(tmp_path):
    """Two offsets on the same click design: the DIFFERENCE must track, independent of the latency constant."""
    bpm = 128.0
    period = 60.0 / bpm
    r1 = detect_offset_audio(_click_track(tmp_path, bpm, 0.05), bpm)
    r2 = detect_offset_audio(_click_track(tmp_path, bpm, 0.25), bpm)
    assert r1 is not None and r2 is not None
    delta = _circular_err(r2["offset_sec"] - r1["offset_sec"], 0.20, period)
    assert delta < 0.012, f"relative tracking off by {delta*1000:.1f}ms"


def test_returns_none_on_bad_bpm(tmp_path):
    path = _click_track(tmp_path, 128.0, 0.1)
    assert detect_offset_audio(path, 0) is None
    assert detect_offset_audio(path, None) is None


def test_offset_in_beat_range(tmp_path):
    bpm = 174.0
    r = detect_offset_audio(_click_track(tmp_path, bpm, 0.1), bpm)
    assert r is not None
    assert 0.0 <= r["offset_sec"] < 60.0 / bpm
    assert 0.0 <= r["confidence"] <= 1.0

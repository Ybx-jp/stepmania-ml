"""Beat<->time timing spine for data-layer v2 (the beat-synchronous re-grid).

Generalizes the parser's single constant hop (`hop = sr*60/(avg_bpm*4)`, one hop per song) into a
piecewise-linear beat<->time map built from the full `#BPMS` (+ `#STOPS`) events the parser already produces.
This is the SHARED primitive both halves of v2 call: variable-BPM re-gridding and the finer (48th) subdivision
are the same surgery — frames follow the real beat timeline instead of a constant-tempo hop.

Ported + hardened from the validated core in `probe_meter_equivariant_sb.py` (which caught the "first-BPM-only
drifts to noise" bug); adds `#STOPS` and the inverse `time_to_beat`. Beat 0 is anchored at t=0 (offset-free —
the global `#OFFSET` is a downstream audio-alignment shift, not part of the timing geometry). See
`notes/data_layer_v2_scope.md` phase 1.
"""
from typing import List, Sequence, Union
import numpy as np

Number = Union[float, np.ndarray]


class TimingMap:
    """Piecewise-linear beat<->time over a song's `#BPMS`/`#STOPS`.

    Within BPM segment i (starting at beat b_i, tempo v_i), time advances at 60/v_i seconds per beat. A stop at
    beat s inserts `dur` seconds of dead time that delays every beat STRICTLY AFTER s (a note ON beat s plays at
    the moment the stop begins, before the pause — StepMania semantics).
    """

    def __init__(self, timing_events: List, min_bpm: float = 1e-6):
        bpms = sorted((e.beat, e.value) for e in timing_events
                      if getattr(e, "event_type", None) == "bpm" and e.value > min_bpm)
        if not bpms:
            raise ValueError("TimingMap requires at least one positive #BPMS event")
        # a BPM before beat 0 is unusual; clamp the first segment to start at beat 0 so the map spans [0, inf)
        if bpms[0][0] > 0:
            bpms = [(0.0, bpms[0][1])] + bpms
        self._seg_beats = np.array([b for b, _ in bpms], dtype=np.float64)
        self._seg_bpms = np.array([v for _, v in bpms], dtype=np.float64)
        # cumulative PURE-TEMPO time at each segment start (stops added separately)
        self._seg_times = np.zeros(len(bpms), dtype=np.float64)
        for i in range(1, len(bpms)):
            db = self._seg_beats[i] - self._seg_beats[i - 1]
            self._seg_times[i] = self._seg_times[i - 1] + db * 60.0 / self._seg_bpms[i - 1]

        stops = sorted((e.beat, e.value) for e in timing_events
                       if getattr(e, "event_type", None) == "stop" and e.value > 0)
        self._stop_beats = np.array([b for b, _ in stops], dtype=np.float64)
        self._stop_durs = np.array([d for _, d in stops], dtype=np.float64)
        # cumulative stop time to add for a beat > stop_beat
        self._stop_cum = np.cumsum(self._stop_durs) if len(stops) else np.array([])

    # ------------------------------------------------------------------ beat -> time
    def beat_to_time(self, beat: Number) -> Number:
        beat = np.asarray(beat, dtype=np.float64)
        seg = np.searchsorted(self._seg_beats, beat, side="right") - 1
        seg = np.clip(seg, 0, len(self._seg_beats) - 1)
        t = self._seg_times[seg] + (beat - self._seg_beats[seg]) * 60.0 / self._seg_bpms[seg]
        t = t + self._stop_time_before(beat)
        return t if t.ndim else float(t)

    def _stop_time_before(self, beat: np.ndarray) -> np.ndarray:
        """Total stop seconds delaying a note at `beat` (stops STRICTLY before it)."""
        if not len(self._stop_beats):
            return np.zeros_like(np.asarray(beat, dtype=np.float64))
        k = np.searchsorted(self._stop_beats, beat, side="left")  # # stops with stop_beat < beat
        return np.where(k > 0, self._stop_cum[np.clip(k - 1, 0, None)], 0.0)

    # ------------------------------------------------------------------ time -> beat
    def time_to_beat(self, time: Number) -> Number:
        """Inverse. Inside a stop's dead interval, returns the stopped beat (time is ambiguous there)."""
        time = np.asarray(time, dtype=np.float64)
        # time at each segment start INCLUDING stops that occur before the segment start
        seg_time_full = self._seg_times + self._stop_time_before(self._seg_beats + 1e-12)
        seg = np.searchsorted(seg_time_full, time, side="right") - 1
        seg = np.clip(seg, 0, len(self._seg_beats) - 1)
        # subtract stop time accrued up to this segment, then invert the linear tempo
        base_t = seg_time_full[seg]
        beat = self._seg_beats[seg] + (time - base_t) * self._seg_bpms[seg] / 60.0
        # a beat can't fall before its segment start (clamp negatives from being inside a stop)
        beat = np.maximum(beat, self._seg_beats[seg])
        return beat if beat.ndim else float(beat)

    # ------------------------------------------------------------------ v2 frame grid
    def frame_beats(self, total_beats: float, subdiv: int = 12) -> np.ndarray:
        """The v2 grid in BEAT space: one frame every 1/subdiv beat (subdiv=12 -> 48th grid)."""
        n = int(np.floor(total_beats * subdiv)) + 1
        return np.arange(n, dtype=np.float64) / subdiv

    def frame_times(self, total_beats: float, subdiv: int = 12) -> np.ndarray:
        """The v2 grid in TIME space (seconds) — where to sample audio for each beat-synchronous frame."""
        return self.beat_to_time(self.frame_beats(total_beats, subdiv))

    @property
    def total_stop_seconds(self) -> float:
        return float(self._stop_durs.sum()) if len(self._stop_durs) else 0.0


def resample_frames(features: np.ndarray, src_times: np.ndarray, dst_times: np.ndarray) -> np.ndarray:
    """Linearly resample per-frame features (n_src, D) sampled at `src_times` onto `dst_times` (n_dst,) -> (n_dst, D).

    The v2 phase-2b primitive: extract audio features at a FINE CONSTANT hop (src_times = arange(n)*hop/sr), then
    map them onto the BEAT-SYNCHRONOUS frame times (TimingMap.frame_times) so a note's cell reads the audio at its
    TRUE musical moment on a variable-BPM song — instead of a single-avg-hop that drifts. Column-wise np.interp
    (clamps at the ends). Pairs with TimingMap.frame_times(subdiv=12).
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim == 1:
        features = features[:, None]
    out = np.empty((len(dst_times), features.shape[1]), dtype=np.float64)
    for d in range(features.shape[1]):
        out[:, d] = np.interp(dst_times, src_times, features[:, d])
    return out.astype(np.float32)

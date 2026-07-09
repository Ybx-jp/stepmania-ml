"""Audio-only START-OFFSET detector for bring-your-own-song generation (`scripts/generate.py`).

THE PROBLEM (notes/byo_offset_detection_findings.md): `generate.py` grids the audio from t=0, but the generator
choreographs on its `metric_phase` (within-beat position) channel, so if the audio's downbeats don't sit on the
frame grid the model chores to a PHANTOM grid = "deaf, no choreography" (user by-ear on the personal set). Fixing
it needs the sub-beat phase that aligns the audio to the beat grid at the (user-supplied) BPM — i.e. StepMania's
`#OFFSET`, recovered from audio.

THE METHOD (the ONE that beat the oracle; three "principled" rivals — DFT-phase, kick-band, kick-tiebreak — all
LOST on the `~/sm-personal` + training-pack oracle, so do NOT reintroduce them):
  full-band onset-strength envelope -> fold its energy modulo one beat into a phase histogram (a "pulse train"
  correlation) -> the peak phase is where the beats sit -> subtract a fixed ~31 ms latency (spectral flux peaks
  just AFTER the transient) -> the within-beat downbeat phase.

Validated (held-out, 24 personal songs): median ~7 ms, ~80% within 40 ms. The remaining ~20% SLIP a half/quarter
beat (a genuine beat-vs-offbeat ambiguity onset energy can't resolve) — detectable as a near-equal rival peak a
half-beat away, surfaced as `is_confident=False` so the caller can warn + suggest a manual `--offset`.

Returns the phase in SECONDS in [0, beat_period): the time of the first downbeat relative to audio t=0. The caller
(generate.py) uses it two ways that MUST move together (conventions verified in the parser + extractor):
  - EXTRACTION anchor: skip `offset_sec` of audio so frame 0 lands on a beat (stub chart offset = +offset_sec,
    which `AudioFeatureExtractor.extract_from_chart` honors via its `start_sample` skip-if-positive).
  - PLAYBACK sync: write `#OFFSET = -offset_sec` (StepMania convention: #OFFSET = -(time of beat 0)) so the
    untrimmed copied audio still plays in time.
"""
from __future__ import annotations

import numpy as np

# The onset-strength (spectral-flux) peak lags the true transient by a roughly fixed interval; this constant was
# calibrated on a train split and held out on the personal oracle (the plain full-band method + this single latency
# beat every "principled" alternative). Subtract it from the detected peak phase to recover the true beat position.
LATENCY_SEC = 0.031


def detect_offset_audio(audio_path: str, bpm: float, sr: int = 22050, hop_length: int = 128,
                        n_bins: int = 256):
    """Detect the within-beat downbeat phase (StepMania start offset) from audio, given the BPM.

    Args:
        audio_path: path to the song audio.
        bpm: the song BPM (user-supplied; estimation is unreliable — see [[byo-audio-bpm-footgun]]).
        sr / hop_length: onset-envelope extraction grid (a FINE fixed hop, independent of the chart grid, for
            sub-beat resolution — 128 @ 22050 ≈ 5.8 ms/frame).
        n_bins: phase-histogram resolution over one beat.

    Returns a dict, or None if the audio/BPM is unusable:
        offset_sec:   the first downbeat's time in [0, beat_period), seconds. Feed as the extraction skip; write
                      -offset_sec as the .sm #OFFSET.
        beat_period:  60/bpm, seconds.
        confidence:   1 - rival/peak of the half-beat-away histogram bin; ~1 = clean, ~0 = a beat/offbeat slip.
        is_confident: confidence above the slip threshold (else the caller should warn + suggest --offset).
    """
    if bpm is None or not np.isfinite(bpm) or bpm <= 0:
        return None
    import librosa
    y, _ = librosa.load(audio_path, sr=sr)
    if y.size == 0:
        return None
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    if env.size == 0 or not np.any(env > 0):
        return None

    beat_period = 60.0 / float(bpm)
    times = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop_length)
    # fold onset energy into a within-beat phase histogram (the pulse-train correlation, evaluated for every phase
    # at once): each frame's onset strength votes for its phase bin; the beats stack up at the true phase.
    bins = np.floor((times % beat_period) / beat_period * n_bins).astype(int) % n_bins
    hist = np.zeros(n_bins)
    np.add.at(hist, bins, env)
    # circular smoothing (a short boxcar via FFT) so a single noisy bin can't win the peak
    kw = max(3, n_bins // 64)
    kern = np.ones(kw) / kw
    hist = np.real(np.fft.ifft(np.fft.fft(hist) * np.fft.fft(kern, n_bins)))

    peak = int(np.argmax(hist))
    phase_peak = (peak + 0.5) / n_bins * beat_period          # bin-center time, seconds
    offset_sec = float((phase_peak - LATENCY_SEC) % beat_period)  # latency-correct + wrap into [0, beat_period)

    # a "slip" = two near-equal pulse peaks a HALF beat apart (the beat/offbeat ambiguity). Grade it.
    half = (peak + n_bins // 2) % n_bins
    peak_val = float(hist[peak])
    rival_val = float(hist[half])
    confidence = float(1.0 - rival_val / (peak_val + 1e-12))
    return {
        "offset_sec": offset_sec,
        "beat_period": beat_period,
        "confidence": confidence,
        "is_confident": bool(confidence > 0.15),   # <15% gap to the half-beat rival => likely a slip
    }

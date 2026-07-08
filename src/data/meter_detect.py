"""Meter / subdivision detection from audio — the duple-vs-triplet classifier.

Extracted verbatim from the validated prototype `probe_meter_equivariant_sb.py` (2026-07-04, meter arc)
so the decode pipeline can reuse the SAME statistic the probe validated (rather than re-deriving an
unvalidated variant on the highres features). See `notes/meter_4_4_assumption_scope.md` §A.

The signal: fold a FINE onset envelope (128-hop ≈5.8 ms) into within-beat PHASE using the chart's full
`#BPMS` map (drift-free beat<->time), histogram into 12 cells/beat = LCM(4,3), and take the DFT. DFT
magnitudes are rotation-invariant so the (unreliable) beat OFFSET drops out:
  duple energy = |H[2]|+|H[4]| (eighth, sixteenth combs) ; triple = |H[3]|+|H[6]| (triplet, sextuplet).
  triple_pref = (triple - duple) / (triple + duple)  in [-1, 1] ;  meter = triple if triple_pref > 0.

Validated: triple_pref vs the INDEPENDENT chart triplet_frac Spearman +0.47 (n=18 prototype) / +0.81 on a
triplet-enriched set. It cleanly separates the triplet seed songs (Equinox +0.22, My Christmas list +0.16,
subconsciousness +0.79) from duple songs (-0.37 … -0.68). Two harness bugs were caught building it (drifting
single-BPM phase; offset-sensitive fixed cells) — only the rotation-invariant DFT reading is the real test.

Primary entry point: `detect_triple_pref(sm_file)` -> reading dict (or None if no audio/BPM).
`chart_triplet_frac(txt)` is the chart-derived ground-truth fraction (needs a reference chart).
"""
from __future__ import annotations
import os
import re
from fractions import Fraction
import numpy as np

SR, HOP = 22050, 128


def _read(f: str) -> str:
    try:
        return open(f, encoding='utf-8', errors='ignore').read()
    except Exception:
        return ''


def chart_triplet_frac(txt: str) -> float:
    """Fraction of a chart's notes that land on TRIPLET beat-positions (denominator divisible by 3).
    The reliable ground-truth meter label — needs a parsed .sm (not available for a brand-new song)."""
    best = 0.0
    for m in re.finditer(r'#NOTES:(.*?);', txt, re.S):
        parts = m.group(1).split(':')
        if len(parts) < 6 or 'dance-single' not in parts[0]:
            continue
        tri = binv = 0
        for meas in parts[5].split(','):
            lines = [ln for ln in meas.splitlines() if ln.strip() and not ln.strip().startswith('//')]
            L = len(lines)
            for i, ln in enumerate(lines):
                if L and re.search(r'[124]', ln):
                    d = Fraction(i, L).denominator
                    tri += (d % 3 == 0)
                    binv += (d % 3 != 0)
        if tri + binv > 50:
            best = max(best, tri / (tri + binv))
    return best


def bpm_map(txt: str):
    """Full #BPMS map -> (start_beats, start_times, bpms) for EXACT drift-free beat<->time.
    (Using only the first BPM over a multi-minute song drifts the phase into noise — the harness bug.)"""
    m = re.search(r'#BPMS:([^;]*);', txt)
    if not m or not m.group(1).strip():
        return None
    segs = []
    for s in m.group(1).split(','):
        try:
            b, v = s.split('=')
            segs.append((float(b), float(v)))
        except Exception:
            pass
    segs = [s for s in sorted(segs) if s[1] > 0]
    if not segs:
        return None
    beats = np.array([s[0] for s in segs])
    bpms = np.array([s[1] for s in segs])
    times = np.zeros(len(segs))
    for i in range(1, len(segs)):
        times[i] = times[i - 1] + (beats[i] - beats[i - 1]) * 60.0 / bpms[i - 1]
    return beats, times, bpms


def _time_to_beat(t, bm):
    beats, times, bpms = bm
    seg = np.searchsorted(times, t, side='right') - 1
    seg = np.clip(seg, 0, len(beats) - 1)
    return beats[seg] + (t - times[seg]) * bpms[seg] / 60.0


def _music_path(sm_file: str, txt: str):
    m = re.search(r'#MUSIC:([^;]*);', txt)
    if not m or not m.group(1).strip():
        return None
    p = os.path.join(os.path.dirname(sm_file), m.group(1).strip())
    return p if os.path.exists(p) else None


def phase_hist(sm_file: str):
    """12-cell within-beat onset-mass histogram from the fine (128-hop) onset envelope. None if no audio/BPM."""
    import librosa
    txt = _read(sm_file)
    bm = bpm_map(txt)
    mus = _music_path(sm_file, txt)
    if bm is None or not mus:
        return None
    y, sr = librosa.load(mus, sr=SR, mono=True)
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    t = np.arange(len(env)) * HOP / sr                      # frame times (s)
    beat = _time_to_beat(t, bm)                             # EXACT beat via full BPM map (drift-free)
    phase = np.mod(beat, 1.0)                               # within-beat phase [0,1)
    cell = np.mod(np.round(phase * 12).astype(int), 12)     # nearest of 12 cells
    h = np.zeros(12)
    for c, e in zip(cell, env):
        h[c] += e
    return h


def strong_readings(h: np.ndarray) -> dict:
    """ROTATION-INVARIANT meter detection via the DFT of the within-beat onset histogram.
    Bin k of a 12-pt DFT = k cycles/beat: duple structure at {2 (eighth), 4 (16th)}, triple at {3, 6}.
    Magnitudes ignore absolute phase, so #OFFSET is irrelevant. The downbeat is recovered from bin-1's phase."""
    H = np.fft.rfft(h)                                       # bins 0..6
    mag = np.abs(H)
    duple = mag[2] + mag[4]
    triple = mag[3] + mag[6]
    tp = (triple - duple) / (triple + duple + 1e-9)         # triple-preference in [-1, 1]
    meter = 'triple' if tp > 0 else 'duple'
    shift = int(np.round(np.angle(H[1]) / (2 * np.pi) * 12)) % 12
    ha = np.roll(h, -shift)
    sb_duple = (ha[0] + ha[6]) / (ha[0] + ha[3] + ha[6] + ha[9] + 1e-9)   # ≡ current SB (fine env, duple grid)
    sb_triple = ha[0] / (ha[0] + ha[4] + ha[8] + 1e-9)
    sb_eq = sb_triple if meter == 'triple' else sb_duple
    return dict(meter=meter, sb_duple=sb_duple, sb_triple=sb_triple, sb_eq=sb_eq, triple_pref=tp)


def detect_triple_pref(sm_file: str) -> dict | None:
    """Audio-derived meter reading for one song. Returns the `strong_readings` dict
    (keys: meter, triple_pref, sb_duple, sb_triple, sb_eq) or None if audio/BPM is unavailable.

    Deployable to a brand-new song (needs only the audio + `#BPMS`/`#OFFSET`, both required for the governor
    anyway) — the classifier for the per-song duple/triplet b_trip switch. `triple_pref > 0` => triplet-feel.
    """
    h = phase_hist(sm_file)
    if h is None:
        return None
    return strong_readings(h)


def detect_triple_pref_audio(audio_path: str, bpm: float) -> dict | None:
    """The BYO-audio sibling of `detect_triple_pref` for `scripts/generate.py`: read the meter directly from an
    audio file + a single constant BPM (no `.sm` to source `#BPMS`/`#OFFSET` from). Shares `strong_readings`
    (the rotation-invariant DFT) so the classifier is identical to the export path; only the beat mapping differs
    (constant BPM here vs the full `bpm_map` in `phase_hist`). Returns the `strong_readings` dict or None on failure.
    """
    import librosa
    if not bpm or bpm <= 0:
        return None
    try:
        y, sr = librosa.load(audio_path, sr=SR, mono=True)
    except Exception:
        return None
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    t = np.arange(len(env)) * HOP / sr          # frame times (s)
    beat = t * bpm / 60.0                        # constant-BPM beat (BYO-audio has no #BPMS drift map)
    cell = np.mod(np.round(np.mod(beat, 1.0) * 12).astype(int), 12)  # nearest of 12 within-beat cells
    h = np.zeros(12)
    for c, e in zip(cell, env):
        h[c] += e
    return strong_readings(h)

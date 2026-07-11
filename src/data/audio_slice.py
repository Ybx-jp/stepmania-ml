"""Trim a local audio file to a time range and re-encode to Vorbis .ogg.

Used by --trim-audio in scripts/generate.py and scripts/pull_audio.py. Trimming
happens on the LOCAL file (whether given directly or pulled from a URL) BEFORE any
charting: the trimmed .ogg becomes the audio the generator runs on, so bpm, duration,
offset detection, features and generation all see only the clip. Because offset
detection runs on that clip, cutting the intro anchors #OFFSET to the clip's first
beat. BPM is unchanged (trimming doesn't alter tempo).

Timestamp forms accepted: 'SS[.sss]', 'M:SS[.sss]', 'H:MM:SS[.sss]' — e.g.
'4', '0:04', '2:14.5', '1:02:03'. Range spec: 'START' (start..end-of-file) or
'START,END' (the interior). An empty field takes its default ('' or ',END' => 0;
'START,' => end-of-file).
"""
import hashlib
import shutil
import subprocess
from pathlib import Path

# Cache trimmed clips so charting one range at several difficulties re-encodes once.
CACHE = Path.home() / ".cache" / "stepmania-chart-gen" / "trimmed"


def parse_timestamp(s: str) -> float:
    """'M:SS'/'H:MM:SS'/'SS' -> seconds (float). Raises ValueError on bad input."""
    s = s.strip()
    if not s:
        raise ValueError("empty timestamp")
    parts = s.split(":")
    if len(parts) > 3:
        raise ValueError(f"bad timestamp {s!r} (too many ':' fields)")
    try:
        fields = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"bad timestamp {s!r} (non-numeric field)")
    if any(f < 0 for f in fields):
        raise ValueError(f"bad timestamp {s!r} (negative field)")
    secs = 0.0
    for f in fields:          # fold: SS, M:SS, H:MM:SS all handled by acc*60 + field
        secs = secs * 60 + f
    return secs


def parse_trim_spec(spec: str):
    """'START' or 'START,END' -> (start_s, end_s_or_None). Validates end > start."""
    parts = spec.split(",")
    if len(parts) > 2:
        raise ValueError(f"bad --trim-audio {spec!r} (expected START or START,END)")
    start = parse_timestamp(parts[0]) if parts[0].strip() else 0.0
    end = None
    if len(parts) == 2 and parts[1].strip():
        end = parse_timestamp(parts[1])
    if end is not None and end <= start:
        raise ValueError(f"--trim-audio end ({end:g}s) must be after start ({start:g}s)")
    return start, end


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "'ffmpeg' not found on PATH — install it with: "
            "conda install -c conda-forge ffmpeg   (or: sudo apt install ffmpeg)")


def _probe_duration(path) -> float:
    """Audio length in seconds via ffprobe, or None if it can't be read."""
    if shutil.which("ffprobe") is None:
        return None
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def _resolve_range(src, start: float, end):
    """Validate/clamp (start, end) against the source duration when knowable."""
    dur = _probe_duration(src)
    if dur is not None:
        if start >= dur:
            raise ValueError(f"--trim-audio start ({start:g}s) is past the audio length ({dur:g}s)")
        if end is not None and end > dur:
            end = dur           # clamp a too-long end down to the real end rather than erroring
    return start, end


def _run_trim(src, dst, start: float, end, quality: int) -> None:
    # Output-side -ss (after -i) forces an exact decode from 0 for a sample-accurate cut;
    # -t is the output DURATION, so [start, start+dur] == [start, end]. -vn drops any video.
    cmd = ["ffmpeg", "-nostdin", "-y", "-i", str(src), "-ss", f"{start:.6f}"]
    if end is not None:
        cmd += ["-t", f"{end - start:.6f}"]
    cmd += ["-vn", "-c:a", "libvorbis", "-q:a", str(quality), str(dst)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("ffmpeg trim failed:\n" + (result.stderr.strip() or result.stdout.strip()))


def trim_to(src, dst, start: float, end, quality: int = 6) -> Path:
    """Trim `src` to [start, end] (end=None => end-of-file) writing Vorbis .ogg at `dst`."""
    _require_ffmpeg()
    start, end = _resolve_range(src, start, end)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run_trim(src, dst, start, end, quality)
    if not dst.is_file():
        raise RuntimeError(f"ffmpeg finished but expected file is missing: {dst}")
    return dst


def _tag(t: float) -> str:
    """Compact seconds tag for cache filenames (2.5 -> '2p5', 4.0 -> '4')."""
    return (f"{t:.3f}".rstrip("0").rstrip(".") or "0").replace(".", "p")


def ensure_trimmed(src, start: float, end, quality: int = 6, cache_dir=CACHE) -> Path:
    """Trim `src` to a per-(source,range) cache path, reusing it if already present."""
    src = Path(src)
    h = hashlib.sha1(str(src.resolve()).encode()).hexdigest()[:8]
    end_tag = _tag(end) if end is not None else "end"
    out = Path(cache_dir) / f"{src.stem}__{_tag(start)}-{end_tag}__{h}.ogg"
    if out.is_file():
        return out
    return trim_to(src, out, start, end, quality)

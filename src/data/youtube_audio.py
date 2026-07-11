"""Download audio from a YouTube (or any yt-dlp-supported) URL as a Vorbis .ogg.

StepMania and this project's feature pipeline want Vorbis-in-Ogg audio; yt-dlp
fetches the best audio stream and ffmpeg transcodes it to Vorbis. Both binaries
must be on PATH (see README). A JavaScript runtime (deno) lets yt-dlp reach
YouTube's full-quality audio formats — without one it falls back to a lower-quality
client. None of that is required to *import* this module; it's checked at call time.

Used by:
  - scripts/pull_audio.py  (the standalone CLI)
  - scripts/generate.py    (so --audio accepts a URL directly)
"""
import re
import shutil
import subprocess
from pathlib import Path

# Cache pulled audio here so charting one song at several difficulties (or a whole
# pack via batch_generate.py) re-downloads it only ONCE — keyed by the video id.
DEFAULT_CACHE = Path.home() / ".cache" / "stepmania-chart-gen" / "youtube"

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def is_url(s: str) -> bool:
    """True if `s` looks like an http(s) URL rather than a local file path."""
    return bool(_URL_RE.match(s.strip()))


def _sanitize(name: str) -> str:
    """Strip filesystem-hostile characters (same set generate.py uses for song dirs)."""
    return re.sub(r'[<>:"/\\|?*]', "_", name).strip() or "audio"


def _require_tools() -> None:
    """Fail with an actionable message if yt-dlp / ffmpeg are not installed."""
    for name, hint in (
        ("yt-dlp", "pip install yt-dlp"),
        ("ffmpeg", "conda install -c conda-forge ffmpeg   (or: sudo apt install ffmpeg)"),
    ):
        if shutil.which(name) is None:
            raise RuntimeError(f"'{name}' not found on PATH — install it with: {hint}")


def _capture(cmd) -> str:
    """Run a yt-dlp query whose stdout we need (metadata), raising on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp exited {result.returncode}:\n"
            + (result.stderr.strip() or result.stdout.strip()))
    return result.stdout


def _extract_meta(url: str):
    """Return (video_id, title) via a metadata-only call — no download."""
    out = _capture([
        "yt-dlp", "--skip-download", "--no-playlist",
        "--print", "%(id)s", "--print", "%(title)s", url,
    ])
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"could not read video metadata for {url}")
    vid = lines[0].strip()
    title = lines[1].strip() if len(lines) > 1 else vid
    return vid, title


def _download(url: str, out_template: str, quality: int) -> None:
    """Download+transcode to Vorbis .ogg. Progress streams to the terminal."""
    cmd = [
        "yt-dlp",
        "-x",                          # audio only
        "--audio-format", "vorbis",    # -> Vorbis codec in an .ogg container
        "--audio-quality", str(quality),
        "--no-playlist",               # a single URL even if it points into a playlist
        "-o", out_template,
        url,
    ]
    result = subprocess.run(cmd)       # inherit stdio so the progress bar shows
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp exited {result.returncode} (see output above)")


def download_audio(url: str, output: str = None, outdir: str = ".", quality: int = 6) -> Path:
    """CLI-style pull: write <title>.ogg into `outdir`, or an exact `output` filename.

    Returns the path to the produced .ogg. Because we choose the literal output
    filename (rather than a %(title)s template), the returned path is exact.
    """
    _require_tools()
    if output:
        out_path = Path(output)
        if out_path.suffix.lower() != ".ogg":
            out_path = out_path.with_suffix(".ogg")
    else:
        _, title = _extract_meta(url)
        out_path = Path(outdir) / f"{_sanitize(title)}.ogg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # yt-dlp appends the audio ext, so hand it a template without the .ogg suffix.
    _download(url, str(out_path.with_suffix("")) + ".%(ext)s", quality)
    if not out_path.is_file():
        raise RuntimeError(f"download finished but expected file is missing: {out_path}")
    return out_path


def fetch_to_cache(url: str, cache_dir=DEFAULT_CACHE, quality: int = 6):
    """Pull audio to a per-video-id cache, reusing it if already present.

    Returns (ogg_path, video_title). This is the entry point generate.py uses so
    repeated runs on the same URL (e.g. one song at several difficulties) hit the
    network at most once.
    """
    _require_tools()
    vid, title = _extract_meta(url)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{vid}.ogg"
    if out_path.is_file():
        return out_path, title
    _download(url, str(cache_dir / f"{vid}.%(ext)s"), quality)
    if not out_path.is_file():
        raise RuntimeError(f"download finished but expected file is missing: {out_path}")
    return out_path, title

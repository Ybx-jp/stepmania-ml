#!/usr/bin/env python3
"""Pull the audio track from a YouTube video and save it as a Vorbis .ogg file.

StepMania loads Vorbis-in-Ogg audio, so this is the format the rest of the
chart-generation pipeline expects. `scripts/generate.py --audio <URL>` can also
take a URL directly (it reuses this module); this CLI is for pulling audio on its own.

Requires yt-dlp + ffmpeg on PATH; install deno (`conda install -c conda-forge deno`)
so yt-dlp can reach YouTube's full-quality audio formats.

Examples:
  python scripts/pull_audio.py "https://www.youtube.com/watch?v=XXXX"
  python scripts/pull_audio.py URL -o song.ogg
  python scripts/pull_audio.py URL --outdir data/audio --quality 8
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.youtube_audio import download_audio
from src.data.audio_slice import parse_trim_spec, trim_to


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a YouTube video's audio as a Vorbis .ogg file.",
    )
    parser.add_argument("url", help="YouTube video URL (or any yt-dlp-supported URL)")
    parser.add_argument(
        "-o", "--output",
        help="Exact output filename (e.g. song.ogg). Overrides --outdir/title naming.",
    )
    parser.add_argument(
        "--outdir", default=".",
        help="Directory to write into when -o is not given (default: current dir).",
    )
    parser.add_argument(
        "-q", "--quality", type=int, default=6, choices=range(0, 11), metavar="0-10",
        help="Vorbis quality; higher = better/larger (default: 6, ~192kbps).",
    )
    parser.add_argument(
        "--trim-audio", "--trim_audio", dest="trim_audio", default=None, metavar="START[,END]",
        help="keep only a time range: 'START' (to end) or 'START,END', as SS / M:SS / H:MM:SS "
             "(e.g. '0:04' or '0:04,2:14').",
    )
    args = parser.parse_args()

    try:
        out = download_audio(args.url, output=args.output, outdir=args.outdir, quality=args.quality)
        if args.trim_audio:
            start, end = parse_trim_spec(args.trim_audio)
            # ffmpeg can't read and write the same file, so trim to a sibling temp then swap it in.
            tmp = out.with_name(out.stem + ".trimtmp.ogg")
            trim_to(out, tmp, start, end, quality=args.quality)
            tmp.replace(out)
    except (RuntimeError, ValueError) as e:
        sys.exit(f"error: {e}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()

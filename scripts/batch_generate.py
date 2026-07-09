#!/usr/bin/env python3
"""Batch-generate charts over a folder of songs that each ship a REFERENCE chart (audio + .sm/.ssc).

Purpose: regenerate a whole pack with the deployed generator, pulling each song's TRUE timing (BPM + #OFFSET)
straight from its hand-authored reference chart. This dogfoods the shipped single-song CLI — it invokes
`scripts/generate.py` once per song (same interpreter) rather than re-implementing the decode, so the batch can
never drift from what the user actually plays.

Why reference timing instead of the audio auto-detector: for a pack that HAS reference charts, the reference
#OFFSET is the ground truth (it's the very oracle the detector was validated against), so passing `--bpm` + `--offset`
sidesteps the detector's ~20% half-beat-slip entirely. Pass `--use_detector` to instead let generate.py auto-detect
the offset from audio (to by-ear-test the detector itself).

Skips (unsupported by the single-BPM tool): variable-BPM songs (>1 distinct #BPMS) and songs with #STOPS.

Usage:
  python scripts/batch_generate.py --src ~/sm-personal --out ~/sm-generated/personal_regen --difficulty Hard
  python scripts/batch_generate.py --src ~/sm-personal --out /tmp/x --dry_run          # just show the parse plan
  python scripts/batch_generate.py --src ~/sm-personal --out /tmp/x --limit 2          # first 2 songs (smoke)
"""
import argparse
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

AUDIO_EXT = (".ogg", ".mp3", ".wav")


def _tag(text, tag):
    """First `#TAG:value;` (case-insensitive), value stripped; None if absent/empty."""
    m = re.search(rf"#{tag}:([^;]*);", text, re.IGNORECASE)
    return m.group(1).strip() if m and m.group(1).strip() else None


def parse_reference(chart_path):
    """Read title / music / bpm(s) / offset / stops from a .sm/.ssc. Returns a dict + a skip reason (or None)."""
    text = Path(chart_path).read_text(encoding="utf-8", errors="ignore")
    title = _tag(text, "TITLE")
    music = _tag(text, "MUSIC")
    offset = _tag(text, "OFFSET")
    bpms_raw = _tag(text, "BPMS")
    stops_raw = _tag(text, "STOPS")
    # #BPMS = "beat=bpm,beat=bpm,..."; collect the distinct bpm values
    bpm_vals = []
    if bpms_raw:
        for pair in bpms_raw.split(","):
            if "=" in pair:
                try:
                    bpm_vals.append(round(float(pair.split("=")[1]), 2))
                except ValueError:
                    pass
    skip = None
    if not bpm_vals:
        skip = "no #BPMS"
    elif len(set(bpm_vals)) > 1:
        skip = f"variable BPM ({sorted(set(bpm_vals))})"
    elif stops_raw:
        skip = "has #STOPS"
    return {
        "title": title, "music": music,
        "bpm": bpm_vals[0] if bpm_vals else None,
        "offset": float(offset) if offset is not None else None,
    }, skip


def find_songs(src):
    """Map each song FOLDER (containing a chart) -> its reference chart path (prefer .sm over .ssc)."""
    charts = glob.glob(os.path.join(src, "**", "*.sm"), recursive=True) + \
             glob.glob(os.path.join(src, "**", "*.ssc"), recursive=True)
    by_folder = {}
    for c in sorted(charts):
        folder = os.path.dirname(c)
        # prefer .sm; only overwrite an .ssc entry with a .sm
        if folder not in by_folder or (c.lower().endswith(".sm") and by_folder[folder].lower().endswith(".ssc")):
            by_folder[folder] = c
    return by_folder


def find_audio(folder, music):
    if music:
        p = os.path.join(folder, os.path.basename(music))
        if os.path.isfile(p):
            return p
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(AUDIO_EXT):
            return os.path.join(folder, f)
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=os.path.expanduser("~/sm-personal"),
                   help="root folder of songs (each song folder has audio + a .sm/.ssc reference)")
    p.add_argument("--out", default=os.path.expanduser("~/sm-generated/personal_regen"),
                   help="GROUP output folder (each song is nested inside)")
    p.add_argument("--difficulty", default="Hard", choices=["Beginner", "Easy", "Medium", "Hard"])
    p.add_argument("--features", default="highres_v2", help="passthrough to generate.py (default the v2 canonical)")
    p.add_argument("--use_detector", action="store_true",
                   help="let generate.py AUTO-detect the offset from audio instead of using the reference #OFFSET "
                        "(to by-ear-test the detector). Default: pass the reference #OFFSET (authoritative).")
    p.add_argument("--overwrite", action="store_true", help="regenerate even if the song's out folder already exists")
    p.add_argument("--limit", type=int, default=None, help="only the first N eligible songs (smoke test)")
    p.add_argument("--dry_run", action="store_true", help="print the plan (parsed timing + skip decisions), generate nothing")
    args = p.parse_args()

    gen_py = str(Path(__file__).resolve().parent / "generate.py")
    songs = find_songs(args.src)
    if not songs:
        raise SystemExit(f"no .sm/.ssc charts found under {args.src}")

    print(f"{'song':<38} {'bpm':>7} {'#OFFSET':>8} {'status':>10}  note")
    print("-" * 88)
    done = skipped = failed = 0
    for folder, chart in sorted(songs.items()):
        meta, skip = parse_reference(chart)
        name = (meta["title"] or os.path.basename(folder))[:37]
        if skip:
            print(f"{name:<38} {'-':>7} {'-':>8} {'SKIP':>10}  {skip}")
            skipped += 1
            continue
        audio = find_audio(folder, meta["music"])
        if audio is None:
            print(f"{name:<38} {meta['bpm']:>7.1f} {'-':>8} {'SKIP':>10}  no audio in folder")
            skipped += 1
            continue
        off = meta["offset"] if meta["offset"] is not None else 0.0
        song_out = Path(args.out) / (meta["title"] or os.path.basename(folder))
        if song_out.exists() and not args.overwrite and not args.dry_run:
            print(f"{name:<38} {meta['bpm']:>7.1f} {off:>8.3f} {'exists':>10}  (--overwrite to redo)")
            done += 1
            continue

        cmd = [sys.executable, gen_py, "--audio", audio, "--difficulty", args.difficulty,
               "--features", args.features, "--bpm", str(meta["bpm"]), "--out", args.out,
               "--inherit_from", chart]
        if meta["title"]:
            cmd += ["--title", meta["title"]]
        if not args.use_detector:
            cmd += ["--offset", str(off)]     # authoritative reference offset (skip the detector's slip risk)

        if args.limit is not None and (done + failed) >= args.limit:
            break
        if args.dry_run:
            print(f"{name:<38} {meta['bpm']:>7.1f} {off:>8.3f} {'PLAN':>10}  "
                  f"{'auto-offset' if args.use_detector else 'ref-offset'} | {os.path.basename(audio)}")
            continue

        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            note = next((ln for ln in reversed(r.stdout.splitlines()) if ln.startswith("wrote ")), "ok")
            print(f"{name:<38} {meta['bpm']:>7.1f} {off:>8.3f} {'OK':>10}  {note[6:80] if note.startswith('wrote') else note}")
            done += 1
        else:
            tail = (r.stderr.strip().splitlines() or ["?"])[-1][:60]
            print(f"{name:<38} {meta['bpm']:>7.1f} {off:>8.3f} {'FAIL':>10}  {tail}")
            failed += 1

    print("-" * 88)
    print(f"{'DRY RUN — ' if args.dry_run else ''}generated/kept={done}  skipped={skipped}  failed={failed}  "
          f"-> {args.out}")


if __name__ == "__main__":
    main()

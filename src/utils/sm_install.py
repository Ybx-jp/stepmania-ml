"""
Install exported sample folders into a StepMania-readable songs directory.

The export scripts write playable folders under `outputs/...`. StepMania, however, only reads
song folders laid out as `<songs_root>/<group>/<song>/*.sm` (groups required, no sub-groups —
see any StepMania `Songs/instructions.txt`). Rather than `sudo cp -r` into the root-owned
install `Songs/` every time, we copy into a user-owned dir that StepMania also scans via the
`AdditionalSongFolders` preference. No sudo, no system files touched.

A "group" is auto-detected as any directory that directly contains at least one *song folder*
(a folder holding a `.sm`/`.ssc`). This handles both export layouts:
  - export_typed_samples.py: `out_dir/<NN_song>/chart.sm`        -> out_dir itself is the group
  - export_reranked.py:      `out_dir/{best,first}/<NN_song>/...` -> best/ and first/ are groups

Group names are prefixed with the out_dir's basename so different experiments don't collide in
the song wheel (e.g. `reranked_hard_best`, `reranked_hard_first`, `typed_samples`).
"""

import os
import shutil
from pathlib import Path

# Resolution order for the destination: explicit arg > $SM_SONGS_DIR > the AdditionalSongFolders
# default we set up. Keep in sync with Preferences.ini AdditionalSongFolders.
DEFAULT_SONGS_DIR = os.path.expanduser(os.environ.get("SM_SONGS_DIR", "~/sm-generated"))


def _has_simfile(d: Path) -> bool:
    return d.is_dir() and any(
        f.suffix.lower() in (".sm", ".ssc") for f in d.iterdir() if f.is_file()
    )


def _is_group(d: Path) -> bool:
    """A group directly contains at least one song folder (a folder with a simfile)."""
    return d.is_dir() and any(_has_simfile(c) for c in d.iterdir() if c.is_dir())


def _group_name(out_dir: Path, group: Path) -> str:
    if group == out_dir:
        return out_dir.name
    rel = group.relative_to(out_dir).as_posix().replace("/", "_")
    return f"{out_dir.name}_{rel}"


def install_to_stepmania(out_dir, songs_dir: str = None) -> list:
    """
    Copy every group found under `out_dir` into `songs_dir` (replacing any same-named group).
    Returns the list of installed destination paths. Raises if no groups are found.
    """
    out_dir = Path(out_dir).resolve()
    songs_root = Path(os.path.expanduser(songs_dir or DEFAULT_SONGS_DIR))
    songs_root.mkdir(parents=True, exist_ok=True)

    # Candidate dirs: out_dir plus all descendant dirs; keep those that are groups.
    candidates = [out_dir] + [p for p in sorted(out_dir.rglob("*")) if p.is_dir()]
    groups = [d for d in candidates if _is_group(d)]
    if not groups:
        raise RuntimeError(f"No StepMania groups (dir-of-song-folders) found under {out_dir}")

    installed = []
    for group in groups:
        name = _group_name(out_dir, group)
        dest = songs_root / name
        if dest.exists():
            shutil.rmtree(dest)
        # Copy only the song folders (skip stray files like logs at the group level).
        dest.mkdir(parents=True)
        for song in sorted(c for c in group.iterdir() if _has_simfile(c)):
            shutil.copytree(song, dest / song.name)
        installed.append(dest)

    return installed

"""Read a source `.sm`/`.ssc` chart's PRESENTATION-header tags (and the media files they reference) so a
GENERATED chart can inherit the original's look — its banner, background, `#BGCHANGES` music video, CD title,
subtitle, sample-preview window, etc.

Why a standalone reader (not `StepManiaParser`): the parser only carries title/artist/genre/credit (it exists to
extract note data + timing for the model), so it drops every presentation tag. This module greps the raw tags
purely for pass-through into `sm_writer`.

What is DELIBERATELY NOT inheritable: `#BPMS` and `#STOPS`. The generator places notes on its OWN constant-BPM
grid with no knowledge of source tempo changes/stops; carrying the source's timing tags would slide the generated
notes off the audio at every change. Timing is the generator's to own. Only cosmetic/metadata tags are inherited
(background video timing is in beats, so it can drift if the BPM differs — but it's purely visual, never gameplay).
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Dict, List

# Presentation/metadata tags safe to copy from a source chart onto a generated one, in StepMania's header order
# (NotesWriterSM.cpp). #TITLE/#ARTIST/#MUSIC/#OFFSET are set explicitly by the caller; #BPMS/#STOPS are excluded
# on purpose (see module docstring).
INHERITABLE_TAGS: List[str] = [
    "SUBTITLE", "TITLETRANSLIT", "SUBTITLETRANSLIT", "ARTISTTRANSLIT",
    "GENRE", "CREDIT", "BANNER", "BACKGROUND", "CDTITLE",
    "SAMPLESTART", "SAMPLELENGTH", "DISPLAYBPM",
    "BGCHANGES", "FGCHANGES",
]

# tags whose value names a media file (to copy alongside the chart so the reference resolves in the new folder)
_MEDIA_FILE_TAGS = ("BANNER", "BACKGROUND", "CDTITLE")
_MEDIA_CHANGE_TAGS = ("BGCHANGES", "FGCHANGES")  # comma-separated "beat=file=rate=...=effect" entries; file = field 1


def read_header_tags(path: str, tags: List[str] = INHERITABLE_TAGS) -> Dict[str, str]:
    """Extract the given `#TAG:value;` header values from a .sm/.ssc file. Empty tags are skipped. Returns a
    {TAG: value} dict (values verbatim, semicolon-terminated content stripped)."""
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    txt = re.sub(r"//.*", "", txt)  # strip // line comments (SM allows them mid-file)
    out: Dict[str, str] = {}
    for tag in tags:
        m = re.search(rf"#{tag}\s*:(.*?);", txt, re.DOTALL)
        if m:
            val = m.group(1).strip()
            if val:
                out[tag] = val
    return out


def referenced_media(tags: Dict[str, str]) -> List[str]:
    """Filenames referenced by presentation tags (banner/background/cdtitle + the BG/FG change videos), de-duped
    in first-seen order, so the caller can copy them next to the generated chart."""
    files: List[str] = []
    for t in _MEDIA_FILE_TAGS:
        if tags.get(t):
            files.append(tags[t])
    for t in _MEDIA_CHANGE_TAGS:
        v = tags.get(t)
        if not v:
            continue
        for change in v.split(","):
            fields = change.split("=")
            if len(fields) >= 2 and fields[1].strip():
                files.append(fields[1].strip())
    seen, out = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def copy_media(tags: Dict[str, str], src_dir: str, dst_dir: str) -> List[str]:
    """Copy every media file referenced by `tags` from `src_dir` into `dst_dir` (only those that actually exist).
    Returns the list of filenames copied (so an inherited #BGCHANGES video ends up in the new song folder)."""
    copied: List[str] = []
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    for name in referenced_media(tags):
        src = src_dir / name
        if src.is_file():
            try:
                shutil.copy2(src, dst_dir / name)
                copied.append(name)
            except Exception:
                pass
    return copied


def find_sibling_chart(audio_path: str) -> str | None:
    """The '.sm/.ssc next to the audio' used by `--inherit_from auto`: return the first chart file in the audio's
    own folder (prefer .ssc over .sm — richer), or None."""
    folder = Path(audio_path).parent
    for pat in ("*.ssc", "*.sm"):
        hits = sorted(folder.glob(pat))
        if hits:
            return str(hits[0])
    return None

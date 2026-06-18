"""
Render a (T, 4) chart tensor back to a playable StepMania .sm file.

This is the inverse of StepManiaParser.convert_to_tensor for Phase 1 scope
(steps and jumps only, fixed BPM, 16th-note resolution). It is the Stage 0
deliverable that lets generated charts be written out and played / re-parsed.

Resolution contract (must match the parser, timesteps_per_beat=4):
    - 4 beats per measure -> 16 timesteps (rows) per measure
    - measure m, row l  ->  timestep index 16*m + l
    - each row is 4 chars over {'0','1'}, panels [Left, Down, Up, Right]

See docs/phase2_generative_design.md.
"""

from typing import Optional, Union
import math

import numpy as np
import torch

ROWS_PER_MEASURE = 16  # 4 beats * timesteps_per_beat(4); 16th-note resolution
NUM_PANELS = 4


def _chart_to_measures(chart: np.ndarray, typed: bool = False) -> str:
    """Convert a (T, 4) chart array into the .sm measure block (no trailing ';').

    typed=False: binary taps (cell > 0.5 -> '1'). typed=True: keep symbols
    0..4 (none/tap/hold-head/tail/roll-head) verbatim.
    """
    if typed:
        bits = np.clip(np.rint(np.asarray(chart)), 0, 4).astype(np.int64)
    else:
        bits = (chart > 0.5).astype(np.int64)
    T = bits.shape[0]
    num_measures = max(1, math.ceil(T / ROWS_PER_MEASURE))
    padded = num_measures * ROWS_PER_MEASURE

    if padded > T:  # pad final measure with empty rows
        bits = np.vstack([bits, np.zeros((padded - T, NUM_PANELS), dtype=np.int64)])

    measures = []
    for m in range(num_measures):
        block = bits[m * ROWS_PER_MEASURE:(m + 1) * ROWS_PER_MEASURE]
        rows = ["".join(str(int(v)) for v in row) for row in block]
        measures.append("\n".join(rows))

    # Measures are comma-separated; the parser splits the notes body on ','.
    return "\n,\n".join(measures)


def tensor_to_sm(
    chart: Union[np.ndarray, torch.Tensor],
    bpm: float,
    title: str = "Generated Chart",
    artist: str = "stepmania-chart-generator",
    music: str = "audio.ogg",
    difficulty_name: str = "Medium",
    difficulty_value: int = 5,
    offset: float = 0.0,
    author: str = "phase2-generator",
    typed: bool = False,
) -> str:
    """Render a (T, 4) chart tensor to a complete .sm file as a string.

    Args:
        chart: (T, 4) tensor/array, panels [Left, Down, Up, Right]. Binary taps by
            default; if typed=True, cells are symbols 0..4 (none/tap/hold-head/tail/roll-head).
        bpm: fixed BPM for the chart (Phase 1 scope).
        title/artist/music/offset: simfile header metadata.
        difficulty_name: one of Beginner/Easy/Medium/Hard/Challenge.
        difficulty_value: numeric difficulty (meter).
        author: chart author credit (line 2 of #NOTES).

    Returns:
        The full .sm file contents as a string.
    """
    return _sm_header(bpm, title, artist, music, offset) + _notes_block(
        chart, difficulty_name, difficulty_value, author, typed=typed
    )


def _sm_header(bpm, title, artist, music, offset) -> str:
    return (
        f"#TITLE:{title};\n"
        f"#ARTIST:{artist};\n"
        f"#MUSIC:{music};\n"
        f"#OFFSET:{offset};\n"
        f"#SAMPLESTART:0.000;\n"
        f"#SAMPLELENGTH:10.000;\n"
        f"#SELECTABLE:YES;\n"
        f"#BPMS:0.000={float(bpm)};\n"
    )


def _notes_block(chart, difficulty_name, difficulty_value, author, typed: bool = False) -> str:
    """One #NOTES section. #NOTES 5-line header: style:author:difficulty:meter:radar:."""
    if isinstance(chart, torch.Tensor):
        arr = chart.detach().cpu().numpy()
    else:
        arr = np.asarray(chart)
    if arr.ndim != 2 or arr.shape[1] != NUM_PANELS:
        raise ValueError(f"chart must be (T, {NUM_PANELS}), got {arr.shape}")
    measures = _chart_to_measures(arr, typed=typed)
    return (
        "#NOTES:\n"
        "     dance-single:\n"
        f"     {author}:\n"
        f"     {difficulty_name}:\n"
        f"     {int(difficulty_value)}:\n"
        "     0.0,0.0,0.0,0.0,0.0:\n"
        f"{measures}\n"
        ";\n"
    )


def charts_to_sm(charts, bpm, title="Generated Chart",
                 artist="stepmania-chart-generator", music="audio.ogg", offset=0.0,
                 typed: bool = False) -> str:
    """Render multiple difficulty charts into one .sm file (e.g. generated + original).

    Args:
        charts: list of dicts, each {chart, difficulty_name, difficulty_value, author?}.
        typed: if True, charts carry symbols 0..4 (taps/holds/rolls) instead of binary.
    Returns the full .sm contents.
    """
    out = _sm_header(bpm, title, artist, music, offset)
    for c in charts:
        out += _notes_block(c["chart"], c["difficulty_name"], c["difficulty_value"],
                            c.get("author", "phase2-generator"), typed=typed)
    return out


def write_sm(chart: Union[np.ndarray, torch.Tensor], path: str, bpm: float, **kwargs) -> str:
    """Render `chart` to a .sm file at `path`. Extra kwargs forwarded to tensor_to_sm.

    Returns the path written.
    """
    content = tensor_to_sm(chart, bpm=bpm, **kwargs)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

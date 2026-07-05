"""
Render a (T, 4) chart tensor back to a playable StepMania .sm file.

This is the inverse of StepManiaParser.convert_to_tensor for Phase 1 scope
(steps and jumps only, fixed BPM, 16th-note resolution). It is the Stage 0
deliverable that lets generated charts be written out and played / re-parsed.

Resolution contract (must match the parser's timesteps_per_beat):
    - 4 beats per measure -> 4*timesteps_per_beat rows per measure
      (timesteps_per_beat=4 -> 16 rows = 16th grid; =12 -> 48 rows = the data-layer-v2 48th grid, so triplets at
       1/3-of-beat land on exact rows 16/48 & 32/48 instead of being floored to the 16th grid)
    - measure m, row l  ->  timestep index rows_per_measure*m + l
    - each row is 4 chars over {'0','1'}, panels [Left, Down, Up, Right]

See docs/phase2_generative_design.md, notes/data_layer_v2_scope.md.
"""

from typing import Optional, Union
import math

import numpy as np
import torch

ROWS_PER_MEASURE = 16  # DEFAULT: 4 beats * timesteps_per_beat(4); 16th-note resolution. Override via timesteps_per_beat.
NUM_PANELS = 4


def _rows_per_measure(timesteps_per_beat: int = 4) -> int:
    """Rows per 4/4 measure = 4 beats * timesteps_per_beat (16 for the 16th grid, 48 for the v2 48th grid)."""
    return 4 * int(timesteps_per_beat)


def _chart_to_measures(chart: np.ndarray, typed: bool = False, rows_per_measure: int = ROWS_PER_MEASURE) -> str:
    """Convert a (T, 4) chart array into the .sm measure block (no trailing ';').

    typed=False: binary taps (cell > 0.5 -> '1'). typed=True: keep symbols
    0..4 (none/tap/hold-head/tail/roll-head) verbatim. `rows_per_measure` = 4*timesteps_per_beat sets the grid
    (16 = 16th, 48 = the data-layer-v2 48th grid); it MUST match the grid the (T,4) frames were generated on.
    """
    if typed:
        bits = np.clip(np.rint(np.asarray(chart)), 0, 4).astype(np.int64)
    else:
        bits = (chart > 0.5).astype(np.int64)
    T = bits.shape[0]
    num_measures = max(1, math.ceil(T / rows_per_measure))
    padded = num_measures * rows_per_measure

    if padded > T:  # pad final measure with empty rows
        bits = np.vstack([bits, np.zeros((padded - T, NUM_PANELS), dtype=np.int64)])

    measures = []
    for m in range(num_measures):
        block = bits[m * rows_per_measure:(m + 1) * rows_per_measure]
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
    timesteps_per_beat: int = 4,
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
        timesteps_per_beat: the grid the (T,4) frames were generated on (4 = 16th, 12 = the v2 48th grid).

    Returns:
        The full .sm file contents as a string.
    """
    return _sm_header(bpm, title, artist, music, offset) + _notes_block(
        chart, difficulty_name, difficulty_value, author, typed=typed,
        rows_per_measure=_rows_per_measure(timesteps_per_beat)
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


def _notes_block(chart, difficulty_name, difficulty_value, author, typed: bool = False,
                 rows_per_measure: int = ROWS_PER_MEASURE) -> str:
    """One #NOTES section. #NOTES 5-line header: style:author:difficulty:meter:radar:."""
    if isinstance(chart, torch.Tensor):
        arr = chart.detach().cpu().numpy()
    else:
        arr = np.asarray(chart)
    if arr.ndim != 2 or arr.shape[1] != NUM_PANELS:
        raise ValueError(f"chart must be (T, {NUM_PANELS}), got {arr.shape}")
    measures = _chart_to_measures(arr, typed=typed, rows_per_measure=rows_per_measure)
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
                 typed: bool = False, timesteps_per_beat: int = 4) -> str:
    """Render multiple difficulty charts into one .sm file (e.g. generated + original).

    Args:
        charts: list of dicts, each {chart, difficulty_name, difficulty_value, author?}.
        typed: if True, charts carry symbols 0..4 (taps/holds/rolls) instead of binary.
        timesteps_per_beat: the grid ALL charts were generated on (4 = 16th, 12 = the v2 48th grid). ALL charts in
            one .sm share one measure resolution, so a v2 export must have the original A/B chart re-quantized on
            the SAME 48th grid (via StepManiaParser.for_v2) — do not mix grids within one file.
    Returns the full .sm contents.
    """
    rpm = _rows_per_measure(timesteps_per_beat)
    out = _sm_header(bpm, title, artist, music, offset)
    for c in charts:
        out += _notes_block(c["chart"], c["difficulty_name"], c["difficulty_value"],
                            c.get("author", "phase2-generator"), typed=typed, rows_per_measure=rpm)
    return out


def write_sm(chart: Union[np.ndarray, torch.Tensor], path: str, bpm: float, **kwargs) -> str:
    """Render `chart` to a .sm file at `path`. Extra kwargs forwarded to tensor_to_sm.

    Returns the path written.
    """
    content = tensor_to_sm(chart, bpm=bpm, **kwargs)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

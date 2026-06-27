"""Smoke tests for the bring-your-own-audio generator (scripts/generate.py)."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_hop_formula_matches_parser():
    """generate.py's hop must equal the parser's BPM-aligned 16th-note hop, or the model
    sees an off-grid frame rate."""
    from scripts.generate import SR, TIMESTEPS_PER_BEAT, build_stub_chart
    for bpm in (120.0, 150.0, 174.0):
        hop = int(SR * 60 / (bpm * TIMESTEPS_PER_BEAT))
        stub = build_stub_chart("song.ogg", bpm, duration=60.0, hop_length=hop)
        # parser formula: int(target_sr * 60 / (avg_bpm * timesteps_per_beat))
        assert stub.hop_length == int(22050 * 60 / (bpm * 4))
        assert stub.song_length_seconds == 60.0
        assert stub.note_data == []  # we generate the notes, not read them


CKPT = PROJECT_ROOT / "checkpoints/gen_motif_full_fixed/best_val.pt"
MANIFOLD = PROJECT_ROOT / "cache/radar_manifold.npz"


@pytest.mark.skipif(not (CKPT.is_file() and MANIFOLD.is_file()),
                    reason="needs the deployed checkpoint + manifold (not in CI without weights)")
def test_generate_from_audio_roundtrips(tmp_path):
    """End-to-end: a synthetic audio file -> a .sm that re-parses cleanly with balanced holds."""
    import soundfile as sf
    from src.data.stepmania_parser import StepManiaParser

    audio = tmp_path / "tone.wav"
    sr = 22050
    t = np.linspace(0, 12.0, int(sr * 12.0))
    sf.write(audio, (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)

    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/generate.py"),
         "--audio", str(audio), "--difficulty", "Medium", "--bpm", "120",
         "--out", str(out), "--seed", "0"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=300,
    )
    assert r.returncode == 0, f"generate.py failed:\n{r.stdout}\n{r.stderr}"
    sm = out / "chart.sm"
    assert sm.is_file(), "no chart.sm written"

    chart = StepManiaParser(min_song_length=1, max_song_length=100000).parse_file(str(sm))
    assert chart is not None, "generated .sm did not re-parse"
    assert chart.note_data, "no charts in the generated .sm"
    p = StepManiaParser(min_song_length=1, max_song_length=100000)
    tensor = p.convert_to_tensor_extended(chart, chart.note_data[0])[0]
    assert (tensor == 2).sum() == (tensor == 3).sum(), "hold heads != tails (unbalanced spans)"

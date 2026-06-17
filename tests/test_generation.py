"""Stage 0 generation infrastructure: tokenizer + .sm writer round-trips."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.generation.tokenizer import (
    ChartTokenizer,
    NUM_PANEL_STATES,
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    VOCAB_SIZE,
)
from src.generation.sm_writer import tensor_to_sm, ROWS_PER_MEASURE
from src.data.stepmania_parser import StepManiaParser, StepManiaChart, NoteData


def _random_chart(T, seed=0, max_simultaneous=2):
    """Random (T, 4) binary chart with at most `max_simultaneous` panels per row."""
    rng = np.random.default_rng(seed)
    chart = np.zeros((T, 4), dtype=np.float32)
    for t in range(T):
        if rng.random() < 0.4:  # ~40% of rows have a step
            k = rng.integers(1, max_simultaneous + 1)
            panels = rng.choice(4, size=int(k), replace=False)
            chart[t, panels] = 1.0
    return chart


# ---- tokenizer ------------------------------------------------------------------

def test_vocab_constants():
    assert NUM_PANEL_STATES == 16
    assert (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN) == (16, 17, 18)
    assert VOCAB_SIZE == 19


def test_single_state_roundtrip():
    # All 16 panel-states encode/decode losslessly.
    for s in range(NUM_PANEL_STATES):
        row = ChartTokenizer.token_to_panel_state(s)
        assert ChartTokenizer.panel_state_to_token(row) == s


def test_known_encoding():
    # [L,D,U,R] = [0,1,0,1] -> Down(bit1)+Right(bit3) = 0b1010 = 10
    assert ChartTokenizer.panel_state_to_token([0, 1, 0, 1]) == 10
    assert ChartTokenizer.panel_state_to_token([1, 0, 0, 0]) == 1
    np.testing.assert_array_equal(ChartTokenizer.token_to_panel_state(10), [0, 1, 0, 1])


def test_full_chart_tokenizer_roundtrip():
    chart = _random_chart(200, seed=1)
    tokens = ChartTokenizer.encode(chart)
    assert tokens.shape == (200,)
    assert tokens.max().item() < NUM_PANEL_STATES  # no special tokens
    decoded = ChartTokenizer.decode(tokens)
    np.testing.assert_array_equal(decoded, chart)


def test_special_tokens_added_and_stripped():
    chart = _random_chart(32, seed=2)
    tokens = ChartTokenizer.encode(chart, add_special=True)
    assert tokens[0].item() == BOS_TOKEN
    assert tokens[-1].item() == EOS_TOKEN
    assert tokens.shape == (34,)
    decoded = ChartTokenizer.decode(tokens)  # specials dropped
    np.testing.assert_array_equal(decoded, chart)


# ---- .sm writer -----------------------------------------------------------------

def _reparse_sm_notes(content, T, bpm):
    """Parse the #NOTES body of `content` back to a (T, 4) tensor, bypassing audio.

    Builds a StepManiaChart with timesteps_total=T so we isolate the note-encoding
    round-trip from audio-duration-derived length.
    """
    parser = StepManiaParser()
    note_list = parser._parse_notes_sm(content)
    assert len(note_list) == 1, f"expected 1 dance-single chart, got {len(note_list)}"
    chart_meta = StepManiaChart(
        title="", artist="", audio_file="", bpm=bpm, offset=0.0,
        sample_start=0.0, sample_length=0.0, timing_events=[],
        note_data=note_list, song_length_seconds=0.0,
        timesteps_total=T, hop_length=0,
    )
    return parser.convert_to_tensor(chart_meta, note_list[0])


def test_writer_parser_roundtrip_exact():
    # T a multiple of 16 -> writer pads nothing; round-trip must be identity.
    T = ROWS_PER_MEASURE * 8  # 128
    chart = _random_chart(T, seed=3)
    content = tensor_to_sm(chart, bpm=120.0, difficulty_name="Hard", difficulty_value=8)
    reparsed = _reparse_sm_notes(content, T=T, bpm=120.0)
    assert reparsed.shape == (T, 4)
    np.testing.assert_array_equal(reparsed, chart)


def test_writer_pads_partial_final_measure():
    # T not a multiple of 16 -> writer pads to a full measure; first T rows preserved.
    T = ROWS_PER_MEASURE * 3 + 5  # 53
    chart = _random_chart(T, seed=4)
    content = tensor_to_sm(chart, bpm=150.0)
    padded_T = ROWS_PER_MEASURE * 4  # 64
    reparsed = _reparse_sm_notes(content, T=padded_T, bpm=150.0)
    np.testing.assert_array_equal(reparsed[:T], chart)
    assert reparsed[T:].sum() == 0  # padding rows are empty


def test_writer_produces_parseable_header():
    chart = _random_chart(16, seed=5)
    content = tensor_to_sm(chart, bpm=128.0, title="T", artist="A", music="song.ogg")
    for field in ("#TITLE:T;", "#ARTIST:A;", "#MUSIC:song.ogg;", "#BPMS:0.000=128.0;"):
        assert field in content, f"missing {field!r}"
    assert content.count("#NOTES:") == 1
    assert content.rstrip().endswith(";")

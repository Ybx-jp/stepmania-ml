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
from src.generation.sm_writer import tensor_to_sm, charts_to_sm, ROWS_PER_MEASURE
from src.generation.evaluation import onset_density_metrics
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


def test_typed_writer_parser_roundtrip():
    # Typed chart with taps, a hold (2..3), a roll (4..3), and a jump must round-trip
    # exactly through the typed writer -> parser convert_to_tensor_typed.
    from src.generation.sm_writer import charts_to_sm
    T = ROWS_PER_MEASURE * 2  # 32
    chart = np.zeros((T, 4), dtype=np.int64)
    chart[0, 0] = 1               # tap, Left
    chart[4, 1] = 2               # hold head, Down
    chart[8, 1] = 3               # hold tail, Down
    chart[12, 2] = 4              # roll head, Up
    chart[16, 2] = 3              # roll tail, Up
    chart[20, 0] = 1; chart[20, 3] = 1   # jump (Left+Right)
    chart[24, 3] = 2; chart[28, 3] = 3   # another hold, Right

    content = tensor_to_sm(chart, bpm=130.0, typed=True, difficulty_name="Hard", difficulty_value=8)
    parser = StepManiaParser()
    notes = parser._parse_notes_sm(content)
    assert len(notes) == 1
    meta = StepManiaChart(
        title="", artist="", audio_file="", bpm=130.0, offset=0.0, sample_start=0.0,
        sample_length=0.0, timing_events=[], note_data=notes, song_length_seconds=0.0,
        timesteps_total=T, hop_length=0,
    )
    reparsed = parser.convert_to_tensor_typed(meta, notes[0])
    np.testing.assert_array_equal(reparsed, chart)
    # all four non-empty symbols present and preserved
    assert set(np.unique(reparsed)) == {0, 1, 2, 3, 4}


def test_typed_helpers():
    from src.generation.typed import onset_mask, symbol_histogram, NUM_SYMBOLS
    assert NUM_SYMBOLS == 5
    chart = np.zeros((5, 4), dtype=np.int64)
    chart[1, 0] = 2; chart[3, 2] = 1
    np.testing.assert_array_equal(onset_mask(chart), [False, True, False, True, False])
    h = symbol_histogram(chart)
    assert h["tap"] == 1 and h["hold_head"] == 1 and h["none"] == 18


def test_kv_cache_matches_noncached():
    # Cached generation must be bit-identical to non-cached (greedy, fixed onset).
    import torch
    from src.generation.factorized import FactorizedChartGenerator
    torch.manual_seed(0)
    m = FactorizedChartGenerator(audio_dim=23, d_model=64, nhead=4, num_layers=2, onset_layers=1).eval()
    B, T = 2, 96
    audio = torch.randn(B, T, 23)
    diff = torch.tensor([0, 3])
    torch.manual_seed(1)
    override = torch.rand(B, T) > 0.7
    slow = m.generate(audio, diff, onset_override=override, panel_greedy=True)
    fast = m.generate_cached(audio, diff, onset_override=override, panel_greedy=True)
    assert torch.equal(slow, fast), f"{(slow != fast).any(-1).sum().item()} timesteps differ"


def test_writer_produces_parseable_header():
    chart = _random_chart(16, seed=5)
    content = tensor_to_sm(chart, bpm=128.0, title="T", artist="A", music="song.ogg")
    for field in ("#TITLE:T;", "#ARTIST:A;", "#MUSIC:song.ogg;", "#BPMS:0.000=128.0;"):
        assert field in content, f"missing {field!r}"
    assert content.count("#NOTES:") == 1
    assert content.rstrip().endswith(";")


def test_charts_to_sm_two_charts_roundtrip():
    # Two charts in one .sm -> parser sees both, each round-trips exactly.
    T = ROWS_PER_MEASURE * 4
    gen = _random_chart(T, seed=10)
    orig = _random_chart(T, seed=11)
    content = charts_to_sm(
        charts=[
            {"chart": gen, "difficulty_name": "Challenge", "difficulty_value": 9, "author": "generated"},
            {"chart": orig, "difficulty_name": "Hard", "difficulty_value": 7, "author": "original"},
        ],
        bpm=140.0, title="AB", music="song.ogg",
    )
    assert content.count("#NOTES:") == 2
    parser = StepManiaParser()
    notes = parser._parse_notes_sm(content)
    assert len(notes) == 2
    chart_meta = StepManiaChart(
        title="", artist="", audio_file="", bpm=140.0, offset=0.0, sample_start=0.0,
        sample_length=0.0, timing_events=[], note_data=notes, song_length_seconds=0.0,
        timesteps_total=T, hop_length=0,
    )
    np.testing.assert_array_equal(parser.convert_to_tensor(chart_meta, notes[0]), gen)
    np.testing.assert_array_equal(parser.convert_to_tensor(chart_meta, notes[1]), orig)


# ---- onset / density metrics ----------------------------------------------------

def test_onset_metrics_perfect_match():
    chart = _random_chart(64, seed=6)
    m = onset_density_metrics(chart, reference=chart)
    assert m["onset_f1"] == 1.0
    assert m["onset_precision"] == 1.0
    assert m["onset_recall"] == 1.0
    assert m["panel_accuracy_on_onset"] == 1.0
    assert m["density_ratio"] == 1.0


def test_onset_metrics_no_reference():
    chart = np.zeros((32, 4), dtype=np.float32)
    chart[::4, 0] = 1.0  # a step every 4th timestep on Left
    m = onset_density_metrics(chart)
    assert "onset_f1" not in m  # no reference -> density-only
    assert abs(m["gen_density"] - 0.25) < 1e-6
    assert m["n_timesteps"] == 32


def test_onset_metrics_partial_and_panels():
    ref = np.zeros((4, 4), dtype=np.float32)
    gen = np.zeros((4, 4), dtype=np.float32)
    ref[0, 0] = 1; ref[1, 1] = 1; ref[2, 2] = 1            # 3 onsets
    gen[0, 0] = 1; gen[1, 3] = 1; gen[3, 0] = 1            # onsets at 0,1,3
    m = onset_density_metrics(gen, reference=ref)
    # shared onsets at t=0 (match panels) and t=1 (wrong panel); fp at t=3; fn at t=2
    assert m["onset_precision"] == 2 / 3  # tp=2 (t0,t1), fp=1 (t3)
    assert m["onset_recall"] == 2 / 3     # tp=2, fn=1 (t2)
    assert abs(m["panel_accuracy_on_onset"] - 0.5) < 1e-6  # t0 right, t1 wrong


def test_onset_metrics_mask_applied():
    gen = _random_chart(20, seed=7)
    ref = _random_chart(20, seed=8)
    mask = np.zeros(20, dtype=bool)
    mask[:10] = True
    m = onset_density_metrics(gen, reference=ref, mask=mask)
    assert m["n_timesteps"] == 10

"""data-layer-v2 phase 2a: the finer-grid note quantization (StepManiaParser.for_v2 / _beat_to_ts)."""
import numpy as np
from src.data.stepmania_parser import StepManiaParser


def test_legacy_path_is_floor_16th_byte_identical():
    p = StepManiaParser()  # deployed default: timesteps_per_beat=4, floor
    assert p.timesteps_per_beat == 4 and p.round_quantize is False
    for beat in [0.0, 0.25, 0.3, 1.0 / 3, 1.333, 2.667, 7.9]:
        assert p._beat_to_ts(beat) == int(np.floor(beat * 4))


def test_v2_factory_config():
    v = StepManiaParser.for_v2()
    assert v.timesteps_per_beat == 12 and v.round_quantize is True
    v96 = StepManiaParser.for_v2(subdiv=24)
    assert v96.timesteps_per_beat == 24


def test_triplet_lands_exactly_on_48th():
    legacy = StepManiaParser()          # 16th floor
    v2 = StepManiaParser.for_v2()       # 48th round
    # an eighth-note triplet at beat 1/3
    b = 1.0 / 3
    # legacy: floor(4/3)=1 -> quantized back to 1/4 = 0.25 (the 0.083-beat shear)
    assert legacy._beat_to_ts(b) == 1
    assert abs(1 / 4 - b) > 0.08
    # v2: round(12/3)=4 -> quantized back to 4/12 = 1/3 EXACTLY
    ts = v2._beat_to_ts(b)
    assert ts == 4
    assert abs(ts / 12 - b) < 1e-9


def test_sixteenth_still_exact_on_48th():
    v2 = StepManiaParser.for_v2()
    for i in range(4):          # 16th positions 0,1/4,2/4,3/4 all land on 48th cells 0,3,6,9
        b = i / 4
        ts = v2._beat_to_ts(b)
        assert ts == i * 3 and abs(ts / 12 - b) < 1e-9


def test_v2_convert_places_triplet_correctly():
    from src.data.stepmania_parser import StepManiaChart, NoteData, TimingEvent
    # one measure of six eighth-note triplets (taps on panel 0), 4 beats -> 48 cells at subdiv 12
    measure = "\n".join(["1000"] * 6)
    chart = StepManiaChart(title="t", artist="", audio_file="a", bpm=120.0, offset=0.0,
                           sample_start=0.0, sample_length=0.0,
                           timing_events=[TimingEvent(0.0, 120.0, "bpm")], note_data=[],
                           song_length_seconds=2.0, timesteps_total=48, hop_length=1)
    nd = NoteData(difficulty_name="Hard", difficulty_value=8, notes=measure)
    arr = StepManiaParser.for_v2().convert_to_tensor_typed(chart, nd)
    # six triplets over 4 beats -> beats 0, 2/3, 4/3, 2, 8/3, 10/3 -> 48th cells 0,8,16,24,32,40
    hit = np.where(arr[:, 0] != 0)[0]
    assert list(hit) == [0, 8, 16, 24, 32, 40]

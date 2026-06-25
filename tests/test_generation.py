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


def test_pair_holds():
    from src.generation.typed import pair_holds
    chart = np.zeros((10, 4), dtype=np.int64)
    # panel 0: valid hold (head@1 -> tail@4)
    chart[1, 0] = 2; chart[4, 0] = 3
    # panel 1: orphan head (no tail) -> should become a tap
    chart[2, 1] = 2
    # panel 2: orphan tail (no head) -> should become none
    chart[5, 2] = 3
    # panel 3: roll head -> tail (valid)
    chart[0, 3] = 4; chart[3, 3] = 3
    out = pair_holds(chart)
    assert out[1, 0] == 2 and out[4, 0] == 3      # valid hold preserved
    assert out[2, 1] == 1                          # orphan head -> tap
    assert out[5, 2] == 0                          # orphan tail -> none
    assert out[0, 3] == 4 and out[3, 3] == 3       # valid roll preserved
    # every head now has a matching tail per panel
    for p in range(4):
        heads = int(((out[:, p] == 2) | (out[:, p] == 4)).sum())
        tails = int((out[:, p] == 3).sum())
        assert heads == tails, f"panel {p}: {heads} heads vs {tails} tails"


def test_pattern_bias_and_crossover_helpers():
    from src.generation.typed import make_pattern_bias, count_crossovers, NUM_PATTERNS
    # jump bias only boosts multi-panel patterns
    b = make_pattern_bias(jump=2.0)
    assert b.shape == (NUM_PATTERNS,)
    # single-panel patterns (state 1<<p, index state-1): L=0, D=1, U=3, R=7 -> unboosted
    assert b[0] == 0.0 and b[1] == 0.0 and b[3] == 0.0 and b[7] == 0.0
    assert b[(1 | 2) - 1] == 2.0                                # L+D jump (state 3 -> index 2) boosted
    # panel_prefs adds per-panel bias
    pp = make_pattern_bias(panel_prefs=[0, 0, 0, 5.0])          # favor R (panel 3)
    assert pp[(1 << 3) - 1] == 5.0                              # single-R pattern gets +5
    assert pp[0] == 0.0                                         # single-L unaffected
    # crossover counting: L(left foot) then R(right foot) = no cross; L then L = ...
    chart = np.zeros((4, 4), dtype=np.int64)
    chart[0, 3] = 1   # R with left foot (next_foot starts left) -> crossover
    cr, singles = count_crossovers(chart)
    assert singles == 1 and cr == 1


def test_onset_phase_penalty_shifts_on_beat():
    # The metric gate suppresses off-beat onsets, so a higher penalty must raise the fraction of
    # notes that land on the beat (frame index % 4 == 0).
    import numpy as np, torch
    from src.generation.typed_model import LayeredTypedChartGenerator
    torch.manual_seed(0)
    m = LayeredTypedChartGenerator(audio_dim=23, d_model=64, nhead=4, num_layers=2, onset_layers=1).eval()
    B, T = 2, 400
    audio = torch.randn(B, T, 23); diff = torch.tensor([2, 2])

    def on_beat_frac(pen):
        torch.manual_seed(1)
        g = m.generate(audio, diff, onset_sample=True, onset_phase_penalty=pen).numpy()
        on = (g != 0).any(2)
        fr = [(np.where(on[b])[0] % 4 == 0).mean() for b in range(B) if on[b].any()]
        return float(np.mean(fr))

    assert on_beat_frac(2.5) > on_beat_frac(0.0) + 0.1, "phase penalty did not push onsets on-beat"


def test_no_crossovers_decoding():
    import torch
    from src.generation.typed_model import LayeredTypedChartGenerator
    from src.generation.typed import count_crossovers
    torch.manual_seed(0)
    m = LayeredTypedChartGenerator(audio_dim=23, d_model=64, nhead=4, num_layers=2, onset_layers=1).eval()
    B, T = 2, 200
    audio = torch.randn(B, T, 23); diff = torch.tensor([2, 3])
    ov = torch.ones(B, T, dtype=torch.bool)
    g = m.generate(audio, diff, onset_override=ov, pattern_sample=True, pattern_temperature=1.0,
                   no_crossovers=True).numpy()
    total_cross = sum(count_crossovers(g[b])[0] for b in range(B))
    assert total_cross == 0, f"no_crossovers should yield 0 crossovers, got {total_cross}"


def test_no_jump_during_hold():
    # A pad player holding a panel has one free foot, so while a hold is open there must be
    # no jump (>=2 fresh presses on non-held panels). Verify the knob enforces it.
    import torch
    from src.generation.typed_model import LayeredTypedChartGenerator
    torch.manual_seed(0)
    m = LayeredTypedChartGenerator(audio_dim=23, d_model=64, nhead=4, num_layers=2, onset_layers=1).eval()
    B, T = 2, 200
    audio = torch.randn(B, T, 23); diff = torch.tensor([2, 3])
    ov = torch.ones(B, T, dtype=torch.bool)  # force onsets so holds and notes appear
    g = m.generate(audio, diff, onset_override=ov, pattern_sample=True, pattern_temperature=1.5,
                   type_sample=True, type_temperature=1.0, hold_aware=True,
                   no_jump_during_hold=True).numpy()
    for b in range(B):
        held = [False, False, False, False]
        for tt in range(T):
            row = g[b, tt]
            # panels held entering this frame (head opened earlier, not yet tailed)
            fresh = sum(1 for p in range(4) if row[p] in (1, 2, 4) and not held[p])  # fresh presses on free panels
            if any(held):
                assert fresh <= 1, f"jump ({fresh} fresh presses) placed while a hold was open at b={b} t={tt}"
            # update hold automaton: head opens, tail closes
            for p in range(4):
                if row[p] in (2, 4):
                    held[p] = True
                elif row[p] == 3:
                    held[p] = False


def test_no_note_while_both_feet_held_and_jack_cap_survives_holds():
    # Two gaps that CONDITIONING exposed (playtest 2026-06-23): (1) a single fresh press while BOTH feet are
    # pinned by holds is unhittable (total occupancy > 2 feet) -- no_jump_during_hold must count FEET, not just
    # forbid >=2 fresh; (2) a {tap, hold-close} jump reads as a single in the chart, so max_jack_run must cap
    # FRESH single presses, not the pattern, or fast jacks leak through hold-close frames.
    import torch
    from src.generation.typed_model import LayeredTypedChartGenerator
    torch.manual_seed(0)
    m = LayeredTypedChartGenerator(audio_dim=23, d_model=64, nhead=4, num_layers=2, onset_layers=1).eval()
    B, T = 3, 300
    audio = torch.randn(B, T, 23); diff = torch.tensor([2, 3, 3])
    ov = torch.ones(B, T, dtype=torch.bool)                      # dense onsets -> many holds + adjacency
    g = m.generate(audio, diff, onset_override=ov, pattern_sample=True, pattern_temperature=1.5,
                   type_sample=True, type_temperature=1.5, hold_aware=True,   # high type temp -> lots of holds
                   no_jump_during_hold=True, no_cross_during_hold=True, max_jack_run=1).numpy()
    for b in range(B):
        held = [False] * 4; prev_fresh_single = -1
        for tt in range(T):
            row = g[b, tt]
            fresh = [p for p in range(4) if row[p] in (1, 2, 4) and not held[p]]   # panels a foot freshly hits
            occupied = set(p for p in range(4) if held[p]) | set(fresh)             # feet needed this instant
            assert len(occupied) <= 2, f"{len(occupied)} feet required at b={b} t={tt} (note while both feet held)"
            cur = fresh[0] if len(fresh) == 1 else -1                               # a fresh SINGLE press
            if cur >= 0 and cur == prev_fresh_single:
                raise AssertionError(f"fast same-panel fresh jack at b={b} t={tt} despite max_jack_run=1")
            prev_fresh_single = cur                                                 # -1 (jump/hold-close/empty) breaks the run
            for p in range(4):
                if row[p] in (2, 4):
                    held[p] = True
                elif row[p] == 3:
                    held[p] = False


def test_no_cross_during_hold():
    # While a hold pins one foot, the free foot must not fast-cross panels (the B4U "jacks with one
    # foot during a hold"). The flag must REDUCE (never increase) the count of fast free-foot panel
    # changes during an open hold. Comparative test (same seed off vs on) -> robust to the exact
    # free-foot bookkeeping.
    import torch
    from src.generation.typed_model import LayeredTypedChartGenerator
    m = LayeredTypedChartGenerator(audio_dim=23, d_model=64, nhead=4, num_layers=2, onset_layers=1).eval()
    B, T = 8, 200                                # batch big enough that the off-vs-on count is robust to
    audio = torch.randn(B, T, 23); diff = torch.tensor([2, 3] * 4)   # sampling noise at the tiny per-seq counts
    ov = torch.ones(B, T, dtype=torch.bool)  # force onsets so holds and notes appear
    kw = dict(onset_override=ov, pattern_sample=True, pattern_temperature=1.5, type_sample=True,
              type_temperature=1.0, hold_aware=True, no_jump_during_hold=True)

    def fast_hold_crosses(g):
        cnt = 0
        for b in range(B):
            held = [False, False, False, False]; last_p, last_t = -1, -99
            for tt in range(T):
                held_open = any(held); row = g[b, tt]
                taps = [p for p in range(4) if row[p] == 1 and not held[p]]  # free-panel taps
                if held_open and len(taps) == 1:
                    p = taps[0]
                    if last_p >= 0 and (tt - last_t) <= 1 and p != last_p:
                        cnt += 1                       # fast free-foot panel change during a hold
                    last_p, last_t = p, tt
                elif not held_open:
                    last_p = -1
                for p in range(4):
                    if row[p] in (2, 4):
                        held[p] = True
                    elif row[p] == 3:
                        held[p] = False
        return cnt

    torch.manual_seed(0); g_off = m.generate(audio, diff, no_cross_during_hold=False, **kw).numpy()
    torch.manual_seed(0); g_on = m.generate(audio, diff, no_cross_during_hold=True, **kw).numpy()
    c_off, c_on = fast_hold_crosses(g_off), fast_hold_crosses(g_on)
    assert c_on <= c_off, f"no_cross_during_hold increased fast hold crosses: {c_off} -> {c_on}"
    if c_off > 0:                                       # if there were crosses to fix, it should fix some
        assert c_on < c_off, f"no_cross_during_hold did not reduce fast hold crosses: {c_off} -> {c_on}"


def test_max_jack_run_caps_fast_jacks():
    # H13 EXERTION: one foot hammering one panel at 16th speed is brutal; real charts alternate panels
    # (jack-pair-rate ~0.006, max fast run ~1 over 786 charts). With onsets forced every frame (all
    # 16th-adjacent) and max_jack_run=1, no two consecutive single-panel frames may share a panel.
    import torch
    from src.generation.typed_model import LayeredTypedChartGenerator
    torch.manual_seed(0)
    m = LayeredTypedChartGenerator(audio_dim=23, d_model=64, nhead=4, num_layers=2, onset_layers=1).eval()
    B, T = 2, 200
    audio = torch.randn(B, T, 23); diff = torch.tensor([2, 3])
    ov = torch.ones(B, T, dtype=torch.bool)             # every frame an onset -> all pairs are 16th-adjacent
    kw = dict(onset_override=ov, pattern_sample=True, pattern_temperature=1.5)

    def fast_jack_pairs(g):                              # adjacent single-panel frames on the SAME panel
        n = 0                                            # (count ANY active symbol: the cap masks at the
        for b in range(B):                               # pattern/which-panel level, so with hold_aware off
            for tt in range(T - 1):                      # an active panel == a pattern panel regardless of type)
                pa = [p for p in range(4) if g[b, tt][p] != 0]
                pc = [p for p in range(4) if g[b, tt + 1][p] != 0]
                if len(pa) == 1 and len(pc) == 1 and pa[0] == pc[0]:
                    n += 1
        return n

    torch.manual_seed(0); g_off = m.generate(audio, diff, **kw).numpy()
    torch.manual_seed(0); g_on = m.generate(audio, diff, max_jack_run=1, **kw).numpy()
    assert fast_jack_pairs(g_on) == 0, f"max_jack_run=1 should yield 0 fast same-panel jacks, got {fast_jack_pairs(g_on)}"
    assert (g_on != 0).any(), "cap collapsed generation to empty"
    if fast_jack_pairs(g_off) > 0:                       # if there were jacks to fix, the cap fixed them
        assert fast_jack_pairs(g_on) < fast_jack_pairs(g_off)


def test_enforce_playability_jack_cap():
    # The H13 exertion cap is MANDATORY: enforce_playability injects max_jack_run when missing and
    # REFUSES to ship a chart with it disabled (None/0) unless a deviation reason is given.
    import pytest
    from src.generation.playtest_export import enforce_playability, MANDATORY_JACK_CAP
    kw = enforce_playability({})                          # missing -> injected at the default cap
    assert kw["max_jack_run"] == MANDATORY_JACK_CAP
    assert enforce_playability({"max_jack_run": 2})["max_jack_run"] == 2   # a looser positive cap is allowed
    with pytest.raises(SystemExit):                       # disabled (None) -> refuse without an override
        enforce_playability({"max_jack_run": None})
    with pytest.raises(SystemExit):                       # disabled (0) -> refuse without an override
        enforce_playability({"max_jack_run": 0})
    # explicit, user-approved deviation is allowed (e.g. an offline diagnostic measuring the uncapped baseline)
    assert enforce_playability({"max_jack_run": None}, override_reason="diagnostic")["max_jack_run"] is None


def test_onset_phase_alloc():
    # The phase-aware threshold steers the note rhythm distribution to the given shares while preserving
    # the note budget. A single threshold buries 16ths; alloc must (a) place 16th-phase notes at ~the
    # target share, and (b) keep total note count close to what the same threshold gives globally.
    import torch
    from src.generation.typed_model import LayeredTypedChartGenerator
    from src.generation.typed import pair_holds
    m = LayeredTypedChartGenerator(audio_dim=23, d_model=64, nhead=4, num_layers=2, onset_layers=1).eval()
    B, T = 1, 400
    audio = torch.randn(B, T, 23); diff = torch.tensor([3])
    # sparse, selective regime (~20% density) so a single threshold is actually selective and buries the
    # lower-confidence 16ths -- which is the regime real generation runs in.
    with torch.no_grad():
        p = torch.sigmoid(m.onset_logits(m.encode_audio(audio), diff))[0]
    tau = float(torch.quantile(p, 1 - 0.20))
    kw = dict(onset_threshold=tau, pattern_sample=True, pattern_temperature=0.7, type_sample=True,
              type_temperature=0.4, hold_aware=True)

    def phase_counts(g):
        note = np.asarray((pair_holds(g[0].numpy()) != 0).any(1))
        t = np.arange(T)
        return note[t % 4 == 0].sum(), note[t % 4 == 2].sum(), note[(t % 4 == 1) | (t % 4 == 3)].sum()

    torch.manual_seed(0); g_global = m.generate(audio, diff, **kw)
    torch.manual_seed(0); g_alloc = m.generate(audio, diff, onset_phase_alloc=(0.7, 0.25, 0.05), **kw)
    q0, e0, s0 = phase_counts(g_global)
    q1, e1, s1 = phase_counts(g_alloc)
    n_global, n_alloc = int(q0 + e0 + s0), int(q1 + e1 + s1)
    # alloc never EXCEEDS the global budget (it redistributes; band caps can only shrink it -- and in the
    # realistic low-density regime it's preserved, e.g. diag_phase_threshold showed 0.270 -> 0.267).
    assert n_alloc <= n_global + 1, f"alloc exceeded note budget: {n_global} -> {n_alloc}"
    # MECHANISM (model-independent): alloc steers the realized rhythm distribution toward the shares --
    # quarters dominate and 16ths land near the requested 5% (loose: hold-pairing shifts counts a little).
    # (That this BEATS a single threshold is a trained-model property, shown on the real model in
    # diag_phase_threshold.py: 16ths 1.3% -> 4.1%; an untrained model has no phase bias to overcome.)
    assert n_alloc > 0
    assert q1 > e1 and q1 > s1, f"alloc quarter share not dominant: q={q1} e={e1} s={s1}"
    assert 0.02 <= s1 / n_alloc <= 0.12, f"alloc 16th share off target (~0.05): {s1}/{n_alloc}"


def test_onset_phase_calib():
    # The per-phase calibration offset raises 16th onset confidence (b16 > 0) so more 16ths clear the SAME
    # per-song threshold -- unlike the flat alloc quota, the count floats (here we just check it increases
    # the 16th share vs no offset, at a fixed threshold derived from the offset probs).
    import torch
    from src.generation.typed_model import LayeredTypedChartGenerator
    from src.generation.typed import pair_holds
    m = LayeredTypedChartGenerator(audio_dim=23, d_model=64, nhead=4, num_layers=2, onset_layers=1).eval()
    B, T = 1, 400
    audio = torch.randn(B, T, 23); diff = torch.tensor([3])

    def run(calib):
        # threshold computed from the SAME (calibrated) onset logits, as the exporter does
        with torch.no_grad():
            ol = m.onset_logits(m.encode_audio(audio), diff)[0]
            if calib is not None:
                ph = torch.arange(T) % 4
                ol = ol + torch.where(ph == 2, calib[0], torch.where((ph == 1) | (ph == 3), calib[1], 0.0))
            tau = float(torch.quantile(torch.sigmoid(ol), 1 - 0.20))
        g = m.generate(audio, diff, onset_threshold=tau, onset_phase_calib=calib, pattern_sample=True,
                       pattern_temperature=0.7, type_sample=True, type_temperature=0.4, hold_aware=True)
        note = np.asarray((pair_holds(g[0].numpy()) != 0).any(1))
        t = np.arange(T); n = max(int(note.sum()), 1)
        return note[(t % 4 == 1) | (t % 4 == 3)].sum() / n  # 16th fraction

    base = run(None)
    boosted = run((0.0, 2.0))  # strong 16th offset
    assert boosted > base, f"calib b16 did not raise 16th share: {base:.3f} -> {boosted:.3f}"


def test_style_encoder_bottleneck_and_invariance():
    # The reference-chart style encoder pools over time to a single (B,d) latent, and
    # padded frames must not affect that latent (masked mean).
    import torch
    from src.generation.typed_model import LayeredTypedChartGenerator
    torch.manual_seed(0)
    m = LayeredTypedChartGenerator(audio_dim=23, d_model=64, nhead=4, num_layers=2, onset_layers=1).eval()
    B, T = 2, 64
    ref = torch.randint(0, 5, (B, T, 4))
    mask = torch.ones(B, T, dtype=torch.bool)
    s = m.encode_style(ref, mask)
    assert s.shape == (B, 64), "style latent must be a single per-sample vector"
    # garbage in the padded tail must not change the latent when masked out
    mask2 = mask.clone(); mask2[:, T // 2:] = False
    ref2 = ref.clone(); ref2[:, T // 2:] = torch.randint(0, 5, (B, T - T // 2, 4))
    s_a = m.encode_style(ref, mask2)
    s_b = m.encode_style(ref2, mask2)
    assert torch.allclose(s_a, s_b, atol=1e-5), "masked frames leaked into the style latent"


def test_style_conditioning_changes_logits():
    # Conditioning on a reference chart must actually move the model's predictions
    # (vs the null-style path), otherwise the style knob is a no-op.
    import torch
    from src.generation.typed_model import LayeredTypedChartGenerator
    torch.manual_seed(0)
    m = LayeredTypedChartGenerator(audio_dim=23, d_model=64, nhead=4, num_layers=2, onset_layers=1).eval()
    B, T = 2, 32
    audio = torch.randn(B, T, 23); states = torch.randint(0, 5, (B, T, 4)); diff = torch.tensor([1, 2])
    mask = torch.ones(B, T, dtype=torch.bool)
    ref = torch.randint(1, 5, (B, T, 4))  # dense reference
    # nudge the style encoder off its zero-init so conditioning is non-trivial
    for p in m.style_encoder.parameters():
        p.data += 0.1 * torch.randn_like(p)
    ol0, pat0, _ = m(audio, states, diff, mask)                       # null style
    ol1, pat1, _ = m(audio, states, diff, mask, reference=ref, reference_mask=mask)
    assert not torch.allclose(pat0, pat1, atol=1e-4), "style conditioning did not change pattern logits"


def test_style_guidance_runs_and_shapes():
    # CFG guidance on the style knob produces a valid chart and differs from g=1.
    import torch
    from src.generation.typed_model import LayeredTypedChartGenerator
    torch.manual_seed(0)
    m = LayeredTypedChartGenerator(audio_dim=23, d_model=64, nhead=4, num_layers=2, onset_layers=1).eval()
    for p in m.style_encoder.parameters():
        p.data += 0.1 * torch.randn_like(p)
    B, T = 2, 48
    audio = torch.randn(B, T, 23); diff = torch.tensor([1, 2])
    ref = torch.randint(1, 5, (B, T, 4)); mask = torch.ones(B, T, dtype=torch.bool)
    ov = torch.ones(B, T, dtype=torch.bool)
    g1 = m.generate(audio, diff, onset_override=ov, reference=ref, reference_mask=mask, guidance_scale=1.0)
    g3 = m.generate(audio, diff, onset_override=ov, reference=ref, reference_mask=mask, guidance_scale=3.0)
    assert g1.shape == (B, T, 4) and g3.shape == (B, T, 4)
    assert (g1 != g3).any(), "style guidance had no effect on the generated chart"


def test_motif_per_frame_schedule_and_onset_decouple():
    # H15 local-motif: motif accepts a per-frame schedule (B,T,K) as well as a global (B,K) vector, and is
    # DECOUPLED from the onset head (motif shapes which-panels, not density).
    import torch
    from src.generation.typed_model import LayeredTypedChartGenerator, MOTIF_DIM
    torch.manual_seed(0)
    m = LayeredTypedChartGenerator(audio_dim=23, d_model=64, nhead=4, num_layers=2, onset_layers=1).eval()
    m.motif_proj.weight.data += 0.1 * torch.randn_like(m.motif_proj.weight)   # off the zero-init no-op
    m.motif_proj.bias.data += 0.1 * torch.randn_like(m.motif_proj.bias)
    B, T, K = 2, 32, MOTIF_DIM
    audio = torch.randn(B, T, 23); states = torch.randint(0, 5, (B, T, 4)); diff = torch.tensor([1, 2])
    mask = torch.ones(B, T, dtype=torch.bool)
    sched = torch.randn(B, T, K)                                              # varying per-frame schedule

    ol_s, pat_s, typ_s = m(audio, states, diff, mask, motif=sched)
    assert ol_s.shape == (B, T) and pat_s.shape[:2] == (B, T) and typ_s.shape[:2] == (B, T)
    ov = torch.ones(B, T, dtype=torch.bool)
    assert m.generate(audio, diff, onset_override=ov, motif=sched).shape == (B, T, 4)

    # back-compat: a CONSTANT schedule must equal the equivalent global (B,K) vector
    gv = torch.randn(B, K); sched_const = gv.unsqueeze(1).expand(B, T, K).contiguous()
    ol_g, pat_g, _ = m(audio, states, diff, mask, motif=gv)
    ol_c, pat_c, _ = m(audio, states, diff, mask, motif=sched_const)
    assert torch.allclose(pat_g, pat_c, atol=1e-5), "constant schedule != equivalent global vector"

    ol0, pat0, _ = m(audio, states, diff, mask, motif=None)                   # null-motif (CFG token)
    assert not torch.allclose(pat0, pat_s, atol=1e-4), "per-frame motif did not change pattern logits"
    # onset head is DECOUPLED from motif (#1): every motif variant leaves onset logits identical
    assert torch.allclose(ol0, ol_s, atol=1e-6) and torch.allclose(ol0, ol_g, atol=1e-6), \
        "motif leaked into the onset head (must be decoupled)"


def test_figure_token_conditioning_and_onset_decouple():
    # H15 hierarchical: discrete per-section figure tokens (B,T) feed the decoder only, zero-init = warm-start
    # no-op, and never affect the onset head.
    import torch
    from src.generation.typed_model import LayeredTypedChartGenerator, NUM_FIGURE_CLASSES
    torch.manual_seed(0)
    m = LayeredTypedChartGenerator(audio_dim=23, d_model=64, nhead=4, num_layers=2, onset_layers=1).eval()
    B, T = 2, 32
    audio = torch.randn(B, T, 23); states = torch.randint(0, 5, (B, T, 4)); diff = torch.tensor([1, 2])
    mask = torch.ones(B, T, dtype=torch.bool)
    fig = torch.randint(0, NUM_FIGURE_CLASSES, (B, T))
    ol0, pat0, _ = m(audio, states, diff, mask)                              # null figure
    olf, patf, _ = m(audio, states, diff, mask, figure=fig)
    assert torch.allclose(pat0, patf, atol=1e-6), "zero-init figure embedding must be a warm-start no-op"
    m.figure_embedding.weight.data += 0.1 * torch.randn_like(m.figure_embedding.weight)
    olf2, patf2, _ = m(audio, states, diff, mask, figure=fig)
    assert not torch.allclose(pat0, patf2, atol=1e-4), "trained figure token did not change pattern logits"
    assert torch.allclose(ol0, olf2, atol=1e-6), "figure leaked into the onset head (must be decoupled)"
    ov = torch.ones(B, T, dtype=torch.bool)
    assert m.generate(audio, diff, onset_override=ov, figure=fig).shape == (B, T, 4)


def test_hold_aware_decoding_valid():
    # Hold-aware decoding's automaton guarantees no orphan tails (a tail only ever
    # closes an open head) and at most one open hold per panel at a time.
    import torch
    from src.generation.typed_model import LayeredTypedChartGenerator
    torch.manual_seed(0)
    m = LayeredTypedChartGenerator(audio_dim=23, d_model=64, nhead=4, num_layers=2, onset_layers=1).eval()
    B, T = 2, 128
    audio = torch.randn(B, T, 23)
    diff = torch.tensor([1, 2])
    ov = torch.ones(B, T, dtype=torch.bool)  # force onsets so panels receive notes
    g = m.generate(audio, diff, onset_override=ov, greedy=True, type_sample=True,
                   type_temperature=0.5, hold_aware=True).numpy()
    for b in range(B):
        for p in range(4):
            open_ = False
            for s in g[b, :, p]:
                if s in (2, 4):
                    assert not open_, "new head opened while a hold was still open"
                    open_ = True
                elif s == 3:
                    assert open_, "orphan tail (closed a panel with no open head)"
                    open_ = False


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

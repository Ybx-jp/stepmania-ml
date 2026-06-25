#!/usr/bin/env python3
"""
H19 audit — does the figure/motif detector miscount HOLD RELEASES (tails) as struck onsets, manufacturing
phantom jacks/trills?

Mechanism under test (motif_codebook.onset_tokens): `active = arr != 0` lets symbol 3 (hold/roll TAIL) through
as if it were a note, and empty hold-body frames are DROPPED, so a hold's head(2) and tail(3) become ADJACENT
same-panel tokens in the onset sequence -> a phantom "XX" jack pair (or "XYXY" trill across alternating holds).
The body frames are 0 (correctly skipped) — the bug is the RELEASE, not the sustain.

This compares the deployed detector (include_tail=True) against an attack-only variant
(active = (arr!=0)&(arr!=3)) on the REAL exported charts the user played, reporting:
  - tail share of onset tokens
  - whole-chart trill knob k10 and jack<->sweep knob k0 (the eval measures), both variants
  - per-section figure_token distribution (jack/trill fractions), both variants

  python experiments/generation_typed/diag_figure_hold_confound.py [--root outputs/playtest_h15] [--section 64]
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, sys
from collections import Counter
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.stepmania_parser import StepManiaParser
from src.data.groove_radar import calculate_groove_radar_from_chart
from src.generation.typed import panels_to_pattern
from src.generation.motif_codebook import (_canon, name_figure, MotifBasis,
                                           FIGURE_CLASSES, _FIG2IDX)

BASIS = MotifBasis.load(PROJECT_ROOT / "cache/motif_basis.npz")


def onset_tokens_v(typed, include_tail):
    """onset_tokens, but optionally treat the hold/roll TAIL (symbol 3) as NOT an attack."""
    arr = np.asarray(typed)
    active = (arr != 0) if include_tail else ((arr != 0) & (arr != 3))
    onset = active.any(1)
    return panels_to_pattern(active[onset]) if onset.any() else np.empty(0, np.int64)


def hist_from_tokens(s):
    vec = np.zeros(len(BASIS.col_meta)); off = 0
    for W in BASIS.scales:
        vocab = BASIS.codebooks[W]; vidx = {c: k for k, c in enumerate(vocab)}
        cnt = Counter()
        if len(s) >= W:
            for j in range(len(s) - W + 1):
                cnt[_canon(s[j:j + W])] += 1
        block = np.zeros(len(vocab))
        for c, n in cnt.items():
            if c in vidx:
                block[vidx[c]] += n
        tot = block.sum()
        vec[off:off + len(vocab)] = block / tot if tot > 0 else block
        off += len(vocab)
    return vec


def figure_token_v(typed, include_tail, W=3, min_onsets=4):
    toks = onset_tokens_v(typed, include_tail)
    if len(toks) < max(W, min_onsets):
        return 0
    cnt = Counter(_canon(toks[j:j + W]) for j in range(len(toks) - W + 1))
    return _FIG2IDX[name_figure(cnt.most_common(1)[0][0])]


def section_fig_fracs(typed, include_tail, section):
    T = typed.shape[0]
    toks = [figure_token_v(typed[i:i + section], include_tail) for i in range(0, T, section)]
    c = Counter(toks); n = max(len(toks), 1)
    return {FIGURE_CLASSES[i]: c.get(i, 0) / n for i in range(len(FIGURE_CLASSES))}


def radar_of(parser, chart, nd):
    bin_t, hold_info = parser.convert_to_tensor_extended(chart, nd)
    bpm = float(chart.bpm); T = bin_t.shape[0]
    song_len = T * 60.0 / (bpm * 4) if bpm > 0 else 1.0
    return calculate_groove_radar_from_chart(bin_t, hold_info, chart.timing_events, song_len, bpm).to_vector()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/playtest_h15")
    ap.add_argument("--sets", nargs="*", default=["h15_08_motif_trill", "h15_00_base"])
    ap.add_argument("--section", type=int, default=64)
    args = ap.parse_args()
    parser = StepManiaParser()

    for setname in args.sets:
        sms = sorted(glob.glob(str(PROJECT_ROOT / args.root / setname / "*" / "chart.sm")))
        print(f"\n=== {setname}  ({len(sms)} songs) ===")
        agg = {"tail_share": [], "dk10": [], "dk0": [],
               "jack_inc": [], "jack_att": [], "trill_inc": [], "trill_att": []}
        for sm in sms:
            song = Path(sm).parent.name
            chart = parser.parse_file(sm)
            nd = next((n for n in chart.note_data
                       if n.difficulty_name.rstrip(':').strip() == "Challenge"), None)
            if nd is None:
                continue
            typed = parser.convert_to_tensor_typed(chart, nd)
            try:
                radar = radar_of(parser, chart, nd).astype(float)
            except Exception:
                continue

            n_inc = len(onset_tokens_v(typed, True))
            n_att = len(onset_tokens_v(typed, False))
            tail_share = (n_inc - n_att) / max(n_inc, 1)

            k_inc = BASIS.encode(hist_from_tokens(onset_tokens_v(typed, True)), radar)
            k_att = BASIS.encode(hist_from_tokens(onset_tokens_v(typed, False)), radar)
            dk10 = k_inc[10] - k_att[10]   # trill knob shift caused by tails
            dk0 = k_inc[0] - k_att[0]       # jack<->sweep knob shift (k0+ = jack)

            f_inc = section_fig_fracs(typed, True, args.section)
            f_att = section_fig_fracs(typed, False, args.section)

            agg["tail_share"].append(tail_share)
            agg["dk10"].append(dk10); agg["dk0"].append(dk0)
            agg["jack_inc"].append(f_inc["jack"]); agg["jack_att"].append(f_att["jack"])
            agg["trill_inc"].append(f_inc["trill"]); agg["trill_att"].append(f_att["trill"])

            print(f"  {song[:26]:26}  tail/onset {tail_share:5.2f}   "
                  f"trillK {k_inc[10]:+5.2f}->{k_att[10]:+5.2f} (Δ{dk10:+.2f})   "
                  f"jackK {k_inc[0]:+5.2f}->{k_att[0]:+5.2f} (Δ{dk0:+.2f})   "
                  f"jack-sec {f_inc['jack']:.2f}->{f_att['jack']:.2f}  "
                  f"trill-sec {f_inc['trill']:.2f}->{f_att['trill']:.2f}")
        if agg["tail_share"]:
            m = {k: float(np.mean(v)) for k, v in agg.items()}
            print(f"  -- mean: tail/onset {m['tail_share']:.2f} | trillK infl Δ{m['dk10']:+.2f} | "
                  f"jackK infl Δ{m['dk0']:+.2f} | jack-sec {m['jack_inc']:.2f}->{m['jack_att']:.2f} | "
                  f"trill-sec {m['trill_inc']:.2f}->{m['trill_att']:.2f}")


if __name__ == "__main__":
    main()

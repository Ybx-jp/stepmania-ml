#!/usr/bin/env python3
"""
Characterize EXPORTED playtest sets (the actual installed charts) — verify the conditioning LANDED in the .sm
files the user plays, not just in the eval harness. Parses each set's generated 'Challenge' chart and reports
playability/rhythm stats + figure-label fractions (the thing the H15 motif/figure knobs target). No model, no
audio — chart-only, so it's fast and reads the real artifact.

  python experiments/generation_typed/characterize_sets.py [--root outputs/playtest_h15] [--section 64]
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
from src.generation.motif_codebook import onset_tokens, _canon, name_figure, MotifBasis

BASIS = MotifBasis.load(PROJECT_ROOT / "cache/motif_basis.npz")


def chart_radar_and_knobs(parser, chart, nd, typed):
    """Compute the generated chart's OWN groove radar, then the radar-orthogonal motif KNOBS (the exact
    measure eval_motif used: candle=k3, trill=k10, jack<->sweep=k0). Returns (radar5, candleK, trillK, sweepK)
    or None on failure."""
    try:
        bin_t, hold_info = parser.convert_to_tensor_extended(chart, nd)
        bpm = float(chart.bpm); T = bin_t.shape[0]
        song_len = T * 60.0 / (bpm * 4) if bpm > 0 else 1.0
        radar = calculate_groove_radar_from_chart(bin_t, hold_info, chart.timing_events, song_len, bpm).to_vector()
        k = BASIS.encode_chart(typed, radar.astype(float))
        return radar, float(k[3]), float(k[10]), float(k[0])
    except Exception:
        return None


def chart_stats(typed, section):
    """typed (T,4) symbols 0..4 -> dict of playability/rhythm stats + figure fractions."""
    active = typed != 0
    onset = active.any(1)
    n_on = int(onset.sum())
    if n_on == 0:
        return None
    density = float(onset.mean())
    npan = active.sum(1)
    jump = float((npan[onset] >= 2).mean())                       # air proxy
    hold = float((typed == 2).sum() / max(1, (typed != 0).sum())) # freeze proxy (hold-heads / placed symbols)
    # 16th-grid off-beat onset share (chaos/syncopation proxy): frame t%4 in {1,3}
    idx = np.nonzero(onset)[0]
    six16 = float(np.mean((idx % 4 == 1) | (idx % 4 == 3))) if len(idx) else 0.0
    # jack: consecutive single-note onsets on the SAME panel
    singles = [(t, int(np.nonzero(active[t])[0][0])) for t in idx if npan[t] == 1]
    jacks = sum(1 for (a, pa), (b, pb) in zip(singles, singles[1:]) if pa == pb)
    jack = jacks / max(1, len(singles) - 1)
    # figure-family MASS: fraction of ALL W=3 onset windows mapping to each family (sensitive — counts every
    # candle window, not just a section's dominant label which the trill-heavy songs saturate).
    toks = onset_tokens(typed)
    fam = Counter(name_figure(_canon(toks[j:j + 3])) for j in range(len(toks) - 2)) if len(toks) >= 3 else Counter()
    ftot = sum(fam.values())
    f = (lambda k: fam[k] / ftot) if ftot else (lambda k: 0.0)
    return dict(density=density, jump=jump, hold=hold, six16=six16, jack=jack,
                sweep=f('sweep/staircase'), candle=f('candle/cross'), trill=f('trill'),
                jackf=f('jack'), n=n_on)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/playtest_h15")
    ap.add_argument("--section", type=int, default=64)
    args = ap.parse_args()
    parser = StepManiaParser()
    sets = sorted(d for d in glob.glob(f"{args.root}/*") if Path(d).is_dir())

    rows = []
    for sd in sets:
        name = Path(sd).name
        stats = []
        for sm in sorted(glob.glob(f"{sd}/*/chart.sm")):
            try:
                chart = parser.parse_file(sm)
            except Exception:
                continue
            nd = next((n for n in chart.note_data
                       if n.difficulty_name.rstrip(':').strip() == "Challenge"), None)  # generated chart
            if nd is None:
                continue
            typed = parser.convert_to_tensor_typed(chart, nd)
            st = chart_stats(typed, args.section)
            if st:
                rk = chart_radar_and_knobs(parser, chart, nd, typed)
                if rk is not None:
                    radar, st['candleK'], st['trillK'], st['sweepK'] = rk
                    st['chaosR'], st['freezeR'], st['airR'], st['streamR'] = radar[4], radar[3], radar[2], radar[0]
                stats.append(st)
        if not stats:
            continue
        keys = set().union(*[s.keys() for s in stats]) - {'n'}
        agg = {k: float(np.nanmean([s.get(k, np.nan) for s in stats])) for k in keys}
        agg['name'] = name; agg['songs'] = len(stats)
        rows.append(agg)

    # table
    g = lambda r, k: r.get(k, float('nan'))
    hdr = (f"{'set':<26} | {'dens':>5} {'jump':>5} {'hold':>5} {'16th':>5} | "
           f"{'candK':>6} {'trillK':>6} {'sweepK':>6} | {'candF':>5} {'sweepF':>6}")
    print("(K = radar-orthogonal motif KNOB z-score, eval_motif's measure; F = raw figure-family mass)")
    print(hdr); print("-" * len(hdr))
    base = next((r for r in rows if r['name'].endswith('00_base')), None)
    for r in rows:
        print(f"{r['name']:<26} | {g(r,'density'):>5.2f} {g(r,'jump'):>5.2f} {g(r,'hold'):>5.2f} "
              f"{g(r,'six16'):>5.2f} | {g(r,'candleK'):>+6.2f} {g(r,'trillK'):>+6.2f} {g(r,'sweepK'):>+6.2f} | "
              f"{g(r,'candle'):>5.2f} {g(r,'sweep'):>6.2f}")
    if base:
        print("\nΔ motif KNOB vs base (the conditioning target; eval_motif's measure):")
        print(f"{'set':<26} | {'ΔcandK':>7} {'ΔtrillK':>7} {'ΔsweepK':>7}")
        for r in rows:
            if r is base:
                continue
            print(f"{r['name']:<26} | {g(r,'candleK')-g(base,'candleK'):>+7.2f} "
                  f"{g(r,'trillK')-g(base,'trillK'):>+7.2f} {g(r,'sweepK')-g(base,'sweepK'):>+7.2f}")
    print("\nREAD: ΔKNOB > 0 on the conditioned axis = the steering LANDED in the exported charts (candle set -> "
          "ΔcandK>0). This is the same radar-orthogonal knob eval_motif measured (which showed Δ+1.7..+3.9 under "
          "ITS decode settings); compare to confirm the export settings (temp 0.7, radar off) preserve the steer.")


if __name__ == "__main__":
    main()

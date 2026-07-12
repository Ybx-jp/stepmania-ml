#!/usr/bin/env python3
"""Decoded-chart check for the universal window (bridges the onset probe to the by-ear gate).

Reads an --ab_onset_window export (each .sm holds Challenge=WINDOWED / Edit=SINGLE-PASS / <human>) and measures
tail (last 20%) vs body phase-concentration on the ACTUAL DECODED charts -- does the windowed decode produce a
LESS-smeared tail (backbone Herfindahl closer to the human) than single-pass, and does it keep more real tail
notes? The onset probe (probe_universal_window.py) proved the onset head recovers; this confirms the recovery
survives the AR pattern/type decode before we spend the user's ears.
"""
import warnings, os, sys, glob
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
from pathlib import Path
import numpy as np
ROOT = Path('/home/ybx/code/stepmania-chart-generator'); sys.path.insert(0, str(ROOT))
from src.data.stepmania_parser import StepManiaParser

SUBDIV = 12
JITTER = {1, 5, 7, 11}; TRIPLET = {2, 4, 8, 10}
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('outputs/typed_samples')
# Challenge = windowed (the fix); Edit = single-pass (today's default); the third slot = the human original.
ARM_LABEL = {'Challenge': 'WINDOWED(fix)', 'Edit': 'single-pass'}


def herf(phases):
    if len(phases) == 0:
        return np.nan
    h = np.array([(phases == k).sum() for k in range(SUBDIV)], float) / len(phases)
    return (h ** 2).sum()


def main():
    P = StepManiaParser.for_v2(subdiv=SUBDIV, min_song_length=30.0, max_song_length=600.0,
                               min_bpm=40.0, max_bpm=320.0, max_simultaneous=4, gimmick_max_bpm=400.0)
    sms = sorted(glob.glob(str(OUT / '**' / '*.sm'), recursive=True))
    print(f"reading {len(sms)} exported .sm from {OUT}\n")
    print(f"{'song':22s} {'arm':16s} {'notes':>6s} {'tailN':>5s} | {'quarter% t/b':>13s} "
          f"{'jit% t/b':>10s} | {'Herf body':>9s} {'Herf TAIL':>9s} {'vs human':>9s}")
    print("-" * 108)
    for sm in sms:
        c = P.parse_file(sm)
        title = (c.title or Path(sm).stem)[:22]
        # human Herfindahl(TAIL) reference from the non-generated slot
        hum_tailH = np.nan
        rows = []
        for nd in c.note_data:
            dn = (nd.difficulty_name or '').rstrip(':').strip()
            t = np.asarray(P.convert_to_tensor_typed(c, nd))
            pres = ((t == 1) | (t == 2) | (t == 4)).any(1)
            idx = np.nonzero(pres)[0]; T = len(pres)
            if len(idx) == 0:
                continue
            pos = idx / max(T - 1, 1)
            body = idx[pos < 0.8] % SUBDIV; tail = idx[pos >= 0.8] % SUBDIV
            qb = (body == 0).mean() * 100 if len(body) else np.nan
            qt = (tail == 0).mean() * 100 if len(tail) else np.nan
            jb = np.isin(body, list(JITTER)).mean() * 100 if len(body) else np.nan
            jt = np.isin(tail, list(JITTER)).mean() * 100 if len(tail) else np.nan
            rec = dict(dn=dn, n=len(idx), tailN=len(tail), qb=qb, qt=qt, jb=jb, jt=jt,
                       hb=herf(body), ht=herf(tail))
            if dn in ARM_LABEL:
                rows.append(rec)
            else:
                hum_tailH = rec['ht']  # the human slot
        for r in rows:
            vs = (r['ht'] - hum_tailH) if not np.isnan(hum_tailH) else np.nan
            print(f"{title:22s} {ARM_LABEL[r['dn']]:16s} {r['n']:>6d} {r['tailN']:>5d} | "
                  f"{r['qb']:5.0f}/{r['qt']:<5.0f}    {r['jb']:4.0f}/{r['jt']:<4.0f}   | "
                  f"{r['hb']:9.3f} {r['ht']:9.3f} {vs:+9.3f}")
        if not np.isnan(hum_tailH):
            print(f"{'':22s} {'(human TAIL Herf':16s} {hum_tailH:.3f})")
        print()
    print("Read: WINDOWED tail Herfindahl should be HIGHER than single-pass (less smear) and CLOSER to human (vs~0);")
    print("      quarter%(tail) higher + jitter%(tail) lower = the backbone survived the decode into the tail.")


if __name__ == '__main__':
    main()

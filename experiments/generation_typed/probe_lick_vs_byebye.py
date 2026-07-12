#!/usr/bin/env python3
"""Length-vs-song control: run the H-subtail metrics on TWO long songs generated via the SAME deployed v2 path
(generate.py, Hard, GLOBAL/no-harm) — Bye Bye (174bpm, 1056 beats) and Lick the Rainbow (128bpm, ~896 beats).
Both are ~2x the v2 ~458-beat trained context. If BOTH show the dense quarter-less 8th-wash tail => LENGTH artifact
(generalizes, worth fixing). If ONLY Bye Bye => a Bye-Bye SONG artifact.

Metrics per chart, tail(last 20%) vs body(first 80%): pure-48th jitter {1,5,7,11}, phase Herfindahl concentration,
quarter-backbone share, density-by-position."""
import warnings, os, sys
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
from pathlib import Path
import numpy as np
ROOT = Path('/home/ybx/code/stepmania-chart-generator'); sys.path.insert(0, str(ROOT))
from src.data.stepmania_parser import StepManiaParser

SUBDIV = 12; NBINS = 10
JITTER = {1, 5, 7, 11}; TRIPLET = {2, 4, 8, 10}
CHARTS = [
    ('Bye Bye  (174bpm)', '/home/ybx/sm-generated/stamina_probe/Bye Bye GLOBAL/chart.sm'),
    ('Lick GLOBAL (base)', '/home/ybx/sm-generated/stamina_probe/Lick GLOBAL/chart.sm'),
    ('Lick HANGOVER', '/home/ybx/sm-generated/stamina_probe/Lick HANGOVER/chart.sm'),
]


def present(sm):
    P = StepManiaParser.for_v2(subdiv=SUBDIV, min_song_length=30.0, max_song_length=600.0,
                               min_bpm=40.0, max_bpm=320.0, max_simultaneous=4, gimmick_max_bpm=400.0)
    c = P.parse_file(sm); nd = next((n for n in c.note_data if n.difficulty_name), None)
    t = np.asarray(P.convert_to_tensor_typed(c, nd))
    return ((t == 1) | (t == 2) | (t == 4)).any(1)


def main():
    print(f"LENGTH-vs-SONG control | subdiv {SUBDIV} | jitter phases {sorted(JITTER)}\n")
    # (1) jitter by position
    print("(1) pure-48th JITTER% by song-position decile")
    print(f"{'chart':18s} {'Tfr':>6s} {'jit%':>5s} |" + "".join(f" b{i}" for i in range(NBINS)))
    print("-" * 72)
    for label, sm in CHARTS:
        if not Path(sm).is_file():
            print(f"{label:18s} (missing: {sm})"); continue
        pres = present(sm); idx = np.nonzero(pres)[0]; T = len(pres)
        is_jit = np.isin(idx % SUBDIV, list(JITTER))
        bins = np.clip((idx / max(T - 1, 1) * NBINS).astype(int), 0, NBINS - 1)
        rates = [np.isin(idx[bins == b] % SUBDIV, list(JITTER)).mean() if (bins == b).any() else np.nan
                 for b in range(NBINS)]
        bstr = " ".join(f"{r*100:2.0f}" if not np.isnan(r) else " ." for r in rates)
        print(f"{label:18s} {T:6d} {is_jit.mean()*100:5.1f} | {bstr}")

    # (1b) QUARTER-backbone share by position decile (the collapse/recovery signal)
    print("\n(1b) QUARTER-backbone% (phase==0) by song-position decile  (tail recovery = last 2 bins rise)")
    print(f"{'chart':18s} {'tailQ':>5s} |" + "".join(f" b{i}" for i in range(NBINS)))
    print("-" * 72)
    for label, sm in CHARTS:
        if not Path(sm).is_file():
            print(f"{label:18s} (missing)"); continue
        pres = present(sm); idx = np.nonzero(pres)[0]; T = len(pres)
        bins = np.clip((idx / max(T - 1, 1) * NBINS).astype(int), 0, NBINS - 1); ph = idx % SUBDIV
        q = [(ph[bins == b] == 0).mean() if (bins == b).any() else np.nan for b in range(NBINS)]
        tailq = np.nanmean(q[8:])
        qstr = " ".join(f"{v*100:2.0f}" if not np.isnan(v) else " ." for v in q)
        print(f"{label:18s} {tailq*100:4.0f}% | {qstr}")

    # (2) smear: tail vs body phase distribution
    print("\n(2) SMEAR: body vs TAIL phase distribution (H=Herfindahl concentration; lower=smear)")
    print(f"{'chart':18s} | {'region':6s} {'quarter%':>8s} {'8th%':>5s} {'16dup%':>6s} {'jit%':>5s} {'H':>6s}")
    print("-" * 62)
    for label, sm in CHARTS:
        if not Path(sm).is_file():
            continue
        pres = present(sm); idx = np.nonzero(pres)[0]; T = len(pres); pos = idx / max(T - 1, 1)
        for reg, mask in [('body', pos < 0.8), ('TAIL', pos >= 0.8)]:
            ph = idx[mask] % SUBDIV; n = len(ph)
            if n == 0:
                continue
            p = np.array([(ph == k).sum() for k in range(SUBDIV)]) / n
            H = (p ** 2).sum()
            print(f"{label:18s} | {reg:6s} {p[0]*100:>8.0f} {p[6]*100:>5.0f} {(p[3]+p[9])*100:>6.0f} "
                  f"{sum(p[k] for k in JITTER)*100:>5.0f} {H:>6.3f}")

    # (3) density by position
    print("\n(3) DENSITY by position (notes/frame x1000)")
    print(f"{'chart':18s} |" + "".join(f" b{i:<2d}" for i in range(NBINS)) + " | b8b9/body")
    print("-" * 74)
    for label, sm in CHARTS:
        if not Path(sm).is_file():
            continue
        pres = present(sm); T = len(pres); idx = np.nonzero(pres)[0]
        allb = np.clip((np.arange(T) / max(T - 1, 1) * NBINS).astype(int), 0, NBINS - 1)
        ntb = np.clip((idx / max(T - 1, 1) * NBINS).astype(int), 0, NBINS - 1)
        dens = np.array([(ntb == b).sum() / max((allb == b).sum(), 1) for b in range(NBINS)])
        body = dens[:8].mean(); tail = dens[8:].mean()
        print(f"{label:18s} |" + "".join(f"{d*1000:4.0f}" for d in dens) + f" | {tail/max(body,1e-9):6.2f}x")
    print("\nVERDICT: both long songs show the tail wash => LENGTH; only Bye Bye => SONG artifact.")


if __name__ == '__main__':
    main()

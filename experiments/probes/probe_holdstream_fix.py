#!/usr/bin/env python3
"""A/B prototype: does `hold_stream_penalty` fix the HOLD-IN-STREAM defect without side effects?

The stream-holdjack probe found the ROOT of the user's "hold with a jack sequence": the type head opens hold-heads
in dense STREAM sections a human keeps hold-free (gen 18% of stream frames vs real ~0%), and the pinned foot then
forces jacks (no_cross_during_hold + fatigue). Fix = `hold_stream_penalty` (typed_model.generate): subtract
penalty * local_onset_density from the hold-head logit -> suppress holds in dense sections, leave SPARSE-section
holds (the real, musical ones) untouched. Decoupled from onset/tau (changes tap-vs-hold ONLY).

This A/B sweeps the penalty and checks, paired per song vs baseline (penalty=0):
  TARGET   hold_in_stream   : gen hold-rate in real-stream frames -> should DROP toward real (~0).
  WIN      freefoot_jack    : free-foot jack rate while a hold pins the other foot -> should DROP (fewer holds).
  GUARD    hold_burst       : one-foot CROSS during a hold (bipedal) -> must NOT rise (don't trade jacks for crosses).
  GUARD    hold_in_sparse   : gen hold-rate in NON-stream frames -> should hold ~flat (keep the musical holds).
  CONTEXT  overall_hold_rate / overall_jack_rate.

Usage: python probe_holdstream_fix.py --data_dir data/ --audio_dir data/ --n 30 --k 2 --penalties 0,3,6
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, sys, csv
from pathlib import Path
import numpy as np, torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(PROJECT_ROOT / 'experiments' / 'realism_critic'))
from src.utils.reproducibility import set_seed
from choreography_metrics import note_starts                                          # noqa: E402
from bipedal_metrics import pinned_mask, stats as bipedal_stats                        # noqa: E402
from probe_quality_features import (load_val_dataset, build_songs, canonical_gen_typed,  # noqa: E402
                                    load_generator, DEPLOYED_CHECKPOINT)
from probe_stream_holdjack import (real_stream_windows, single_panel,                  # noqa: E402
                                   freefoot_during_hold)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--checkpoint', default=DEPLOYED_CHECKPOINT)
    p.add_argument('--n', type=int, default=30); p.add_argument('--k', type=int, default=2)
    p.add_argument('--max_len', type=int, default=768); p.add_argument('--difficulty', type=int, default=3)
    p.add_argument('--min_len', type=int, default=6); p.add_argument('--max_gap', type=int, default=2)
    p.add_argument('--penalties', default='0,3,6')                # baseline first, then the fix doses
    p.add_argument('--out', default='outputs/probe_results/holdstream_fix.csv')
    return p.parse_args()


def local_density(gen, win=16):
    """Local onset fraction per frame (matches the model's stream_gate pre-floor)."""
    on = note_starts(gen).any(1).astype(float)
    k = np.ones(win) / win
    return np.convolve(on, k, mode='same')


def measure(gen, spans, floor=0.25):
    """All A/B metrics for one generated typed chart, given the song's real-stream frame spans."""
    pin = pinned_mask(gen); sp, _ = single_panel(gen); T = gen.shape[0]
    inwin = np.zeros(T, bool)
    for a, b in spans:
        inwin[a:b + 1] = True
    hold_in_stream = pin[inwin].any(1).mean() if inwin.any() else np.nan
    hold_in_sparse = pin[~inwin].any(1).mean() if (~inwin).any() else np.nan
    # PRINCIPLED guard: hold rate in genuinely LOW-density frames (density<floor, where the gate is exactly 0 ->
    # these holds MUST survive) vs HIGH-density frames (density>=floor, the targeted streams).
    dens = local_density(gen)
    lo, hi = dens < floor, dens >= floor
    hold_in_lowdens = pin[lo].any(1).mean() if lo.any() else np.nan
    hold_in_highdens = pin[hi].any(1).mean() if hi.any() else np.nan
    fj, fm = freefoot_during_hold(gen); freefoot_jack = fj / (fj + fm) if (fj + fm) else np.nan
    hb = bipedal_stats(gen); hold_burst = hb['hold_burst'] if np.isfinite(hb['hold_burst']) else 0.0
    ns = note_starts(gen); n_notes = int(ns.sum()); n_heads = int(((gen == 2) | (gen == 4)).sum())
    # overall jack rate over consecutive single-onset pairs
    onsets = [t for t in range(T) if ns[t].any()]; jack = cross = 0
    for i in range(1, len(onsets)):
        t0, t1 = onsets[i - 1], onsets[i]; p0, p1 = sp[t0], sp[t1]
        if p0 >= 0 and p1 >= 0 and (t1 - t0) <= 4:
            jack += (p0 == p1); cross += (p0 != p1)
    return {'hold_in_stream': hold_in_stream, 'hold_in_lowdens': hold_in_lowdens, 'hold_in_highdens': hold_in_highdens,
            'hold_in_sparse': hold_in_sparse, 'freefoot_jack': freefoot_jack,
            'hold_burst': hold_burst, 'hold_rate': n_heads / n_notes if n_notes else 0.0,
            'jack_rate': jack / (jack + cross) if (jack + cross) else np.nan}


def main():
    args = parse_args(); set_seed(args.seed)
    penalties = [float(x) for x in args.penalties.split(',')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device} | n={args.n} K={args.k} | penalties={penalties}")
    val_ds = load_val_dataset(args.data_dir, args.audio_dir, args.seed)
    songs = build_songs(val_ds, args.n, args.difficulty, args.max_len)
    model = load_generator(args.checkpoint, 42, device)
    print(f"songs={len(songs)}\n")

    METS = ['hold_in_stream', 'hold_in_highdens', 'hold_in_lowdens', 'hold_in_sparse', 'freefoot_jack',
            'hold_burst', 'hold_rate', 'jack_rate']
    # per (penalty, metric) -> list of per-song means (paired across penalties by song index)
    data = {pen: {m: [] for m in METS} for pen in penalties}
    rows = []
    for i, s in enumerate(songs, 1):
        spans = real_stream_windows(s['real_typed'], args.min_len, args.max_gap)
        if not spans:
            continue
        row = {'title': s['title'], 'bpm': s['bpm']}
        for pen in penalties:
            ov = {'hold_stream_penalty': pen} if pen else None
            acc = {m: [] for m in METS}
            for _ in range(args.k):
                gen = canonical_gen_typed(model, s, device, decode_overrides=ov)
                mm = measure(gen, spans)
                for m in METS:
                    acc[m].append(mm[m])
            for m in METS:
                v = float(np.nanmean(acc[m])); data[pen][m].append(v); row[f'p{pen}_{m}'] = v
        rows.append(row)
        b, f = data[penalties[0]], data[penalties[-1]]
        print(f"  [{i}/{len(songs)}] {s['title'][:22]:22s} hold_in_stream {b['hold_in_stream'][-1]:.2f}"
              f"->{f['hold_in_stream'][-1]:.2f}  freefoot_jack {b['freefoot_jack'][-1]:.2f}->{f['freefoot_jack'][-1]:.2f}"
              f"  hold_burst {b['hold_burst'][-1]:.2f}->{f['hold_burst'][-1]:.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} songs)\n")

    def perm_paired(d, n=10000):  # one-sided (mean<0, i.e. a DROP) sign-flip permutation
        d = np.asarray(d, float); d = d[np.isfinite(d)]; d = d[d != 0]
        if len(d) < 6:
            return float(np.mean(d)) if len(d) else np.nan, np.nan
        rng = np.random.default_rng(0); obs = float(np.mean(d))
        null = np.array([np.mean(d * rng.choice([-1, 1], len(d))) for _ in range(n)])
        return obs, float((null <= obs).mean())     # p that a drop this large is chance

    base = penalties[0]
    print("=" * 90)
    print(f"  ARM MEANS (paired, {len(rows)} songs)     [baseline penalty={base}]")
    print("=" * 90)
    hdr = "  metric            " + "".join(f"  pen={pen:<6}" for pen in penalties)
    print(hdr); print("-" * len(hdr))
    for m in METS:
        line = f"  {m:16s}" + "".join(f"  {np.nanmean(data[pen][m]):<10.3f}" for pen in penalties)
        print(line)
    print("\n  PAIRED DELTAS vs baseline (want: hold_in_stream & freefoot_jack DROP; hold_burst & hold_in_sparse ~flat):")
    for pen in penalties[1:]:
        print(f"  --- penalty {pen} ---")
        for m in METS:
            d = [f - b for f, b in zip(data[pen][m], data[base][m])]
            obs, p = perm_paired(d)
            direction = 'DROP' if obs < 0 else 'rise'
            note = ''
            if m in ('hold_in_stream', 'hold_in_highdens', 'freefoot_jack'):
                note = ('  <- TARGET ' + ('✓' if (np.isfinite(p) and p < 0.05 and obs < 0) else '(n.s.)'))
            if m == 'hold_burst':   # a RISE in one-foot crosses = regression (traded jacks for crosses)
                note = '  <- GUARD ' + ('REGRESSED' if obs > 0.02 else 'OK')
            if m == 'hold_in_lowdens':   # PRINCIPLED guard: low-density holds must SURVIVE (gate is 0 there)
                note = '  <- GUARD ' + ('REGRESSED (lost musical holds)' if obs < -0.015 else 'OK')
            print(f"    {m:16s} Δ={obs:+.3f} ({direction}, p={p:.3f}){note}")


if __name__ == '__main__':
    main()

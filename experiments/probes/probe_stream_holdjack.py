#!/usr/bin/env python3
"""Does the generator substitute HOLD+JACK where a human charts a STREAM? (user by-ear observation, 2026-07)

The claim: in sections a human charts as a flowing STREAM (alternating single taps, two feet), the model instead
opens a HOLD (pins a foot) and JACKS (same-panel repeats) on the free foot. This is INVISIBLE to the existing
battery: hold_burst counts one-foot CROSSES during a hold (dist>=1.4); a jack is dist-0, so hold+jack scores ZERO
on hold_burst.

MECHANISM (two decode forces that both fire ONLY when a hold is open -> convert an intended stream into a jack):
  1. no_cross_during_hold (typed_model.py): at a 16th gap it HARD-FORBIDS different-panel singles and ALLOWS the
     jack. So a streaming cross under a hold is -inf; the jack is the surviving legal move.
  2. fatigue governor (foot_fatigue_design.md:170): a one-foot WIDE stream costs travel_weight*dist > jack_weight,
     so it PREFERS the jack during a hold.
So the HOLD is the trigger; the jack is downstream. Root question: WHY is a hold open where a human streams? =
the TYPE head opening a hold-head in a stream section (a positional mis-placement a GLOBAL holdrate averages away
-- which is why the global holdrate-vs-bpm probe came back null; this is a different, positional question).

DESIGN (experiment-design Rule 5 = align to what REAL does; Rule 1 = the metric must see the LOCAL property):
  - Find REAL STREAM windows: maximal runs of >=min_len consecutive real single-TAP onsets, ALTERNATING panel,
    gap<=max_gap frames (8th/16th), NO holds. "A human streams here."
  - At the SAME frames, measure GEN:  hold_frac (did gen open a hold there?), jack_rate (same-panel onset pairs),
    holdjack_rate (jack WHILE a hold is open = the exact substitution).
  - CONTRAST in-stream vs OUT-of-stream (gen's own jack rate elsewhere): elevation IN stream windows = a positional
    substitution, NOT the already-known GLOBAL jack-heaviness (jack_heaviness_findings.md).
  - CAUSAL CHAIN test: within stream windows, gen jack rate | hold-open  vs  | no-hold. If jacking is much higher
    when gen has a hold open, the (mis-placed) hold is what triggers the jack (the guard/governor mechanism above).
  - CONTROL: real's own jack/hold rate in its stream windows ~ 0 (by construction) -- sanity that windows are clean.

Generation = the DEPLOYED canonical decode via the shared helpers (Rule 14). Cheap: n~40, K~2 (positional metric
pools many windows/song).

Usage: python probe_stream_holdjack.py --data_dir data/ --audio_dir data/ --difficulty 3 --n 40 --k 2
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, sys, csv
from pathlib import Path
import numpy as np, torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(1, str(Path(__file__).resolve().parent))  # probes dir (sibling probe imports)
sys.path.insert(0, str(PROJECT_ROOT / 'experiments' / 'realism_critic'))
from src.utils.reproducibility import set_seed
from choreography_metrics import note_starts                                          # noqa: E402
from bipedal_metrics import pinned_mask, foot_moves                                    # noqa: E402
from probe_quality_features import (load_val_dataset, build_songs, canonical_gen_typed,  # noqa: E402
                                    spearman, load_generator, DEPLOYED_CHECKPOINT)

DIFF_NAMES = ['Beginner', 'Easy', 'Medium', 'Hard']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--checkpoint', default=DEPLOYED_CHECKPOINT)
    p.add_argument('--n', type=int, default=40); p.add_argument('--k', type=int, default=2)
    p.add_argument('--max_len', type=int, default=768); p.add_argument('--difficulty', type=int, default=3)
    p.add_argument('--min_len', type=int, default=6)   # a stream = >=6 alternating single-tap onsets
    p.add_argument('--max_gap', type=int, default=2)   # 8th (2) / 16th (1) spacing; not slower
    p.add_argument('--out', default='outputs/probe_results/stream_holdjack.csv')
    return p.parse_args()


def single_panel(typed):
    """(T,) panel index for frames with EXACTLY ONE note-start; -1 if none, -2 if a jump (>=2)."""
    ns = note_starts(typed); T = ns.shape[0]; out = np.full(T, -1)
    for t in range(T):
        w = np.where(ns[t])[0]
        out[t] = w[0] if len(w) == 1 else (-2 if len(w) >= 2 else -1)
    return out, ns.any(1)


def freefoot_during_hold(typed):
    """HOLD-CENTRIC (the PRIMARY test of 'a hold WITH a jack sequence'): when a hold pins one foot, does the FREE
    foot JACK (same panel, dist 0) or ALTERNATE/stream (different panel) on consecutive notes? Uses EVERY hold as
    data (not just holds inside a real-stream window). The jack-analog of hold_burst (which only counts crosses).
    Returns (jack, move) counts over free-foot consecutive-note pairs while the other foot is pinned."""
    moves = foot_moves(typed); jack = move = 0
    for f in (0, 1):
        seq = sorted(moves[f], key=lambda x: x[1])
        for i in range(1, len(seq)):
            (p0, t0, _), (p1, t1, op1) = seq[i - 1], seq[i]
            if op1 and 0 < (t1 - t0) <= 4:      # other foot pinned by a hold, playable spacing
                if p0 == p1:
                    jack += 1
                else:
                    move += 1
    return jack, move


def real_stream_windows(real, min_len, max_gap):
    """Maximal runs of consecutive real single-TAP onsets, alternating panel, gap<=max_gap, NO hold.
    Returns list of (t_start, t_end) frame spans."""
    sp, _ = single_panel(real); T = real.shape[0]
    onsets = [(t, sp[t]) for t in range(T) if (real[t] != 0).any()]   # (frame, panel/-1/-2) in time order
    is_tap = lambda t, p: p >= 0 and real[t, p] == 1                   # pure tap single (symbol 1)
    windows, run = [], []
    def flush():
        if len(run) >= min_len:
            windows.append((run[0][0], run[-1][0]))
    for (t, p) in onsets:
        if not run:
            run = [(t, p)] if is_tap(t, p) else []
            continue
        pt, pp = run[-1]
        if is_tap(t, p) and p != pp and (t - pt) <= max_gap:
            run.append((t, p))
        else:
            flush(); run = [(t, p)] if is_tap(t, p) else []
    flush()
    return windows


def gen_window_stats(gen, spans, pin, sp):
    """Pooled gen behavior INSIDE the given (real-stream) frame spans."""
    inwin = np.zeros(gen.shape[0], bool)
    for a, b in spans:
        inwin[a:b + 1] = True
    hold_frames = int(pin[inwin].any(1).sum()); tot_frames = int(inwin.sum())
    # onset pairs within windows: classify jack (same panel) vs cross (diff), + whether a hold is open at the 2nd note
    onsets = [t for t in range(gen.shape[0]) if inwin[t] and (gen[t] != 0).any()]
    jack = cross = hj_jack = hj_pairs = nohold_jack = nohold_pairs = 0
    for i in range(1, len(onsets)):
        t0, t1 = onsets[i - 1], onsets[i]
        p0, p1 = sp[t0], sp[t1]
        if p0 >= 0 and p1 >= 0 and (t1 - t0) <= 4:          # two consecutive single presses, playable spacing
            is_jack = (p0 == p1)
            jack += is_jack; cross += (not is_jack)
            if pin[t1].any():                                # a hold open at the 2nd note (the substitution regime)
                hj_pairs += 1; hj_jack += is_jack
            else:
                nohold_pairs += 1; nohold_jack += is_jack
    return dict(hold_frames=hold_frames, tot_frames=tot_frames, jack=jack, cross=cross,
                hj_jack=hj_jack, hj_pairs=hj_pairs, nohold_jack=nohold_jack, nohold_pairs=nohold_pairs)


def gen_outstream_jack(gen, spans, sp):
    """Gen jack rate OUTSIDE the real-stream windows (the baseline: is jacking ELEVATED in-stream?)."""
    inwin = np.zeros(gen.shape[0], bool)
    for a, b in spans:
        inwin[a:b + 1] = True
    onsets = [t for t in range(gen.shape[0]) if (not inwin[t]) and (gen[t] != 0).any()]
    jack = cross = 0
    for i in range(1, len(onsets)):
        t0, t1 = onsets[i - 1], onsets[i]; p0, p1 = sp[t0], sp[t1]
        if p0 >= 0 and p1 >= 0 and (t1 - t0) <= 4:
            jack += (p0 == p1); cross += (p0 != p1)
    return jack, cross


def rate(a, b):
    return a / b if b else np.nan


def main():
    args = parse_args(); set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device} | {DIFF_NAMES[args.difficulty]} | n={args.n} K={args.k} "
          f"| stream>=%d onsets, gap<=%d" % (args.min_len, args.max_gap))

    val_ds = load_val_dataset(args.data_dir, args.audio_dir, args.seed)
    songs = build_songs(val_ds, args.n, args.difficulty, args.max_len)
    print(f"songs={len(songs)}\n")
    model = load_generator(args.checkpoint, 42, device)

    rows = []
    for i, s in enumerate(songs, 1):
        spans = real_stream_windows(s['real_typed'], args.min_len, args.max_gap)
        if not spans:
            print(f"  [{i}/{len(songs)}] {s['title'][:24]:24s} -- no real stream windows, skip")
            continue
        stream_frames = sum(b - a + 1 for a, b in spans)
        # pool gen behavior over K generations
        acc = None; out_j = out_c = 0; g_ffj = g_ffm = 0
        for _ in range(args.k):
            gen = canonical_gen_typed(model, s, device)
            pin = pinned_mask(gen); sp, _ = single_panel(gen)
            st = gen_window_stats(gen, spans, pin, sp)
            acc = st if acc is None else {k: acc[k] + st[k] for k in acc}
            oj, oc = gen_outstream_jack(gen, spans, sp); out_j += oj; out_c += oc
            fj, fm = freefoot_during_hold(gen); g_ffj += fj; g_ffm += fm       # HOLD-CENTRIC (all holds)
        # real's own behavior in ITS stream windows (control ~ 0)
        rpin = pinned_mask(s['real_typed']); rsp, _ = single_panel(s['real_typed'])
        rst = gen_window_stats(s['real_typed'], spans, rpin, rsp)
        r_ffj, r_ffm = freefoot_during_hold(s['real_typed'])                    # HOLD-CENTRIC real reference
        rows.append({
            'title': s['title'], 'bpm': s['bpm'], 'n_windows': len(spans), 'stream_frames': stream_frames,
            # HOLD-CENTRIC (PRIMARY): free-foot jack fraction while the other foot is pinned by a hold, gen vs real
            'gen_freefoot_jack': rate(g_ffj, g_ffj + g_ffm), 'real_freefoot_jack': rate(r_ffj, r_ffm + r_ffj),
            'gen_ff_pairs': g_ffj + g_ffm, 'real_ff_pairs': r_ffj + r_ffm,
            'gen_hold_in_stream': rate(acc['hold_frames'], acc['tot_frames']),        # mis-placed hold rate
            'gen_jack_in_stream': rate(acc['jack'], acc['jack'] + acc['cross']),       # jack substitution
            'gen_jack_out_stream': rate(out_j, out_j + out_c),                          # baseline elsewhere
            'gen_jack_when_hold': rate(acc['hj_jack'], acc['hj_pairs']),               # causal: jack | hold open
            'gen_jack_no_hold': rate(acc['nohold_jack'], acc['nohold_pairs']),         # jack | no hold, in-stream
            'gen_holdjack_frac': rate(acc['hj_jack'], acc['jack'] + acc['cross']),     # the exact hold+jack event
            'real_hold_in_stream': rate(rst['hold_frames'], rst['tot_frames']),        # control ~0
            'real_jack_in_stream': rate(rst['jack'], rst['jack'] + rst['cross']),      # control ~0
            'hj_pairs': acc['hj_pairs'], 'nohold_pairs': acc['nohold_pairs'],
        })
        r = rows[-1]
        print(f"  [{i}/{len(songs)}] {s['title'][:22]:22s} win={len(spans):2d} "
              f"gen: hold_in_str={r['gen_hold_in_stream']:.2f} jack_in={r['gen_jack_in_stream']:.2f} "
              f"jack_out={r['gen_jack_out_stream']:.2f} | jack|hold={r['gen_jack_when_hold']:.2f} "
              f"jack|nohold={r['gen_jack_no_hold']:.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} songs with stream windows)")

    def col(k):
        return np.array([r[k] for r in rows], float)
    def summ(name, k):
        v = col(k); v = v[np.isfinite(v)]
        return f"  {name:34s} mean={np.mean(v):.3f}  median={np.median(v):.3f}  (n={len(v)})"

    def wilcoxon(a, b):
        """Paired one-sided (a>b) sign-rank via a permutation of signs (no scipy dep)."""
        d = a - b; d = d[np.isfinite(d)]; d = d[d != 0]
        if len(d) < 6:
            return np.nan, np.nan
        obs = np.sum(d[d > 0]) if False else float(np.mean(d))
        rng = np.random.default_rng(0); n = len(d); null = []
        for _ in range(10000):
            null.append(np.mean(d * rng.choice([-1, 1], n)))
        null = np.array(null); return obs, float((null >= obs).mean())

    print("\n" + "=" * 84)
    print("  PRIMARY (hold-centric): when a HOLD pins one foot, does the FREE foot JACK (gen) or STREAM (real)?")
    print("=" * 84)
    print(summ("gen free-foot JACK | hold open", 'gen_freefoot_jack') + "   <- 'a hold WITH a jack sequence'")
    print(summ("real free-foot JACK | hold open", 'real_freefoot_jack') + "   <- humans alternate/stream instead")
    gfj, rfj = col('gen_freefoot_jack'), col('real_freefoot_jack')
    obs0, p0 = wilcoxon(gfj, rfj)
    print(f"  mean(gen - real) = {obs0:+.3f}, one-sided perm p={p0:.3f}"
          f"  {'GEN JACKS the free foot where humans STREAM' if (np.isfinite(p0) and p0 < 0.05) else 'no gap'}")
    print(f"  (pooled free-foot-during-hold pairs: gen {int(np.nansum(col('gen_ff_pairs')))}, "
          f"real {int(np.nansum(col('real_ff_pairs')))})")

    print("\n" + "=" * 84)
    print("  SECONDARY (positional): does GEN open holds/jacks in frames where REAL charts a clean stream?")
    print("=" * 84)
    print(summ("gen HOLD rate in real-stream", 'gen_hold_in_stream') + "   <- mis-placed holds (real~0)")
    print(summ("  (real hold rate, control)", 'real_hold_in_stream'))
    print(summ("gen JACK rate in real-stream", 'gen_jack_in_stream'))
    print(summ("gen JACK rate OUT of stream", 'gen_jack_out_stream') + "   <- baseline (global jack-heaviness)")
    print(summ("  (real jack rate, control)", 'real_jack_in_stream'))

    ji, jo = col('gen_jack_in_stream'), col('gen_jack_out_stream')
    obs, p = wilcoxon(ji, jo)
    print(f"\n  ELEVATION (positional, not global): mean(jack_in - jack_out) = {obs:+.3f}, one-sided perm p={p:.3f}"
          f"  {'ELEVATED in stream' if (np.isfinite(p) and p < 0.05) else 'not elevated'}")

    jh, jn = col('gen_jack_when_hold'), col('gen_jack_no_hold')
    obs2, p2 = wilcoxon(jh, jn)
    print("\n  CAUSAL CHAIN — inside real-stream windows, is gen's jacking TRIGGERED by an open hold?")
    print(summ("    gen jack | HOLD open", 'gen_jack_when_hold'))
    print(summ("    gen jack | no hold  ", 'gen_jack_no_hold'))
    print(f"    mean(jack|hold - jack|nohold) = {obs2:+.3f}, one-sided perm p={p2:.3f}"
          f"  {'HOLD triggers the jack' if (np.isfinite(p2) and p2 < 0.05) else 'no hold-jack coupling'}")
    tot_hj = int(np.nansum(col('hj_pairs'))); tot_nh = int(np.nansum(col('nohold_pairs')))
    print(f"    (pooled onset-pairs in-stream: {tot_hj} under a hold, {tot_nh} not)")

    # does the substitution scale with BPM? (ties back to the fast-song defect)
    bpm = col('bpm')
    print(f"\n  vs BPM: Spearman(bpm, gen_holdjack_frac) = {spearman(col('gen_holdjack_frac'), bpm):+.3f}   "
          f"Spearman(bpm, gen_hold_in_stream) = {spearman(col('gen_hold_in_stream'), bpm):+.3f}")


if __name__ == '__main__':
    main()

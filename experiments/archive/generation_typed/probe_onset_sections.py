#!/usr/bin/env python3
"""PROBE 3: section-level onset CAUSALITY + density MISALLOCATION (the user's playtest complaints).

Two questions the prior onset work never tested (it measured placement quality and jacks SEPARATELY):

 3A. CAUSAL: is the head's jack-heaviness "the pattern head making the best of a bad situation" -- i.e.
     does the ONSET head's blocky rhythm CAUSE the jacks? Decompose (not just correlate): per section bin
     by onset-BLOCKINESS (mean consecutive-onset-run length, gap<=4), then compare JACKINESS (mean
     same-panel-run length) of MODEL vs REAL within each bin.
       - model has MORE high-blockiness sections than real -> the ONSET head hands over blocky rhythm.
       - at the SAME blockiness model jacks MORE than real -> the PATTERN head adds excess on its own.

 3B. MISALLOCATION: "awkward sections over-noted, active sections empty." Per section correlate MODEL
     density vs REAL density (does it put notes where the human did?) and each vs the model's own p_onset
     (audio salience). Report over/under-placed section shares.

Native decode (own onset head, radar-conditioned, density matched to REAL, governor OFF), so this is the
deployed onset head's behavior. Section = fixed window of --win frames (default 32 = 2 beats). Pooled by
difficulty (Rule 12; lean Med+Hard, Easy n=1).

  python experiments/generation_typed/probe_onset_sections.py [--songs 16] [--win 32]
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.typed import pair_holds
from src.generation.playability_metrics import ACTIVE_SYMBOLS
from src.generation.playtest_export import enforce_playability
from compare_foot_physics import load_songs, DIFF_NAMES


def _run_len_per_pos(positions, same_panel_of=None, gap=4):
    """For sorted onset frame indices, return an array giving the run length each position belongs to.
    A run = consecutive positions within `gap` frames; if same_panel_of given, also same panel (jacks)."""
    n = len(positions)
    out = np.ones(n, dtype=float)
    if n == 0:
        return out
    i = 0
    while i < n:
        j = i
        while (j + 1 < n and positions[j + 1] - positions[j] <= gap
               and (same_panel_of is None or same_panel_of[j + 1] == same_panel_of[i])):
            j += 1
        out[i:j + 1] = j - i + 1
        i = j + 1
    return out


def section_stats(chart):
    """Per-onset (frame) arrays: blockiness (consecutive-onset run len) and jackiness (same-panel run len,
    singles only; jumps get run length 1). Returns (frames, blockiness, jackiness)."""
    chart = np.asarray(chart)
    active = [(t, [k for k in range(chart.shape[1]) if chart[t, k] in ACTIVE_SYMBOLS]) for t in range(chart.shape[0])]
    active = [(t, a) for t, a in active if a]
    if not active:
        return np.array([]), np.array([]), np.array([])
    frames = np.array([t for t, _ in active])
    block = _run_len_per_pos(frames)                                  # onset-run length (panel-agnostic)
    panel = np.array([a[0] if len(a) == 1 else -1 for _, a in active])  # single panel, or -1 for jumps
    jack = _run_len_per_pos(frames, same_panel_of=panel)             # same-panel run (jumps break runs)
    jack[panel < 0] = 1.0                                            # a jump is not a jack
    return frames, block, jack


def gen_native(model, song, device):
    real = np.asarray(song['real'])
    tgt = float(np.isin(real, ACTIVE_SYMBOLS).any(1).mean())
    T = song['len']; bpm = song['bpm']
    audio = torch.from_numpy(song['audio']).unsqueeze(0).to(device)
    dt = torch.tensor([song['diff']], device=device)
    radar_t = torch.from_numpy(song['radar']).unsqueeze(0).to(device)
    with torch.no_grad():
        memory = model.encode_audio(audio)
        p = torch.sigmoid(model.onset_logits(memory, dt, radar=radar_t)[0]).cpu().numpy()
    tau = float(np.quantile(p, 1 - tgt)) if tgt > 0 else 0.5
    gk = dict(onset_threshold=tau, lengths=torch.tensor([T], device=device), type_sample=True,
              type_temperature=0.4, pattern_sample=True, pattern_temperature=0.7, max_jack_run=2,
              bpm=bpm, radar=radar_t, fatigue_penalty=None)
    enforce_playability(gk); set_seed(42)
    with torch.no_grad():
        g = model.generate(audio, dt, **gk)[0].cpu().numpy()
    return pair_holds(g), real, p[:T]


def windowed(frames, vals, T, win):
    """Mean of per-onset `vals` within each window (NaN if no onsets), aligned to window index."""
    nW = (T + win - 1) // win
    out = np.full(nW, np.nan)
    if len(frames):
        wid = (frames // win).astype(int)
        for w in range(nW):
            m = wid == w
            if m.any():
                out[w] = vals[m].mean()
    return out


def win_density(chart, T, win):
    press = np.isin(np.asarray(chart), ACTIVE_SYMBOLS).any(1)
    nW = (T + win - 1) // win
    return np.array([press[w * win:(w + 1) * win].mean() for w in range(nW)])


def pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) < 1e-9 or np.std(b[m]) < 1e-9:
        return float('nan')
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/gen_motif_full_fixed/best_val.pt")
    ap.add_argument("--songs", type=int, default=16)
    ap.add_argument("--by", default="rich")
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--win", type=int, default=32)
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    songs = load_songs(args, device)
    audio_dim = songs[0]['audio'].shape[1]
    model = LayeredTypedChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)['model_state_dict']); model.eval()

    # collect per-window rows by difficulty
    rows = defaultdict(lambda: {"mblock": [], "mjack": [], "rblock": [], "rjack": [],
                                "mdens": [], "rdens": [], "pon": []})
    for s in songs:
        g, real, p = gen_native(model, s, device); T = s['len']
        for chart, bk, jk in ((g, "mblock", "mjack"), (real, "rblock", "rjack")):
            fr, block, jack = section_stats(chart)
            rows[s['diff']][bk].extend(windowed(fr, block, T, args.win))
            rows[s['diff']][jk].extend(windowed(fr, jack, T, args.win))
        rows[s['diff']]["mdens"].extend(win_density(g, T, args.win))
        rows[s['diff']]["rdens"].extend(win_density(real, T, args.win))
        rows[s['diff']]["pon"].extend([p[w * args.win:(w + 1) * args.win].mean()
                                       for w in range((T + args.win - 1) // args.win)])

    print(f"\nPROBE 3 — SECTION-LEVEL onset causality + misallocation  {len(songs)} songs, win={args.win} frames")
    print("native deployed onset head, density matched to REAL, governor OFF.\n")

    for d in sorted(rows):
        r = {k: np.array(v, dtype=float) for k, v in rows[d].items()}
        n = len(r["mdens"])
        if n == 0:
            continue
        print(f"================ {DIFF_NAMES[d]} ({n} windows) ================")

        # --- 3A: decompose jacks — control for LOCAL DENSITY (onset-run saturates in rich charts) ---
        # At MATCHED local density: model jackiness ~ real -> jacks are onset-amount-driven (onset head /
        # best-of-bad-situation); model > real at every density -> the PATTERN head adds excess jacks.
        print("3A CAUSAL — jackiness (mean same-panel run len) controlled for LOCAL DENSITY:")
        print(f"   overall jackiness: model {np.nanmean(r['mjack']):.2f}  vs  real {np.nanmean(r['rjack']):.2f}"
              f"   (model jacks more?)")
        print(f"   corr(local density, jackiness): model {pearson(r['mdens'], r['mjack']):+.2f}  "
              f"real {pearson(r['rdens'], r['rjack']):+.2f}   (denser -> jackier?)")
        edges = [0.0, 0.2, 0.35, 0.5, 1.01]                           # local-density bins
        print(f"   {'density bin':>14} {'modelJack':>10} {'realJack':>9} {'mN':>5} {'rN':>5}")
        for lo, hi in zip(edges[:-1], edges[1:]):
            mm = (r["mdens"] >= lo) & (r["mdens"] < hi) & np.isfinite(r["mjack"])
            rm = (r["rdens"] >= lo) & (r["rdens"] < hi) & np.isfinite(r["rjack"])
            mj = r["mjack"][mm].mean() if mm.any() else float('nan')
            rj = r["rjack"][rm].mean() if rm.any() else float('nan')
            print(f"   {f'[{lo},{hi})':>14} {mj:>10.2f} {rj:>9.2f} {int(mm.sum()):>5} {int(rm.sum()):>5}")

        # --- 3B: density misallocation vs real + audio salience ---
        print("3B MISALLOC — section density: does the model put notes where REAL/audio do?")
        print(f"   corr(model_dens, real_dens) = {pearson(r['mdens'], r['rdens']):+.2f}   (1=perfect allocation)")
        print(f"   corr(model_dens, p_onset)   = {pearson(r['mdens'], r['pon']):+.2f}   "
              f"vs corr(real_dens, p_onset) = {pearson(r['rdens'], r['pon']):+.2f}")
        diff = r["mdens"] - r["rdens"]
        over = float((diff > 0.15).mean()); under = float((diff < -0.15).mean())
        print(f"   sections over-placed (model-real > .15): {100*over:.0f}%   "
              f"under-placed (< -.15): {100*under:.0f}%   |mean abs misalloc| {np.nanmean(np.abs(diff)):.3f}")
        print()

    print("READ 3A: at MATCHED local density, if model jackiness > real in every bin -> the PATTERN head adds "
          "excess jacks (not just best-of-a-bad-situation). If model~real per bin and model only differs in the "
          "density MIX -> it's onset-amount driven. READ 3B: low corr(model,real) quantifies misallocation; model "
          "tracking p_onset MORE than real -> chasing audio salience where the human used musical judgement.")


if __name__ == "__main__":
    main()

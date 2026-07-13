#!/usr/bin/env python3
"""PROBE 4: does the FATIGUE GOVERNOR let pattern_temperature go higher?

The H2 "cap pattern_temperature at 0.6-0.85 for arrow coherence" finding PREDATES the per-foot fatigue
governor (notes/foot_fatigue_design.md: exertion with EXPONENTIAL decay; a sustained jack escalates the
penalty). Hypothesis (user): the governor now supplies the coherence floor H2's cap was protecting, so
temperature can rise -- reducing the head's jack-heaviness and adding jumps (Probe 1) -- WITHOUT the
incoherence blowup, because the governor's exponential jack penalty + per-foot fatigue catch it.

2-D contrast (experiment-design Rule 11 — change one thing per axis): pattern_temperature × governor
{OFF (fatigue_penalty=None), ON (fatigue_penalty=2, the release default)}. Native decode adhering to
conditioning-mechanics: own onset head, radar-conditioned, tau from the SAME conditioned logits (§6),
density matched to REAL, bpm passed so the governor is not silent (§8b). BOTH arms keep the MANDATORY
max_jack_run=2 hard cap; the swept governor knob is the SOFT exponential fatigue_penalty.

Metrics: jackDist (len2/len3/≥4 vs real) + maxRun (jack tail), jump% + mJumpStrm (Probe 1 said temp adds
jumps), and a panel-transition ENTROPY "scramble" proxy (higher = more random arrows = the H2 incoherence
the cap protected). Density is the isolation check. COHERENCE IS ULTIMATELY BY-EAR (Rule 8) -> --export_sm.

  python experiments/generation_typed/probe_temp_governor.py [--songs 16] [--temps 0.7,0.85,1.0,1.2,1.5] [--export_sm 2]
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
from src.generation.playability_metrics import chart_metrics, same_panel_run_lengths, run_length_shares, ACTIVE_SYMBOLS
from src.generation.playtest_export import enforce_playability
from src.generation.sm_writer import tensor_to_sm
from compare_foot_physics import load_songs, summarize, DIFF_NAMES
from compare_native import jack_distance

DIFF_NAMES_L = DIFF_NAMES


def _floats(s):
    return [float(x) for x in s.split(",") if x.strip() != ""]


def transition_entropy(chart):
    """Conditional entropy H(next panel-state | prev), over consecutive active frames (gap<=4). Higher =
    more unpredictable arrow transitions = scramblier (a rough proxy for the H2 'incoherence' a cap guards;
    real charts are structured -> lower). bits."""
    chart = np.asarray(chart)
    seq = []
    for t in range(chart.shape[0]):
        bits = sum((1 << k) for k in range(chart.shape[1]) if chart[t, k] in ACTIVE_SYMBOLS)
        if bits:
            seq.append((t, bits))
    if len(seq) < 3:
        return float('nan')
    trans = defaultdict(lambda: defaultdict(int))
    for (t0, a), (t1, b) in zip(seq[:-1], seq[1:]):
        if t1 - t0 <= 4:
            trans[a][b] += 1
    tot = sum(sum(d.values()) for d in trans.values())
    if tot == 0:
        return float('nan')
    H = 0.0
    for a, d in trans.items():
        na = sum(d.values()); pa = na / tot
        ha = -sum((c / na) * np.log2(c / na) for c in d.values())
        H += pa * ha
    return float(H)


def gen(model, song, temp, fatigue, device):
    """Native decode at pattern_temperature=temp, fatigue_penalty=fatigue (None=off). Density matched to REAL."""
    real = np.asarray(song['real'])
    tgt = float(np.isin(real, ACTIVE_SYMBOLS).any(1).mean())
    T = song['len']; bpm = song['bpm']
    audio = torch.from_numpy(song['audio']).unsqueeze(0).to(device)
    dt = torch.tensor([song['diff']], device=device)
    radar_t = torch.from_numpy(song['radar']).unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.sigmoid(model.onset_logits(model.encode_audio(audio), dt, radar=radar_t)[0]).cpu().numpy()
    tau = float(np.quantile(p, 1 - tgt)) if tgt > 0 else 0.5
    gk = dict(onset_threshold=tau, lengths=torch.tensor([T], device=device), type_sample=True,
              type_temperature=0.4, pattern_sample=True, pattern_temperature=temp, max_jack_run=2,
              bpm=bpm, radar=radar_t, fatigue_penalty=fatigue)
    enforce_playability(gk, override_reason="deliberate pattern_temperature sweep (probe 4)")
    set_seed(42)
    with torch.no_grad():
        g = model.generate(audio, dt, **gk)[0].cpu().numpy()
    return pair_holds(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/gen_motif_full_fixed/best_val.pt")
    ap.add_argument("--songs", type=int, default=16)
    ap.add_argument("--by", default="rich")
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--temps", type=_floats, default=[0.7, 0.85, 1.0, 1.2, 1.5])
    ap.add_argument("--fatigue", type=float, default=2.0)
    ap.add_argument("--export_sm", type=int, default=0, help="export this many songs at gov-ON temps 0.85/1.0/1.2")
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    songs = load_songs(args, device)
    audio_dim = songs[0]['audio'].shape[1]
    model = LayeredTypedChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)['model_state_dict']); model.eval()

    # REAL reference, by difficulty
    ref = defaultdict(lambda: {"runs": [], "m": [], "ent": []})
    for s in songs:
        real = np.asarray(s['real'])
        ref[s['diff']]["runs"].extend(same_panel_run_lengths(real))
        ref[s['diff']]["m"].append(chart_metrics(real)); ref[s['diff']]["ent"].append(transition_entropy(real))
    diffs = sorted(ref)

    arms = [("gov OFF", None), (f"gov ON(λ={args.fatigue})", args.fatigue)]
    print(f"\nPROBE 4 — pattern_temperature × FATIGUE GOVERNOR  {len(songs)} songs (by={args.by}), native")
    print("Does the governor's exponential jack penalty let temperature rise without the jack/scramble blowup?")
    print("transEnt = panel-transition entropy (bits; higher = scramblier ~ H2 incoherence). Coherence is by-ear (Rule 8).\n")

    for d in diffs:
        rm = summarize(ref[d]["m"]); rsh = run_length_shares(ref[d]["runs"])
        rent = float(np.nanmean(ref[d]["ent"]))
        print(f"================ {DIFF_NAMES_L[d]} (n={len(ref[d]['m'])}) ================")
        print(f"  REAL: len2 {100*rsh['len2_share']:.0f} len3 {100*rsh['len3_share']:.0f} >=4 {100*rsh['ge4_share']:.0f}"
              f"  maxRun {rsh['max_run']}  jump% {100*rm['jump_rate']:.0f}  mJumpStrm {rm['max_jump_stream']:.1f}  transEnt {rent:.2f}")
        print(f"  {'arm':>12} {'patT':>5} {'jackDist':>8} {'len3%':>6} {'>=4%':>6} {'maxRun':>7} {'jump%':>6} {'mJStrm':>7} {'transEnt':>8} {'dens':>6}")
        for arm_name, fat in arms:
            for temp in args.temps:
                runs, mets, ents = [], [], []
                for s in songs:
                    if s['diff'] != d:
                        continue
                    g = gen(model, s, temp, fat, device)
                    runs.extend(same_panel_run_lengths(g)); mets.append(chart_metrics(g)); ents.append(transition_entropy(g))
                sh = run_length_shares(runs); sm = summarize(mets)
                jd = jack_distance(runs, ref[d]["runs"])
                print(f"  {arm_name:>12} {temp:>5.2f} {jd:>8.3f} {100*sh['len3_share']:>6.1f} {100*sh['ge4_share']:>6.1f} "
                      f"{sh['max_run']:>7d} {100*sm['jump_rate']:>6.1f} {sm['max_jump_stream']:>7.1f} "
                      f"{np.nanmean(ents):>8.2f} {sm['density']:>6.3f}")
            print()

    # ---- export for the by-ear coherence call (Rule 8): gov-ON at candidate temps -----------------
    if args.export_sm:
        out_dir = PROJECT_ROOT / "outputs" / "probe_temp_governor"; out_dir.mkdir(parents=True, exist_ok=True)
        for si, s in enumerate(songs[:args.export_sm]):
            for temp in (0.70, 0.85, 1.00, 1.20):
                g = gen(model, s, temp, args.fatigue, device)
                sm = tensor_to_sm(g, s['bpm'], title=f"song{si}_govON_T{temp}",
                                  difficulty_name=DIFF_NAMES_L[s['diff']], typed=True)
                (out_dir / f"song{si}_govON_T{temp:.2f}.sm").write_text(sm)
        print(f"Exported gov-ON charts at T=0.7/0.85/1.0/1.2 for {args.export_sm} song(s) to {out_dir} — PLAY them (Rule 8).")

    print("\nREAD: if gov-ON holds jackDist/maxRun/transEnt ~flat (and near real) as temp rises while jumps climb"
          " toward real, the governor SUPPORTS higher temp -> raise it (then confirm coherence BY EAR). If transEnt"
          " climbs with temp even gov-ON, the governor bounds FATIGUE/jacks but not musical scramble -> temp still"
          " capped (the H2 concern survives the governor) — by-ear is the decider.")


if __name__ == "__main__":
    main()

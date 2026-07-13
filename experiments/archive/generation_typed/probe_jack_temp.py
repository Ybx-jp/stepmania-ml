#!/usr/bin/env python3
"""PROBE 1 of the jack-heaviness investigation: is the learned head's excess of long same-panel
runs a PATTERN-HEAD TEMPERATURE issue? (notes/foot_physics_baseline.md axis-split: native model_raw
makes len3 ~2x real, >=4 ~3-4x real.)

Prior (style_decoding.md): greedy = 88% jacks; pattern_temperature=1.0 dropped jack RATE to real's
0.20; the shipped default 0.7 (H2 coherence range 0.6-0.85) is LOWER -> jackier. This measures the
RUN-LENGTH distribution (the actual jackDist metric) directly across temperature.

ISOLATION (experiment-design Rule 11): ONE variable = pattern_temperature. Everything else native and
identical: own onset head (radar-conditioned, density matched to REAL), governor OFF (so the jacks are
the PATTERN HEAD's raw tendency, not the fatigue governor's doing -- a separate control), type_temperature
0.4 fixed, SAME seed per (song,temp) so only the temperature scaling of pat_logits differs. We watch
jump% too (Rule 1: a temperature change has side effects -- confirm it isn't wrecking jumps to "fix" jacks).

  python experiments/generation_typed/probe_jack_temp.py [--songs 16] [--temps 0.7,0.85,1.0,1.2,1.5]
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
from compare_foot_physics import load_songs, summarize, DIFF_NAMES
from compare_native import jack_distance


def _floats(s):
    return [float(x) for x in s.split(",") if x.strip() != ""]


def gen_at_temp(model, song, temp, device):
    """Native decode (own onsets, density matched to REAL, governor OFF) at pattern_temperature=temp."""
    real = np.asarray(song['real'])
    tgt = float(np.isin(real, ACTIVE_SYMBOLS).any(1).mean())     # match REAL press density
    T = song['len']; bpm = song['bpm']; diff = song['diff']
    audio = torch.from_numpy(song['audio']).unsqueeze(0).to(device)
    dt = torch.tensor([diff], device=device)
    radar_t = torch.from_numpy(song['radar']).unsqueeze(0).to(device)
    with torch.no_grad():
        memory = model.encode_audio(audio)
        p = torch.sigmoid(model.onset_logits(memory, dt, radar=radar_t)[0]).cpu().numpy()
    tau = float(np.quantile(p, 1 - tgt)) if tgt > 0 else 0.5
    gk = dict(onset_threshold=tau, lengths=torch.tensor([T], device=device), type_sample=True,
              type_temperature=0.4, pattern_sample=True, pattern_temperature=temp, max_jack_run=2,
              bpm=bpm, radar=radar_t, fatigue_penalty=None)        # governor OFF -> isolate the head
    enforce_playability(gk)
    set_seed(42)                                                  # identical draws across temps -> only temp differs
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
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    songs = load_songs(args, device)
    audio_dim = songs[0]['audio'].shape[1]
    model = LayeredTypedChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)['model_state_dict']); model.eval()

    # REAL reference run-length shares + jump%, by difficulty
    real_runs = defaultdict(list); real_m = defaultdict(list)
    for s in songs:
        real_runs[s['diff']].extend(same_panel_run_lengths(np.asarray(s['real'])))
        real_m[s['diff']].append(chart_metrics(np.asarray(s['real'])))
    diffs = sorted(real_runs)

    print(f"\nPATTERN-TEMPERATURE SWEEP (jack-heaviness)  {len(songs)} songs (by={args.by})")
    print("native, governor OFF (isolate the head), density matched to REAL. shipped default = 0.7.")
    print("Lower jackDist = run-length closer to real. Watch jump% for side effects (Rule 1).\n")

    for d in diffs:
        rl = run_length_shares(real_runs[d]); rj = summarize(real_m[d])['jump_rate']
        print(f"=== {DIFF_NAMES[d]} (n={len(real_m[d])})   REAL: len2 {100*rl['len2_share']:.1f}  "
              f"len3 {100*rl['len3_share']:.1f}  >=4 {100*rl['ge4_share']:.1f}  maxRun {rl['max_run']}  jump% {100*rj:.1f} ===")
        print(f"  {'patT':>5} {'len2%':>6} {'len3%':>6} {'>=4%':>6} {'maxRun':>7} {'jump%':>6} {'jackDist':>9}")
        for temp in args.temps:
            runs, mets = [], []
            for s in songs:
                if s['diff'] != d:
                    continue
                g = gen_at_temp(model, s, temp, device)
                runs.extend(same_panel_run_lengths(g)); mets.append(chart_metrics(g))
            sh = run_length_shares(runs); jr = summarize(mets)['jump_rate']
            jd = jack_distance(runs, real_runs[d])
            star = " *" if temp == 0.7 else "  "
            print(f"  {temp:>5.2f}{star}{100*sh['len2_share']:>6.1f} {100*sh['len3_share']:>6.1f} "
                  f"{100*sh['ge4_share']:>6.1f} {sh['max_run']:>7d} {100*jr:>6.1f} {jd:>9.3f}")
        print()

    print("READ: if jackDist falls monotonically as patT rises toward/above 1.0 (len3/>=4 -> real), the "
          "jack-heaviness is a PATTERN-HEAD temperature issue (shipped 0.7 too greedy). If jump% craters or "
          "arrow coherence is the cost, that's the H2 trade-off -- note it. Residual jacks at high patT -> the "
          "ONSET head (rhythm/spacing) is the next variable to isolate.")


if __name__ == "__main__":
    main()

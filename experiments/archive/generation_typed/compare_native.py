#!/usr/bin/env python3
"""NATIVE-MODE foot-physics vs learned-head comparison (fixes the onset_override OOD flaw).

`compare_foot_physics.py` forced the REAL onsets onto the model via onset_override, which puts its
pattern head OOD (a dense external onset stream it never sees in deployment) and INFLATES jacks
(maxRun 24 vs real 4 -- see the retraction in notes/foot_physics_baseline.md). Here the learned model
runs in its DEPLOYED mode instead:

  * its OWN onset head decides WHERE notes go (radar-conditioned, tau from the conditioned logits per
    conditioning-mechanics §6), with the onset COUNT matched to the real chart's press density so density
    is held for the comparison but the POSITIONS are the model's own (in-distribution -- the OOD failure
    was forcing positions, not matching a count);
  * the per-foot fatigue governor + mandatory playability on (the release default), exactly as shipped.

foot_phys is then handed the MODEL's own onset positions, so the A/B isolates PANEL CHOICE at an
in-distribution density. Metric: same-panel run-length shares + jump-stream vs REAL, stratified by
difficulty (the SAME distance_to_real as the override harness, reused -- Rule 14).

foot_phys defaults to its CALIBRATED knobs (beta=2.0, jump_bias=0.0; calib_foot_physics.py).

  python experiments/generation_typed/compare_native.py [--songs 16] [--fatigue 2.0]
         [--beta 2.0] [--jump_bias 0.0] [--export_sm 2]
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
from src.generation.baselines import FootPhysicsBaseline
from src.generation.radar_manifold import RadarManifold
from src.generation.playability_metrics import chart_metrics, same_panel_run_lengths, run_length_shares, ACTIVE_SYMBOLS
from src.generation.sm_writer import tensor_to_sm
from src.generation.playtest_export import enforce_playability
# reuse the override harness's loader + helpers so the comparisons can't drift (Rule 14)
from compare_foot_physics import load_songs, summarize, DIFF_NAMES


def jump_distance(gen_metrics, real_metrics):
    """JUMP axis = jump_rate + jump-stream length vs real. This is the KNOWN under-jump gap
    (a separate air/density thread, conditioning-mechanics §8d) -- reported, NOT the footwork verdict."""
    s, rs = summarize(gen_metrics), summarize(real_metrics)
    return (abs(s['jump_rate'] - rs['jump_rate']) / max(rs['jump_rate'], .05)
            + abs(s['max_jump_stream'] - rs['max_jump_stream']) / max(rs['max_jump_stream'], 1))


def jack_distance(gen_runs, real_runs):
    """JACK axis = full same-panel run-length distribution (len2 + len3 + >=4 shares) vs real.
    THIS is the footwork-placement-STYLE question: does the generator arrange same-panel runs like a
    human (mostly 2, some 3, rare longer)? The >=4 share moves freely in native mode (no cap binds the
    model), so it's included -- unlike the override harness where G2 excluded the cap-pinned jacks."""
    gsh, rsh = run_length_shares(gen_runs), run_length_shares(real_runs)
    return (abs(gsh['len2_share'] - rsh['len2_share'])
            + abs(gsh['len3_share'] - rsh['len3_share'])
            + abs(gsh['ge4_share'] - rsh['ge4_share']))


def gen_native(model, song, args, device):
    """Deployed-mode decode for the model (own onset head, density matched to REAL), foot_phys on the
    model's own onsets. Returns ({name: (T,4)}, om_model)."""
    real = np.asarray(song['real'])
    om_real = np.isin(real, ACTIVE_SYMBOLS).any(1)
    tgt = float(om_real.mean())                                   # match REAL press density (count, not positions)
    T = song['len']; bpm = song['bpm']; diff = song['diff']
    audio = torch.from_numpy(song['audio']).unsqueeze(0).to(device)
    dt = torch.tensor([diff], device=device)
    radar_t = torch.from_numpy(song['radar']).unsqueeze(0).to(device)
    lengths = torch.tensor([T], device=device)

    # tau from the SAME radar-conditioned onset logits the decode uses (conditioning-mechanics §6)
    with torch.no_grad():
        memory = model.encode_audio(audio)
        p = torch.sigmoid(model.onset_logits(memory, dt, radar=radar_t)[0]).cpu().numpy()
    tau = float(np.quantile(p, 1 - tgt)) if tgt > 0 else 0.5
    om_model = p > tau                                            # the model's NATIVE onset positions

    fp = FootPhysicsBaseline(beta=args.beta, jump_bias=args.jump_bias)
    out = {"real": real, "foot_phys": fp.generate(om_model, diff, bpm, rng=np.random.default_rng(0))}
    common = dict(onset_threshold=tau, lengths=lengths, type_sample=True, type_temperature=0.4,
                  pattern_sample=True, pattern_temperature=0.7, max_jack_run=2, bpm=bpm, radar=radar_t)
    for name, fat in (("model_raw", None), ("model_gov", args.fatigue)):
        gk = dict(common); gk["fatigue_penalty"] = fat
        enforce_playability(gk)
        set_seed(42)                                             # identical draws -> only the governor differs
        with torch.no_grad():
            g = model.generate(audio, dt, **gk)[0].cpu().numpy()
        out[name] = pair_holds(g)
    return out, om_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/gen_motif_full_fixed/best_val.pt")
    ap.add_argument("--songs", type=int, default=16)
    ap.add_argument("--by", default="rich")
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--fatigue", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=2.0, help="foot_phys inverse-temp (calibrated default)")
    ap.add_argument("--jump_bias", type=float, default=0.0, help="foot_phys jump log-bias (calibrated default)")
    ap.add_argument("--export_sm", type=int, default=0)
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    songs = load_songs(args, device)
    audio_dim = songs[0]['audio'].shape[1]
    model = LayeredTypedChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)['model_state_dict']); model.eval()

    names = ["real", "foot_phys", "model_raw", "model_gov"]
    gen_names = ["foot_phys", "model_raw", "model_gov"]
    by_diff = defaultdict(lambda: {n: [] for n in names})
    runs_by_diff = defaultdict(lambda: {n: [] for n in names})
    iso = {n: {"drop": 0, "reshape": 0, "tot": 0} for n in gen_names}   # vs the model's own onsets
    exported = 0
    out_dir = PROJECT_ROOT / "outputs" / "compare_native"

    for si, song in enumerate(songs):
        charts, om = gen_native(model, song, args, device)
        for n in names:
            by_diff[song['diff']][n].append(chart_metrics(charts[n]))
            runs_by_diff[song['diff']][n].extend(same_panel_run_lengths(charts[n]))
        for n in gen_names:
            c = np.asarray(charts[n]); nz = (c != 0).any(1); pm = np.isin(c, ACTIVE_SYMBOLS).any(1)
            iso[n]["drop"] += int((om & ~nz).sum()); iso[n]["reshape"] += int((om & nz & ~pm).sum())
            iso[n]["tot"] += int(om.sum())
        if exported < args.export_sm:
            out_dir.mkdir(parents=True, exist_ok=True)
            for n in names:
                sm = tensor_to_sm(charts[n], song['bpm'], title=f"song{si}_{n}",
                                  difficulty_name=DIFF_NAMES[song['diff']], typed=(n != "foot_phys"))
                (out_dir / f"song{si}_{n}.sm").write_text(sm)
            exported += 1

    # ---- report (stratified by difficulty) --------------------------------------
    print(f"\nNATIVE-MODE  FOOT-PHYSICS vs LEARNED HEAD  [{args.ckpt}]  {len(songs)} songs (by={args.by})")
    print("model = own onset head (radar-conditioned, density matched to REAL), governor+playability on. NO onset_override.")
    print(f"foot_phys (calibrated): beta={args.beta}, jump_bias={args.jump_bias}, on the MODEL's own onsets.")
    print("maxRun is the retraction's crux: native model should be ~4-6 (real ~4-5), NOT the override harness's 17-24.\n")
    jump_d = defaultdict(dict); jack_d = defaultdict(dict)        # AXIS-SPLIT: jump gap vs jack STYLE
    for d in sorted(by_diff):
        rows = by_diff[d]; rruns = runs_by_diff[d]
        if not rows["real"]:
            continue
        print(f"--- {DIFF_NAMES[d]} (n={len(rows['real'])}) ---")
        print(f"  {'gen':>10} {'dens':>6} {'jump%':>6} {'mJumpStrm':>10} {'mJackRun':>9} {'jack>=4%':>9}")
        for n in names:
            s = summarize(rows[n])
            print(f"  {n:>10} {s['density']:>6.3f} {100*s['jump_rate']:>6.1f} {s['max_jump_stream']:>10.1f} "
                  f"{s['max_jack_run']:>9.1f} {100*s['jack_ge4_share']:>9.1f}")
        print(f"  {'gen':>10} {'len2%':>6} {'len3%':>6} {'>=4%':>6} {'maxRun':>7} {'nRuns':>6}")
        for n in names:
            sh = run_length_shares(rruns[n])
            print(f"  {n:>10} {100*sh['len2_share']:>6.1f} {100*sh['len3_share']:>6.1f} {100*sh['ge4_share']:>6.1f} "
                  f"{sh['max_run']:>7d} {sh['n_runs_ge2']:>6d}")
        print(f"  {'':>10} {'jumpDist':>9} {'jackDist':>9}   (jumpDist=known under-jump gap; jackDist=footwork STYLE)")
        for n in gen_names:
            jd = jump_distance(rows[n], rows["real"]); kd = jack_distance(rruns[n], rruns["real"])
            jump_d[n][d] = jd; jack_d[n][d] = kd
            print(f"  {n:>10} {jd:>9.3f} {kd:>9.3f}")
        print()

    print("OVERALL (mean over difficulties; lower = closer to real):")
    print(f"  {'gen':>10} {'jumpDist':>9} {'jackDist':>9}")
    for n in gen_names:
        jds, kds = list(jump_d[n].values()), list(jack_d[n].values())
        print(f"  {n:>10} {np.mean(jds):>9.3f} {np.mean(kds):>9.3f}" if jds else f"  {n:>10}  --")
    print("  jumpDist: the KNOWN model under-jump gap (a SEPARATE air/density thread -- do NOT read as footwork).")
    print("  jackDist: the same-panel run-length STYLE -- the actual 'whose footwork is more human' question.")

    # isolation: foot_phys uses the model's onsets, so it's exactly om; model paths reshape a few via holds
    print("\nISOLATION (vs the MODEL's own onsets): drop = onset->nothing (leak); reshape = onset->hold-close (benign).")
    for n in gen_names:
        tot = max(iso[n]["tot"], 1)
        print(f"  {n:>10}: drop {iso[n]['drop']}/{tot} ({100*iso[n]['drop']/tot:.2f}%)   "
              f"reshape->hold {iso[n]['reshape']} ({100*iso[n]['reshape']/tot:.2f}%)")
    if args.export_sm:
        print(f"\nExported {exported} song(s) x {len(names)} charts to {out_dir} -- PLAY them (Rule 8).")
    print("\nREAD (DEPLOYED model, two axes): jackDist is the footwork-STYLE verdict -- whose same-panel run-length")
    print("distribution is closer to human. jumpDist is the model's known under-jump gap, shown for context but NOT")
    print("the footwork question (§8d: a separate air/density thread). Lower jackDist = more human run-length style;")
    print("note foot_phys is degenerately len2-only (it can't make real's 3-jacks), so a low jackDist there is a")
    print("smaller AGGREGATE error, not human footwork -- read the len2/len3/>=4 row, not just the scalar.")


if __name__ == "__main__":
    main()

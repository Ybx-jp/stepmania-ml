#!/usr/bin/env python3
"""
COMPARE the learned pattern head against the foot-physics baseline at PANEL CHOICE.

The hypothesis (notes/foot_physics_baseline.md): given the SAME onsets, does the
learned pattern head produce more human-like footwork than the hand-coded
min-exertion foot policy? To isolate panel choice from density (experiment-design
Rule 11), every generator is fed the REAL chart's onset mask:

  * real        - the human source chart (the reference)
  * foot_phys   - FootPhysicsBaseline (no learned model; pure foot physics)
  * model_raw   - the learned model, onset_override=real onsets, NO fatigue governor
  * model_gov   - the learned model, onset_override=real onsets, + fatigue governor

Density is therefore (near-)identical across all four -- a built-in check that the
isolation held. We then compare footwork realism vs REAL, STRATIFIED by difficulty
(Rule 12): chart_metrics (jump%, maxJumpStream, maxJackRun, jack>=4%) and the
same-panel run-length shares (len2/len3/>=4). Metrics come from the shared
playability_metrics module (Rule 14), so numbers are comparable to the fatigue
calibration history.

  python experiments/generation_typed/compare_foot_physics.py [--songs 12] [--by rich]
         [--fatigue 2.0] [--export_sm 2]

Metrics are blind to musicality -- use --export_sm and PLAY the result (Rule 8).
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, torch, yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
from src.data.song_selection import select_by_groove
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.typed import pair_holds
from src.generation.baselines import FootPhysicsBaseline
from src.generation.playability_metrics import (
    chart_metrics, same_panel_run_lengths, run_length_shares, ACTIVE_SYMBOLS)
from src.generation.sm_writer import tensor_to_sm
from src.generation.playtest_export import enforce_playability

DIFF_NAMES = ["Beginner", "Easy", "Medium", "Hard"]


def load_songs(args, device):
    """Load real charts + audio for the comparison (mirrors calib_foot_fatigue)."""
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    tf, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                   use_metric_phase=True, use_highres_onset=True))
    _, vds, _ = create_datasets(train_files=tf, val_files=vf, test_files=[], audio_dir="data/",
                                max_sequence_length=msl, feature_extractor=ext, cache_dir='cache/samples_v3')
    vds.warm_cache(show_progress=False)
    order = select_by_groove(vds, n=args.songs, by=args.by)  # footwork-rich charts stress the comparison
    songs = []
    for i in order[:args.songs]:
        s = vds[i]; m = vds.valid_samples[i]; T = min(int(s['mask'].sum().item()), args.max_len)
        nd = next((nn for nn in m['chart'].note_data if nn.difficulty_name == m['difficulty_name']), None)
        if nd is None:
            continue
        real = vds.parser.convert_to_tensor_typed(m['chart'], nd)[:T]
        # the chart's OWN measured groove radar -- the exact 5-vec the model was TRAINED to condition on
        # (dataset.py feeds meta['groove_radar'].to_vector()); the fair "reproduce THIS chart" conditioning.
        songs.append({'audio': s['audio'][:T].numpy().astype(np.float32), 'len': T,
                      'diff': int(m['difficulty_class']), 'bpm': float(m['chart'].bpm), 'real': real,
                      'radar': s['groove_radar'].numpy().astype(np.float32)})
    return songs


def generate_all(model, song, device, args, rng):
    """Run every generator on the SAME PRESS mask. Returns ({name: (T,4) chart}, om).

    The shared onset mask is PRESS frames only (ACTIVE_SYMBOLS = tap/hold-head/roll).
    The hold-TAIL (symbol 3) is a release, not an onset, so it is excluded -- using
    `real != 0` would mark every hold release as an onset and make foot_phys plant a
    tap on each release, leaking the "same onsets" isolation on exactly the hold-rich
    charts by='rich' selects (experiment-design G1).
    """
    real = np.asarray(song['real'])
    om = np.isin(real, ACTIVE_SYMBOLS).any(1)                       # (T,) real PRESS mask
    T = song['len']; bpm = song['bpm']; diff = song['diff']
    audio = torch.from_numpy(song['audio']).unsqueeze(0).to(device)
    dt = torch.tensor([diff], device=device)
    override = torch.from_numpy(om).unsqueeze(0).to(device)
    lengths = torch.tensor([T], device=device)

    foot_phys = FootPhysicsBaseline(beta=args.beta, jump_bias=args.jump_bias)  # G4: calibratable knobs
    out = {"real": real, "foot_phys": foot_phys.generate(om, diff, bpm, rng=rng)}
    # G6 FAIR TEST: feed the model the chart's OWN measured radar so it's conditioned to make
    # the SAME footwork family (voltage/air/stream/freeze) the by='rich' set was selected for.
    # Without it the model makes a generic-difficulty chart and "loses" on conditioning it never got.
    # Onsets are overridden (p_onset=None, no tau), so radar steers ONLY the pattern head (which
    # panels) -- density isolation is untouched (conditioning-mechanics: onset path bypassed under override).
    radar_t = None if args.no_radar else torch.from_numpy(song['radar']).unsqueeze(0).to(device)
    # model_raw = no FATIGUE governor, but still the MANDATORY playability layer (hold-aware,
    # no jump/cross during hold, jack cap 2): "raw" means governor-free, NOT unconstrained.
    common = dict(onset_override=override, lengths=lengths, type_sample=True, type_temperature=0.4,
                  pattern_sample=True, pattern_temperature=0.7, max_jack_run=2, bpm=bpm, radar=radar_t)
    for name, fat in (("model_raw", None), ("model_gov", args.fatigue)):
        gk = dict(common); gk["fatigue_penalty"] = fat
        enforce_playability(gk)
        set_seed(42)  # identical pattern-sampling draws -> the only diff is the governor
        with torch.no_grad():
            g = model.generate(audio, dt, **gk)[0].cpu().numpy()
        out[name] = pair_holds(g)
    return out, om


def summarize(per_song_metrics):
    """Mean of each chart_metrics field over a list of dicts."""
    keys = ["density", "jump_rate", "max_jump_stream", "max_jack_run", "jack_ge4_share"]
    return {k: float(np.mean([m[k] for m in per_song_metrics])) for k in keys}


def distance_to_real(gen_metrics, real_metrics, gen_runs, real_runs):
    """Footwork distance to real over the dims that govern FOOTWORK and can MOVE.

    Excluded:
      - the jack dims (max_jack_run, jack_ge4_share, run-length >=4 share): cap-pinned
        at max_jack_run=2 for ALL generators while real is uncapped, so they're constants
        that neither discriminate nor match real (experiment-design G2).
      - jump_RATE (the jump SHARE): the model is a KNOWN under-jumper and jump rate is
        radar/air-conditioning-dominated, not a footwork-placement property -- the
        conditioning-mechanics ref is explicit that calibrating to "match real jump%" is
        the WRONG target (§8d/§9). We keep max_jump_STREAM (jump-stream RUN-LENGTH, which
        the per-note governor DOES target) and the same-panel run-length shares len2/len3.
    """
    s, rs = summarize(gen_metrics), summarize(real_metrics)
    gsh, rsh = run_length_shares(gen_runs), run_length_shares(real_runs)
    return (abs(s['max_jump_stream'] - rs['max_jump_stream']) / max(rs['max_jump_stream'], 1)
            + abs(gsh['len2_share'] - rsh['len2_share'])
            + abs(gsh['len3_share'] - rsh['len3_share']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/gen_motif_full_fixed/best_val.pt")
    ap.add_argument("--songs", type=int, default=12)
    ap.add_argument("--by", default="rich")
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--fatigue", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=1.0,
                    help="foot_phys inverse-temperature; CALIBRATE on the rich-Hard set before "
                         "concluding the head beats it (experiment-design G4)")
    ap.add_argument("--jump_bias", type=float, default=-2.0, help="foot_phys additive jump log-bias (G4)")
    ap.add_argument("--no_radar", action="store_true",
                    help="DON'T feed the model each chart's radar -- reproduces the old confounded run (G6 A/B)")
    ap.add_argument("--export_sm", type=int, default=0, help="export this many songs' charts to .sm for ear-check")
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    songs = load_songs(args, device)
    audio_dim = songs[0]['audio'].shape[1]
    model = LayeredTypedChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)['model_state_dict']); model.eval()

    names = ["real", "foot_phys", "model_raw", "model_gov"]
    gen_names = ["foot_phys", "model_raw", "model_gov"]             # consume om; real is the reference
    by_diff = defaultdict(lambda: {n: [] for n in names})           # difficulty -> name -> [metric dicts]
    runs_by_diff = defaultdict(lambda: {n: [] for n in names})      # difficulty -> name -> [run lengths] (G3)
    iso = {n: {"drop": 0, "reshape": 0, "tot": 0} for n in gen_names}  # (G5) dropped onsets vs hold-reshaping
    rng = np.random.default_rng(0)
    exported = 0
    out_dir = PROJECT_ROOT / "outputs" / "compare_foot_physics"

    for si, song in enumerate(songs):
        charts, om = generate_all(model, song, device, args, rng)
        for n in names:
            by_diff[song['diff']][n].append(chart_metrics(charts[n]))
            runs_by_diff[song['diff']][n].extend(same_panel_run_lengths(charts[n]))
        # (G5) isolation check, decomposed. An onset frame can end up:
        #   - DROPPED  (model emits nothing): a TRUE density leak -> confound, gate on this.
        #   - RESHAPED (only a hold-tail, symbol 3): the model chose a hold-CLOSE there; density is
        #     preserved (it emitted a symbol), the foot event is a release not a press. Benign, but it's
        #     a holds-vs-taps ASYMMETRY vs foot_phys (which makes no holds) -> reported, not gated.
        for n in gen_names:
            c = np.asarray(charts[n])
            nz = (c != 0).any(1)                                  # any symbol -> contributes to density
            pm = np.isin(c, ACTIVE_SYMBOLS).any(1)               # a fresh press (foot strike)
            iso[n]["drop"] += int((om & ~nz).sum())              # onset with NOTHING placed (real leak)
            iso[n]["reshape"] += int((om & nz & ~pm).sum())      # onset became a hold-close tail (benign)
            iso[n]["tot"] += int(om.sum())
        if exported < args.export_sm:
            out_dir.mkdir(parents=True, exist_ok=True)
            for n in names:
                typed = n != "foot_phys"                              # foot_phys is binary taps
                sm = tensor_to_sm(charts[n], song['bpm'], title=f"song{si}_{n}",
                                  difficulty_name=DIFF_NAMES[song['diff']], typed=typed)
                (out_dir / f"song{si}_{n}.sm").write_text(sm)
            exported += 1

    # ---- report (stratified by difficulty; experiment-design Rule 12) ------------
    print(f"\nFOOT-PHYSICS vs LEARNED HEAD  [{args.ckpt}]  {len(songs)} songs (by={args.by}), onsets=REAL presses")
    radar_msg = "OFF (--no_radar; CONFOUNDED control)" if args.no_radar else "ON (each chart's own measured radar -- G6 fair test)"
    print(f"model radar conditioning: {radar_msg}")
    print(f"foot_phys knobs: beta={args.beta}, jump_bias={args.jump_bias} (calibrate before concluding -- G4).")
    print("Density matches across the 3 GENERATORS by construction; 'real' carries hold tails so its density is higher.")
    print("Columns marked * are cap-pinned at max_jack_run=2 for all generators (real uncapped) -> excluded from dist (G2).\n")
    per_diff_dist = defaultdict(dict)
    for d in sorted(by_diff):
        rows = by_diff[d]; rruns = runs_by_diff[d]
        if not rows["real"]:
            continue
        print(f"--- {DIFF_NAMES[d]} (n={len(rows['real'])}) ---")
        print(f"  {'gen':>10} {'dens':>6} {'jump%':>6} {'mJumpStrm':>10} {'mJackRun*':>10} {'jack>=4%*':>10}")
        for n in names:
            s = summarize(rows[n])
            print(f"  {n:>10} {s['density']:>6.3f} {100*s['jump_rate']:>6.1f} {s['max_jump_stream']:>10.1f} "
                  f"{s['max_jack_run']:>10.1f} {100*s['jack_ge4_share']:>10.1f}")
        # same-panel run-length shares, SAME strata (the headline pattern-realism metric -- G3)
        print(f"  {'gen':>10} {'len2%':>6} {'len3%':>6} {'>=4%*':>6} {'maxRun':>7} {'nRuns':>6}")
        for n in names:
            sh = run_length_shares(rruns[n])
            print(f"  {n:>10} {100*sh['len2_share']:>6.1f} {100*sh['len3_share']:>6.1f} {100*sh['ge4_share']:>6.1f} "
                  f"{sh['max_run']:>7d} {sh['n_runs_ge2']:>6d}")
        # distance to real over the FREE dims only (jump%, mJumpStrm, len2, len3) -- G2/G3
        print("  dist->real (footwork dims: mJumpStrm, len2, len3; jump%/jacks excluded -- G2/G6):")
        for n in gen_names:
            dist = distance_to_real(rows[n], rows["real"], rruns[n], rruns["real"])
            per_diff_dist[n][d] = dist
            print(f"     {n:>10}  {dist:.3f}")
        print()

    # overall = mean of the per-difficulty distances (Rule 12: average distances, not pooled charts)
    print("OVERALL dist->real (mean over difficulties; lower = closer to human footwork):")
    for n in gen_names:
        ds = list(per_diff_dist[n].values())
        print(f"  {n:>10}  {np.mean(ds):.3f}" if ds else f"  {n:>10}  --")

    # (G5) isolation summary: DROPPED onsets are the density confound to gate on; RESHAPED (hold-close)
    # onsets keep density and are a holds-vs-taps asymmetry vs foot_phys, reported for context.
    DROP_TOL = 0.005   # >0.5% dropped onsets = a genuine density leak that confounds the footwork metrics
    print("\nISOLATION CHECK (G5):  drop = onset with NOTHING placed (density LEAK);  reshape = onset became a")
    print("hold-close tail (density preserved; foot_phys makes no holds so it's structurally 0 here).")
    confounded = False
    for n in gen_names:
        tot = max(iso[n]["tot"], 1)
        df, rf = iso[n]["drop"] / tot, iso[n]["reshape"] / tot
        flag = "   <- CONFOUND: dropped onsets exceed 0.5%" if df > DROP_TOL else ""
        confounded |= df > DROP_TOL
        print(f"  {n:>10}: drop {iso[n]['drop']}/{tot} ({100*df:.2f}%)   reshape->hold {iso[n]['reshape']} ({100*rf:.2f}%){flag}")
    if not confounded:
        print("  -> no generator drops >0.5% of onsets: density isolation HOLDS. The reshape% is holds the")
        print("     model used where foot_phys cannot -- a caveat for foot_phys on freeze-rich charts, not a leak.")

    if args.export_sm:
        print(f"\nExported {exported} song(s) x {len(names)} charts to {out_dir} -- PLAY them (metrics are blind to feel).")
    print("\nREAD: if model_raw's dist->real < foot_phys's, the learned head adds footwork realism the physics "
          "policy lacks; if they tie, the head mostly re-derived the physics. model_gov shows the governor's "
          "effect at fixed onsets. CAVEAT: foot_phys runs at default beta/jump_bias -- a loss for it is not "
          "committable until those are calibrated on this set (--beta/--jump_bias, G4).")


if __name__ == "__main__":
    main()

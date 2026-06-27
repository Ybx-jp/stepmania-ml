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
from src.generation.playability_metrics import chart_metrics, same_panel_run_lengths, run_length_shares
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
        songs.append({'audio': s['audio'][:T].numpy().astype(np.float32), 'len': T,
                      'diff': int(m['difficulty_class']), 'bpm': float(m['chart'].bpm), 'real': real})
    return songs


def generate_all(model, song, device, fatigue, rng):
    """Run every generator on the SAME (real) onset mask. Returns {name: (T,4) chart}."""
    real = song['real']
    om = (real != 0).any(1)                                          # (T,) real onset mask
    T = song['len']; bpm = song['bpm']; diff = song['diff']
    audio = torch.from_numpy(song['audio']).unsqueeze(0).to(device)
    dt = torch.tensor([diff], device=device)
    override = torch.from_numpy(om).unsqueeze(0).to(device)
    lengths = torch.tensor([T], device=device)

    out = {"real": real, "foot_phys": FootPhysicsBaseline().generate(om, diff, bpm, rng=rng)}
    common = dict(onset_override=override, lengths=lengths, type_sample=True, type_temperature=0.4,
                  pattern_sample=True, pattern_temperature=0.7, max_jack_run=2, bpm=bpm)
    for name, fat in (("model_raw", None), ("model_gov", fatigue)):
        gk = dict(common); gk["fatigue_penalty"] = fat
        enforce_playability(gk)
        set_seed(42)  # identical pattern-sampling draws -> the only diff is the governor
        with torch.no_grad():
            g = model.generate(audio, dt, **gk)[0].cpu().numpy()
        out[name] = pair_holds(g)
    return out


def summarize(per_song_metrics):
    """Mean of each chart_metrics field over a list of dicts."""
    keys = ["density", "jump_rate", "max_jump_stream", "max_jack_run", "jack_ge4_share"]
    return {k: float(np.mean([m[k] for m in per_song_metrics])) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/gen_motif_full_fixed/best_val.pt")
    ap.add_argument("--songs", type=int, default=12)
    ap.add_argument("--by", default="rich")
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--fatigue", type=float, default=2.0)
    ap.add_argument("--export_sm", type=int, default=0, help="export this many songs' charts to .sm for ear-check")
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    songs = load_songs(args, device)
    audio_dim = songs[0]['audio'].shape[1]
    model = LayeredTypedChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)['model_state_dict']); model.eval()

    names = ["real", "foot_phys", "model_raw", "model_gov"]
    by_diff = defaultdict(lambda: {n: [] for n in names})           # difficulty -> name -> [metric dicts]
    runs_by_name = {n: [] for n in names}                           # pooled same-panel run lengths
    rng = np.random.default_rng(0)
    exported = 0
    out_dir = PROJECT_ROOT / "outputs" / "compare_foot_physics"

    for si, song in enumerate(songs):
        charts = generate_all(model, song, device, args.fatigue, rng)
        for n in names:
            by_diff[song['diff']][n].append(chart_metrics(charts[n]))
            runs_by_name[n].extend(same_panel_run_lengths(charts[n]))
        if exported < args.export_sm:
            out_dir.mkdir(parents=True, exist_ok=True)
            for n in names:
                typed = n != "foot_phys"                              # foot_phys is binary taps
                sm = tensor_to_sm(charts[n], song['bpm'], title=f"song{si}_{n}",
                                  difficulty_name=DIFF_NAMES[song['diff']], typed=typed)
                (out_dir / f"song{si}_{n}.sm").write_text(sm)
            exported += 1

    # ---- report (stratified by difficulty) --------------------------------------
    print(f"\nFOOT-PHYSICS vs LEARNED HEAD  [{args.ckpt}]  {len(songs)} songs (by={args.by}), onsets=REAL")
    print("Density should match across generators (isolation check). Compare footwork vs the 'real' row.\n")
    for d in sorted(by_diff):
        rows = by_diff[d]
        if not rows["real"]:
            continue
        print(f"--- {DIFF_NAMES[d]} (n={len(rows['real'])}) ---")
        print(f"  {'gen':>10} {'dens':>6} {'jump%':>6} {'mJumpStrm':>10} {'mJackRun':>9} {'jack>=4%':>9}")
        for n in names:
            s = summarize(rows[n])
            print(f"  {n:>10} {s['density']:>6.3f} {100*s['jump_rate']:>6.1f} {s['max_jump_stream']:>10.1f} "
                  f"{s['max_jack_run']:>9.1f} {100*s['jack_ge4_share']:>9.1f}")
        print()

    print("SAME-PANEL RUN-LENGTH SHARES (pooled, among runs >= 2) -- vs the 'real' row:")
    print(f"  {'gen':>10} {'len2%':>6} {'len3%':>6} {'>=4%':>6} {'maxRun':>7} {'nRuns':>6}")
    for n in names:
        sh = run_length_shares(runs_by_name[n])
        print(f"  {n:>10} {100*sh['len2_share']:>6.1f} {100*sh['len3_share']:>6.1f} {100*sh['ge4_share']:>6.1f} "
              f"{sh['max_run']:>7d} {sh['n_runs_ge2']:>6d}")

    # distance-to-real (footwork only; density is held by construction)
    print("\nDISTANCE TO REAL (sum of |Δ| over jump%, mJumpStrm, mJackRun, jack>=4%, normalized):")
    real_all = [m for d in by_diff for m in by_diff[d]["real"]]
    rs = summarize(real_all)
    for n in names:
        all_n = [m for d in by_diff for m in by_diff[d][n]]
        s = summarize(all_n)
        dist = (abs(s['jump_rate'] - rs['jump_rate']) / max(rs['jump_rate'], .05)
                + abs(s['max_jump_stream'] - rs['max_jump_stream']) / max(rs['max_jump_stream'], 1)
                + abs(s['max_jack_run'] - rs['max_jack_run']) / max(rs['max_jack_run'], 1)
                + abs(s['jack_ge4_share'] - rs['jack_ge4_share']))
        print(f"  {n:>10}  dist→real = {dist:.3f}")
    if args.export_sm:
        print(f"\nExported {exported} song(s) x {len(names)} charts to {out_dir} -- PLAY them (metrics are blind to feel).")
    print("\nREAD: if model_raw is closer to real than foot_phys, the learned head adds footwork realism the "
          "physics policy lacks; if foot_phys matches, the head mostly learned the physics. model_gov shows the "
          "governor's effect at fixed onsets.")


if __name__ == "__main__":
    main()

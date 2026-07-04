#!/usr/bin/env python
"""SMOKE test for the SONG-FEATURES -> GOOD-SETTINGS-REGION exploration (2026-07-03, descriptive-first).

Goal of the thread: map, per song, WHERE in decode-settings space the GOOD charts live, and which SONG
FEATURES predict that region. This smoke run proves the pipeline on a tiny grid and — critically — checks the
graded critic for FLOOR-COMPRESSION (does p_gen separate across cells beyond the per-cell best-of-k noise?)
BEFORE we spend compute on the full BPM-stratified sweep.

Settings axis = the user's MANIFOLD path (`--style`, conditional-fill + ellipsoid projection), NOT the disabled
`--radar` mean-pin. Faithful to export_typed_samples.py's manifold branch: build_target -> snapped 5-vec passed
as radar=, gen_density from the manifold, tau from the SAME conditioned + phase-calib onset logits. Scored
best-of-k with the GRADED (non-saturating) critic. Reuses the probe_quality_features harness for song/critic I/O.
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import argparse, sys
from pathlib import Path
import numpy as np, torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.generation.decode_defaults import CANONICAL_DECODE
from src.generation.decode_harness import conditioned_p_onset, compute_tau, load_generator, DEPLOYED_CHECKPOINT
from src.generation.playtest_export import enforce_playability
from src.generation.typed import pair_holds
from src.generation.radar_manifold import RadarManifold
from src.models import LateFusionClassifier
from probe_quality_features import DIFF_NAMES, to_binary, critic_score, load_val_dataset, build_songs

GRADED_CRITIC = 'checkpoints/realism_critic_graded/best_val.pt'


def gen_at_style(model, s, style_spec, guidance, manifold, device):
    """ONE generation for song s conditioned on a MANIFOLD --style spec at CFG `guidance`.
    Mirrors export_typed_samples' manifold path exactly. Returns (binary_grid, target_info)."""
    phase_calib = CANONICAL_DECODE['onset_phase_calib']            # (0.0, 1.0) the 16th-unlock
    tvec, tinfo = manifold.build_target(style_spec, s['difficulty'])   # conditional-fill + ellipsoid projection
    radar_for_gen = torch.from_numpy(tvec).unsqueeze(0).to(device)
    gen_density = tinfo['density'] if tinfo['density'] is not None else s['real_density']
    audio = torch.from_numpy(s['audio']).unsqueeze(0).to(device)
    diff = torch.tensor([s['difficulty']], device=device)
    with torch.no_grad():
        memory = model.encode_audio(audio)
        p_onset = conditioned_p_onset(model, memory, diff, radar=radar_for_gen, style=None,
                                      guidance=guidance, phase_calib=phase_calib, extra_offset=None)
    tau = compute_tau(p_onset, gen_density)                        # tau from the SAME conditioned+calib logits
    gk = dict(onset_threshold=tau, bpm=s['bpm'], type_sample=True, pattern_sample=True,
              radar=radar_for_gen, style=None, motif=None, figure=None, guidance_scale=guidance,
              **CANONICAL_DECODE)
    enforce_playability(gk, None)                                  # FORCES hold_aware / no_jump / no_cross on
    with torch.no_grad():
        g = model.generate(audio, diff, lengths=torch.tensor([s['T']], device=device), **gk)[0].cpu().numpy()
    return to_binary(pair_holds(g[:s['T']])), tinfo


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--critic', default=GRADED_CRITIC, help='default = the GRADED (non-saturating) critic')
    p.add_argument('--checkpoint', default=DEPLOYED_CHECKPOINT)
    p.add_argument('--n', type=int, default=3, help='#Hard songs (smoke=3)')
    p.add_argument('--k', type=int, default=4, help='#gens/cell, graded-critic-averaged (smoke=4; full sweep=8)')
    p.add_argument('--max_len', type=int, default=768)
    p.add_argument('--cache_dir', default='cache/samples_v3')
    p.add_argument('--difficulty', type=int, default=3, help='3=Hard')
    p.add_argument('--chaos', default='0.2,0.5,0.9', help='--style chaos values to sweep (voltage/air pinned to the loved corner)')
    p.add_argument('--guidance', default='1.5,3.0', help='CFG guidance values to sweep')
    p.add_argument('--out', default='cache/goodregion_smoke.csv')
    return p.parse_args()


def main():
    args = parse_args(); set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    chaos_vals = [float(x) for x in args.chaos.split(',')]
    guid_vals = [float(x) for x in args.guidance.split(',')]
    print(f"device={device} | GRADED critic={args.critic}")
    print(f"grid: chaos={chaos_vals} x guidance={guid_vals} | k={args.k}/cell | voltage=0.7,air=0.5 pinned (the loved corner)")

    val_ds = load_val_dataset(args.data_dir, args.audio_dir, args.seed, args.cache_dir)
    songs = build_songs(val_ds, args.n, args.difficulty, args.max_len)
    print("songs (Hard): " + ", ".join(f"{s['title'][:16]}(bpm{s['bpm']:.0f})" for s in songs))

    manifold = RadarManifold.load(Path('cache/radar_manifold.npz'))   # the SAME saved manifold export uses
    ck = torch.load(args.critic, map_location=device, weights_only=False)
    critic = LateFusionClassifier(ck['config']).to(device); critic.load_state_dict(ck['model_state_dict']); critic.eval()
    model = load_generator(args.checkpoint, 42, device)

    rows = []
    for s in songs:
        a23 = s['audio'][:, :23]
        p_real, m_real = critic_score(critic, a23, s['real'], device)
        print(f"\n=== {s['title'][:30]}  bpm={s['bpm']:.0f}  real_dens={s['real_density']:.3f}  P_real={p_real:.3f} ===")
        for ch in chaos_vals:
            for g in guid_vals:
                spec = f"chaos={ch},voltage=0.7,air=0.5"
                ps, ms = [], []; tinfo = None
                for _ in range(args.k):
                    grid, tinfo = gen_at_style(model, s, spec, g, manifold, device)
                    pr, mr = critic_score(critic, a23, grid, device); ps.append(pr); ms.append(mr)
                tgt = tinfo['target']   # realized on-manifold 5-vec [stream, voltage, air, freeze, chaos]
                row = dict(title=s['title'], bpm=s['bpm'], real_density=s['real_density'], p_real=p_real,
                           chaos=ch, guidance=g, mahal=tinfo['mahalanobis'], projected=tinfo['projected'],
                           real_chaos=float(tgt[4]), real_voltage=float(tgt[1]), real_air=float(tgt[2]),
                           p_gen=float(np.mean(ps)), p_gen_sd=float(np.std(ps)), m_gen=float(np.mean(ms)))
                rows.append(row)
                print(f"   chaos={ch:.1f} g={g:.1f}  P_gen={row['p_gen']:.3f}±{row['p_gen_sd']:.3f}  "
                      f"M_gen={row['m_gen']:+.2f}  mahal={row['mahal']:.2f}{' [proj]' if row['projected'] else ''}")

    import csv
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nwrote {args.out} ({len(rows)} cells)")

    # ---- FLOOR-COMPRESSION diagnostic: does the surface separate BEYOND the per-cell best-of-k noise? ----
    print("\n" + "=" * 72 + "\nFLOOR-COMPRESSION CHECK (per song): between-cell spread vs within-cell noise")
    print("  a surface is INFORMATIVE only if between-cell RANGE >> mean within-cell SD (noise floor)")
    for s in songs:
        srows = [r for r in rows if r['title'] == s['title']]
        pg = np.array([r['p_gen'] for r in srows]); noise = np.mean([r['p_gen_sd'] for r in srows])
        rng = pg.max() - pg.min(); best = max(srows, key=lambda r: r['p_gen'])
        ratio = rng / (noise + 1e-9)
        verdict = 'INFORMATIVE' if ratio > 2 else ('MARGINAL' if ratio > 1 else 'FLOOR-COMPRESSED')
        print(f"  {s['title'][:26]:28s} range={rng:.3f}  noise={noise:.3f}  ratio={ratio:.1f} -> {verdict:16s}"
              f"  argmax@ chaos={best['chaos']:.1f},g={best['guidance']:.1f} (P={best['p_gen']:.3f})")


if __name__ == '__main__':
    main()

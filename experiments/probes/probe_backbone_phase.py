#!/usr/bin/env python
"""ATTRIBUTION: why does the backbone flip 1/4 -> 1/16 when conditioning is cranked past a song's tolerance?
(2026-07-03, user question.) Sweep CFG `guidance` at the milestone HIGH-CHAOS --style spec and measure the
generated backbone's phase shares (quarter vs 16th-offbeat) under a one-variable ABLATION LADDER:

  FULL       deployed stack (governor ON, 16th-unlock ON)          -> the baseline flip
  GOV_OFF    fatigue/stamina governor OFF                          -> flip persists => governor NOT the cause
  CALIB_OFF  onset_phase_calib=(0,0) (16th-unlock OFF)             -> flip vanishes => the calib is the cause
  BOTH_OFF   governor OFF + calib OFF                              -> still flips => raw CFG-amplified chaos

Whichever arm FLATTENS the quarter->16th flip owns the mechanism. Phase metric + conditioning are the deployed
harness (phase_shares; conditioned_p_onset+compute_tau with the SAME calib baked into tau per arm). Real Hard
backbone ~ (q=0.71, e8=0.25, s16=0.04).
"""
import warnings, os
warnings.filterwarnings('ignore')
import argparse, sys
from pathlib import Path
import numpy as np, torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(1, str(Path(__file__).resolve().parent))  # probes dir (sibling probe imports)
from src.utils.reproducibility import set_seed
from src.generation.decode_defaults import CANONICAL_DECODE
from src.generation.decode_harness import conditioned_p_onset, compute_tau, load_generator, DEPLOYED_CHECKPOINT, phase_shares
from src.generation.playtest_export import enforce_playability
from src.generation.typed import pair_holds
from src.generation.radar_manifold import RadarManifold
from probe_quality_features import load_val_dataset, build_songs

# ablation arm -> (phase_calib, decode_overrides). calib is baked into BOTH tau and the decode (consistency).
ARMS = {
    'FULL':      (CANONICAL_DECODE['onset_phase_calib'], {}),
    'GOV_OFF':   (CANONICAL_DECODE['onset_phase_calib'], {'fatigue_penalty': None, 'stamina_ceiling': None}),
    'CALIB_OFF': ((0.0, 0.0),                            {}),
    'BOTH_OFF':  ((0.0, 0.0),                            {'fatigue_penalty': None, 'stamina_ceiling': None}),
}


def gen_typed(model, s, spec, guidance, manifold, device, phase_calib, overrides, seed):
    """One generation at a --style spec + guidance under an ablation. Returns (typed_chart, realized_density)."""
    set_seed(seed)                                              # shared RNG across arms -> variance reduction
    tvec, tinfo = manifold.build_target(spec, s['difficulty'])
    radar = torch.from_numpy(tvec).unsqueeze(0).to(device)
    gen_density = tinfo['density'] if tinfo['density'] is not None else s['real_density']
    audio = torch.from_numpy(s['audio']).unsqueeze(0).to(device)
    diff = torch.tensor([s['difficulty']], device=device)
    with torch.no_grad():
        memory = model.encode_audio(audio)
        p_onset = conditioned_p_onset(model, memory, diff, radar=radar, style=None,
                                      guidance=guidance, phase_calib=phase_calib)   # calib baked into tau
    tau = compute_tau(p_onset, gen_density)
    gk = dict(onset_threshold=tau, bpm=s['bpm'], type_sample=True, pattern_sample=True,
              radar=radar, style=None, motif=None, figure=None, guidance_scale=guidance, **CANONICAL_DECODE)
    gk['onset_phase_calib'] = phase_calib                       # arm's calib into the decode too
    gk.update(overrides)                                        # governor-off ablation (labeled, one variable)
    enforce_playability(gk, None)
    with torch.no_grad():
        g = model.generate(audio, diff, lengths=torch.tensor([s['T']], device=device), **gk)[0].cpu().numpy()
    typed = pair_holds(g[:s['T']])
    return typed, float((typed != 0).any(1).mean())


def q_s16(typed):
    q, e8, s16 = phase_shares(np.where(np.asarray(typed).any(-1))[0])
    return q, s16


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint', default=DEPLOYED_CHECKPOINT)
    p.add_argument('--n', type=int, default=2, help='#Hard songs')
    p.add_argument('--k', type=int, default=3, help='#gens/cell (phase-share averaged)')
    p.add_argument('--max_len', type=int, default=768)
    p.add_argument('--cache_dir', default='cache/samples_v3')
    p.add_argument('--spec', default='chaos=0.9,voltage=0.7,air=0.5,freeze=0.5', help='the milestone HIGH-chaos --style spec')
    p.add_argument('--guidance', default='1.0,1.5,2.0,3.0', help='CFG guidance sweep (the flip driver)')
    p.add_argument('--out', default='cache/backbone_phase.csv')
    return p.parse_args()


def main():
    args = parse_args(); set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    guids = [float(x) for x in args.guidance.split(',')]
    print(f"device={device} | spec='{args.spec}' | guidance sweep={guids} | k={args.k} | arms={list(ARMS)}")
    print("backbone flip = quarter-share q DROPS while 16th-share s16 RISES as guidance climbs. real Hard q~0.71 s16~0.04")

    val_ds = load_val_dataset(args.data_dir, args.audio_dir, args.seed, args.cache_dir)
    songs = build_songs(val_ds, args.n, difficulty=3, max_len=args.max_len)
    manifold = RadarManifold.load(Path('cache/radar_manifold.npz'))
    model = load_generator(args.checkpoint, 42, device)

    rows = []
    for s in songs:
        # anchor: the REAL chart's own backbone phase
        rq, re8, rs16 = phase_shares(np.where(s['real_typed'].any(-1))[0]) if 'real_typed' in s else (np.nan,)*3
        print(f"\n=== {s['title'][:30]}  bpm={s['bpm']:.0f}  REAL backbone q={rq:.2f} s16={rs16:.2f} ===")
        print(f"{'arm':10s} " + " ".join(f"g{gg:<10.1f}" for gg in guids))
        for arm, (calib, ov) in ARMS.items():
            cells = []
            for gg in guids:
                qs, s16s, ds = [], [], []
                for j in range(args.k):
                    typed, dens = gen_typed(model, s, args.spec, gg, manifold, device, calib, ov, args.seed + j)
                    q, s16 = q_s16(typed); qs.append(q); s16s.append(s16); ds.append(dens)
                q_m, s16_m, d_m = np.mean(qs), np.mean(s16s), np.mean(ds)
                cells.append((gg, q_m, s16_m, d_m))
                rows.append(dict(title=s['title'], arm=arm, guidance=gg, q=q_m, s16=s16_m, density=d_m))
            print(f"{arm:10s} " + " ".join(f"q{q:.2f}/s{s16:.2f}" for _, q, s16, _ in cells))

    import csv
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nwrote {args.out} ({len(rows)} cells)")

    # ---- ATTRIBUTION: per arm, how much does the backbone flip across the guidance sweep? ----
    print("\n" + "=" * 74 + "\nATTRIBUTION: flip magnitude per arm = mean over songs of [ s16(g_max) - s16(g_min) ]")
    print("  (also quarter drop). A near-ZERO flip in an arm => that arm's disabled mechanism OWNED the flip.")
    for arm in ARMS:
        d_s16, d_q = [], []
        for s in songs:
            a = [r for r in rows if r['arm'] == arm and r['title'] == s['title']]
            a.sort(key=lambda r: r['guidance'])
            d_s16.append(a[-1]['s16'] - a[0]['s16']); d_q.append(a[-1]['q'] - a[0]['q'])
        print(f"  {arm:10s}  Δs16={np.mean(d_s16):+.3f}   Δquarter={np.mean(d_q):+.3f}")


if __name__ == '__main__':
    main()

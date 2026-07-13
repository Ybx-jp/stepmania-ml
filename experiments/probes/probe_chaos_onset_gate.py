#!/usr/bin/env python
"""Phase-0 PROBE for the chaos×onset GATE (scope: notes/chaos_onset_gate_scope.md; the "harness it completely"
ceiling-raiser). Decides PROBE-vs-TRAIN: can a DECODE-TIME gate (no retrain) un-smear the overload songs at the
GOOD peak setting (chaos=0.9, g=1.5) without degrading the good ones?

MECHANISM (decode_harness.chaos_onset_gate_offset, single-sourced): replace the FLAT 16th-unlock (a global
additive bias CFG smears uniformly — H4/referee) with a LOCAL audio-keyed off-beat lift
`Δlogit[t] = gain·chaos·offbeat[t]·saliency[t]`, saliency = norm01 max(dim41 highres-onset, dim35 perc). Fed to
BOTH tau (conditioned_p_onset extra_offset=) AND generate(onset_logit_offset=) — same coupling as harm_calib.

ARMS (ONE variable each vs canonical BASE; shared RNG across arms — experiment-design Rule 11). Every arm keeps
the CANONICAL 16th-unlock (0,1.0) ON — an earlier version turned it OFF in the gate arm, confounding unlock-off
WITH the gate (a canonical-defaults + one-change violation); the good-song 16th collapse was the unlock removal,
not the gate. Corrected:
  BASE     deployed (unlock 0,1.0, NO gate)                                  (reproduces what was played)
  ADD      + additive content gate (unlock STILL on)                        (isolates the gate's ADD effect)
  DESMEAR  + subtract off-beat logits in LOW-saliency zones (unlock on)     (the dead-zone smear guard)

SONGS (Rule 12 stratify, Rule 5 real anchor): the 4 by-ear-LABELED taste songs, each at its HARDEST difficulty:
  OVERLOAD (must un-smear): High School Love, Love Vacation
  GOOD     (must NOT drop): TimeToEye, Grand Chariot

METRIC: the referee's OVERLOAD DETECTOR (goodregion_findings.md) — on_grid_share + sixteenth_anchoring
(<~0.3 = the smear cliff) + realized density + q/s16 vs the song's REAL chart. **Rule 8: the ASCII onset grid is
dumped FIRST** (the thread's cautionary lesson — one grid settled what 3 scalar iterations couldn't). By-ear is
the binding gate; this probe only locates whether the gate MOVES anchoring the right way per song.
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
from src.generation.decode_harness import (conditioned_p_onset, compute_tau, load_generator,
                                           DEPLOYED_CHECKPOINT, phase_shares, chaos_onset_gate_offset)
from src.generation.playtest_export import enforce_playability
from src.generation.typed import pair_holds
from src.generation.radar_manifold import RadarManifold
from probe_quality_features import load_val_dataset
from probe_backbone_tolerance import on_grid_share, sixteenth_anchoring

TARGETS = {  # substring -> group label (by-ear labels from playtest_log 2026-07-04)
    'high school love': 'OVERLOAD', 'love_vacation': 'OVERLOAD',
    'timetoeye': 'GOOD', 'grand chariot': 'GOOD',
}


def select_songs(val_ds, max_len):
    """Pick each TARGET song at its HARDEST available difficulty (matches the deployed --hardest export)."""
    best = {}  # key substring -> (difficulty_class, idx, meta)
    for i, meta in enumerate(val_ds.valid_samples):
        hay = f"{meta['chart'].title or ''} {meta['chart_file']}".lower()
        for key in TARGETS:
            if key in hay:
                dc = int(meta['difficulty_class'])
                if key not in best or dc > best[key][0]:
                    best[key] = (dc, i, meta)
    songs = []
    for key, (dc, i, meta) in best.items():
        s = val_ds[i]; T = min(int(s['mask'].sum().item()), max_len)
        nd = next((nn for nn in meta['chart'].note_data if nn.difficulty_name == meta['difficulty_name']
                   and nn.difficulty_value == meta['difficulty_value']), None)
        if nd is None or T < 64:
            continue
        tf = np.asarray(val_ds.parser.convert_to_tensor_typed(meta['chart'], nd))[:T]
        songs.append({'audio': s['audio'][:T].numpy().astype(np.float32), 'real_typed': tf,
                      'difficulty': dc, 'bpm': float(meta['chart'].bpm), 'T': T,
                      'real_density': float((tf != 0).any(1).mean()),
                      'group': TARGETS[key], 'title': (meta['chart'].title or Path(meta['chart_file']).stem)})
    return songs


def gen(model, s, spec, guidance, manifold, device, phase_calib, gate_gain, seed, desmear=False):
    """One gen at (spec, guidance). gate_gain>0 builds the chaos×onset offset (ADD or DESMEAR) and threads it
    into tau AND decode."""
    set_seed(seed)
    tvec, tinfo = manifold.build_target(spec, s['difficulty'])
    radar = torch.from_numpy(tvec).unsqueeze(0).to(device)
    chaos = float(tvec[4])                                       # resolved manifold chaos (scales the ADD gate)
    gen_density = tinfo['density'] if tinfo['density'] is not None else s['real_density']
    audio = torch.from_numpy(s['audio']).unsqueeze(0).to(device)
    diff = torch.tensor([s['difficulty']], device=device)
    gate_t = None
    if gate_gain and gate_gain > 0:
        gate_t = torch.from_numpy(chaos_onset_gate_offset(s['audio'], gate_gain, chaos, desmear=desmear)).to(device)
    with torch.no_grad():
        memory = model.encode_audio(audio)
        # tau sees the SAME phase_calib AND gate offset the decode uses (conditioning-mechanics §3/§6)
        p_onset = conditioned_p_onset(model, memory, diff, radar=radar, style=None,
                                      guidance=guidance, phase_calib=phase_calib, extra_offset=gate_t)
    tau = compute_tau(p_onset, gen_density)
    gk = dict(onset_threshold=tau, bpm=s['bpm'], type_sample=True, pattern_sample=True,
              radar=radar, style=None, motif=None, figure=None, guidance_scale=guidance, **CANONICAL_DECODE)
    gk['onset_phase_calib'] = phase_calib
    gk['onset_logit_offset'] = gate_t                           # the gate into the decode too (None if off)
    enforce_playability(gk, None)
    with torch.no_grad():
        g = model.generate(audio, diff, lengths=torch.tensor([s['T']], device=device), **gk)[0].cpu().numpy()
    typed = pair_holds(g[:s['T']])
    return typed, float((typed != 0).any(1).mean())


def ascii_grid(typed, measures=8):
    """Dump the onset PHASE grid, 16 frames (1 measure) per line: Q=quarter 8=eighth :=16th-offbeat ·=empty.
    The eye reads a backbone (Q every 4) vs a uniform 16th smear (: everywhere) instantly (Rule 8)."""
    on = np.asarray(typed).any(-1)
    out = []
    for m in range(measures):
        row = []
        for k in range(16):
            t = m * 16 + k
            if t >= len(on):
                break
            ph = t % 4
            row.append(('Q' if ph == 0 else '8' if ph == 2 else ':') if on[t] else '·')
        out.append('|' + ''.join(row))
    return '\n'.join(out)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint', default=DEPLOYED_CHECKPOINT)
    p.add_argument('--k', type=int, default=2, help='#gens/arm (anchoring averaged; grid from the first)')
    p.add_argument('--max_len', type=int, default=768)
    p.add_argument('--cache_dir', default='cache/samples_v3')
    p.add_argument('--spec', default='chaos=0.9,voltage=0.7,air=0.5,freeze=0.5')
    p.add_argument('--guidance', type=float, default=1.5, help='the GOOD peak guidance (H14/referee)')
    p.add_argument('--gate_gain', type=float, default=3.0, help='ADD-gate strength (replace-unlock variant)')
    p.add_argument('--desmear_gain', type=float, default=4.0, help='DESMEAR subtract strength (keep-unlock variant)')
    p.add_argument('--measures', type=int, default=8, help='#measures in the ASCII grid dump')
    p.add_argument('--out', default='outputs/probe_results/chaos_onset_gate.csv')
    return p.parse_args()


def main():
    args = parse_args(); set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device} | spec='{args.spec}' g={args.guidance} | gate_gain={args.gate_gain} | k={args.k}")
    print("ARMS (all keep canonical unlock): BASE=deployed · ADD=+additive gate · DESMEAR=+low-saliency subtract")
    print("overload-detector: anchoring<~0.3 = smear cliff (goodregion_findings.md). real Hard q~0.71 s16~0.04\n")

    val_ds = load_val_dataset(args.data_dir, args.audio_dir, args.seed, args.cache_dir)
    songs = select_songs(val_ds, args.max_len)
    found = {s['title'] for s in songs}
    print(f"selected {len(songs)} songs: " + ", ".join(f"{s['title']}[{s['group']}]" for s in songs))
    manifold = RadarManifold.load(Path('cache/radar_manifold.npz'))
    model = load_generator(args.checkpoint, 42, device)

    # arm -> (phase_calib, gate_gain, desmear). EVERY arm keeps the CANONICAL unlock (0,1.0) so each changes
    # exactly ONE thing from BASE (experiment-design Rule 11; the earlier (0,0) replace-arm confounded unlock-off
    # WITH the gate). BASE = deployed. ADD = + additive content gate. DESMEAR = + subtract in low-saliency zones.
    UNLOCK = CANONICAL_DECODE['onset_phase_calib']
    ARMS = {'BASE':    (UNLOCK, 0.0,               False),
            'ADD':     (UNLOCK, args.gate_gain,    False),
            'DESMEAR': (UNLOCK, args.desmear_gain, True)}
    rows = []
    for s in songs:
        rq, _, rs16 = phase_shares(np.where(s['real_typed'].any(-1))[0])
        ra = sixteenth_anchoring(s['real_typed']); rg = on_grid_share(s['real_typed'])
        print("\n" + "=" * 78)
        print(f"{s['title']}  [{s['group']}]  diff={s['difficulty']} bpm={s['bpm']:.0f}  "
              f"REAL: on_grid={rg:.2f} anchor={ra:.2f} q={rq:.2f} s16={rs16:.2f}")
        for arm, (calib, gain, dsm) in ARMS.items():
            og, an, de, qs, s16s = [], [], [], [], []
            grid0 = None
            for j in range(args.k):
                typed, dens = gen(model, s, args.spec, args.guidance, manifold, device, calib, gain, args.seed + j, desmear=dsm)
                if j == 0:
                    grid0 = ascii_grid(typed, args.measures)
                og.append(on_grid_share(typed)); an.append(sixteenth_anchoring(typed)); de.append(dens)
                q, _, s16 = phase_shares(np.where(typed.any(-1))[0]); qs.append(q); s16s.append(s16)
            an_m, og_m = np.nanmean(an), np.nanmean(og)
            print(f"\n  --- {arm}: on_grid={og_m:.2f}  anchor={an_m:.2f}  q={np.mean(qs):.2f} "
                  f"s16={np.mean(s16s):.2f}  dens={np.mean(de):.2f} (real {s['real_density']:.2f}) ---")
            print(grid0)
            rows.append(dict(title=s['title'], group=s['group'], arm=arm, on_grid=round(og_m, 3),
                             anchor=round(an_m, 3), q=round(float(np.mean(qs)), 3),
                             s16=round(float(np.mean(s16s)), 3), density=round(float(np.mean(de)), 3)))

    # ---- verdict: per song, on_grid/anchor/s16 across arms. Each arm = ONE change from BASE. The DESMEAR win =
    # OVERLOAD un-smears (anchor↑, s16→real) WHILE GOOD keeps its s16 (the unlock is still on, untouched). anchor is
    # only the OVERLOAD detector (goodregion); s16 vs BASE/real is the "kept the loved syncopation" read; ears bind. ----
    print("\n" + "=" * 78 + "\nVERDICT — per song, per arm:  on_grid / anchor / s16   (real s16 in header)")
    print("  DESMEAR target: OVERLOAD anchor climbs out of <0.3 AND GOOD s16 ~ preserved (vs BASE, unlock untouched).")
    for s in songs:
        _, _, rs16 = phase_shares(np.where(s['real_typed'].any(-1))[0])
        line = f"  [{s['group']:8s}] {s['title'][:26]:26s} real_s16={rs16:.2f} | "
        for arm in ARMS:
            r = next(rr for rr in rows if rr['title'] == s['title'] and rr['arm'] == arm)
            line += f"{arm}: {r['on_grid']:.2f}/{r['anchor'] if not np.isnan(r['anchor']) else float('nan'):.2f}/{r['s16']:.2f}  "
        print(line)
    print("\nDECISION: DESMEAR un-smears OVERLOAD (anchor↑) while GOOD s16 stays near BASE/real -> shippable opt-in "
          "tolerance-guard (prefer probe). If DESMEAR also flattens GOOD s16, or can't un-smear -> decode exhausted, "
          "escalate to the Phase-1 retrain. BY-EAR is the binding gate on the GOOD songs (export + play before committing).")

    import csv
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nwrote {args.out} ({len(rows)} rows)")


if __name__ == '__main__':
    main()

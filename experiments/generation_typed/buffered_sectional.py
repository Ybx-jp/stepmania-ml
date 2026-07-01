#!/usr/bin/env python3
"""
Buffered-sectional generation (H11 fix, decode-time, no retrain). The boundary-reset probe showed
flushing AR context at section boundaries makes the model re-choreograph (overshooting real because the
hard cold-start is abrupt). This is the playable version: generate each section INDEPENDENTLY with a
discarded WARMUP buffer (absorbs the cold-start transient) and COOLDOWN buffer (absorbs H5 end-fade), keep
only the clean middle, concatenate. Addresses cross-boundary momentum + cold-start + end-fade at once.

Reports transition-responsiveness (baseline vs sectional vs real) and exports baseline-vs-sectional for
an A/B playtest (groove-validated, Hard songs with real section structure).
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, re, shutil, sys
from pathlib import Path
import numpy as np, torch, yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import split_chart_files
from src.data.dataset import StepManiaDataset, DIFFICULTY_NAMES
from src.data.song_selection import select_by_groove
from src.generation.decode_harness import (
    CANONICAL_DECODE, conditioned_p_onset, make_feature_extractor, load_generator)
from src.generation.typed import pair_holds
from src.generation.sm_writer import charts_to_sm
from diag_transitions_freerun import foote_boundaries, responsiveness, SSM_DIMS, descriptor, W


def responsiveness_nodens(typed, bnds, rng, T):
    """Same responsiveness metric but with DENSITY dropped from the descriptor (dim 0) — only jump_frac +
    L/D/U/R panel mix. Isolates PATTERN-HEAD choreography transitions from the breathe arc's density modulation
    (which mechanically inflates the density-inclusive metric). rng seeded the SAME as the full metric for paired
    comparison."""
    def mag_nd(c):
        a = descriptor(typed, c - W, c); b = descriptor(typed, c, c + W)
        return None if (a is None or b is None) else float(np.abs(a[1:] - b[1:]).sum())  # a[1:] drops density
    bm = [m for c in bnds for m in [mag_nd(int(c))] if m is not None]
    rand = rng.integers(W, T - W, size=max(len(bnds) * 3, 30))
    rm = [m for c in rand for m in [mag_nd(int(c))] if m is not None]
    return (np.mean(bm), np.mean(rm)) if (bm and rm) else None

# CANONICAL DEFAULTS (2026-06-28) — the deployed model + the exporter's playtest-validated full-governor
# stack, NOT the stale June-21 palette. Source of truth: export_typed_samples.py argparse defaults +
# notes/governor_release_region.md + the conditioning-mechanics skill §7-§8. Re-running H11 on anything
# less is the recurring "ran with a stale subset of settings" failure (wrong model, 41-dim features,
# pattern_temperature 0.7, no governor) that invalidates the delta.
GEN_CKPT = "checkpoints/gen_motif_full_fixed/best_val.pt"   # the deployed 42-dim H19 highres retrain (was gen_stage1)
# the canonical decode palette (single source; can no longer drift from the deployed regime) + playability.
# bpm is per-song -> added at each call site (governor needs it).
CALIB = CANONICAL_DECODE["onset_phase_calib"]   # the 16th-unlock; tau MUST use the SAME offset (conditioned_p_onset does)
DECODE = dict(
    **CANONICAL_DECODE,
    type_sample=True, pattern_sample=True,
    hold_aware=True, no_jump_during_hold=True, no_cross_during_hold=True,  # MANDATORY playability
)


def calibrated_p_onset(model, aud, diff, calib, device=None):
    """sigmoid(onset logits + the SAME per-phase calib offset the decode uses) -> p for the tau quantile.
    Thin wrapper over the shared harness (was a hand-rolled copy of conditioned_p_onset; `device` now unused)."""
    return conditioned_p_onset(model, model.encode_audio(aud), diff, phase_calib=calib)


def safe_name(s):
    return (re.sub(r'[^\w\- ]+', '', (s or 'x').strip(), flags=re.UNICODE).strip() or 'x')[:60]


def generate_sectional(model, audio_full, diff, orig, bnds, W_in, W_out, device, bpm, decode, seed=42):
    """Generate each section [a,b) independently over audio [a-W_in, b+W_out) (cold start), keep [a,b).
    tau is computed from the SLICED audio's own onset (the encoder re-encodes the slice -> the full-song
    p_on doesn't match it), and targets each section's OWN real density (so generated sections track the
    real intensity arc). Warmup/cooldown buffers absorb cold-start, end-fade, AND slice edge-effects."""
    T = audio_full.shape[1]
    segs = [0] + sorted(int(b) for b in bnds) + [T]
    out = np.zeros((T, 4), dtype=np.int64)
    for i in range(len(segs) - 1):
        a, b = segs[i], segs[i + 1]
        if b <= a:
            continue
        lo, hi = max(0, a - W_in), min(T, b + W_out)
        aud = audio_full[:, lo:hi]
        rd_seg = float((np.asarray(orig)[a:b] != 0).any(1).mean())   # this section's real density
        with torch.no_grad():
            p_seg = calibrated_p_onset(model, aud, diff, CALIB, device)   # same calib offset as the decode (16th-unlock)
        tau = float(np.quantile(p_seg, 1 - rd_seg)) if rd_seg > 0 else 1.0
        set_seed(seed)
        g = model.generate(aud, diff, lengths=torch.tensor([hi - lo], device=device),
                           onset_threshold=tau, bpm=bpm, **decode)[0].cpu().numpy()
        out[a:b] = g[a - lo: a - lo + (b - a)]   # keep the clean middle (skip warmup, before cooldown)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_songs', type=int, default=6); ap.add_argument('--max_len', type=int, default=1024)
    ap.add_argument('--warmup', type=int, default=24); ap.add_argument('--cooldown', type=int, default=16)
    ap.add_argument('--out_dir', default='outputs/sectional'); ap.add_argument('--install', action='store_true')
    ap.add_argument('--governor_off', action='store_true',
                    help='ablation: strip the decode-time governor (fatigue/stamina/breathe), keep everything else '
                         'canonical (calib, temps, playability). Isolates whether the breathe arc inflates the '
                         'responsiveness metric via its density modulation.')
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    decode = dict(DECODE)
    if args.governor_off:                              # one labeled change: governor off, all else canonical
        decode.update(fatigue_penalty=None, stamina_ceiling=None, stamina_breathe=0.0)
    _, vf, _ = split_chart_files(random_state=42)  # discover data/ + seeded split (harness)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    # 42-dim HIGHRES features (use_highres_onset=True, cache samples_v3) = what gen_motif_full_fixed expects.
    fspec = make_feature_extractor("highres")  # harness = single source of the feature ladder
    ds = StepManiaDataset(chart_files=vf[:args.num_songs * 40], audio_dir="data/", max_sequence_length=msl,
                          feature_extractor=fspec.extractor, cache_dir=fspec.cache_dir)
    gen = load_generator(GEN_CKPT, fspec.audio_dim, device)  # builds + loads (strict=False) + .eval()
    order = select_by_groove(ds, by='rich', difficulty=3)   # rich Hard songs with real structure

    out = Path(args.out_dir)
    for t in ('sectional', 'baseline'): (out / t).mkdir(parents=True, exist_ok=True)
    R = {k: {'b': [], 'r': []} for k in ('real', 'baseline', 'sectional')}
    Rnd = {k: {'b': [], 'r': []} for k in ('real', 'baseline', 'sectional')}  # density-DROPPED descriptor (pattern only)
    rows = []; used = 0
    for i in order:
        if used >= args.num_songs: break
        meta = ds.valid_samples[i]; sample = ds[i]; T = min(int(sample['mask'].sum().item()), args.max_len)
        if T < 256: continue
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        typed_r = np.asarray(ds.parser.convert_to_tensor_typed(meta['chart'], nd))[:T]
        bnds = foote_boundaries(sample['audio'][:T, SSM_DIMS].numpy())
        if len(bnds) < 2 or (typed_r != 0).any(1).sum() < 32: continue
        audio = sample['audio'][:T].unsqueeze(0).to(device); diff = torch.tensor([meta['difficulty_class']], device=device)
        bpm = float(meta['chart'].bpm)   # governor needs real BPM (press-rate); silent without it
        rd = float((typed_r != 0).any(1).mean())
        with torch.no_grad():
            p_on = calibrated_p_onset(gen, audio, diff, CALIB, device)   # tau uses the SAME 16th-unlock offset as decode
            tau = float(np.quantile(p_on, 1 - rd))
            set_seed(42)
            g0 = gen.generate(audio, diff, lengths=torch.tensor([T], device=device), onset_threshold=tau, bpm=bpm, **decode)[0].cpu().numpy()
        g1 = generate_sectional(gen, audio, diff, typed_r, bnds, args.warmup, args.cooldown, device, bpm, decode)
        charts = {'baseline': pair_holds(g0), 'sectional': pair_holds(g1)}
        per_song = {}; per_song_nd = {}
        for name, typed in [('real', typed_r), ('baseline', charts['baseline']), ('sectional', charts['sectional'])]:
            res = responsiveness(typed, bnds, np.random.default_rng(0), T)
            if res:
                R[name]['b'].append(res[0]); R[name]['r'].append(res[1])
                per_song[name] = res[0] - res[1]              # FULL responsiveness = @boundary - @random
            res_nd = responsiveness_nodens(typed, bnds, np.random.default_rng(0), T)  # density-DROPPED (pattern only)
            if res_nd:
                Rnd[name]['b'].append(res_nd[0]); Rnd[name]['r'].append(res_nd[1])
                per_song_nd[name] = res_nd[0] - res_nd[1]
        title = meta['chart'].title or Path(meta['chart_file']).stem
        rows.append((safe_name(title)[:24], len(bnds), per_song, per_song_nd))
        # export A/B
        music = os.path.basename(meta['audio_file']); dn = DIFFICULTY_NAMES[meta['difficulty_class']]
        for tag in ('baseline', 'sectional'):
            folder = out / tag / f"{used:02d}_{safe_name(title)}"; folder.mkdir(parents=True, exist_ok=True)
            if os.path.exists(meta['audio_file']):
                try: shutil.copy2(meta['audio_file'], folder / music)
                except Exception: pass
            sm = charts_to_sm(charts=[
                {"chart": charts[tag], "difficulty_name": "Challenge", "difficulty_value": nd.difficulty_value, "author": f"h11-{tag}"},
                {"chart": typed_r, "difficulty_name": dn, "difficulty_value": nd.difficulty_value, "author": "original"},
            ], bpm=bpm, title=f"{title} ({tag})", artist=meta['chart'].artist or "", music=music, offset=float(meta['chart'].offset), typed=True)
            (folder / "chart.sm").write_text(sm, encoding="utf-8")
        used += 1

    gov = ("GOVERNOR OFF (ablation): fatigue/stamina/breathe stripped" if args.governor_off
           else "full governor (fatigue=2 free=6 / stamina=50 tau=8 breathe=1.2)")
    print(f"\n=== Buffered-sectional ({used} songs, warmup={args.warmup} cooldown={args.cooldown}) ===")
    print(f"CONFIG: gen_motif_full_fixed (42-dim highres) + pattern_temp=1.0, type_temp=0.4, max_jack_run=2, "
          f"onset_phase_calib={CALIB} (16th-unlock).\n        {gov}.\n")
    # PER-SONG responsiveness FIRST — the pooled mean is known noisy & song-set-dependent (the Rule 11 trap
    # that gave the June-21 run a confidently-noisy non-answer). The per-song spread is the honest readout.
    # TWO descriptors: FULL [density,jump,L,D,U,R] vs NO-DENSITY [jump,L,D,U,R] (pattern-head choreography only).
    print("PER-SONG responsiveness — FULL descriptor | (no-density: pattern-only)")
    print(f"{'song':<24} {'bnds':>4}   {'real':>15} {'baseline':>15} {'sectional':>15}")
    print("-" * 80)
    for n, b, ps, psd in rows:
        def c(k): return (f"{ps.get(k, float('nan')):>+.3f}|{psd.get(k, float('nan')):>+.3f}"
                          if k in ps else "      -        ")
        print(f"{n:<24} {b:>4}   {c('real'):>15} {c('baseline'):>15} {c('sectional'):>15}")
    print("-" * 80)
    for label, RR in (("FULL descriptor [density,jump,L,D,U,R]", R),
                      ("NO-DENSITY [jump,L,D,U,R] — pattern-head choreography ONLY", Rnd)):
        print(f"\n{label}")
        print(f"{'chart':<12} {'@boundary':>10} {'@random':>9} {'responsiveness':>15}")
        print("-" * 50)
        for name in ('real', 'baseline', 'sectional'):
            bb, rr = np.mean(RR[name]['b']), np.mean(RR[name]['r'])
            print(f"{name:<12} {bb:>10.3f} {rr:>9.3f} {bb - rr:>15.3f}")
        print("-" * 50)
    print("\nREAD: if FULL baseline≈real but NO-DENSITY baseline<<real, the governor's gain is DENSITY (breathe arc), "
          "NOT the pattern head; the NO-DENSITY full-vs-gov-off gap = the pattern head's true transition contribution.")
    print(f"Exported {args.out_dir}/{{baseline,sectional}} for A/B playtest.")
    if args.install:
        from src.utils.sm_install import install_to_stepmania
        for d in install_to_stepmania(args.out_dir): print("installed:", d)


if __name__ == '__main__':
    main()

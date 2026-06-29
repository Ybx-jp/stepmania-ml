#!/usr/bin/env python3
"""PROBE: ONSET-HEAD PHRASING COHERENCE (diagnostic, no-retrain, no-generation, 2026-06-28).

GOAL (user): characterize whether the DEPLOYED onset head derives its OWN coherent choreographic phrasing —
NOT whether it copies a real chart's onsets (that's a faint sanity band only). The reference for "did it do the
right thing" is the MUSICAL EVENT (audio), not a reference chart. Four observable behaviors:

  1. BOUNDARY-SNAP   : at a section boundary, does the allocation shift ON the boundary frame (a step), or smear
                       (the breathing arc is a ~6-measure boxsmooth by construction)? metric = transition WIDTH
                       + LAG of the p_onset allocation envelope vs the Foote boundary.
  2. BURST-IN-QUIET  : in low-energy phrases, does p_onset spike for a sparse PERC burst AND a sparse HARM burst
                       (allocate for the lone percussion hit / lone melodic note) while staying ~empty otherwise?
                       metric = corr(p_onset, perc) & corr(p_onset, harm) restricted to bottom-quartile-energy
                       frames + the "emptiness-respect" mean p_onset in calm frames. (Symmetric perc/harm — the
                       sharp test of H-onset-perc-bias: responds to perc-in-quiet but not harm-in-quiet = the
                       melodic under-placement defect.)
  3. CLEAN-TAIL      : does allocation END when the music ends, or over-run ~a measure past audio-end (the known
                       knob-invariant defect)? metric = frames of realized onsets PAST the detected audio-end.
  4. PERC<->HARM FLUIDITY : can emphasis shift between percussive and harmonic onsets WITHIN a song / phrase?
                       metric = within-song RANGE of (corr_perc - corr_harm) across sliding windows. (diag_breathe_
                       energy already refuted a SONG-LEVEL perc bias — so we measure the TEMPORAL range, the new
                       within-song-rebalancing angle, not the global mean.)

Deployed onset path (the `generation-defaults` skill / export native mode): gen_motif_full_fixed (42-dim highres),
radar=None/style=None, onset_phase_calib=(0,1.0) applied to the logits (tau uses the same offset). 42-dim layout:
MFCC0=energy dim0, onset_env dim13, perc_onset dim35, harm_onset dim36, highres_onset dim41, chroma 23-34.

  python experiments/generation_typed/probe_phrasing_coherence.py [--match "high school love,kneeso,deja loin"]
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, sys
from pathlib import Path
import numpy as np, torch, yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.playability_metrics import ACTIVE_SYMBOLS
from diag_transitions_freerun import foote_boundaries, SSM_DIMS

# 42-dim highres channel indices (verified from src/data/audio_features.py assembly order)
I_ENERGY, I_ONSETENV, I_PERC, I_HARM, I_HIRES = 0, 13, 35, 36, 41
CALIB = (0.0, 1.0)   # deployed onset_phase_calib (b8, b16) — applied to logits; tau uses the same offset


def boxsmooth(x, w):
    w = max(int(w), 1); k = 2 * w + 1
    return np.convolve(np.pad(x, w, mode='edge'), np.ones(k) / k, mode='valid')


def z(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def norm01(x):
    lo, hi = np.min(x), np.max(x)
    return (x - lo) / (hi - lo + 1e-9)


def ncorr(a, b):
    if len(a) < 8 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def calibrated_p_onset(model, audio, diff, device, extra=None):
    """Deployed native p_onset: difficulty-only onset logits + the onset_phase_calib offset (+ an optional
    per-frame EXTRA logit offset, e.g. the hand-crafted sparse-harm-in-quiet calibrator), sigmoid."""
    with torch.no_grad():
        ol = model.onset_logits(model.encode_audio(audio), diff, radar=None, style=None)[0]
        b8, b16 = CALIB; ph = torch.arange(ol.shape[0], device=device) % 4
        ol = ol + torch.where(ph == 2, b8, torch.where((ph == 1) | (ph == 3), b16, 0.0))
        if extra is not None:
            ol = ol + torch.from_numpy(extra.astype(np.float32)).to(device)
        return torch.sigmoid(ol).cpu().numpy()


def gate01(feat, quiet_feat):
    """The smoothed, 0-1-normed 'is this frame in the gated band' source signal. quiet_feat:
      'energy' = dim-0 total loudness (the ORIGINAL gate; MISSES a piano solo — energy-loud, perc-absent).
      'perc'   = dim-35 percussive onset (the FIX; a piano solo is perc-ABSENT → its band fires IN the solo)."""
    src = feat[:, I_PERC] if quiet_feat == 'perc' else feat[:, I_ENERGY]
    return norm01(boxsmooth(src, 16))


def sparse_harm_offset(feat, harm_gain, quiet_q, quiet_feat='energy'):
    """STEP-1 hand-crafted calibrator: un-bury a sparse HARMONIC onset in a QUIET phrase (the mirror of the
    head's existing sparse-perc response). offset[t] = harm_gain · quiet_gate[t] · harm[t], a LOGIT boost that
    fires only where the GATE band is low AND a harmonic onset is present. quiet_feat picks the gate band
    (energy=original loudness gate; perc=percussion-absence gate, the piano-solo fix)."""
    g = gate01(feat, quiet_feat)
    q = np.percentile(g, quiet_q)
    quiet_gate = np.clip((q - g) / (q + 1e-6), 0.0, 1.0)          # 0 above the quiet quantile, →1 as gate→0
    return harm_gain * quiet_gate * feat[:, I_HARM]              # harm (dim36) already 0–1


# ---- the four axes -------------------------------------------------------------------------------
def axis1_boundary_snap(env, bnds, W=48):
    """LAG + WIDTH of the allocation-envelope step at each boundary. env = smoothed allocation signal.
    width = # frames the |gradient| stays above half its local peak (narrow = snappy step)."""
    g = np.abs(np.gradient(env)); lags, widths, deltas = [], [], []
    for b in bnds:
        lo, hi = max(0, b - W), min(len(env), b + W)
        seg = g[lo:hi]
        if seg.size < 8 or seg.max() < 1e-9:
            continue
        c = lo + int(np.argmax(seg))
        lags.append(c - int(b))
        widths.append(int((seg > seg.max() / 2).sum()))
        deltas.append(abs(env[hi - 1] - env[lo]))
    if not lags:
        return None
    return dict(lag=float(np.median(lags)), width=float(np.median(widths)), delta=float(np.mean(deltas)))


def axis2_burst_in_quiet(p, perc, harm, energy01, qlo=25):
    quiet = energy01 < np.percentile(energy01, qlo)
    if quiet.sum() < 16:
        return None
    cp, ch = ncorr(p[quiet], perc[quiet]), ncorr(p[quiet], harm[quiet])
    calm = quiet & (perc < np.percentile(perc, 50)) & (harm < np.percentile(harm, 50))
    burst = quiet & ((perc > np.percentile(perc, 80)) | (harm > np.percentile(harm, 80)))
    return dict(corr_perc=cp, corr_harm=ch,
                p_calm=float(p[calm].mean()) if calm.sum() else np.nan,
                p_burst=float(p[burst].mean()) if burst.sum() else np.nan, n_quiet=int(quiet.sum()))


def axis3_clean_tail(p, realized, energy01, tau, fhz):
    act = boxsmooth(energy01, 8)
    above = np.where(act > 0.10)[0]                      # last frame with audible loudness (10% of range)
    if above.size == 0:
        return None
    audio_end = int(above[-1])
    fired = np.where(realized[:] > 0)[0]
    last_onset = int(fired[-1]) if fired.size else audio_end
    over_f = max(0, last_onset - audio_end)
    return dict(audio_end=audio_end, last_onset=last_onset, overrun_frames=over_f,
                overrun_measures=over_f / 16.0, p_after=float(p[audio_end + 1:].mean()) if audio_end + 1 < len(p) else 0.0)


def axis4_percharm_fluidity(p, perc, harm, win=64, hop=16):
    diffs, centers = [], []
    for s in range(0, len(p) - win, hop):
        e = s + win
        cp, ch = ncorr(p[s:e], perc[s:e]), ncorr(p[s:e], harm[s:e])
        if not (np.isnan(cp) or np.isnan(ch)):
            diffs.append(cp - ch); centers.append(s + win // 2)
    if len(diffs) < 4:
        return None
    d = np.array(diffs)
    return dict(rng=float(d.max() - d.min()), std=float(d.std()), mean=float(d.mean()),
                series=d, centers=np.array(centers))


def load_songs(match, max_len):
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    _, vf, _ = create_data_splits(cf, random_state=42)
    subs = [s.strip().lower() for s in match.split(',') if s.strip()]
    vf = [f for f in vf if any(s in f.lower() for s in subs)]
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                   use_metric_phase=True, use_highres_onset=True))
    # cache_dir=None on PURPOSE: the shared cache/samples_v3 is keyed by INTEGER INDEX (sample_{idx}.pt) with
    # NO identity check, so a --match subset reads STALE features from a prior run's different file ordering
    # (verified: HSL served kneeso's audio). Fresh extraction (3 songs) is cheap and collision-proof.
    _, vds, _ = create_datasets(train_files=[], val_files=vf, test_files=[], audio_dir="data/",
                                max_sequence_length=msl, feature_extractor=ext, cache_dir=None)
    out = []
    for i in range(len(vds)):
        s = vds[i]; m = vds.valid_samples[i]
        if m['difficulty_class'] != 3:                   # Hard only
            continue
        T = min(int(s['mask'].sum().item()), max_len)
        nd = next((n for n in m['chart'].note_data if n.difficulty_name == m['difficulty_name']), None)
        if nd is None:
            continue
        real = np.asarray(vds.parser.convert_to_tensor_typed(m['chart'], nd))[:T]
        out.append(dict(title=str(getattr(m['chart'], 'title', None) or Path(m['chart_file']).stem),
                        audio=s['audio'][:T].numpy().astype(np.float32), T=T,
                        bpm=float(m['chart'].bpm), real=real, diff=3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/gen_motif_full_fixed/best_val.pt")
    ap.add_argument("--match", default="high school love,kneeso,deja loin")
    ap.add_argument("--max_len", type=int, default=1440)
    ap.add_argument("--harm_gain", type=float, default=0.0,
                    help="STEP-1 sparse-harm-in-quiet calibrator: logit boost ∝ quiet_gate·harm. 0 = off (pure "
                         "diagnostic). >0 prints a base-vs-calibrated A/B on axis-2 + density. ~2.0 to start.")
    ap.add_argument("--quiet_q", type=float, default=40.0, help="gate percentile defining 'quiet' for the calibrator")
    ap.add_argument("--quiet_feat", choices=["energy", "perc"], default="energy",
                    help="gate band: 'energy' (dim0, ORIGINAL — misses an energy-loud piano solo) or 'perc' "
                         "(dim35 percussion-absence, the FIX — fires IN a perc-absent melodic solo).")
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    songs = load_songs(args.match, args.max_len)
    if not songs:
        print(f"no Hard songs matched {args.match!r}"); return
    model = LayeredTypedChartGenerator(audio_dim=songs[0]['audio'].shape[1], d_model=128,
                                       num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)['model_state_dict']); model.eval()

    print("\nONSET PHRASING-COHERENCE DIAGNOSTIC (deployed head, native + calib (0,1.0); reference = AUDIO events).")
    print("Objective = the model's OWN coherent allocation vs musical events — NOT fidelity to the real chart.\n")
    fig, axes = plt.subplots(len(songs), 2, figsize=(15, 3.2 * len(songs)), squeeze=False)
    for row, s in zip(axes, songs):
        T = s['T']; feat = s['audio']; fhz = s['bpm'] * 4 / 60
        diff = torch.tensor([s['diff']], device=device)
        audio = torch.from_numpy(feat).unsqueeze(0).to(device)
        p = calibrated_p_onset(model, audio, diff, device)
        real_d = float((np.isin(s['real'], ACTIVE_SYMBOLS)).any(1).mean())
        tau = float(np.quantile(p, 1 - real_d)) if real_d > 0 else 0.5
        realized = (p > tau).astype(float)

        energy01 = norm01(boxsmooth(feat[:, I_ENERGY], 16))
        perc, harm = feat[:, I_PERC], feat[:, I_HARM]
        p_alloc = boxsmooth(p, 16)                        # model allocation envelope
        p_breathe = boxsmooth(p, 96)                      # the breathe boxsmooth (for the snap contrast)
        real_dens_env = boxsmooth(np.isin(s['real'], ACTIVE_SYMBOLS).any(1).astype(float), 16)
        bnds = foote_boundaries(feat[:, SSM_DIMS])

        a1 = axis1_boundary_snap(p_alloc, bnds); a1b = axis1_boundary_snap(p_breathe, bnds)
        a1r = axis1_boundary_snap(real_dens_env, bnds)
        a2 = axis2_burst_in_quiet(p, perc, harm, energy01)
        a3 = axis3_clean_tail(p, realized, energy01, tau, fhz)
        a4 = axis4_percharm_fluidity(p, perc, harm)

        print(f"=== {s['title'][:46]}  (T={T}, bpm={s['bpm']:.0f}, {len(bnds)} boundaries) ===")
        if a1 and a1b and a1r:
            print(f"  1 BOUNDARY-SNAP   p_onset: lag {a1['lag']:+.0f}f width {a1['width']:.0f}f "
                  f"({a1['width']/16:.2f}meas) | breathe width {a1b['width']:.0f}f | real width {a1r['width']:.0f}f")
        if a2:
            print(f"  2 BURST-IN-QUIET  corr(p,perc)={a2['corr_perc']:+.2f}  corr(p,harm)={a2['corr_harm']:+.2f}  "
                  f"| p_calm={a2['p_calm']:.3f}  p_burst={a2['p_burst']:.3f}  (n_quiet={a2['n_quiet']})")
        if a3:
            print(f"  3 CLEAN-TAIL      audio_end@{a3['audio_end']}  last_onset@{a3['last_onset']}  "
                  f"OVER-RUN {a3['overrun_frames']}f ({a3['overrun_measures']:.2f}meas)  p_after={a3['p_after']:.3f}")
        if a4:
            print(f"  4 PERC<->HARM     range(corr_perc-corr_harm)={a4['rng']:.2f}  std={a4['std']:.2f}  "
                  f"mean={a4['mean']:+.2f}  ({'fluid' if a4['rng'] > 0.5 else 'locked-ish'})")
        if args.harm_gain > 0:                            # STEP-1 A/B: the hand-crafted sparse-harm-in-quiet offset
            off = sparse_harm_offset(feat, args.harm_gain, args.quiet_q, args.quiet_feat)
            p_adj = calibrated_p_onset(model, audio, diff, device, extra=off)
            tau_adj = float(np.quantile(p_adj, 1 - real_d)) if real_d > 0 else 0.5
            a2c = axis2_burst_in_quiet(p_adj, perc, harm, energy01)
            # density: realized note-rate inside the GATED frames (did the calibrator actually allocate there?)
            g = gate01(feat, args.quiet_feat)
            qn = g < np.percentile(g, args.quiet_q)
            dq_b = float((p[qn] > tau).mean()); dq_a = float((p_adj[qn] > tau_adj).mean())
            d_b = float((p > tau).mean()); d_a = float((p_adj > tau_adj).mean())
            # DECISIVE: fraction of the offset MASS that lands on MELODIC-DOMINANT frames (high harm, low perc —
            # the piano-solo signature). High = the gate targets the solo; low (energy gate) = it leaks elsewhere.
            mel = (harm > np.percentile(harm, 75)) & (perc < np.percentile(perc, 50))
            frac_mel = float(off[mel].sum() / (off.sum() + 1e-9))
            # where the boost mass is centered in the song (frame), to read 'in the solo' vs 'after it'
            centroid = float((np.arange(T) * off).sum() / (off.sum() + 1e-9)) if off.sum() > 0 else np.nan
            print(f"  ── CALIB(gain={args.harm_gain}, gate={args.quiet_feat}) axis-2  corr_harm "
                  f"{a2['corr_harm']:+.2f}→{a2c['corr_harm']:+.2f}"
                  f"  p_calm→burst {a2['p_calm']:.2f}/{a2['p_burst']:.2f} → {a2c['p_calm']:.2f}/{a2c['p_burst']:.2f}"
                  f"  | gated-dens {dq_b:.3f}→{dq_a:.3f}  global-dens {d_b:.3f}→{d_a:.3f}"
                  f"  | offset→melodic {frac_mel:.2f}  centroid@{centroid:.0f}/{T}")
        print()

        ax = row[0]; t = np.arange(T)
        ax.plot(t, z(energy01), color='#999', lw=1, label='energy (MFCC0)')
        ax.plot(t, z(p_alloc), color='#1f77b4', lw=1.4, label='p_onset alloc (box16)')
        ax.plot(t, z(boxsmooth(perc, 16)), color='#d62728', lw=0.9, alpha=.7, label='perc onset')
        ax.plot(t, z(boxsmooth(harm, 16)), color='#2ca02c', lw=0.9, alpha=.7, label='harm onset')
        for b in bnds:
            ax.axvline(b, color='k', alpha=0.25, lw=0.8)
        if a3:
            ax.axvline(a3['audio_end'], color='purple', ls='--', lw=1.3, label='audio-end')
            ax.axvline(a3['last_onset'], color='orange', ls=':', lw=1.3, label='last onset')
        ax.set_title(f"{s['title'][:40]} — phrasing (boundaries=grey)", fontsize=9)
        ax.legend(fontsize=6, ncol=3, loc='upper right'); ax.set_xlabel("frame (16th)")

        ax2 = row[1]
        if a4:
            ax2.plot(a4['centers'], a4['series'], color='#9467bd', lw=1.2)
            ax2.axhline(0, color='k', lw=0.7)
            ax2.fill_between(a4['centers'], 0, a4['series'], where=a4['series'] > 0, color='#d62728', alpha=.25)
            ax2.fill_between(a4['centers'], 0, a4['series'], where=a4['series'] < 0, color='#2ca02c', alpha=.25)
            for b in bnds:
                ax2.axvline(b, color='k', alpha=0.2, lw=0.7)
            ax2.set_title(f"perc↔harm emphasis  corr_perc−corr_harm  (range {a4['rng']:.2f})", fontsize=9)
            ax2.set_xlabel("frame (16th)"); ax2.set_ylabel("+perc / −harm")
    fig.tight_layout()
    outp = PROJECT_ROOT / "outputs" / "phrasing_coherence.png"
    fig.savefig(outp, dpi=110); print(f"saved plot -> {outp}")


if __name__ == "__main__":
    main()

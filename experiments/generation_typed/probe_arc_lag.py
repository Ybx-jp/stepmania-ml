#!/usr/bin/env python3
"""PROBE: WHERE does the "lag adapting to phase changes" come from? (HSL piano-solo cold-start +
slow-to-back-off, observed in unlock16_b20, 2026-06-28).

Three candidate lag sources, which make DIFFERENT predictions:
  (A) the breathing arc's intensity envelope = a CENTERED boxsmooth of p_onset over stamina_breathe_win=96
      frames (~6 measures). A centered smoother is ZERO-PHASE: it can only WIDEN (blur symmetrically) the
      response, it CANNOT create a one-sided lag -- if anything it ANTICIPATES. (This arc postdates H11, so
      H11 never tested it.)
  (B) the onset head's intrinsic p_onset envelope (audio-only, non-causal -> should NOT be causally late).
  (C) the AR pattern head carrying the prior section's pattern across a boundary = CAUSAL, only-late = a true
      one-sided lag. This is **H11 (`notes/h11_transitions_findings.md`), ALREADY characterized**: teacher-
      forced loss at boundaries is flat (not a representational deficit), free-run under-transitions (AR drift),
      "cold-start = the t=0 special case." H11's *magnitude* metric is BLIND to direction; H11's pooled free-run
      gap is NOISY/song-set-dependent (`buffered_sectional.py`) -> do NOT re-run a pooled responsiveness average.

So this probe is the cheap, no-generation, TIMING + DIRECTION cut H11 lacks, on the COMPLAINT song (HSL; Rule
5/11 = the population that exhibits it), isolating the variable I changed this session (breathe).

DECISION RULE (fixed before reading, Rule 9):
  - breathe env LEADS or matches p_onset (centroid lag <= 0)  -> breathe is ZERO-PHASE as predicted, it CANNOT
    be the cold-start cause (a knob fix to stamina_breathe_win would only change blur width, not the lag).
  - p_onset does NOT lag the audio energy (lag ~ 0)            -> the onset head isn't late either.
  - => the remaining one-sided lag is the PATTERN AR -> cite H11, it's a documented post-0.1.0 thread. No new
    mechanism, no retrain decision here.
  - If instead p_onset or the breathe env DOES lag the audio -> that's the new, cheaper-to-fix source; escalate
    to the generation confirmation (breathe ON vs OFF realized-density timing).

  python experiments/generation_typed/probe_arc_lag.py [--match "high school love,kneeso"] [--win 96]
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, sys
from pathlib import Path
import numpy as np, torch, yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.playability_metrics import ACTIVE_SYMBOLS

# Foote-novelty boundary detector — REUSED verbatim from diag_transitions_freerun.py (Rule 14: same
# section-boundary definition as the prior H11 work, not a hand-rolled one).
L = 32
SSM_DIMS = list(range(0, 13)) + list(range(23, 35))   # MFCC(timbre) + chroma(harmony)


def foote_boundaries(feat, topk=6):
    f = feat - feat.mean(0, keepdims=True); f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
    S = f @ f.T; a = np.arange(2 * L)
    sign = np.where((a[:, None] < L) == (a[None, :] < L), 1.0, -1.0)
    gw = np.exp(-((a[:, None] - L + .5) ** 2 + (a[None, :] - L + .5) ** 2) / (2 * (L / 2) ** 2))
    ker = sign * gw; T = len(feat); nov = np.zeros(T)
    for t in range(L, T - L):
        nov[t] = (S[t - L:t + L, t - L:t + L] * ker).sum()
    nov = np.maximum(nov, 0); pos = nov[nov > 0]
    pk, props = find_peaks(nov, distance=2 * L, prominence=max(pos.std() if pos.size else 0, 1e-6))
    if len(pk) > topk:
        pk = pk[np.argsort(props['prominences'])[::-1][:topk]]
    return np.sort(pk)


def boxsmooth(x, w):
    w = max(int(w), 1); k = 2 * w + 1
    xp = np.pad(x, w, mode='edge')
    return np.convolve(xp, np.ones(k) / k, mode='valid')


def z(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def lag_of(sig, ref, max_lag):
    """How much `sig` LAGS `ref`, in frames (+ = sig later). argmax_L corr(ref[t], sig[t+L])."""
    N = min(len(ref), len(sig)); ref, sig = ref[:N], sig[:N]   # chart tensor can be 1 frame short of audio
    best, bc = 0, -2.0
    for Lg in range(-max_lag, max_lag + 1):
        if Lg >= 0:
            a, b = ref[:N - Lg], sig[Lg:]
        else:
            a, b = ref[-Lg:], sig[:N + Lg]
        if len(a) < 8 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
            continue
        c = float(np.corrcoef(a, b)[0, 1])
        if c > bc:
            bc, best = c, Lg
    return best, bc


def load_match(match, max_len, device):
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    tf, vf, _ = create_data_splits(cf, random_state=42)
    subs = [s.strip().lower() for s in match.split(',') if s.strip()]
    vf = [f for f in vf if any(s in f.lower() for s in subs)]
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                   use_metric_phase=True, use_highres_onset=True))
    _, vds, _ = create_datasets(train_files=[], val_files=vf, test_files=[], audio_dir="data/",
                                max_sequence_length=msl, feature_extractor=ext, cache_dir='cache/samples_v3')
    vds.warm_cache(show_progress=False)
    out = []
    for i in range(len(vds)):
        s = vds[i]; m = vds.valid_samples[i]
        if m['difficulty_class'] != 3:                    # Hard only (the played set)
            continue
        T = min(int(s['mask'].sum().item()), max_len)
        title = getattr(m['chart'], 'title', None) or Path(m['chart_file']).stem   # title lives on the chart obj
        nd = next((n for n in m['chart'].note_data if n.difficulty_name == m['difficulty_name']), None)
        if nd is None:
            continue
        real = vds.parser.convert_to_tensor_typed(m['chart'], nd)[:T]
        out.append({'title': str(title), 'audio': s['audio'][:T].numpy().astype(np.float32), 'len': T,
                    'diff': 3, 'bpm': float(m['chart'].bpm), 'real': np.asarray(real),
                    'radar': s['groove_radar'].numpy().astype(np.float32)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/gen_motif_full_fixed/best_val.pt")
    ap.add_argument("--match", default="high school love,kneeso")  # HSL (complaint) + japa1 (contrast)
    ap.add_argument("--max_len", type=int, default=1440)
    ap.add_argument("--win", type=int, default=96)          # the breathe boxsmooth window (stamina_breathe_win)
    ap.add_argument("--max_lag", type=int, default=64)      # +/- 16 beats of search
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    songs = load_match(args.match, args.max_len, device)
    if not songs:
        print(f"no Hard songs matched {args.match!r}"); return
    audio_dim = songs[0]['audio'].shape[1]
    model = LayeredTypedChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=4, onset_layers=2).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)['model_state_dict']); model.eval()

    print(f"\nPROBE — intensity-envelope LAG at phase changes (no-gen core). win(breathe)={args.win}f, "
          f"max_lag={args.max_lag}f (~{args.max_lag/4:.0f} beats)")
    print("Reference = audio energy (MFCC0). + lag = signal is LATE vs the audio. Foote section boundaries marked.\n")

    fig, axes = plt.subplots(len(songs), 1, figsize=(13, 3.4 * len(songs)), squeeze=False)
    for ax, s in zip(axes[:, 0], songs):
        T = s['len']; feat = s['audio']
        audio_e = boxsmooth(feat[:, 0], 16)               # MFCC0 ~ log-energy, lightly smoothed (musical intensity)
        bnds = foote_boundaries(feat[:, SSM_DIMS])
        a = torch.from_numpy(feat).unsqueeze(0).to(device)
        dt = torch.tensor([s['diff']], device=device)
        rad = torch.from_numpy(s['radar']).unsqueeze(0).to(device)
        with torch.no_grad():
            p_onset = torch.sigmoid(model.onset_logits(model.encode_audio(a), dt, radar=rad)[0]).cpu().numpy()[:T]
        p_sm = boxsmooth(p_onset, 16)                      # onset head intensity (lightly smoothed)
        p_breathe = boxsmooth(p_onset, args.win)           # the ARC's signal (centered = zero-phase)
        real_d = boxsmooth(np.isin(s['real'], ACTIVE_SYMBOLS).any(1).astype(float), args.win)  # human density env

        l_pon, c_pon = lag_of(p_sm, audio_e, args.max_lag)            # onset head vs audio
        l_bre, c_bre = lag_of(p_breathe, audio_e, args.max_lag)       # arc signal vs audio
        l_b_vs_p, c_bp = lag_of(p_breathe, p_sm, args.max_lag)        # arc vs p_onset (zero-phase check)
        l_real, _ = lag_of(real_d, audio_e, args.max_lag)             # human density vs audio (reference)
        fhz = s['bpm'] * 4 / 60

        def fr(x):
            return f"{x:+d}f ({x/4:+.1f}beat, {x/fhz*1000:+.0f}ms)"
        print(f"=== {s['title'][:40]}  (T={T}, bpm={s['bpm']:.0f}) ===")
        print(f"  onset p_onset  lags audio by   {fr(l_pon)}   r={c_pon:.2f}")
        print(f"  breathe env    lags audio by   {fr(l_bre)}   r={c_bre:.2f}")
        print(f"  breathe env    lags p_onset by {fr(l_b_vs_p)}   r={c_bp:.2f}   (<=0 => zero-phase, CANNOT cause the lag)")
        print(f"  [ref] human density lags audio by {fr(l_real)}\n")

        Tp = min(len(audio_e), len(p_sm), len(p_breathe), len(real_d))   # chart env can be a few frames short
        t = np.arange(Tp)
        ax.plot(t, z(audio_e[:Tp]), color='#888', lw=1, label='audio energy (MFCC0)')
        ax.plot(t, z(p_sm[:Tp]), color='#1f77b4', lw=1.2, label='p_onset (onset head)')
        ax.plot(t, z(p_breathe[:Tp]), color='#d62728', lw=1.5, label=f'breathe env (box{args.win})')
        ax.plot(t, z(real_d[:Tp]), color='#2ca02c', lw=1, ls='--', label='human density')
        for b in bnds:
            ax.axvline(b, color='k', alpha=0.25, lw=0.8)
        ax.set_title(f"{s['title'][:48]}  |  p_onset lag {l_pon:+d}f, breathe-vs-p_onset {l_b_vs_p:+d}f")
        ax.set_xlabel("frame (16th)"); ax.set_ylabel("z"); ax.legend(fontsize=7, ncol=4, loc='upper right')
    fig.tight_layout()
    outp = PROJECT_ROOT / "outputs" / "arc_lag_envelopes.png"
    fig.savefig(outp, dpi=110); print(f"saved plot -> {outp}")
    print("\nREAD: if breathe-vs-p_onset lag <= 0 (zero-phase confirmed) AND p_onset lag vs audio ~ 0, then "
          "neither the arc nor the onset head produces the one-sided cold-start lag -> it is the AR PATTERN head "
          "(H11, already characterized; post-0.1.0). If p_onset/breathe DO lag audio, that is the new source.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Does the stamina BREATHING arc's WHOLE-SONG z-normalization mis-scope LONG songs? (explore/taste-critic)

Hypothesis (user, 2026-07-11): the stamina system was tuned on the train/val corpus, which is GATED to <=130s.
The Stage-3 breathing ceiling z-normalizes the onset-energy envelope over the ENTIRE song (typed_model.py:693-699):
  z = (env - mean_wholesong) / std_wholesong ;  ceiling = base*(1 + breathe*z), clamp min=floor*base
On a ~1-arc short song that's a fair local reference. On a long verse/chorus/BREAKDOWN song the whole-song mean is
a MIXTURE, so a quiet breakdown reads deeply-negative z -> ceiling slammed to the floor -> max thinning exactly
where a human charts interesting sparse choreography ("neuters the onset head").

CHEAP DECISIVE PROBE (no AR loop, no chart generation): for each song compute the deployed p_onset (v2 onset head,
sliding-window so the tail isn't a PE artifact), then the breathing ceiling TWO ways -- deployed GLOBAL-z vs a
LOCAL-z (rolling ~one-training-song-span window, the proposed fix) -- and measure their DIVERGENCE vs song length.
Prediction: ~0 divergence on <=130s songs (global==local), large + more floored sections on long songs.
NOTE: this measures the CEILING (the mechanism). Whether it BITES also needs E_slow > ceiling (the AR-loop
generation ON/OFF/flat probe is the follow-up); a floored ceiling is the necessary condition.

Usage: OMP_NUM_THREADS=4 python experiments/generation_typed/probe_stamina_longsong.py
"""
import warnings, os, sys, glob, argparse
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F

ROOT = Path('/home/ybx/code/stepmania-chart-generator'); sys.path.insert(0, str(ROOT))
from src.data.stepmania_parser import StepManiaParser
from src.generation.decode_harness import make_feature_extractor, load_generator, conditioned_p_onset, MODEL_ARCH

V2_CKPT = "checkpoints/gen_motif_v2_48th_cont/best_val.pt"
V2_CTX, V2_MSL, SUBDIV = 5504, 5400, 12
BASE, BREATHE, FLOOR_FRAC, BREATHE_WIN = 50.0, 1.2, 0.4, 96   # canonical stamina (generation-defaults)
F16 = SUBDIV // 4
PHRASE_W = BREATHE_WIN * F16                                  # 288 frames (subdiv-scaled), the deployed phrase window
HARD = 3
PERSONAL = Path('/home/ybx/sm-personal')
EXCLUDE = {'lick the rainbow', 'pump up the jam', 'jealous',
           'stereo sayan 3d [lezbeepic remix]', 'heroes'}     # user-flagged bad/varbpm (2026-07-11)
AUDIO_EXT = ['.ogg', '.mp3', '.wav', '.flac', '.m4a']


def find_audio(chart_audio, sm_path):
    d = sm_path.parent
    if chart_audio and (d / chart_audio).exists(): return d / chart_audio
    for e in AUDIO_EXT:
        g = sorted(d.glob('*' + e))
        if g: return g[0]
    return None


def boxcar(x, w):
    """centered moving average over 2w+1 (count_include_pad=False), matching typed_model.py:691."""
    return F.avg_pool1d(x.view(1, 1, -1), kernel_size=2 * w + 1, stride=1, padding=w,
                        count_include_pad=False).view(-1)


def rolling_mean_std(env, win):
    """Local mean/std over a centered rolling window of `win` frames, via boxcar of x and x^2."""
    w = max(win // 2, 1)
    mu = boxcar(env, w)
    ex2 = boxcar(env * env, w)
    var = (ex2 - mu * mu).clamp(min=1e-6)
    return mu, var.sqrt()


def breathing_ceiling(p_onset, local_win=None):
    env = boxcar(p_onset, PHRASE_W)
    if local_win is None:                                    # DEPLOYED: whole-song z
        mu = env.mean(); std = env.var(unbiased=False).clamp(min=1e-6).sqrt()
        z = (env - mu) / std
    else:                                                    # PROPOSED FIX: local rolling z
        mu, std = rolling_mean_std(env, local_win)
        z = (env - mu) / std
    ceil = (BASE * (1.0 + BREATHE * z)).clamp(min=BASE * FLOOR_FRAC)
    return ceil, z


@torch.no_grad()
def p_onset_for(model, device, audio_np):
    T = audio_np.shape[0]
    if T > int(model.pos_encoding.pe.size(1)):
        from src.generation.transformer import PositionalEncoding
        model.pos_encoding = PositionalEncoding(model.pos_encoding.pe.shape[-1], max_len=T + 128).to(device)
    audio = torch.from_numpy(audio_np.astype(np.float32)).unsqueeze(0).to(device)
    memory = model.encode_audio(audio)
    diff = torch.tensor([HARD], device=device)
    p = conditioned_p_onset(model, memory, diff, phase_calib=(0.0, 1.0), subdiv=SUBDIV, window=V2_MSL)
    return np.asarray(p).reshape(-1)


def collect_songs(args):
    songs = []   # (label, sm_path)
    # LONG: personal songs that HAVE a chart + audio, minus the excluded set
    for smf in sorted(glob.glob(str(PERSONAL / '**' / '*.sm'), recursive=True)):
        if Path(smf).parent.name.lower() in EXCLUDE: continue
        songs.append(('personal', Path(smf)))
    # SHORT: a few train-corpus songs (<=130s by the default gate) for the low-length anchor
    short = []
    for smf in glob.glob(f"{args.data_dir}/**/*.sm", recursive=True):
        if len(short) >= args.n_short: break
        short.append(('train', Path(smf)))
    return short + songs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data'); ap.add_argument('--audio_dir', default='data')
    ap.add_argument('--n_short', type=int, default=8)
    ap.add_argument('--local_win', type=int, default=3600)   # ~one training-song span (300 beats @ subdiv12)
    ap.add_argument('--max_songs', type=int, default=24)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = StepManiaParser.for_v2(subdiv=SUBDIV, min_song_length=30.0, max_song_length=600.0,
                                    min_bpm=40.0, max_bpm=320.0, max_simultaneous=4, gimmick_max_bpm=400.0)
    spec = make_feature_extractor("highres_v2")
    v2_arch = dict(MODEL_ARCH, max_len=V2_CTX)
    model = load_generator(ROOT / V2_CKPT, spec.audio_dim, device, arch=v2_arch)
    print(f"local_win={args.local_win} frames | phrase_w={PHRASE_W} | base={BASE} floor={BASE*FLOOR_FRAC}\n")

    rows = []
    for label, smf in collect_songs(args):
        if len(rows) >= args.max_songs: break
        try:
            chart = parser.parse_file(str(smf))
            if chart is None: continue
            ap_ = find_audio(chart.audio_file, smf)
            if ap_ is None: continue
            feats = spec.extractor.extract_from_chart(str(ap_), chart)
            if feats is None: continue
            a = feats.get_aligned_features()
            if not np.all(np.isfinite(a)) or a.shape[0] < 256: continue
            model.pos_encoding = model.pos_encoding.__class__(model.pos_encoding.pe.shape[-1],
                                                              max_len=V2_CTX).to(device)  # reset PE per song
            p = p_onset_for(model, device, a)
        except Exception as e:
            print(f"  [skip] {smf.parent.name[:30]}: {e}"); continue
        T = len(p); dur = T * chart.hop_length / 22050.0
        pt = torch.from_numpy(p.astype(np.float32))
        cg, zg = breathing_ceiling(pt, None)
        cl, _ = breathing_ceiling(pt, args.local_win)
        floor = BASE * FLOOR_FRAC + 1e-3
        rows.append({'label': label, 'song': smf.parent.name[:30], 'T': T, 'dur': dur, 'bpm': chart.bpm,
                     'div': float((cg - cl).abs().mean() / BASE),
                     'floor_g': float((cg <= floor).float().mean()),
                     'floor_l': float((cl <= floor).float().mean()),
                     'zrange': float(np.percentile(zg.numpy(), 95) - np.percentile(zg.numpy(), 5))})
        r = rows[-1]
        print(f"  {r['label']:8s} {r['song']:30s} {r['dur']:5.0f}s T={T:5d} bpm={r['bpm']:5.1f} | "
              f"div={r['div']:.3f} floorG={r['floor_g']*100:4.0f}% floorL={r['floor_l']*100:4.0f}% "
              f"zrng={r['zrange']:.2f}")

    if not rows: print("no songs"); return
    print("\n" + "=" * 78)
    for grp in ('train', 'personal'):
        g = [r for r in rows if r['label'] == grp]
        if not g: continue
        dv = np.array([r['div'] for r in g]); fg = np.array([r['floor_g'] for r in g])
        fl = np.array([r['floor_l'] for r in g]); du = np.array([r['dur'] for r in g])
        print(f"{grp:8s} n={len(g):2d} | dur {du.mean():4.0f}s | global-vs-local ceiling div {dv.mean():.3f} | "
              f"floored(global) {fg.mean()*100:4.0f}% vs (local) {fl.mean()*100:4.0f}%  "
              f"[extra floored by global: {(fg-fl).mean()*100:+4.0f}%]")
    allr = rows
    du = np.array([r['dur'] for r in allr]); dv = np.array([r['div'] for r in allr])
    if len(allr) > 3 and du.std() > 1:
        print(f"\ncorr(duration, global-vs-local divergence) = {np.corrcoef(du, dv)[0,1]:+.3f}  (predict +)")
        fex = np.array([r['floor_g'] - r['floor_l'] for r in allr])
        print(f"corr(duration, EXTRA sections floored by global) = {np.corrcoef(du, fex)[0,1]:+.3f}  (predict +)")


if __name__ == '__main__':
    main()

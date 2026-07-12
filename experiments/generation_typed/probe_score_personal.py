#!/usr/bin/env python3
"""CHEAP DIAGNOSTIC (explore/taste-critic-quality-resolution): does the taste critic have
RESOLUTION on GOOD charts, and did the GRADED critic fix the near-binary rail problem?

Scores every hand-made chart in ~/sm-personal with BOTH critics:
  - BINARY  (checkpoints/realism_critic)         -> P(real) = softmax[:,1]   AND logit margin
  - GRADED  (checkpoints/realism_critic_graded)  -> logit margin (its trained score)

For each real chart it also builds the train_graded_critic PANEL-SCRAMBLE ladder
[0, .2, .45, .7, 1.0] (density/timing/audio held FIXED; only arrow-choice taste degrades) and
scores every rung. This is the exact eval the graded critic was optimized against -- now run on
the OUT-OF-DISTRIBUTION personal set it never trained on.

Replicates eval_taste_current.py's feeding: 42-dim highres features (dims 0..22 == the base-23
critic space), first max_len frames, binary note-PRESENCE grid.
"""
import warnings, os, sys, glob, json
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
from pathlib import Path
import numpy as np, torch

ROOT = Path('/home/ybx/code/stepmania-chart-generator')
sys.path.insert(0, str(ROOT))
from src.data.stepmania_parser import StepManiaParser
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
from src.models import LateFusionClassifier

PERSONAL = Path('/home/ybx/sm-personal')
BINARY = ROOT / 'checkpoints/realism_critic/best_val.pt'
GRADED = ROOT / 'checkpoints/realism_critic_graded/best_val.pt'
MAX_LEN = 768
PANEL_LADDER = [0.0, 0.2, 0.45, 0.7, 1.0]
AUDIO_EXT = ['.ogg', '.mp3', '.wav', '.flac', '.m4a']


def to_binary(typed):
    t = np.asarray(typed); return ((t == 1) | (t == 2) | (t == 4)).astype(np.float32)


def corrupt_panels_frac(chart, frac, rng):
    if frac <= 0: return chart.copy()
    out = chart.copy(); rows = np.where(chart.any(1))[0]
    pick = rng.random(len(rows)) < frac
    for t, hit in zip(rows, pick):
        if not hit: continue
        k = int(chart[t].sum()); out[t] = 0.0
        out[t, rng.choice(4, size=k, replace=False)] = 1.0
    return out


def find_audio(chart_audio, sm_path):
    d = sm_path.parent
    if chart_audio:
        p = d / chart_audio
        if p.exists(): return p
        stem = Path(chart_audio).stem
        for e in AUDIO_EXT:
            p = d / (stem + e)
            if p.exists(): return p
    # fallback: any audio in dir
    for e in AUDIO_EXT:
        g = sorted(d.glob('*' + e))
        if g: return g[0]
    return None


def load_critic(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    c = LateFusionClassifier(ck['config']).to(device)
    c.load_state_dict(ck['model_state_dict']); c.eval()
    return c


@torch.no_grad()
def score(critic, a23, chart, device):
    """Returns (P_real, margin) for a single (T,23) audio + (T,4) chart."""
    a = torch.from_numpy(a23).unsqueeze(0).to(device)
    c = torch.from_numpy(chart).unsqueeze(0).to(device)
    m = torch.ones(1, a.shape[1], device=device)
    logits = critic(a, c, m)
    if isinstance(logits, dict): logits = logits['logits']
    logits = logits[0]
    p = float(torch.softmax(logits, 0)[1])
    margin = float(logits[1] - logits[0])
    return p, margin


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = StepManiaParser.for_inference()   # v1 16th grid (critic space) + WIDENED gates so real BYO songs parse
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                   use_metric_phase=True, use_highres_onset=True))
    binc = load_critic(BINARY, device); gradc = load_critic(GRADED, device)
    rng = np.random.default_rng(42)
    print(f"device={device}  binary={BINARY.name}  graded={GRADED.name}")

    sm_files = sorted(glob.glob(str(PERSONAL / '**' / '*.sm'), recursive=True))
    print(f"found {len(sm_files)} .sm files under {PERSONAL}\n")

    rows = []          # per-chart handmade rows
    ladder_bin = [[] for _ in PANEL_LADDER]   # binary P(real) per level (pooled)
    ladder_grd = [[] for _ in PANEL_LADDER]   # graded margin per level (pooled)
    mono_bin = mono_grd = nlad = 0

    for smf in sm_files:
        smf = Path(smf)
        try:
            chart = parser.parse_file(str(smf))
            if chart is None or not chart.note_data:
                print(f"  [skip] parse-empty: {smf.parent.name}"); continue
            audio_path = find_audio(chart.audio_file, smf)
            if audio_path is None:
                print(f"  [skip] no audio: {smf.parent.name}"); continue
            feats = ext.extract_from_chart(str(audio_path), chart)
            if feats is None:
                print(f"  [skip] feat-fail: {smf.parent.name}"); continue
            a_full = feats.get_aligned_features()   # (Ta, 42)
            _bpms = sorted({round(e.value, 2) for e in chart.timing_events
                            if e.event_type == 'bpm' and e.value > 0})
            varbpm = len(_bpms) > 1
        except Exception as e:
            print(f"  [skip] {smf.parent.name}: {e}"); continue

        for nd in chart.note_data:
            if not nd.difficulty_name: continue
            try:
                typed = parser.convert_to_tensor_typed(chart, nd)
            except Exception as e:
                print(f"  [skip diff] {smf.parent.name}/{nd.difficulty_name}: {e}"); continue
            real = to_binary(typed)
            T = min(real.shape[0], a_full.shape[0], MAX_LEN)
            if T < 64: continue
            real = real[:T]; a23 = a_full[:T, :23].astype(np.float32)
            if not real.any(): continue

            pb, mb = score(binc, a23, real, device)
            _, mg = score(gradc, a23, real, device)
            rows.append({'song': smf.parent.name, 'diff': nd.difficulty_name,
                         'val': nd.difficulty_value, 'T': T, 'varbpm': varbpm,
                         'bin_p': pb, 'bin_m': mb, 'grd_m': mg,
                         'dens': float(real.any(1).mean())})

            # panel-scramble ladder (only on the hardest/one chart per song is enough, but do each)
            lb, lg = [], []
            for j, frac in enumerate(PANEL_LADDER):
                ch = corrupt_panels_frac(real, frac, rng)
                pb_, _ = score(binc, a23, ch, device)
                _, mg_ = score(gradc, a23, ch, device)
                ladder_bin[j].append(pb_); ladder_grd[j].append(mg_)
                lb.append(pb_); lg.append(mg_)
            mono_bin += int(np.all(np.diff(lb) <= 1e-6))
            mono_grd += int(np.all(np.diff(lg) <= 1e-6))
            nlad += 1

    if not rows:
        print("NO charts scored."); return

    # ---- per-chart table ----
    rows.sort(key=lambda r: r['bin_p'])
    print("\n" + "=" * 96)
    print(f"{'song':32s} {'diff':10s} {'lvl':>3s} {'T':>4s} {'dens':>5s} {'BINp':>6s} {'BINm':>7s} {'GRDm':>7s} {'vB':>2s}")
    print("-" * 96)
    for r in rows:
        print(f"{r['song'][:32]:32s} {r['diff'][:10]:10s} {r['val']:>3d} {r['T']:>4d} "
              f"{r['dens']:>5.2f} {r['bin_p']:>6.3f} {r['bin_m']:>+7.2f} {r['grd_m']:>+7.2f} "
              f"{'Y' if r['varbpm'] else '':>2s}")

    bp = np.array([r['bin_p'] for r in rows]); gm = np.array([r['grd_m'] for r in rows])
    print("=" * 96)
    print(f"\nHAND-MADE charts (n={len(rows)}):  BIN P(real) mean={bp.mean():.3f} "
          f"median={np.median(bp):.3f}  [train REAL rung baseline = 0.823]")
    print(f"  BIN P(real) distribution:  <0.1: {np.mean(bp<0.1)*100:.0f}%   "
          f"mid 0.1-0.9: {np.mean((bp>=0.1)&(bp<=0.9))*100:.0f}%   >0.9: {np.mean(bp>0.9)*100:.0f}%   "
          f"[train REAL: 9% / 14% / 77%]")
    print(f"  GRADED margin mean={gm.mean():+.2f} median={np.median(gm):+.2f} "
          f"range [{gm.min():+.2f}, {gm.max():+.2f}]")

    # ---- ladder (graded-ness gate) ----
    print(f"\nPANEL-SCRAMBLE LADDER (pooled over n={nlad} charts) -- density/timing/audio FIXED, taste degrades:")
    print(f"  frac scrambled : " + "  ".join(f"{f:>6.2f}" for f in PANEL_LADDER))
    print(f"  BINARY P(real) : " + "  ".join(f"{np.mean(x):>6.3f}" for x in ladder_bin)
          + f"   (monotone {mono_bin}/{nlad})")
    print(f"  GRADED margin  : " + "  ".join(f"{np.mean(x):>+6.2f}" for x in ladder_grd)
          + f"   (monotone {mono_grd}/{nlad})")
    bin_spread = np.mean(ladder_bin[0]) - np.mean(ladder_bin[-1])
    grd_spread = np.mean(ladder_grd[0]) - np.mean(ladder_grd[-1])
    print(f"  real->full-corrupt spread:  BINARY {bin_spread:+.3f} (P)   GRADED {grd_spread:+.2f} (margin)")
    # how much of the drop happens only at the HEAVY end (saturation signature)?
    print(f"  early sensitivity (real -> 20% scrambled):  "
          f"BINARY {np.mean(ladder_bin[0])-np.mean(ladder_bin[1]):+.3f}   "
          f"GRADED {np.mean(ladder_grd[0])-np.mean(ladder_grd[1]):+.2f}")

    json.dump({'rows': rows,
               'ladder_bin': [list(map(float, x)) for x in ladder_bin],
               'ladder_grd': [list(map(float, x)) for x in ladder_grd],
               'panel_ladder': PANEL_LADDER},
              open('/tmp/claude-1000/-home-ybx-code-stepmania-chart-generator/9dbf8d8e-76da-4725-9dd7-cc1e1f46dbd8/scratchpad/personal_scores.json', 'w'), indent=2)
    print("\nsaved -> scratchpad/personal_scores.json")


if __name__ == '__main__':
    main()

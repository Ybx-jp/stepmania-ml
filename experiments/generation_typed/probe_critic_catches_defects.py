#!/usr/bin/env python3
"""E1.2-style: does the E1.1 48th graded critic AGREE WITH THE EAR + CATCH the defects the 2026-07-11 playtest named?

Ear ground truth (playtest_log 2026-07-11): Switch GLOBAL>OFF; Bye Bye GLOBAL>OFF (off "disjointed"); Bye Bye worst
overall for spurious sub-16ths. Architectural prediction: the presence-based critic CAN see #1 sub-16ths (its 0.98
jitter strength) + #2 quiet under-charge, but is BLIND to #3 hold foot-speed (a hold-type phenomenon).

Scores each chart with the graded_v2 critic (margin) in SLIDING 2304-frame windows across the WHOLE song, so we can
see whether the TAIL scores lower (the "worst near the end" defect). Reports mean/min/tail margin + sub-16th fraction,
then: (a) per-song GLOBAL-vs-OFF agreement, (b) corr(margin, sub16 fraction) [negative = critic penalizes the defect],
(c) does the tail window score below the body, (d) REAL (human) vs generated ranking.
"""
import warnings, os, sys, glob, re
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
from pathlib import Path
import numpy as np, torch

ROOT = Path('/home/ybx/code/stepmania-chart-generator'); sys.path.insert(0, str(ROOT))
from src.data.stepmania_parser import StepManiaParser
from src.generation.decode_harness import make_feature_extractor
from src.models import LateFusionClassifier

CRITIC = ROOT / "checkpoints/realism_critic_graded_v2/best_val.pt"
PROBE = Path('/home/ybx/sm-generated/stamina_probe')
PERSONAL = Path('/home/ybx/sm-personal')
SUBDIV = 12; SIXTEENTH = {0, 3, 6, 9}; WIN = 2304; STRIDE = 1152
AUDIO_EXT = ['.ogg', '.mp3', '.wav', '.flac', '.m4a']
REAL_DIR = {'Calling': PERSONAL / "Yb's Home Cooked/Calling (Lose My Mind)",
            'Switch': PERSONAL / "Hardcore Xtreme/Switch", 'Bye Bye': PERSONAL / "Hardcore Xtreme/Bye Bye"}


def to_binary(t):
    t = np.asarray(t); return ((t == 1) | (t == 2) | (t == 4)).astype(np.float32)


def find_audio(chart_audio, sm_path):
    d = sm_path.parent
    if chart_audio and (d / chart_audio).exists(): return d / chart_audio
    for e in AUDIO_EXT:
        g = sorted(d.glob('*' + e))
        if g: return g[0]
    return None


def densest_nd(chart):
    best, bn = None, -1
    for nd in chart.note_data:
        if not nd.difficulty_name: continue
        n = int((chart_typed(chart, nd)).any(1).sum())
        if n > bn: best, bn = nd, n
    return best


def chart_typed(chart, nd):
    return to_binary(PARSER.convert_to_tensor_typed(chart, nd))


@torch.no_grad()
def margins_over(critic, a23, chart, device):
    """Sliding-window margins across the whole song."""
    T = min(a23.shape[0], chart.shape[0])
    a23, chart = a23[:T], chart[:T]
    out = []
    starts = list(range(0, max(T - WIN, 0) + 1, STRIDE)) or [0]
    if starts[-1] + WIN < T: starts.append(T - WIN)
    for s in starts:
        aw = a23[s:s + WIN]; cw = chart[s:s + WIN]
        if cw.shape[0] < 128 or not cw.any(): continue
        a = torch.from_numpy(aw).unsqueeze(0).to(device); c = torch.from_numpy(cw).unsqueeze(0).to(device)
        m = torch.ones(1, aw.shape[0], device=device)
        lg = critic(a, c, m)
        if isinstance(lg, dict): lg = lg['logits']
        out.append(float(lg[0, 1] - lg[0, 0]))
    return out


def sub16_frac(chart):
    rows = np.where(chart.any(1))[0]
    if len(rows) == 0: return 0.0
    return float(np.mean([(r % SUBDIV) not in SIXTEENTH for r in rows]))


def main():
    global PARSER
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PARSER = StepManiaParser.for_v2(subdiv=SUBDIV, min_song_length=30.0, max_song_length=600.0,
                                    min_bpm=40.0, max_bpm=320.0, max_simultaneous=4, gimmick_max_bpm=400.0)
    ext = make_feature_extractor("highres_v2").extractor
    ck = torch.load(CRITIC, map_location=device, weights_only=False)
    critic = LateFusionClassifier(ck['config']).to(device); critic.load_state_dict(ck['model_state_dict']); critic.eval()
    print(f"critic={CRITIC.parent.name} (val_mono={ck.get('val_mono')})\n")

    items = []  # (song, arm, sm_path)
    for song in ('Calling', 'Switch', 'Bye Bye'):
        for arm in ('OFF', 'GLOBAL', 'LOCAL'):
            p = PROBE / f"{song} {arm}" / "chart.sm"
            if p.is_file(): items.append((song, arm, p))
        rd = REAL_DIR[song]; g = sorted(glob.glob(str(rd / '*.sm')))
        if g: items.append((song, 'REAL', Path(g[0])))

    rows = []
    for song, arm, smp in items:
        try:
            chart = PARSER.parse_file(str(smp))
            if chart is None: print(f"  [skip] {song} {arm}: parse None"); continue
            ap = find_audio(chart.audio_file, smp)
            feats = ext.extract_from_chart(str(ap), chart) if ap else None
            if feats is None: print(f"  [skip] {song} {arm}: no feats"); continue
            a23 = feats.get_aligned_features()[:, :23].astype(np.float32)
            nd = densest_nd(chart)
            if nd is None: continue
            cb = chart_typed(chart, nd)
            ms = margins_over(critic, a23, cb, device)
            if not ms: continue
            T = min(a23.shape[0], cb.shape[0])
            rows.append({'song': song, 'arm': arm, 'mean': float(np.mean(ms)), 'min': float(np.min(ms)),
                         'tail': float(ms[-1]), 'body': float(np.mean(ms[:-1])) if len(ms) > 1 else float(ms[0]),
                         'sub16': sub16_frac(cb[:T]), 'nwin': len(ms), 'notes': int(cb[:T].any(1).sum())})
        except Exception as e:
            print(f"  [skip] {song} {arm}: {e}"); continue

    print(f"{'song':9s} {'arm':7s} {'notes':>6s} {'mean_m':>7s} {'min_m':>7s} {'tail_m':>7s} {'body_m':>7s} {'sub16%':>6s} {'win':>3s}")
    print("-" * 74)
    for r in sorted(rows, key=lambda x: (x['song'], x['arm'])):
        print(f"{r['song']:9s} {r['arm']:7s} {r['notes']:>6d} {r['mean']:>+7.2f} {r['min']:>+7.2f} "
              f"{r['tail']:>+7.2f} {r['body']:>+7.2f} {r['sub16']*100:>5.1f}% {r['nwin']:>3d}")

    print("\n=== (a) per-song GLOBAL vs OFF (ear: global>=off on Switch/Bye Bye) ===")
    for song in ('Calling', 'Switch', 'Bye Bye'):
        d = {r['arm']: r for r in rows if r['song'] == song}
        if 'GLOBAL' in d and 'OFF' in d:
            gm, om = d['GLOBAL']['mean'], d['OFF']['mean']
            print(f"  {song:9s} GLOBAL {gm:+.2f} vs OFF {om:+.2f}  -> {'AGREES (G>=O)' if gm >= om else 'DISAGREES (G<O)'}"
                  + (f" | REAL {d['REAL']['mean']:+.2f}" if 'REAL' in d else ""))

    gen = [r for r in rows if r['arm'] != 'REAL']
    if len(gen) > 3:
        mm = np.array([r['mean'] for r in gen]); s16 = np.array([r['sub16'] for r in gen])
        if s16.std() > 1e-6:
            print(f"\n=== (b) corr(mean margin, sub-16th fraction) over gens = {np.corrcoef(mm, s16)[0,1]:+.3f}  "
                  f"(NEGATIVE = critic penalizes the #1 defect) ===")
        tail = np.array([r['tail'] for r in gen]); body = np.array([r['body'] for r in gen])
        print(f"=== (c) tail vs body margin (mean over gens): tail {tail.mean():+.2f} vs body {body.mean():+.2f}  "
              f"[{(tail-body).mean():+.2f}]  (NEGATIVE = critic scores the tail worse, matching 'sub-16ths near the end') ===")
    real = [r for r in rows if r['arm'] == 'REAL']; gg = [r for r in rows if r['arm'] != 'REAL']
    if real and gg:
        print(f"\n=== (d) REAL (human) mean margin {np.mean([r['mean'] for r in real]):+.2f} "
              f"vs generated {np.mean([r['mean'] for r in gg]):+.2f}  (REAL should be higher) ===")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Bipedal kinematics metric — the CORRECT movement geometry (the distance-only version was a dud: with two
alternating feet, L,R,L in 16ths is easy, so raw distance mismeasures effort). The body is two feet that
alternate across the pad; a HOLD pins one foot. The awkward signal is ONE foot forced to stream across
different panels fast — e.g. L-hold + U,D,U 16ths (left foot pinned, right foot alone does U->D->U).

Assign feet via an alternation automaton that respects holds, then per foot measure consecutive-note
moves: distance (pad 2D) and ioi (frames). Metrics:
  - per_foot_vel   : mean dist/ioi over each foot's consecutive notes (alternation keeps this low).
  - fast_cross     : fraction of per-foot moves that are dist>=1.4 (different/opposite panel) AND ioi<=2
                     (16th/8th) -> a fast one-foot cross.
  - hold_burst     : fast_cross rate restricted to moves where the OTHER foot is pinned by a hold
                     (the L-hold + U,D,U case). Real charts should keep this LOW.

REAL vs gen_stage1 vs panel-shuffled null.
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, sys
from pathlib import Path
import numpy as np, torch, yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits
from src.data.dataset import StepManiaDataset
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.typed import pair_holds

GEN_CKPT = "checkpoints/gen_stage1/best_val.pt"
COORDS = np.array([[-1, 0], [0, -1], [0, 1], [1, 0]], dtype=np.float64)  # L,D,U,R


def pinned_mask(typed):
    """(T,4) bool: panel occupied by a foot from hold-head (2/4) through tail (3)."""
    T = typed.shape[0]; pin = np.zeros((T, 4), bool)
    for p in range(4):
        col = typed[:, p]; t = 0
        while t < T:
            if col[t] in (2, 4):
                tt = t + 1
                while tt < T and col[tt] != 3:
                    tt += 1
                pin[t:min(tt + 1, T), p] = True; t = tt + 1
            else:
                t += 1
    return pin


def foot_moves(typed):
    """Assign feet (alternate on taps; holds pin a foot) -> per-foot list of (panel, frame, other_pinned)."""
    typed = np.asarray(pair_holds(typed)); T = typed.shape[0]
    pin = pinned_mask(typed)
    next_foot = 0; hold_owner = {}; moves = {0: [], 1: []}
    for t in range(T):
        for p in list(hold_owner):              # release ended holds
            if not pin[t, p]:
                del hold_owner[p]
        presses = [p for p in range(4) if typed[t, p] in (1, 2, 4)]
        if not presses:
            continue
        busy = set(hold_owner.values())
        for p in presses:
            if typed[t, p] in (2, 4):           # hold head -> pick a free foot, pin it
                f = next_foot if next_foot not in busy else 1 - next_foot
                hold_owner[p] = f
            elif len(busy) == 1:                 # one foot pinned -> tap goes to the free foot
                f = 1 - next(iter(busy))
            else:
                f = next_foot
            other_pinned = (1 - f) in set(hold_owner.values())
            moves[f].append((p, t, other_pinned))
            next_foot = 1 - f
    return moves


def stats(typed):
    moves = foot_moves(typed)
    vels, fast, n, hb_fast, hb_n = [], 0, 0, 0, 0
    for f in (0, 1):
        seq = sorted(moves[f], key=lambda x: x[1])
        for i in range(len(seq) - 1):
            (p0, t0, _), (p1, t1, op1) = seq[i], seq[i + 1]
            d = np.linalg.norm(COORDS[p1] - COORDS[p0]); ioi = t1 - t0
            if ioi <= 0:
                continue
            vels.append(d / ioi); n += 1
            is_fast = (d >= 1.4 and ioi <= 2)
            fast += is_fast
            if op1:                              # the OTHER foot was pinned during this move
                hb_n += 1; hb_fast += is_fast
    return dict(vel=np.mean(vels) if vels else np.nan,
                fast=fast / n if n else np.nan,
                hold_burst=hb_fast / hb_n if hb_n else np.nan,
                hb_n=hb_n)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--num_songs', type=int, default=40)
    ap.add_argument('--max_len', type=int, default=1024); args = ap.parse_args()
    set_seed(42); rng = np.random.default_rng(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    _, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True, use_metric_phase=True))
    ds = StepManiaDataset(chart_files=vf[:args.num_songs * 4], audio_dir="data/", max_sequence_length=msl,
                          feature_extractor=ext, cache_dir='cache/samples_v2')
    gen = LayeredTypedChartGenerator(audio_dim=41, d_model=128, num_layers=4, onset_layers=2).to(device)
    gen.load_state_dict(torch.load(GEN_CKPT, map_location=device)['model_state_dict']); gen.eval()

    SRC = ['REAL', 'gen_stage1', 'REAL panel-shuffled']
    acc = {s: {k: [] for k in ('vel', 'fast', 'hold_burst', 'hb_n')} for s in SRC}
    seen, used = set(), 0
    for i in range(len(ds.valid_samples)):
        if used >= args.num_songs: break
        meta = ds.valid_samples[i]
        if meta['chart_file'] in seen: continue
        sample = ds[i]; T = min(int(sample['mask'].sum().item()), args.max_len)
        if T < 128: continue
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        orig = np.asarray(ds.parser.convert_to_tensor_typed(meta['chart'], nd))[:T]
        if (orig != 0).any(1).sum() < 8: continue
        audio = sample['audio'][:T].unsqueeze(0).to(device); diff = torch.tensor([meta['difficulty_class']], device=device)
        with torch.no_grad():
            p_on = torch.sigmoid(gen.onset_logits(gen.encode_audio(audio), diff))[0].cpu().numpy()
            tau = float(np.quantile(p_on, 1 - float((orig != 0).any(1).mean())))
            g = gen.generate(audio, diff, lengths=torch.tensor([T], device=device), onset_threshold=tau,
                             type_sample=True, type_temperature=0.4, hold_aware=True, pattern_sample=True,
                             pattern_temperature=0.7, no_jump_during_hold=True)[0].cpu().numpy()
        g = pair_holds(g)
        # panel-shuffled null: random panel per single-note frame (holds kept as-is)
        shuf = orig.copy()
        for t in range(T):
            ps = np.where(orig[t] != 0)[0]
            if len(ps) == 1 and orig[t, ps[0]] == 1:
                shuf[t, ps[0]] = 0; shuf[t, rng.integers(4)] = 1
        for src, chart in [('REAL', orig), ('gen_stage1', g), ('REAL panel-shuffled', shuf)]:
            s = stats(chart)
            for k in ('vel', 'fast', 'hold_burst', 'hb_n'):
                acc[src][k].append(s[k])
        seen.add(meta['chart_file']); used += 1

    print(f"\n=== Bipedal kinematics ({used} songs) ===")
    print(f"{'source':<22} {'per_foot_vel':>13} {'fast_cross%':>12} {'hold_burst%':>12} {'hold_moves':>11}")
    print("-" * 74)
    for s in SRC:
        v = np.nanmean(acc[s]['vel']); fa = 100 * np.nanmean(acc[s]['fast'])
        hb = 100 * np.nanmean(acc[s]['hold_burst']); hbn = int(np.nansum(acc[s]['hb_n']))
        print(f"{s:<22} {v:>13.3f} {fa:>11.1f}% {hb:>11.1f}% {hbn:>11}")
    print("-" * 74)
    print("hold_burst% = one foot fast-crossing while the other is pinned (the L-hold+U,D,U awkwardness).")
    print("Real should keep fast_cross/hold_burst LOW; if gen_stage1 >> real, it choreographs un-dance-ably.")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
H11 free-running confirmation. The teacher-forced probe (diag_transitions.py) showed the model's loss is
~flat at audio section-boundaries -> transitions aren't a representational deficit; the failure is
free-running AR drift (the pattern head's momentum). This tests that directly: does the GENERATED chart
CHANGE its choreography at audio boundaries the way REAL charts do, or drift through them?

For each audio section-boundary, take a window before & after and a choreography descriptor
[density, jump_frac, L,D,U,R panel mix]; |before - after| L1 = "transition magnitude" (how much the chart
changes there). Compare to RANDOM non-boundary positions (control). Real charts should change MORE at
boundaries than randomly (they follow the song's form). If the GENERATED chart's boundary-vs-random
"transition responsiveness" is LOWER than real's, the model under-transitions = AR drift (H11 confirmed).
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, sys
from pathlib import Path
import numpy as np, torch, yaml
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits
from src.data.dataset import StepManiaDataset
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.typed import pair_holds

GEN_CKPT = "checkpoints/gen_stage1/best_val.pt"
L = 32; W = 32          # Foote half-kernel (8 beats = section-scale); descriptor window
TOPK = 5                # keep only the strongest few boundaries per song (real sections, not local noise)
SSM_DIMS = list(range(0, 13)) + list(range(23, 35))


def foote_boundaries(feat):
    f = feat - feat.mean(0, keepdims=True); f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
    S = f @ f.T; a = np.arange(2 * L)
    sign = np.where((a[:, None] < L) == (a[None, :] < L), 1.0, -1.0)
    gw = np.exp(-((a[:, None] - L + .5) ** 2 + (a[None, :] - L + .5) ** 2) / (2 * (L / 2) ** 2))
    ker = sign * gw; T = len(feat); nov = np.zeros(T)
    for t in range(L, T - L):
        nov[t] = (S[t - L:t + L, t - L:t + L] * ker).sum()
    nov = np.maximum(nov, 0); pos = nov[nov > 0]
    pk, props = find_peaks(nov, distance=2 * L, prominence=max(pos.std() if pos.size else 0, 1e-6))
    if len(pk) > TOPK:                                    # keep the TOPK most prominent (real section changes)
        pk = pk[np.argsort(props['prominences'])[::-1][:TOPK]]
    return np.sort(pk)


def descriptor(typed, lo, hi):
    seg = np.asarray(typed)[lo:hi]
    starts = (seg == 1) | (seg == 2) | (seg == 4)        # note-starts per panel
    note = starts.any(1); n = int(note.sum())
    if n < 3:
        return None
    density = note.mean()
    jump = (starts.sum(1) >= 2).sum() / n                # jump fraction
    panel = starts.sum(0) / max(starts.sum(), 1)         # L,D,U,R mix
    return np.concatenate([[density, jump], panel])       # 6-dim


def mag(typed, c):                                        # transition magnitude across position c
    a, b = descriptor(typed, c - W, c), descriptor(typed, c, c + W)
    return None if (a is None or b is None) else float(np.abs(a - b).sum())


def responsiveness(typed, bnds, rng, T):
    bm = [m for b in bnds for m in [mag(typed, b)] if m is not None]
    rand = rng.integers(W, T - W, size=max(len(bnds) * 3, 30))
    rm = [m for c in rand for m in [mag(typed, int(c))] if m is not None]
    if not bm or not rm:
        return None
    return np.mean(bm), np.mean(rm)                       # (boundary mag, random mag)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--num_songs', type=int, default=30)
    ap.add_argument('--max_len', type=int, default=1024); ap.add_argument('--min_difficulty', type=int, default=2)
    args = ap.parse_args()
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

    R = {'real': {'b': [], 'r': []}, 'gen': {'b': [], 'r': []}}
    used = 0
    for i in range(len(ds.valid_samples)):
        if used >= args.num_songs: break
        meta = ds.valid_samples[i]
        if meta['difficulty_class'] < args.min_difficulty: continue
        sample = ds[i]; T = min(int(sample['mask'].sum().item()), args.max_len)
        if T < 256: continue
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        typed_r = np.asarray(ds.parser.convert_to_tensor_typed(meta['chart'], nd))[:T]
        if (typed_r != 0).any(1).sum() < 32: continue
        bnds = foote_boundaries(sample['audio'][:T, SSM_DIMS].numpy())
        if len(bnds) < 2: continue
        audio = sample['audio'][:T].unsqueeze(0).to(device); diff = torch.tensor([meta['difficulty_class']], device=device)
        with torch.no_grad():
            p_on = torch.sigmoid(gen.onset_logits(gen.encode_audio(audio), diff))[0].cpu().numpy()
            tau = float(np.quantile(p_on, 1 - float((typed_r != 0).any(1).mean())))
            g = gen.generate(audio, diff, lengths=torch.tensor([T], device=device), onset_threshold=tau,
                             type_sample=True, type_temperature=0.4, hold_aware=True, pattern_sample=True,
                             pattern_temperature=0.7, no_jump_during_hold=True, no_cross_during_hold=True)[0].cpu().numpy()
        for name, typed in [('real', typed_r), ('gen', pair_holds(g))]:
            res = responsiveness(typed, bnds, np.random.default_rng(0), T)
            if res:
                R[name]['b'].append(res[0]); R[name]['r'].append(res[1])
        used += 1

    print(f"\n=== H11 free-running transition responsiveness ({used} songs) ===")
    print("how much the CHOREOGRAPHY changes across a position (L1 of [density,jump,L,D,U,R]).")
    print("real should change MORE at boundaries than at random; if GEN's (boundary-random) < real's, it drifts.\n")
    print(f"{'chart':<8} {'@boundary':>10} {'@random':>9} {'responsiveness':>15}")
    print("-" * 46)
    for name in ['real', 'gen']:
        b, r = np.mean(R[name]['b']), np.mean(R[name]['r'])
        print(f"{name:<8} {b:>10.3f} {r:>9.3f} {b - r:>15.3f}")
    print("-" * 46)
    print("responsiveness = @boundary - @random. real >> gen => the generator under-transitions (AR drift, H11).")


if __name__ == '__main__':
    main()

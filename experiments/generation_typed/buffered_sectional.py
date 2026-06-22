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
import argparse, glob, re, shutil, sys
from pathlib import Path
import numpy as np, torch, yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits
from src.data.dataset import StepManiaDataset, DIFFICULTY_NAMES
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
from src.data.song_selection import select_by_groove
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.typed import pair_holds
from src.generation.sm_writer import charts_to_sm
from diag_transitions_freerun import foote_boundaries, responsiveness, SSM_DIMS

GEN_CKPT = "checkpoints/gen_stage1/best_val.pt"
DECODE = dict(type_sample=True, type_temperature=0.4, hold_aware=True, pattern_sample=True,
              pattern_temperature=0.7, no_jump_during_hold=True, no_cross_during_hold=True)


def safe_name(s):
    return (re.sub(r'[^\w\- ]+', '', (s or 'x').strip(), flags=re.UNICODE).strip() or 'x')[:60]


def generate_sectional(model, audio_full, diff, orig, bnds, W_in, W_out, device, seed=42):
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
            p_seg = torch.sigmoid(model.onset_logits(model.encode_audio(aud), diff))[0].cpu().numpy()
        tau = float(np.quantile(p_seg, 1 - rd_seg)) if rd_seg > 0 else 1.0
        set_seed(seed)
        g = model.generate(aud, diff, lengths=torch.tensor([hi - lo], device=device),
                           onset_threshold=tau, **DECODE)[0].cpu().numpy()
        out[a:b] = g[a - lo: a - lo + (b - a)]   # keep the clean middle (skip warmup, before cooldown)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_songs', type=int, default=6); ap.add_argument('--max_len', type=int, default=1024)
    ap.add_argument('--warmup', type=int, default=24); ap.add_argument('--cooldown', type=int, default=16)
    ap.add_argument('--out_dir', default='outputs/sectional'); ap.add_argument('--install', action='store_true')
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    _, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True, use_metric_phase=True))
    ds = StepManiaDataset(chart_files=vf[:args.num_songs * 40], audio_dir="data/", max_sequence_length=msl,
                          feature_extractor=ext, cache_dir='cache/samples_v2')
    gen = LayeredTypedChartGenerator(audio_dim=41, d_model=128, num_layers=4, onset_layers=2).to(device)
    gen.load_state_dict(torch.load(GEN_CKPT, map_location=device)['model_state_dict']); gen.eval()
    order = select_by_groove(ds, by='rich', difficulty=3)   # rich Hard songs with real structure

    out = Path(args.out_dir)
    for t in ('sectional', 'baseline'): (out / t).mkdir(parents=True, exist_ok=True)
    R = {k: {'b': [], 'r': []} for k in ('real', 'baseline', 'sectional')}
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
        rd = float((typed_r != 0).any(1).mean())
        with torch.no_grad():
            p_on = torch.sigmoid(gen.onset_logits(gen.encode_audio(audio), diff))[0].cpu().numpy()
            tau = float(np.quantile(p_on, 1 - rd))
            set_seed(42)
            g0 = gen.generate(audio, diff, lengths=torch.tensor([T], device=device), onset_threshold=tau, **DECODE)[0].cpu().numpy()
        g1 = generate_sectional(gen, audio, diff, typed_r, bnds, args.warmup, args.cooldown, device)
        charts = {'baseline': pair_holds(g0), 'sectional': pair_holds(g1)}
        for name, typed in [('real', typed_r), ('baseline', charts['baseline']), ('sectional', charts['sectional'])]:
            res = responsiveness(typed, bnds, np.random.default_rng(0), T)
            if res: R[name]['b'].append(res[0]); R[name]['r'].append(res[1])
        title = meta['chart'].title or Path(meta['chart_file']).stem
        rows.append((safe_name(title)[:24], len(bnds)))
        # export A/B
        bpm = float(meta['chart'].bpm); music = os.path.basename(meta['audio_file']); dn = DIFFICULTY_NAMES[meta['difficulty_class']]
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

    print(f"\n=== Buffered-sectional ({used} songs, warmup={args.warmup} cooldown={args.cooldown}) ===")
    print(f"songs (boundaries): " + ", ".join(f"{n}({b})" for n, b in rows))
    print(f"\n{'chart':<12} {'@boundary':>10} {'@random':>9} {'responsiveness':>15}  (real target +0.128)")
    print("-" * 52)
    for name in ('real', 'baseline', 'sectional'):
        b, r = np.mean(R[name]['b']), np.mean(R[name]['r'])
        print(f"{name:<12} {b:>10.3f} {r:>9.3f} {b - r:>15.3f}")
    print("-" * 52)
    print(f"sectional responsiveness near real (vs baseline ~0) = it re-choreographs at transitions.")
    print(f"Exported {args.out_dir}/{{baseline,sectional}} for A/B playtest.")
    if args.install:
        from src.utils.sm_install import install_to_stepmania
        for d in install_to_stepmania(args.out_dir): print("installed:", d)


if __name__ == '__main__':
    main()

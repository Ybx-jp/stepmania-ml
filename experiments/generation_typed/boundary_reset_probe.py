#!/usr/bin/env python3
"""
H11 fix probe (cheap, no retrain): does flushing the pattern head's AR context at section boundaries make
the generator re-choreograph? generate(boundary_reset=<frames>) drops self-attn note-history at each
boundary (keeps audio cross-attn). Measure transition-responsiveness (diag_transitions_freerun) for
gen-no-reset vs gen-reset vs real. If reset moves responsiveness toward real's, the reset direction works
-> build the buffered-sectional version (warmup/outro buffers) for clean playable output.

NOTE: judge by the metric only -- the raw reset reintroduces cold-start at each boundary (the buffered
version fixes that). This just validates the direction.
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, sys
from pathlib import Path
import numpy as np, torch, yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits
from src.data.dataset import StepManiaDataset
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.typed import pair_holds
from diag_transitions_freerun import foote_boundaries, responsiveness, SSM_DIMS

GEN_CKPT = "checkpoints/gen_stage1/best_val.pt"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--num_songs', type=int, default=30)
    ap.add_argument('--max_len', type=int, default=1024); ap.add_argument('--min_difficulty', type=int, default=2)
    args = ap.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    _, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True, use_metric_phase=True))
    ds = StepManiaDataset(chart_files=vf[:args.num_songs * 4], audio_dir="data/", max_sequence_length=msl,
                          feature_extractor=ext, cache_dir='cache/samples_v2')
    gen = LayeredTypedChartGenerator(audio_dim=41, d_model=128, num_layers=4, onset_layers=2).to(device)
    gen.load_state_dict(torch.load(GEN_CKPT, map_location=device)['model_state_dict']); gen.eval()

    R = {k: {'b': [], 'r': []} for k in ('real', 'gen', 'gen_reset')}
    decode = dict(type_sample=True, type_temperature=0.4, hold_aware=True, pattern_sample=True,
                  pattern_temperature=0.7, no_jump_during_hold=True, no_cross_during_hold=True)
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
        lengths = torch.tensor([T], device=device)
        with torch.no_grad():
            p_on = torch.sigmoid(gen.onset_logits(gen.encode_audio(audio), diff))[0].cpu().numpy()
            tau = float(np.quantile(p_on, 1 - float((typed_r != 0).any(1).mean())))
            set_seed(42)
            g0 = gen.generate(audio, diff, lengths=lengths, onset_threshold=tau, **decode)[0].cpu().numpy()
            set_seed(42)
            g1 = gen.generate(audio, diff, lengths=lengths, onset_threshold=tau,
                              boundary_reset=list(bnds), **decode)[0].cpu().numpy()
        for name, typed in [('real', typed_r), ('gen', pair_holds(g0)), ('gen_reset', pair_holds(g1))]:
            res = responsiveness(typed, bnds, np.random.default_rng(0), T)
            if res:
                R[name]['b'].append(res[0]); R[name]['r'].append(res[1])
        used += 1

    print(f"\n=== H11 boundary-reset probe ({used} songs) ===")
    print("transition responsiveness = choreography change @boundary - @random (real target +0.128)\n")
    print(f"{'chart':<12} {'@boundary':>10} {'@random':>9} {'responsiveness':>15}")
    print("-" * 50)
    for name in ('real', 'gen', 'gen_reset'):
        b, r = np.mean(R[name]['b']), np.mean(R[name]['r'])
        print(f"{name:<12} {b:>10.3f} {r:>9.3f} {b - r:>15.3f}")
    print("-" * 50)
    print("gen_reset responsiveness > gen, toward real => flushing AR context at boundaries works")
    print("=> build buffered-sectional (warmup/outro buffers) for clean playable output.")


if __name__ == '__main__':
    main()

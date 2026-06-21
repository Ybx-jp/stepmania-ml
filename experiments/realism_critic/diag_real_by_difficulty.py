#!/usr/bin/env python3
"""
Diagnostic: does the taste critic score REAL charts high at every difficulty, or does it
over-reject Hard? Scores real (un-generated) charts' P(real) grouped by difficulty class.

If real Hard scores ~0.8 while generated Hard scores ~0.02 (from export_reranked --difficulty Hard),
the Hard collapse is a genuine gen-vs-real gap (critic trustworthy). If real Hard also scores ~0.02,
the critic simply doesn't generalize to Hard (over-rejection).
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import sys, glob
from pathlib import Path
from collections import defaultdict
import numpy as np, torch, yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits
from src.data.dataset import StepManiaDataset, DIFFICULTY_NAMES
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
from src.models import LateFusionClassifier

CRITIC = "checkpoints/realism_critic/best_val.pt"
MAX_LEN = 1440
PER_CLASS = 25  # how many real charts per difficulty to score


def to_binary(t):
    t = np.asarray(t); return ((t == 1) | (t == 2) | (t == 4)).astype(np.float32)


@torch.no_grad()
def p_real(critic, audio23, chart_bin, device):
    a = torch.from_numpy(audio23).unsqueeze(0).to(device); c = torch.from_numpy(chart_bin).unsqueeze(0).to(device)
    m = torch.ones(1, a.shape[1], device=device)
    logits = critic(a, c, m)
    if isinstance(logits, dict): logits = logits['logits']
    return float(torch.softmax(logits, 1)[0, 1])


def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    _, val_files, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True, use_metric_phase=True))
    ds = StepManiaDataset(chart_files=val_files, audio_dir="data/", max_sequence_length=msl,
                          feature_extractor=ext, cache_dir='cache/samples_v2')

    ck = torch.load(CRITIC, map_location=device, weights_only=False)
    critic = LateFusionClassifier(ck['config']).to(device); critic.load_state_dict(ck['model_state_dict']); critic.eval()

    scores = defaultdict(list)
    for i in range(len(ds.valid_samples)):
        if all(len(scores[c]) >= PER_CLASS for c in range(4)):
            break
        meta = ds.valid_samples[i]
        cls = meta['difficulty_class']
        if len(scores[cls]) >= PER_CLASS:
            continue
        sample = ds[i]; T = min(int(sample['mask'].sum().item()), MAX_LEN)
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None: continue
        audio23 = sample['audio'][:T, :23].numpy().astype(np.float32)
        orig_typed = ds.parser.convert_to_tensor_typed(meta['chart'], nd)[:T]
        scores[cls].append(p_real(critic, audio23, to_binary(orig_typed), device))

    print(f"\n{'difficulty':<12} {'n':>4} {'mean P(real)':>13} {'median':>8} {'min':>7} {'max':>7}")
    print("-" * 56)
    for c in range(4):
        s = np.array(scores[c])
        if len(s) == 0:
            print(f"{DIFFICULTY_NAMES[c]:<12} {0:>4}  (none found)"); continue
        print(f"{DIFFICULTY_NAMES[c]:<12} {len(s):>4} {s.mean():>13.3f} {np.median(s):>8.3f} {s.min():>7.3f} {s.max():>7.3f}")
    print("-" * 56)
    print("Compare 'Hard' mean here to generated-Hard best ~0.02-0.12 (export_reranked --difficulty Hard).")
    print("real Hard high  -> genuine gen-vs-real gap (critic trustworthy at Hard).")
    print("real Hard low   -> critic over-rejects Hard (needs difficulty-balanced training for 2c).")


if __name__ == '__main__':
    main()

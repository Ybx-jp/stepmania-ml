#!/usr/bin/env python3
"""
Decisive decode-vs-architecture test for groove (see groove_periodicity_findings.md). The localization
showed groove (ac_off) is lost at onset placement (the pattern/hold decode is innocent: stage 3=4=0.109).
Open fork: is the loss in the THRESHOLD decode (cheap fix) or in the onset HEAD (architectural)?

Same model, same p_on, same off-beat COUNT (matched to the real chart) — only the PLACEMENT rule changes:
  (a) threshold placement : off-beats = top-n_off off-beat frames by p_on (what generate() does).
  (b) periodic placement  : build a per-MEASURE off-beat template (mean p_on per off-beat slot across
      measures), fire the top-k consistently-active slots in EVERY measure (periodicity by construction,
      using the model's OWN posterior to choose which slots).
On-beat backbone identical (top-n_on on-beat frames by p_on). ac_off computed on note-presence (no
generation needed — decode is ac_off-neutral).

Read: if (b) >> (a) and approaches REAL 0.187 -> the posterior HAS a recurring groove a periodicity-aware
decode can extract -> DECODE-FIXABLE (cheap). If (b) ~ (a) ~ 0.11 -> the head's posterior has no consistent
off-beat template -> onset head doesn't carry groove -> ARCHITECTURAL (neither decode nor expert-data).
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

GEN_CKPT = "checkpoints/gen_stage1/best_val.pt"
OFFBEAT_SLOTS = [j for j in range(16) if j % 4 != 0]  # 12 off-beat positions within a measure


def autocorr(x, lag):
    x = x.astype(np.float64) - x.mean(); d = (x * x).sum()
    return 0.0 if (d < 1e-9 or lag >= len(x)) else float((x[lag:] * x[:-lag]).sum() / d)


def ac_off(sig):
    t = np.arange(len(sig)); s = sig.copy().astype(np.float64); s[t % 4 == 0] = 0.0
    return autocorr(s, 4)


def backbone(p_on, T, n_on):
    t = np.arange(T); on_idx = np.where(t % 4 == 0)[0]; m = np.zeros(T, bool)
    if n_on > 0 and len(on_idx):
        m[on_idx[np.argsort(p_on[on_idx])[::-1][:n_on]]] = True
    return m


def place_threshold(p_on, T, n_off):
    t = np.arange(T); off_idx = np.where(t % 4 != 0)[0]; m = np.zeros(T, bool)
    if n_off > 0 and len(off_idx):
        m[off_idx[np.argsort(p_on[off_idx])[::-1][:n_off]]] = True
    return m


def place_periodic(p_on, T, n_off):
    """Per-measure template: mean p_on per off-beat slot across measures; fire top-k slots every measure."""
    n_meas = T // 16
    m = np.zeros(T, bool)
    if n_meas < 2 or n_off <= 0:
        return place_threshold(p_on, T, n_off)
    # template[slot] = mean p_on at that within-measure off-beat slot across measures
    tmpl = np.array([np.mean([p_on[mm * 16 + j] for mm in range(n_meas)]) for j in OFFBEAT_SLOTS])
    k = max(1, round(n_off / n_meas))                      # slots per measure to hit ~n_off total
    top_slots = [OFFBEAT_SLOTS[i] for i in np.argsort(tmpl)[::-1][:k]]
    for mm in range(n_meas):
        for j in top_slots:
            m[mm * 16 + j] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_songs', type=int, default=40); ap.add_argument('--max_len', type=int, default=1024)
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

    acc = {k: [] for k in ['REAL', '(a) threshold', '(b) periodic', 'null (random off)']}
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
        onset_r = (orig != 0).any(1).astype(np.float64)
        t = np.arange(T)
        n_on = int(onset_r[t % 4 == 0].sum()); n_off = int(onset_r[t % 4 != 0].sum())
        if onset_r.sum() < 16 or n_off < 4: continue
        audio = sample['audio'][:T].unsqueeze(0).to(device); diff = torch.tensor([meta['difficulty_class']], device=device)
        with torch.no_grad():
            p_on = torch.sigmoid(gen.onset_logits(gen.encode_audio(audio), diff))[0].cpu().numpy()
        bb = backbone(p_on, T, n_on)
        mask_a = bb | place_threshold(p_on, T, n_off)
        mask_b = bb | place_periodic(p_on, T, n_off)
        # null: backbone + random off-beats (same count)
        off_idx = np.where(t % 4 != 0)[0]; mn = bb.copy()
        mn[rng.choice(off_idx, size=min(n_off, len(off_idx)), replace=False)] = True
        acc['REAL'].append(ac_off(onset_r))
        acc['(a) threshold'].append(ac_off(mask_a.astype(np.float64)))
        acc['(b) periodic'].append(ac_off(mask_b.astype(np.float64)))
        acc['null (random off)'].append(ac_off(mn.astype(np.float64)))
        seen.add(meta['chart_file']); used += 1

    print(f"\n=== Periodicity-aware decode test ({used} songs) — ac_off, off-beat count matched to REAL ===\n")
    print(f"{'placement':<20} {'ac_off':>8}")
    print("-" * 30)
    for k in ['REAL', '(b) periodic', '(a) threshold', 'null (random off)']:
        print(f"{k:<20} {np.mean(acc[k]):>8.3f}")
    print("-" * 30)
    print("(b)>>(a), toward REAL -> posterior has a recurring groove; DECODE-FIXABLE (cheap).")
    print("(b)~(a)~0.11        -> onset head carries no off-beat template; ARCHITECTURAL.")


if __name__ == '__main__':
    main()

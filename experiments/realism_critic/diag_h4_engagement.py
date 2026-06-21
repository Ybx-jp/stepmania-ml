#!/usr/bin/env python3
"""
Did the H4 high-res-onset retrain (gen_highres) actually USE the new feature, and did chaos become
event-driven? The new conv column's weight norm is only ~11% of the mean (rank 1/42), suggesting the
warm-start barely engaged it. Confirm behaviorally:

(A) Feature ablation — teacher-forced: zero dim 41, measure KL on onset+pattern logits. Low KL = unused.
    (Compare to Stage-1's chroma KL 10.3 "used heavily" vs HPSS 0.29 "near-ignored".)
(B) Chaos behavior — generate under radar chaos=0.9 and measure:
    - on-beat fraction of generated notes (H4 baseline: chaos smears to ~6% on-beat = uniform off-grid)
    - AUC(high-res onset dim41 -> generated OFF-beat notes): if the model uses the feature, its off-beat
      notes land on high-res-onset frames (AUC>0.5 = event-driven); ~0.5 = uniform smear.
    gen_highres vs gen_stage1 (stage1 = 42dim[:41], never sees dim41) isolates the feature's effect.
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import sys, glob
from pathlib import Path
import numpy as np, torch, yaml
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits
from src.data.dataset import StepManiaDataset
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
from src.generation.typed_model import LayeredTypedChartGenerator
from src.generation.typed import pair_holds

import argparse
N_SONGS = 20
MAX_LEN = 1024
HIRES_DIM = 41
CHAOS = 0.9  # radar: [stream, voltage, air, freeze, chaos]
GUIDANCE = 2.0  # CFG amplification for the chaos test (the smear baseline was at g~2, not g=1)


def load_model(ckpt, audio_dim, device):
    m = LayeredTypedChartGenerator(audio_dim=audio_dim, d_model=128, num_layers=4, onset_layers=2).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device)['model_state_dict']); m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--highres_ckpt', default='checkpoints/gen_highres/best_val.pt')
    args = ap.parse_args()
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    _, val_files, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                   use_metric_phase=True, use_highres_onset=True))
    ds = StepManiaDataset(chart_files=val_files[:N_SONGS * 4], audio_dir="data/", max_sequence_length=msl,
                          feature_extractor=ext, cache_dir='cache/samples_v3')

    hires = load_model(args.highres_ckpt, 42, device)
    stage1 = load_model("checkpoints/gen_stage1/best_val.pt", 41, device)
    print(f"highres ckpt: {args.highres_ckpt}")

    abl_kl_onset, abl_kl_pat = [], []
    onbeat = {'gen_highres': [], 'gen_stage1': []}
    offbeat_auc = {'gen_highres': [], 'gen_stage1': []}
    seen, used = set(), 0
    for i in range(len(ds.valid_samples)):
        if used >= N_SONGS: break
        meta = ds.valid_samples[i]
        if meta['chart_file'] in seen: continue
        sample = ds[i]; T = min(int(sample['mask'].sum().item()), MAX_LEN)
        if T < 128: continue
        a42 = sample['audio'][:T].unsqueeze(0).to(device)
        diff = torch.tensor([meta['difficulty_class']], device=device)
        radar = torch.zeros(1, 5, device=device); radar[0, 4] = CHAOS  # high chaos
        t = np.arange(T)

        # (A) ablation on gen_highres: teacher-forced onset + pattern logits, full vs dim41=0
        with torch.no_grad():
            mem = hires.encode_audio(a42)
            ol_full = hires.onset_logits(mem, diff, radar=radar)[0]
            a42z = a42.clone(); a42z[..., HIRES_DIM] = 0.0
            mem_z = hires.encode_audio(a42z)
            ol_z = hires.onset_logits(mem_z, diff, radar=radar)[0]
            # onset KL (Bernoulli per frame)
            pf, pz = torch.sigmoid(ol_full), torch.sigmoid(ol_z)
            kl = (pf * (torch.log(pf + 1e-8) - torch.log(pz + 1e-8))
                  + (1 - pf) * (torch.log(1 - pf + 1e-8) - torch.log(1 - pz + 1e-8))).mean()
            abl_kl_onset.append(float(kl))

        # (B) generate under chaos, both models
        hires_feat = sample['audio'][:T, HIRES_DIM].numpy()
        for name, model, audio_in in [('gen_highres', hires, a42),
                                      ('gen_stage1', stage1, a42[..., :41])]:
            with torch.no_grad():
                p_on = torch.sigmoid(model.onset_logits(model.encode_audio(audio_in), diff, radar=radar))[0].cpu().numpy()
            tau = float(np.quantile(p_on, 1 - 0.20))  # ~20% density target
            with torch.no_grad():
                g = model.generate(audio_in, diff, lengths=torch.tensor([T], device=device),
                                   onset_threshold=tau, type_sample=True, type_temperature=0.4,
                                   hold_aware=True, pattern_sample=True, pattern_temperature=0.7,
                                   no_jump_during_hold=True, radar=radar, guidance_scale=GUIDANCE)[0].cpu().numpy()
            note = (pair_holds(g) != 0).any(1)
            if note.sum() == 0: continue
            onbeat[name].append(float(note[t % 4 == 0].sum()) / float(note.sum()))
            # off-beat frames only: does dim41 discriminate generated note vs no-note?
            off = (t % 2 == 1)
            lab = note[off].astype(int)
            if lab.min() != lab.max():
                offbeat_auc[name].append(roc_auc_score(lab, hires_feat[off]))
        seen.add(meta['chart_file']); used += 1

    print(f"\n=== H4 engagement check ({used} songs, chaos={CHAOS}) ===\n")
    print(f"(A) FEATURE ABLATION (gen_highres, zero dim41) — onset-logit KL: {np.mean(abl_kl_onset):.4f}")
    print(f"    ref: Stage-1 chroma KL 10.3 (used heavily), HPSS 0.29 (near-ignored). Low => unused.\n")
    print(f"(B) CHAOS BEHAVIOR (radar chaos={CHAOS}, CFG guidance={GUIDANCE}):")
    print(f"{'model':<14} {'on-beat%':>9} {'offbeat dim41-AUC':>18}")
    print("-" * 44)
    for name in ['gen_stage1', 'gen_highres']:
        ob = 100 * np.mean(onbeat[name]) if onbeat[name] else float('nan')
        au = np.mean(offbeat_auc[name]) if offbeat_auc[name] else float('nan')
        print(f"{name:<14} {ob:>8.1f}% {au:>18.3f}")
    print("-" * 44)
    print("H4 baseline: chaos smears to ~6% on-beat. If gen_highres off-beats now land on high-res-onset")
    print("frames (AUC>>0.5) AND on-beat% is similar-or-better, the feature made chaos event-driven.")
    print("If AUC~0.5 and KL low, the warm-start didn't engage the feature -> needs from-scratch / reweight.")


if __name__ == '__main__':
    main()

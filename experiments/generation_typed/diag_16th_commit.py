#!/usr/bin/env python3
"""
Step 0 of the v6 retrain scope ([[chaos_retrain_scope]]): confirm WHY the model substitutes 8ths for 16ths
under COHERENT conditioning (the fair-test defect). Decides the retrain lever: objective / feature / data.

PRIMARY (knows-but-loses?) — model + coherent conditioning (radar = the song's own real radar), per-song
density threshold. At real-16th-NOTE frames vs the real-8th-NOTE frames the model places:
  recall_8  = frac of real-8th-note frames that clear tau (model places these)
  recall_16 = frac of real-16th-note frames that clear tau (model MISSES these -> substitution)
  p_on at real-16th-notes vs real-no-note-16th-frames -> does the model KNOW which 16ths are real?
  AUC at 16th frames (real-16th-note vs no-note)
Reads:
  recall_16 << recall_8 AND p_on@real-16th-notes >> p_on@no-note (AUC high) -> KNOWS-BUT-LOSES: the model
    has signal at real 16ths but ranks them below 8ths so the threshold drops them -> OBJECTIVE lever.
  p_on@real-16th-notes ~ no-note (AUC ~0.5) -> CAN'T SEE -> FEATURE lever.

SECONDARY (data starvation?) — raw-parse charts (no model): do Hard charts REJECTED by the >2-simultaneous
filter carry MORE 16ths than kept ones? If yes, the training data under-represents 16ths -> DATA lever
(overlaps the queued hands-filter relaxation).

  python experiments/generation_typed/diag_16th_commit.py --num_songs 50
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
from src.data.dataset import StepManiaDataset, get_difficulty_class
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
from src.generation.typed_model import LayeredTypedChartGenerator
from src.data.stepmania_parser import StepManiaParser

CKPT, AD = "checkpoints/gen_highres_v4/best_val.pt", 42


def auc(scores, labels):
    labels = labels.astype(int)
    n1 = labels.sum(); n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    order = np.argsort(scores); rank = np.empty(len(scores), float); rank[order] = np.arange(len(scores))
    return (rank[labels == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)


def primary(args, device):
    set_seed(42)
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    _, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                   use_metric_phase=True, use_highres_onset=True))
    ds = StepManiaDataset(chart_files=vf[:args.num_songs * 5], audio_dir="data/", max_sequence_length=msl,
                          feature_extractor=ext, cache_dir='cache/samples_v3')
    m = LayeredTypedChartGenerator(audio_dim=AD, d_model=128, num_layers=4, onset_layers=2).to(device)
    m.load_state_dict(torch.load(CKPT, map_location=device)['model_state_dict']); m.eval()

    rec8, rec16, p16n, p16o, p8n, aucs = [], [], [], [], [], []
    used = 0
    for i in range(len(ds.valid_samples)):
        if used >= args.num_songs:
            break
        meta = ds.valid_samples[i]
        if meta['difficulty_class'] < args.min_difficulty:
            continue
        sample = ds[i]; T = min(int(sample['mask'].sum().item()), args.max_len)
        if T < 256:
            continue
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None:
            continue
        orig = np.asarray(ds.parser.convert_to_tensor_typed(meta['chart'], nd))[:T]
        note = (orig != 0).any(1); t = np.arange(T)
        is16 = (t % 4 == 1) | (t % 4 == 3); is8 = (t % 4 == 2)
        if (note & is16).sum() < 16:   # need real 16ths to measure their recall
            continue
        rd = float(note.mean())
        radar = torch.from_numpy(meta['groove_radar'].to_vector().astype(np.float32)).unsqueeze(0).to(device)
        diff = torch.tensor([meta['difficulty_class']], device=device)
        audio = sample['audio'][:T, :AD].unsqueeze(0).to(device)
        with torch.no_grad():  # COHERENT: condition on the song's own real radar
            p = torch.sigmoid(m.onset_logits(m.encode_audio(audio), diff, radar=radar))[0].cpu().numpy()
        tau = float(np.quantile(p, 1 - rd))
        rec8.append((p[note & is8] > tau).mean()); rec16.append((p[note & is16] > tau).mean())
        p16n.append(p[note & is16].mean()); p16o.append(p[(~note) & is16].mean()); p8n.append(p[note & is8].mean())
        aucs.append(auc(p[is16], (note & is16)[is16]))
        used += 1

    print(f"\n=== PRIMARY: knows-but-loses? ({used} songs, coherent conditioning, per-song threshold) ===")
    print(f"  recall of real-8th notes  (placed):  {np.mean(rec8):.3f}")
    print(f"  recall of real-16th notes (placed):  {np.mean(rec16):.3f}   <- if << recall_8, 16ths lose the budget")
    print(f"  p_on @ real-16th NOTES:              {np.mean(p16n):.3f}")
    print(f"  p_on @ real-16th NO-NOTE frames:     {np.mean(p16o):.3f}   (gap = the model KNOWS which 16ths are real)")
    print(f"  p_on @ real-8th  NOTES:              {np.mean(p8n):.3f}   (vs 16th-notes {np.mean(p16n):.3f}: the bias)")
    print(f"  AUC @ 16th frames (real-note vs not): {np.nanmean(aucs):.3f}   (0.5=can't see)")
    knows = np.nanmean(aucs) > 0.6 and np.mean(p16n) > np.mean(p16o) + 0.05
    loses = np.mean(rec16) < np.mean(rec8) - 0.15
    print(f"\n  -> {'KNOWS' if knows else 'CANT-SEE'} the 16ths; {'LOSES the budget to 8ths' if loses else 'recall comparable'}.")
    print(f"     {'KNOWS-BUT-LOSES => OBJECTIVE lever (boost 16th commitment over 8ths).' if (knows and loses) else ''}"
          f"{'CANT-SEE => FEATURE lever.' if not knows else ''}")


def secondary(args):
    p = StepManiaParser()
    files = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    files = files[:args.data_files]
    kept16, rej16 = [], []   # 16th-note fraction of Hard difficulties, kept vs rejected by >2 filter
    for f in files:
        try:
            ch = p.parse_file(f)
        except Exception:
            continue
        if ch is None:
            continue
        for nd in ch.note_data:
            if get_difficulty_class(nd.difficulty_name) < 3:   # Hard only
                continue
            typed = np.asarray(p.convert_to_tensor_typed(ch, nd))
            if typed.ndim != 2 or typed.shape[0] < 64:
                continue
            occ = (typed != 0)
            note = occ.any(1); n = int(note.sum())
            if n < 32:
                continue
            t = np.arange(len(note)); f16 = 100 * note[(t % 4 == 1) | (t % 4 == 3)].sum() / n
            rejected = occ.sum(1).max() > 2   # the >2-simultaneous (hands) rule
            (rej16 if rejected else kept16).append(f16)
    print(f"\n=== SECONDARY: data starvation? (Hard charts, {len(kept16)+len(rej16)} from {len(files)} files) ===")
    print(f"  KEPT  (<=2 panels): {len(kept16):>4} charts, mean 16th-share {np.mean(kept16) if kept16 else 0:.1f}%")
    print(f"  REJECTED (hands):   {len(rej16):>4} charts, mean 16th-share {np.mean(rej16) if rej16 else 0:.1f}%")
    if kept16 and rej16:
        print(f"  -> rejected charts {'HAVE MORE' if np.mean(rej16) > np.mean(kept16)+1 else 'do NOT have more'} 16ths "
              f"=> {'DATA lever in play (hands filter starves 16ths)' if np.mean(rej16) > np.mean(kept16)+1 else 'hands filter is NOT a 16th-data issue'}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_songs', type=int, default=50)
    ap.add_argument('--max_len', type=int, default=1024)
    ap.add_argument('--min_difficulty', type=int, default=2)
    ap.add_argument('--data_files', type=int, default=400)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    primary(args, device)
    secondary(args)


if __name__ == '__main__':
    main()

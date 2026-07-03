#!/usr/bin/env python3
"""Does the AUDIO-ONSET HEAD's placement fidelity degrade on fast songs? (no-generation, ~all val Hard)

The fast-song quality defect is INTRINSIC (governor ruled out) and NOT training coverage. Prime suspect: the
audio-onset head (audio-only, non-causal) places notes worse on fast/dense songs. This tests it DIRECTLY without
generation: run the DEPLOYED conditioned onset path (harness `conditioned_p_onset`) on each song's audio, then
measure how well `p_onset` ranks the REAL chart's onset frames (ROC-AUC + PR-AUC). Correlate that fidelity with BPM,
and partial out onset-density-per-second (the confound: fast songs literally pack more onsets/sec).

  onset-AUC drops with BPM (survives partialling density) -> the ONSET HEAD is the locus of the fast-song defect.
  onset-AUC flat with BPM -> the head is fine; the defect is downstream (pattern/type head) -> needs a gen decomposition.

Generation-free (one forward pass/song) so it runs on ~all val Hard (n~176) — far more power than the n=30 quality
probes. Usage: python probe_onset_head_bpm.py --data_dir data/ --audio_dir data/
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, csv, sys
from pathlib import Path
import numpy as np, torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.generation.decode_defaults import CANONICAL_DECODE
from src.generation.decode_harness import conditioned_p_onset
from probe_quality_features import load_val_dataset, build_songs, load_generator, DEPLOYED_CHECKPOINT


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--difficulty', type=int, default=3)
    p.add_argument('--n', type=int, default=200, help='cap (val has ~176 Hard)'); p.add_argument('--max_len', type=int, default=768)
    p.add_argument('--checkpoint', default=DEPLOYED_CHECKPOINT)
    p.add_argument('--out', default='cache/onset_head_bpm.csv')
    return p.parse_args()


def roc_auc(score, label):
    y = np.asarray(label).astype(int); s = np.asarray(score)
    npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0: return np.nan
    r = np.argsort(np.argsort(s)) + 1
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def pr_auc(score, label):
    y = np.asarray(label).astype(int); order = np.argsort(-np.asarray(score))
    y = y[order]; tp = np.cumsum(y); fp = np.cumsum(1 - y)
    if y.sum() == 0: return np.nan
    prec = tp / (tp + fp); rec = tp / y.sum()
    return float(np.trapz(prec, rec))


def sp(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 6 or np.std(x[ok]) < 1e-9 or np.std(y[ok]) < 1e-9: return np.nan
    return np.corrcoef(np.argsort(np.argsort(x[ok])), np.argsort(np.argsort(y[ok])))[0, 1]


def partial(x, y, z):
    x, y, z = [np.asarray(a, float) for a in (x, y, z)]; ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    rx, ry, rz = [np.argsort(np.argsort(a[ok])).astype(float) for a in (x, y, z)]
    res = lambda a, b: a - (np.polyfit(b, a, 1)[0] * b + np.polyfit(b, a, 1)[1])
    return np.corrcoef(res(rx, rz), res(ry, rz))[0, 1]


def main():
    args = parse_args(); set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_ds = load_val_dataset(args.data_dir, args.audio_dir, args.seed)
    songs = build_songs(val_ds, args.n, args.difficulty, args.max_len)
    print(f"songs={len(songs)} (Hard)")
    model = load_generator(args.checkpoint, 42, device)
    phase_calib = CANONICAL_DECODE['onset_phase_calib']

    rows = []
    for s in songs:
        audio = torch.from_numpy(s['audio']).unsqueeze(0).to(device); diff = torch.tensor([s['difficulty']], device=device)
        with torch.no_grad():
            memory = model.encode_audio(audio)
            p_onset = conditioned_p_onset(model, memory, diff, radar=None, style=None, guidance=1.0, phase_calib=phase_calib)
        real_on = (np.asarray(s['real_typed']) != 0).any(1).astype(int)   # frame has a note (real chart)
        T = min(len(p_onset), len(real_on)); p = p_onset[:T]; y = real_on[:T]
        frame_hz = s['bpm'] * 4.0 / 60.0
        rows.append({'title': s['title'], 'bpm': s['bpm'], 'frame_hz': frame_hz,
                     'onsets_per_sec': float(y.mean() * frame_hz), 'density': float(y.mean()),
                     'onset_auc': roc_auc(p, y), 'onset_prauc': pr_auc(p, y)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"wrote {args.out}")

    bpm = np.array([r['bpm'] for r in rows]); auc = np.array([r['onset_auc'] for r in rows])
    prauc = np.array([r['onset_prauc'] for r in rows]); dens = np.array([r['density'] for r in rows])
    ops = np.array([r['onsets_per_sec'] for r in rows])
    print("\n" + "=" * 74)
    print(f"  ONSET-HEAD PLACEMENT FIDELITY vs BPM  (n={len(rows)} Hard songs, no generation)")
    print("=" * 74)
    print(f"  onset ROC-AUC: mean {np.nanmean(auc):.3f}  |  PR-AUC: mean {np.nanmean(prauc):.3f}")
    print(f"  spearman(bpm, onset ROC-AUC)              = {sp(bpm, auc):+.3f}")
    print(f"  spearman(bpm, onset PR-AUC)               = {sp(bpm, prauc):+.3f}")
    print(f"  spearman(onsets/sec, onset PR-AUC)        = {sp(ops, prauc):+.3f}   (raw onset-density-per-sec effect)")
    print(f"  partial(bpm, PR-AUC | density)            = {partial(bpm, prauc, dens):+.3f}   <- tempo net of density")
    print(f"  partial(bpm, PR-AUC | onsets/sec)         = {partial(bpm, prauc, ops):+.3f}   <- tempo net of onsets/sec")
    order = np.argsort(bpm)
    print("\n  by BPM tertile:")
    for lab, idx in [('slow', order[:len(order)//3]), ('mid', order[len(order)//3:2*len(order)//3]), ('fast', order[2*len(order)//3:])]:
        print(f"    {lab:4s} bpm~{bpm[idx].mean():5.0f}: onset ROC-AUC {np.nanmean(auc[idx]):.3f}  PR-AUC {np.nanmean(prauc[idx]):.3f}  onsets/sec {ops[idx].mean():.1f}")
    print("\n  READ: AUC dropping with BPM (esp. surviving partial|density) => the audio-onset head IS the fast-song")
    print("  locus. Flat AUC => the head is fine; the defect is downstream (pattern/type) and needs a gen decomposition.")


if __name__ == '__main__':
    main()

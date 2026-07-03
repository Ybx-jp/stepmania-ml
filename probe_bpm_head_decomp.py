#!/usr/bin/env python3
"""HEAD DECOMPOSITION of the fast-song defect: onset positions vs which-panel (pattern head).

The fast-song quality defect is INTRINSIC (governor ruled out), NOT coverage, NOT the onset head's PLACEMENT AUC
(flat vs BPM). Prime remaining suspect: the PATTERN (+type) head's arrow-choice at high note density. Direct test via
`onset_override` (caller supplies the onset frames; pattern/type heads pick the panels; stamina auto-skips):

  arm REAL-ONSET : onsets = the REAL chart's frames  (PERFECT timing) + generated panels  -> isolates the pattern head
  arm GEN-ONSET  : onsets = the model's OWN decoded frames (from a canonical pass) + generated panels  -> control,
                   SAME override regime (OOD + stamina-skip) so the only diff vs REAL-ONSET is the onset SOURCE
  (canonical    : the deployed gen-onset+gen-panel chart, scored too as the non-override reference)

READ:
  slope(bpm, q_REAL-ONSET) still strongly negative  => even with PERFECT onsets the generation degrades on fast
    songs -> the PATTERN/TYPE head is the causal locus.
  paired spearman(bpm, q_REAL-ONSET - q_GEN-ONSET) ~ 0  => real onsets don't differentially help fast songs -> the
    onset SOURCE is NOT the fast-song differentiator (consistent with the flat onset-head AUC).

Graded critic (margin); denoised over K; deployed decode via the shared harness (onset_override through
canonical_gen_typed's decode_overrides — Rule 14). CAVEAT: onset_override puts the pattern head slightly OOD
(trained on its own onsets) and skips stamina; that shifts the LEVEL ~BPM-independently, so the SLOPE stays
interpretable and the REAL-vs-GEN delta is regime-matched. Usage:
    python probe_bpm_head_decomp.py --data_dir data/ --audio_dir data/ --n 20 --k 3
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, csv, sys
from pathlib import Path
import numpy as np, torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.models import LateFusionClassifier
from probe_quality_features import (load_val_dataset, build_songs, canonical_gen_typed, critic_score,
                                    to_binary, spearman, load_generator, DEPLOYED_CHECKPOINT)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--difficulty', type=int, default=3)
    p.add_argument('--n', type=int, default=20); p.add_argument('--k', type=int, default=3)
    p.add_argument('--max_len', type=int, default=768)
    p.add_argument('--checkpoint', default=DEPLOYED_CHECKPOINT)
    p.add_argument('--critic', default='checkpoints/realism_critic_graded/best_val.pt')
    p.add_argument('--out', default='cache/bpm_head_decomp.csv')
    return p.parse_args()


def mask_of(typed, T, device):
    m = (np.asarray(typed)[:T] != 0).any(1)
    return torch.from_numpy(m).unsqueeze(0).to(device)   # (1,T) bool


def partial(x, y, z):
    x, y, z = [np.asarray(a, float) for a in (x, y, z)]
    rx, ry, rz = [np.argsort(np.argsort(a)).astype(float) for a in (x, y, z)]
    res = lambda a, b: a - (np.polyfit(b, a, 1)[0] * b + np.polyfit(b, a, 1)[1])
    return np.corrcoef(res(rx, rz), res(ry, rz))[0, 1]


def perm_p(x, y, nperm=5000):
    obs = spearman(x, y); rng = np.random.default_rng(0)
    null = np.array([spearman(x, rng.permutation(y)) for _ in range(nperm)])
    return obs, (np.abs(null) >= abs(obs)).mean()


def main():
    args = parse_args(); set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device} | n={args.n} songs x k={args.k} x 3 passes | critic={args.critic}")
    val_ds = load_val_dataset(args.data_dir, args.audio_dir, args.seed)
    songs = build_songs(val_ds, args.n, args.difficulty, args.max_len)
    print(f"songs={len(songs)}")
    model = load_generator(args.checkpoint, 42, device)
    ck = torch.load(args.critic, map_location=device, weights_only=False)
    critic = LateFusionClassifier(ck['config']).to(device); critic.load_state_dict(ck['model_state_dict']); critic.eval()

    rows = []
    for n, s in enumerate(songs, 1):
        a23 = s['audio'][:, :23]; T = s['T']
        real_mask = mask_of(s['real_typed'], T, device)
        qc, qr, qg = [], [], []
        for _ in range(args.k):
            canon = canonical_gen_typed(model, s, device)                                   # gen-onset + gen-panel
            gen_mask = mask_of(canon, T, device)
            real_ov = canonical_gen_typed(model, s, device, {'onset_override': real_mask})   # real-onset + gen-panel
            gen_ov = canonical_gen_typed(model, s, device, {'onset_override': gen_mask})      # gen-onset(ov) + gen-panel
            qc.append(critic_score(critic, a23, to_binary(canon), device)[1])
            qr.append(critic_score(critic, a23, to_binary(real_ov), device)[1])
            qg.append(critic_score(critic, a23, to_binary(gen_ov), device)[1])
        rows.append({'title': s['title'], 'bpm': s['bpm'], 'real_density': s['real_density'],
                     'q_canon': float(np.mean(qc)), 'q_real_ov': float(np.mean(qr)), 'q_gen_ov': float(np.mean(qg))})
        print(f"  [{n}/{len(songs)}] bpm={s['bpm']:5.0f} {s['title'][:20]:20s} "
              f"canon={np.mean(qc):+.2f} real_ov={np.mean(qr):+.2f} gen_ov={np.mean(qg):+.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"wrote {args.out}")

    bpm = np.array([r['bpm'] for r in rows])
    qc = np.array([r['q_canon'] for r in rows]); qr = np.array([r['q_real_ov'] for r in rows]); qg = np.array([r['q_gen_ov'] for r in rows])
    print("\n" + "=" * 76)
    print("  HEAD DECOMPOSITION — where does the fast-song (BPM) quality drop live?")
    print("=" * 76)
    print(f"  slope(bpm, q_canon    [deployed gen-onset+gen-panel]) = {spearman(bpm, qc):+.3f}")
    print(f"  slope(bpm, q_GEN-ONSET  [override, model onsets]    ) = {spearman(bpm, qg):+.3f}")
    print(f"  slope(bpm, q_REAL-ONSET [override, PERFECT onsets]  ) = {spearman(bpm, qr):+.3f}   "
          f"<- still NEG => PATTERN/TYPE head is the locus")
    dr, dp = perm_p(bpm, qr - qg)
    print(f"\n  paired: spearman(bpm, q_REAL-ONSET - q_GEN-ONSET) = {dr:+.3f}  two-sided p={dp:.3f}  "
          f"{'onsets DO help fast songs (onset source contributes)' if (dr>0 and dp<0.05) else 'onsets do NOT differentially help fast songs (pattern head owns it)'}")
    print(f"  main effect: mean q_REAL-ONSET - q_GEN-ONSET = {(qr-qg).mean():+.3f} (perfect onsets' overall lift)")
    order = np.argsort(bpm)
    print("\n  by BPM tertile:")
    for lab, idx in [('slow', order[:len(order)//3]), ('mid', order[len(order)//3:2*len(order)//3]), ('fast', order[2*len(order)//3:])]:
        print(f"    {lab:4s} bpm~{bpm[idx].mean():5.0f}: canon={qc[idx].mean():+.2f}  gen_ov={qg[idx].mean():+.2f}  "
              f"real_ov={qr[idx].mean():+.2f}  (real-gen Δ={ (qr[idx]-qg[idx]).mean():+.2f})")
    print("\n  READ: if q_REAL-ONSET still slopes strongly negative with BPM, then even PERFECT onsets don't rescue")
    print("  fast songs -> the pattern/type head (arrow choice at high density) is the causal locus.")


if __name__ == '__main__':
    main()

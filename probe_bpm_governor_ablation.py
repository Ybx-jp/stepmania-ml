#!/usr/bin/env python3
"""Does the BPM-coupled GOVERNOR cause the fast-song quality degradation? (mechanism ablation)

Finding (quality_feature_attribution_findings.md): faster Hard songs -> worse generations (BPM r=-0.68, p_fw=0.004,
a GENERATION defect). One candidate mechanism: the per-note fatigue + per-region stamina GOVERNOR is BPM-coupled
(frame_hz = bpm*4/60; conditioning-mechanics §8) and stressed hardest on fast songs.

TEST (one labeled variable = the governor; everything else canonical incl. MANDATORY playability):
  arm ON  = canonical decode (fatigue_penalty=2, stamina_ceiling=50)
  arm OFF = same, governor disabled (fatigue_penalty=None, stamina_ceiling=None)
Per song, K generations each arm, GRADED-critic margin, per-song mean q_on / q_off (DENOISE — the ICC lesson).

PRIMARY (paired, powerful): spearman(bpm, q_off - q_on). If the governor disproportionately hurts fast songs,
removing it helps FAST songs more than slow -> this is POSITIVE. Also report the two slopes (does BPM->quality
FLATTEN governor-off?) and the main effect (mean q shift). Compare SLOPES, not means (governor-off shifts all charts).
  POSITIVE delta-slope + flatter off-slope => governor BPM-coupling is (part of) the cause -> retune its BPM scaling.
  ~zero => intrinsic fast-song difficulty / training coverage, NOT the governor.

Uses the GRADED critic; deployed canonical decode via the shared harness (governor toggled through
canonical_gen_typed's decode_overrides — Rule 14, one code path). Usage:
    python probe_bpm_governor_ablation.py --data_dir data/ --audio_dir data/ --n 30 --k 6
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

GOV_OFF = {'fatigue_penalty': None, 'stamina_ceiling': None}   # the one labeled variable (governor disabled)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--difficulty', type=int, default=3)
    p.add_argument('--n', type=int, default=30); p.add_argument('--k', type=int, default=6)
    p.add_argument('--max_len', type=int, default=768)
    p.add_argument('--checkpoint', default=DEPLOYED_CHECKPOINT)
    p.add_argument('--critic', default='checkpoints/realism_critic_graded/best_val.pt')
    p.add_argument('--out', default='cache/bpm_governor_ablation.csv')
    return p.parse_args()


def q_arm(model, critic, s, a23, device, k, overrides):
    m = [critic_score(critic, a23, to_binary(canonical_gen_typed(model, s, device, overrides)), device)[1]
         for _ in range(k)]
    return float(np.mean(m))


def perm_p(x, y, nperm=5000, one_sided_positive=True):
    obs = spearman(x, y); rng = np.random.default_rng(0)
    null = np.array([spearman(x, rng.permutation(y)) for _ in range(nperm)])
    return obs, ((null >= obs).mean() if one_sided_positive else (np.abs(null) >= abs(obs)).mean())


def main():
    args = parse_args(); set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device} | n={args.n} songs x k={args.k}/arm x 2 arms | critic={args.critic}")
    val_ds = load_val_dataset(args.data_dir, args.audio_dir, args.seed)
    songs = build_songs(val_ds, args.n, args.difficulty, args.max_len)
    print(f"songs={len(songs)}")
    model = load_generator(args.checkpoint, 42, device)
    ck = torch.load(args.critic, map_location=device, weights_only=False)
    critic = LateFusionClassifier(ck['config']).to(device); critic.load_state_dict(ck['model_state_dict']); critic.eval()

    rows = []
    for n, s in enumerate(songs, 1):
        a23 = s['audio'][:, :23]
        q_on = q_arm(model, critic, s, a23, device, args.k, None)      # canonical (governor ON)
        q_off = q_arm(model, critic, s, a23, device, args.k, GOV_OFF)  # governor OFF
        rows.append({'title': s['title'], 'bpm': s['bpm'], 'real_density': s['real_density'],
                     'q_on': q_on, 'q_off': q_off, 'delta_off_minus_on': q_off - q_on})
        print(f"  [{n}/{len(songs)}] bpm={s['bpm']:5.0f} {s['title'][:22]:22s} q_on={q_on:+.2f} q_off={q_off:+.2f} "
              f"Δ(off-on)={q_off-q_on:+.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"wrote {args.out}")

    bpm = np.array([r['bpm'] for r in rows]); q_on = np.array([r['q_on'] for r in rows])
    q_off = np.array([r['q_off'] for r in rows]); delta = q_off - q_on
    slope_on = spearman(bpm, q_on); slope_off = spearman(bpm, q_off)
    d_obs, d_p = perm_p(bpm, delta, one_sided_positive=True)

    print("\n" + "=" * 76)
    print("  GOVERNOR ABLATION — is the BPM->quality degradation caused by the BPM-coupled governor?")
    print("=" * 76)
    print(f"  BPM->quality SLOPE   governor ON : {slope_on:+.3f}")
    print(f"  BPM->quality SLOPE   governor OFF: {slope_off:+.3f}   (FLATTER/less-neg => governor was the cause)")
    print(f"  main effect (mean q_off - q_on)  : {q_off.mean()-q_on.mean():+.3f}   (global governor effect; not the mechanism)")
    print(f"  PRIMARY paired test  spearman(bpm, Δ[off-on]) = {d_obs:+.3f}  one-sided p={d_p:.3f}  "
          f"{'GOVERNOR-CAUSED (removing it helps fast songs more)' if (d_obs>0 and d_p<0.05) else 'NOT governor-caused (intrinsic)'}")
    order = np.argsort(bpm)
    print("\n  by BPM tertile:")
    for lab, idx in [('slow', order[:len(order)//3]), ('mid', order[len(order)//3:2*len(order)//3]), ('fast', order[2*len(order)//3:])]:
        print(f"    {lab:4s} bpm~{bpm[idx].mean():5.0f}: q_on={q_on[idx].mean():+.2f}  q_off={q_off[idx].mean():+.2f}  "
              f"Δ(off-on)={delta[idx].mean():+.2f}")
    print("\n  READ: if Δ(off-on) rises slow->fast (and the paired test is +sig) AND slope_off is flatter than slope_on,")
    print("  the governor's BPM-coupling is (part of) the cause -> retune its BPM scaling. If Δ is flat, it's intrinsic.")


if __name__ == '__main__':
    main()

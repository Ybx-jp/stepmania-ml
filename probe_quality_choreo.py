#!/usr/bin/env python3
"""Which audio features drive per-song CHOREOGRAPHY quality of the canonical-default generator?

The taste critic SATURATES on canonical Hard generations (94% railed to "fake", 0% mid-band -> no dynamic range;
notes/quality_feature_attribution_findings.md). This probe swaps in a GRADED, non-saturating quality proxy: the
per-song CHOREOGRAPHY DISTANCE-TO-REAL, built from the VALIDATED battery (notes/choreography_metrics_findings.md):
  - trans_KL     : KL(gen panel-transition matrix ‖ POOLED-real-Hard transition matrix). The validated which-arrow
                   (H1) metric — gen transitions were shown ~as far from real as a random panel-shuffle.
  - holdburst    : the fast one-foot-cross-during-a-hold rate (bipedal_metrics.stats). The ONE metric shown to
                   PREDICT a play-feel complaint (B4U "crossovers/jacks with one foot during a hold"). Excess over
                   pooled-real = worse.
  - panel_TV     : total-variation between gen and pooled-real panel-usage distribution (gen has a horizontal bias).
  - composite    : z(trans_KL) + z(holdburst_excess) + z(panel_TV)  (equal-weight standardized distance; higher=worse).
Generation replicates the DEPLOYED canonical decode (CANONICAL_DECODE + decode_harness), TYPED (holds kept), via
the shared helpers in probe_quality_features.py (Rule 14 — no re-derivation).

Guards: Rule 11 (gate the proxy's dynamic range AND that it DISCRIMINATES gen from real before trusting it),
Rule 12 (Hard-only by default), Rule 1 (drop per-song-z-scored near-constant regressors), + a family-wise
permutation test so a top |r| at the noise floor is not mistaken for a driver.

Usage: python probe_quality_choreo.py --data_dir data/ --audio_dir data/ --difficulty 3 --n 48
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, sys
from pathlib import Path
import numpy as np, torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'experiments' / 'realism_critic'))
from src.utils.reproducibility import set_seed
# validated choreography battery — reuse the exact functions (Rule 14), don't re-derive:
from choreography_metrics import note_starts, transition_matrix, kl, movement, vel_stats  # noqa: E402
from bipedal_metrics import stats as bipedal_stats                                        # noqa: E402
# shared generation + feature + dataset infra:
from probe_quality_features import (load_val_dataset, build_songs, canonical_gen_typed,   # noqa: E402
                                    cache_features, raw_descriptors, spearman, load_generator, DEPLOYED_CHECKPOINT)

DIFF_NAMES = ['Beginner', 'Easy', 'Medium', 'Hard']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True); p.add_argument('--audio_dir', required=True)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--checkpoint', default=DEPLOYED_CHECKPOINT)
    p.add_argument('--n', type=int, default=48); p.add_argument('--max_len', type=int, default=768)
    p.add_argument('--difficulty', type=int, default=3)     # Hard by default (the saturated tier)
    p.add_argument('--out', default='cache/quality_choreo_hard.csv')
    return p.parse_args()


def panel_dist(typed):
    ns = note_starts(typed); c = ns.sum(0).astype(float)
    return c / c.sum() if c.sum() > 0 else np.full(4, 0.25)


def choreo_raw(typed):
    """Per-chart raw choreography descriptors used to build the distances."""
    ns = note_starts(typed)
    tm = transition_matrix(ns)                         # 4x4 counts
    d, g = movement(ns); v = vel_stats(d, g)
    hb = bipedal_stats(typed)                          # dict: vel, fast, hold_burst, hb_n
    return {'tm': tm, 'panel': panel_dist(typed), 'vel': v['vel'],
            'holdburst': (hb['hold_burst'] if np.isfinite(hb['hold_burst']) else 0.0), 'hb_n': hb['hb_n']}


def tv(p, q):  # total variation between two distributions
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def zscore(a):
    a = np.asarray(a, float); sd = a.std()
    return (a - a.mean()) / sd if sd > 1e-9 else np.zeros_like(a)


def main():
    args = parse_args(); set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device} | checkpoint={args.checkpoint} | difficulty={args.difficulty}")

    val_ds = load_val_dataset(args.data_dir, args.audio_dir, args.seed)
    songs = build_songs(val_ds, args.n, args.difficulty, args.max_len)
    print(f"songs={len(songs)} | dims={songs[0]['audio'].shape[1]}")
    model = load_generator(args.checkpoint, 42, device)

    # ---- 1. generate + measure raw choreography for gen and the song's own real chart ------------------------
    per = []
    for n, s in enumerate(songs, 1):
        gen_typed = canonical_gen_typed(model, s, device)
        cg = choreo_raw(gen_typed); cr = choreo_raw(s['real_typed'])
        feats = {'title': s['title'], 'difficulty': s['difficulty'], 'bpm': s['bpm'], 'real_density': s['real_density']}
        feats.update(cache_features(s['audio'])); feats.update(raw_descriptors(s['audio_file']))
        per.append({'s': s, 'gen': cg, 'real': cr, 'feats': feats})
        print(f"  [{n}/{len(songs)}] {s['title'][:26]:26s} gen holdburst={cg['holdburst']:.3f} (real {cr['holdburst']:.3f})")

    # ---- 2. POOLED real-Hard reference (the stable "what real choreography looks like" target) ---------------
    pooled_tm = sum(p['real']['tm'] for p in per)                       # aggregate transition counts
    pooled_panel = np.mean([p['real']['panel'] for p in per], 0)
    pooled_holdburst = float(np.mean([p['real']['holdburst'] for p in per]))
    pooled_vel = float(np.nanmean([p['real']['vel'] for p in per]))
    print(f"\n  pooled real-Hard: holdburst={pooled_holdburst:.3f}  vel={pooled_vel:.3f}  "
          f"panel(L,D,U,R)=[{','.join(f'{x:.2f}' for x in pooled_panel)}]")

    # ---- 3. distance-to-pooled-real for EACH chart (gen AND real, to check the proxy DISCRIMINATES) ----------
    def distances(c):
        return {'trans_KL': kl(c['tm'], pooled_tm), 'holdburst_excess': c['holdburst'] - pooled_holdburst,
                'panel_TV': tv(c['panel'], pooled_panel), 'vel_gap': abs(c['vel'] - pooled_vel) if np.isfinite(c['vel']) else np.nan}
    for p in per:
        p['dg'] = distances(p['gen']); p['dr'] = distances(p['real'])
    for key in ('trans_KL', 'holdburst_excess', 'panel_TV'):             # equal-weight standardized composite (gen)
        zg = zscore([p['dg'][key] for p in per])
        for p, z in zip(per, zg): p.setdefault('comp', 0.0); p['comp'] = p['comp'] + z
    rows = []
    for p in per:
        r = dict(p['feats']); r.update({f'g_{k}': v for k, v in p['dg'].items()})
        r['choreo_composite'] = p['comp']; rows.append(r)

    keys = list(rows[0].keys())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"wrote {args.out} ({len(rows)} songs x {len(keys)} cols)")

    # ---- 4. PROXY VALIDATION: does the choreography distance separate gen (bad) from real (good)? -------------
    print("\n" + "=" * 78)
    print("  PROXY VALIDATION — the distance must (a) DISCRIMINATE gen>real and (b) VARY across songs (Rule 11)")
    print("=" * 78)
    for key in ('trans_KL', 'holdburst_excess', 'panel_TV', 'vel_gap'):
        g = np.array([p['dg'][key] for p in per], float); r = np.array([p['dr'][key] for p in per], float)
        gm, gs = np.nanmean(g), np.nanstd(g); rm = np.nanmean(r)
        print(f"  {key:18s}: gen {gm:+.3f} (sd {gs:.3f})  vs  real {rm:+.3f}   "
              f"discriminates={'YES' if abs(gm) > abs(rm) + 1e-6 else 'no'}  dyn_range={'ok' if gs > 1e-3 else 'FLAT'}")
    comp = np.array([p['comp'] for p in per], float)
    print(f"  choreo_composite  : sd {comp.std():.3f}  range [{comp.min():+.2f},{comp.max():+.2f}]  (the target)")

    # ---- 5. feature correlations against the choreography quality proxy + permutation floor ------------------
    exclude = {'title', 'choreo_composite'} | {k for k in keys if k.startswith('g_')}
    featkeys = [k for k in keys if k not in exclude]
    def rank_against(tgt):
        out = []
        for k in featkeys:
            vals = np.array([r.get(k, np.nan) for r in rows], float)
            if np.sum(np.isfinite(vals)) < max(6, len(rows) // 2) or np.nanstd(vals) < 1e-6: continue
            rsp = spearman(vals, tgt)
            if np.isfinite(rsp): out.append((k, rsp))
        out.sort(key=lambda x: -abs(x[1])); return out
    tag = lambda k: 'raw ' if k.startswith('raw_') else ('d## ' if (len(k) > 3 and k[1:3].isdigit()) else 'cur ')
    thr = 1.96 / np.sqrt(len(rows))
    for tname, tgt in [('choreo_composite', comp), ('g_trans_KL', np.array([r['g_trans_KL'] for r in rows])),
                       ('g_holdburst_excess', np.array([r['g_holdburst_excess'] for r in rows]))]:
        ranked = rank_against(tgt)
        print("\n" + "=" * 78)
        print(f"  FEATURE -> Spearman(feature, {tname})   [n={len(rows)}; |r|>~{thr:.2f} ~ p<.05 UNCORRECTED]")
        print("=" * 78)
        for k, rsp in ranked[:14]:
            print(f"    {tag(k)}{k:24s} r={rsp:+.3f}")
        # family-wise permutation floor for the single best feature
        cols = {k: np.array([r.get(k, np.nan) for r in rows], float) for k, _ in ranked}
        if ranked:
            obs = abs(ranked[0][1]); rng = np.random.default_rng(0); nmax = []
            V = [v for v in cols.values()]
            for _ in range(1000):
                perm = np.argsort(np.argsort(rng.permutation(tgt)))
                nmax.append(max(abs(np.corrcoef(np.argsort(np.argsort(v)), perm)[0, 1]) for v in V if np.nanstd(v) > 1e-9))
            nmax = np.array(nmax); pfw = (nmax >= obs).mean()
            print(f"  family-wise permutation: best |r|={obs:.3f}, null-max mean {nmax.mean():.3f} / 95th {np.quantile(nmax,0.95):.3f}"
                  f"  -> p_fw={pfw:.3f}  {'SIGNAL' if pfw < 0.05 else 'noise floor'}")
    print("\n  NOTE: if the proxy fails validation (no discrimination / FLAT), the correlations are meaningless.")


if __name__ == '__main__':
    main()

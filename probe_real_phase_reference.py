#!/usr/bin/env python
"""RULE 5 (experiment-design): what do REAL Hard charts do across the chaos axis? The tolerance/good-region
work measures the GENERATED backbone dissolving under cranked chaos+guidance — but H4/H14 warn that regime is
a KNOWN degenerate global-smear, and real charts reportedly raise chaos by ADDING density on a PRESERVED
backbone (chaos<->density +0.63). This bins REAL Hard charts by their own chaos radar and reads off the phase
structure, giving the REFERENCE distribution the generated metric should be compared against (not an arbitrary
threshold). Offline: no model, no generation — parse real charts + compute phase metrics on the REAL typed grid.
"""
import warnings, sys
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from probe_quality_features import load_val_dataset
from probe_backbone_tolerance import _onsets, on_grid_share, quarter_representation, sixteenth_anchoring, ONSET_ENV_DIM


def s16_rate(typed):
    on = _onsets(typed); idx = np.where(on)[0]
    return float(np.mean((idx % 4) % 2 == 1)) if idx.size else np.nan   # fraction of onsets on 16th-offbeats


def main():
    set_seed(42)
    vs = load_val_dataset('data/', 'data/', 42, 'cache/samples_v3')
    rows = []
    for i in range(len(vs)):
        meta = vs.valid_samples[i]
        if int(meta['difficulty_class']) != 3:            # Hard only
            continue
        nd = next((n for n in meta['chart'].note_data if n.difficulty_name == meta['difficulty_name']
                   and n.difficulty_value == meta['difficulty_value']), None)
        if nd is None:
            continue
        s = vs[i]; T = int(s['mask'].sum().item())
        if T < 128:
            continue
        typed = np.asarray(vs.parser.convert_to_tensor_typed(meta['chart'], nd))[:T]
        env = s['audio'][:T, ONSET_ENV_DIM].numpy()
        radar = meta['groove_radar'].to_vector()
        rows.append(dict(chaos=float(radar[4]), density=float((typed != 0).any(1).mean()),
                         on_grid=on_grid_share(typed), qrep=quarter_representation(typed, env),
                         s16=s16_rate(typed), anch=sixteenth_anchoring(typed)))
    print(f"REAL Hard charts: n={len(rows)}\n")

    ch = np.array([r['chaos'] for r in rows])
    qs = np.quantile(ch, [0, 0.25, 0.5, 0.75, 1.0])
    print(f"chaos radar range: {ch.min():.2f}-{ch.max():.2f}  (quartile edges {np.round(qs,2)})")
    print("\nphase structure by REAL chaos QUARTILE (does the backbone survive as chaos rises?):")
    hdr = f"{'chaos bin':16s} {'n':>4} {'chaos':>6} {'density':>8} {'on_grid':>8} {'qrep':>6} {'s16_rate':>9} {'anchor':>7}"
    print(hdr); print('-' * len(hdr))
    for lo, hi, name in [(qs[0], qs[1], 'Q1 (calm)'), (qs[1], qs[2], 'Q2'), (qs[2], qs[3], 'Q3'), (qs[3], qs[4] + 1e-9, 'Q4 (chaotic)')]:
        b = [r for r in rows if lo <= r['chaos'] < hi]
        if not b:
            continue
        m = lambda k: np.nanmean([r[k] for r in b])
        print(f"{name:16s} {len(b):>4} {m('chaos'):>6.2f} {m('density'):>8.3f} {m('on_grid'):>8.2f} "
              f"{m('qrep'):>6.2f} {m('s16'):>9.2f} {m('anch'):>7.2f}")

    from scipy.stats import spearmanr
    print("\nchaos vs each metric (Spearman) — the REAL coupling:")
    for k, label in [('density', 'density (H4 says +0.63: chaos ADDS notes)'),
                     ('on_grid', 'on-grid share (backbone PRESERVED? => ~flat/high)'),
                     ('qrep', 'quarter-rep strict (downbeat coverage)'),
                     ('s16', '16th-offbeat rate (chaos = more off-beats)'),
                     ('anch', '16th anchoring')]:
        x = np.array([r[k] for r in rows]); good = np.isfinite(x)
        rho, p = spearmanr(ch[good], x[good])
        print(f"  chaos -> {k:9s} rho={rho:+.3f} p={p:.3g}   [{label}]")
    print("\nKEY: if on_grid/qrep stay HIGH as chaos rises while density & s16 rise => real charts ADD off-beats on a")
    print("PRESERVED backbone. The generated failure (backbone VACATED to a 1/16-offset spine) is then OFF the real")
    print("manifold => 'tolerance' = distance from THIS real high-chaos phase profile, not an arbitrary threshold.")


if __name__ == '__main__':
    main()

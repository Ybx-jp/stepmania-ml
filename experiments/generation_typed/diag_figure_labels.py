#!/usr/bin/env python3
"""
Hierarchical pick-then-realize PRE-FLIGHT gate (NO training) — is a DISCRETE per-section FIGURE label viable
as the "pick" signal, and does it target the stuck sweep axis? (experiment-design Rules 5,6.)

The continuous local-motif vector rescued candle/trill but NOT jack<->sweep (notes/h15_local_motif_plan.md).
Hypothesis for the hierarchical model: a discrete, COMMITTED per-section figure token (jack/sweep/trill/candle/
jump/step) gives the pattern head a sharper "produce THIS figure" target than a diluted 12-d projection. Before
building it, check on REAL charts (section=64):
  (1) DISTRIBUTION — are there enough SWEEP sections to learn from? (if sweep is ~absent at this scale, it's a
      longer-scale structure -> reframes the fix, and explains why a section vector can't pin it.)
  (2) ALIGNMENT (Rule 5) — do 'sweep'-labeled sections actually have high knob-0 (the sweep axis)? confirms the
      discrete label targets the axis we want to fix.
  (3) WITHIN-CHART VARIETY — do charts span multiple figure classes across sections? (needed for per-section
      figure control to mean anything.)
  (4) DOMINANCE — how dominant is a section's top figure class (is a section one figure, or a mush)?

  /home/ybx/miniconda3/envs/stepmania-chart-gen/bin/python \
      experiments/generation_typed/diag_figure_labels.py [--section 64] [--charts 600]
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, sys, glob
from collections import Counter
from pathlib import Path
import numpy as np, yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.generation.motif_codebook import MotifBasis, onset_tokens, _canon, name_figure

FIGS = ["jack", "sweep/staircase", "trill", "candle/cross", "jump/bracket", "step"]


def section_figure(sec_chart, W=3):
    """Dominant canonical W-window figure family of a section, via name_figure. 'sparse' if too few onsets."""
    toks = onset_tokens(sec_chart)
    if len(toks) < W:
        return "sparse"
    cnt = Counter(_canon(toks[j:j + W]) for j in range(len(toks) - W + 1))
    top_canon, _ = cnt.most_common(1)[0]
    return name_figure(top_canon)


def section_dominance(sec_chart, W=3):
    toks = onset_tokens(sec_chart)
    if len(toks) < W:
        return np.nan
    fams = Counter(name_figure(_canon(toks[j:j + W])) for j in range(len(toks) - W + 1))
    return fams.most_common(1)[0][1] / sum(fams.values())   # share of the dominant family


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", type=int, default=64)
    ap.add_argument("--charts", type=int, default=600)
    ap.add_argument("--min_onsets", type=int, default=8)
    args = ap.parse_args()
    set_seed(42); rng = np.random.default_rng(42)
    basis = MotifBasis.load(PROJECT_ROOT / "cache/motif_basis.npz"); S = args.section

    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    tf, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    tr, _, _ = create_datasets(train_files=tf, val_files=vf, test_files=[], audio_dir="data/",
                               max_sequence_length=msl, cache_dir='cache/samples')
    charts = [(m["chart_tensor"], m["groove_radar"].to_vector().astype(float))
              for m in tr.valid_samples if m.get("chart_tensor") is not None]
    rng.shuffle(charts); charts = charts[:args.charts]
    print(f"figure-label gate: {len(charts)} charts, section={S}\n")

    dist = Counter(); knob0_by_fig = {f: [] for f in FIGS + ["sparse"]}
    dominances = []; charts_classes = []
    for chart, radar in charts:
        T = chart.shape[0]; classes = set()
        for i in range(0, T - 1, S):
            sl = chart[i:i + S]
            if (sl != 0).any(1).sum() < args.min_onsets:
                continue
            fig = section_figure(sl); dist[fig] += 1; classes.add(fig)
            knob0_by_fig.setdefault(fig, []).append(float(basis.encode_chart(sl, radar)[0]))
            d = section_dominance(sl)
            if not np.isnan(d):
                dominances.append(d)
        if classes:
            charts_classes.append(len(classes - {"sparse"}))

    total = sum(dist.values())
    print("(1) FIGURE DISTRIBUTION across sections:")
    for f, n in dist.most_common():
        print(f"    {f:18s} {n:6d}  ({100*n/total:5.1f}%)   mean knob-0(sweep z) {np.mean(knob0_by_fig[f]):+.2f}")
    print(f"\n(2) ALIGNMENT: 'sweep/staircase' sections mean knob-0 = "
          f"{np.mean(knob0_by_fig['sweep/staircase']):+.2f} vs overall "
          f"{np.mean([v for vs in knob0_by_fig.values() for v in vs]):+.2f}  "
          f"(want sweep-class clearly ABOVE overall)")
    print(f"(3) WITHIN-CHART VARIETY: mean distinct non-sparse figure classes per chart = "
          f"{np.mean(charts_classes):.2f}  (want >1 so a per-section schedule is meaningful)")
    print(f"(4) DOMINANCE: section top-figure share = {np.mean(dominances):.2f} "
          f"(how single-figure a section is; ~1 = clean, ~0.3 = mush)")
    sweep_pct = 100 * dist['sweep/staircase'] / total
    print(f"\nVERDICT: sweep sections = {sweep_pct:.1f}% of all. BUILD if sweep is learnably present (say >3%), "
          f"aligned (2), and varied (3>1). If sweep ~absent at S={S}, it's a longer-scale figure -> the section "
          f"is the wrong unit and that's WHY the local vector can't pin it.")


if __name__ == "__main__":
    main()

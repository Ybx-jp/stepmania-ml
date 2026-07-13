#!/usr/bin/env python3
"""
H15 LOCAL-MOTIF pre-flight gates (NO training) — decide section granularity before building per-section
conditioning. See notes/h15_local_motif_plan.md. Cheapest-decisive-first (experiment-design Rules 5,6,7).

GATE A — does local even VARY? Split each real chart into fixed-size SECTIONS (frames), encode each section to
a (K,) motif vector (MotifBasis). Decompose section-vector variance into within-chart vs between-chart:
  within_fraction = within / (within + between)  -> how much motif variation is LOCAL (steerable per-section).
Control (Rule 11): permute a chart's ONSET ORDER then re-section (destroys temporal structure, keeps the global
figure mix). If ORDERED within-chart variance >> SHUFFLED, the local variation is real structure, not sampling
noise. If sections barely differ from the chart's global vector, sectional conditioning has nothing to steer.

GATE B — leakage. A section vector is computed FROM the onsets in its section, so it can LEAK the exact panel at
a frame inside the section (which would make training self-conditioning an artifact). Measure leak as the EXCESS
next-panel predictability of the SECTION vector over the GLOBAL chart vector (the legit style floor):
  excess = acc(section_vec -> panel_at_onset_in_section) - acc(global_vec -> same panel).
Small section -> higher excess -> leakier. Pick the SMALLEST section that VARIES (A) yet has LOW excess (B).

  python \
      experiments/generation_typed/diag_local_motif_gate.py [--sizes 16 32 64 128] [--charts 600]
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, sys, glob
from pathlib import Path
import numpy as np, yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.generation.motif_codebook import MotifBasis, onset_tokens
from src.generation.typed import panels_to_pattern

BASIS = PROJECT_ROOT / "cache/motif_basis.npz"


def chart_sections(chart, size, min_onsets):
    """Yield (T,4) frame-slices of `size` that contain >= min_onsets onsets."""
    T = chart.shape[0]
    for i in range(0, T - 1, size):
        sl = chart[i:i + size]
        if (sl != 0).any(1).sum() >= min_onsets:
            yield sl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 48, 64, 96, 128])
    ap.add_argument("--charts", type=int, default=600)
    ap.add_argument("--min_onsets", type=int, default=8, help="skip sparse sections (need enough onsets to encode)")
    args = ap.parse_args()
    set_seed(42)
    rng = np.random.default_rng(42)
    basis = MotifBasis.load(BASIS); K = basis.K

    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    tf, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    tr, _, _ = create_datasets(train_files=tf, val_files=vf, test_files=[], audio_dir="data/",
                               max_sequence_length=msl, cache_dir='cache/samples')
    charts = [(m["chart_tensor"], m["groove_radar"].to_vector().astype(float))
              for m in tr.valid_samples if m.get("chart_tensor") is not None]
    rng.shuffle(charts); charts = charts[:args.charts]
    print(f"gate: {len(charts)} charts, K={K} knobs, sizes={args.sizes}\n")

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        have_sk = True
    except ImportError:
        have_sk = False
        print("(sklearn missing -> Gate B skipped)\n")

    print(f"{'size':>5} | {'GATE A within-frac':>18} {'ord/shuf var':>14} | "
          f"{'GATE B sec-acc':>14} {'glob-acc':>9} {'excess(leak)':>12} {'maj':>6}")
    print("-" * 92)
    for S in args.sizes:
        # ---- Gate A: per-section vectors, ordered vs onset-shuffled ----
        within_ord, within_shuf, chart_means = [], [], []
        # ---- Gate B: (section_vec, global_vec, panel) pairs ----
        Xsec, Xglob, ypan = [], [], []
        for chart, radar in charts:
            secs_ord = [basis.encode_chart(sl, radar) for sl in chart_sections(chart, S, args.min_onsets)]
            if len(secs_ord) < 2:
                continue
            secs_ord = np.array(secs_ord)                          # (n_sec, K)
            within_ord.append(secs_ord.var(0))                     # per-knob within-chart var
            chart_means.append(secs_ord.mean(0))
            # shuffled control: permute onset ROWS of the chart, rebuild, re-section
            onset_idx = np.nonzero((chart != 0).any(1))[0]
            if len(onset_idx) >= args.min_onsets:
                ch_sh = chart.copy()
                perm = rng.permutation(onset_idx)
                ch_sh[onset_idx] = chart[perm]                     # scramble which onset sits where
                secs_sh = [basis.encode_chart(sl, radar) for sl in chart_sections(ch_sh, S, args.min_onsets)]
                if len(secs_sh) >= 2:
                    within_shuf.append(np.array(secs_sh).var(0))
            # Gate B pairs: global chart vector + each section's vector vs the section's dominant panel figure
            if have_sk:
                gv = basis.encode_chart(chart, radar)
                for sl, sv in zip(chart_sections(chart, S, args.min_onsets), secs_ord):
                    toks = onset_tokens(sl)
                    if len(toks) == 0:
                        continue
                    panel = int(np.bincount(toks, minlength=1).argmax())   # dominant which-panel figure in section
                    Xsec.append(sv); Xglob.append(gv); ypan.append(panel)

        within_ord = np.array(within_ord).mean(0)                  # (K,)
        between = np.array(chart_means).var(0)                     # (K,)
        within_frac = float((within_ord / (within_ord + between + 1e-9)).mean())
        var_ratio = float(within_ord.mean() / (np.array(within_shuf).mean() + 1e-9)) if within_shuf else float('nan')

        sec_acc = glob_acc = maj = excess = float('nan')
        if have_sk and len(set(ypan)) > 1 and len(ypan) > 50:
            Xsec, Xglob, ypan = np.array(Xsec), np.array(Xglob), np.array(ypan)
            idx = np.arange(len(ypan))
            tri, tei = train_test_split(idx, test_size=0.3, random_state=42, stratify=None)
            def acc(X):
                lr = LogisticRegression(max_iter=300, multi_class="auto")
                lr.fit(X[tri], ypan[tri]); return float((lr.predict(X[tei]) == ypan[tei]).mean())
            sec_acc, glob_acc = acc(Xsec), acc(Xglob)
            maj = float(np.bincount(ypan[tei]).max() / len(tei))
            excess = sec_acc - glob_acc

        print(f"{S:>5} | {within_frac:>18.3f} {var_ratio:>14.2f} | "
              f"{sec_acc:>14.3f} {glob_acc:>9.3f} {excess:>12.3f} {maj:>6.3f}")

    print("\nREAD: Gate A wants within-frac HIGH (local variation real) and ord/shuf var >1 (real structure, "
          "not sampling noise). Gate B wants EXCESS LOW (section vec not an answer key beyond the global floor). "
          "Pick the SMALLEST size that satisfies both.")


if __name__ == "__main__":
    main()

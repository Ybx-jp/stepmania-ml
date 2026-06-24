#!/usr/bin/env python3
"""
H15 Phase 1 — MOTIF CODEBOOK mining gate (no training). Builds the multi-scale, symmetry-aware note-pattern
motif vocabulary the Phase-0 gate licensed (notes/h15_motif_findings.md), and checks it's worth conditioning
on BEFORE any retrain.

Representation (user-chosen: absolute + symmetry-aware):
  TOKEN  = absolute which-panels pattern id 0..14 over ONSETS (typed.panels_to_pattern; empties dropped;
           step TYPE and RHYTHM excluded on purpose — holds=freeze knob, rhythm=chaos axis, both already
           conditionable; the codebook isolates the residual the radar can't see).
  MOTIF  = window of W consecutive tokens, W in {2,3,4,6} (multi-scale: short=gestures, long=figures).
  FOLD   = canonicalize under the L<->R mirror (panel map [3,1,2,0] — the STANDARD StepMania mirror; a chart
           and its mirror play identically with feet swapped). This merges LLLL<->RRRR but keeps UUUU distinct
           (L<->R fixes U,D) so the PART-3 charter signatures survive. (The Phase-0 gate's [3,2,1,0] was a
           180deg rotation — fixed here.)

The cheap gate (commit to training only if ALL pass):
  1. INTERPRETABLE  — do frequent canonical motifs read as nameable figures (jack / sweep / staircase /
     crossover / candle)? (printed per scale)
  2. LOW-RANK / MANIFOLD-LIKE — is the per-chart multi-scale motif distribution low-rank (few PCs explain
     most variance)? -> a small "motif-style" coordinate exists to put on the manifold.
  3. RESIDUAL (distinct from radar) — which motif-style PCs are ORTHOGONAL to the radar (|corr| small on all 5
     dims)? Those are the genuinely-new axes the radar can't already reach (the 39% residual).
  4. STABLE — do the top motif-style axes reproduce across a random split-half (loading-vector cosine)? Not
     an overfit artifact.

  /home/ybx/miniconda3/envs/stepmania-chart-gen/bin/python \
      experiments/generation_typed/diag_motif_codebook.py [--scales 2 3 4 6] [--topn 120]
"""
import warnings, os
warnings.filterwarnings('ignore'); os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'
import argparse, glob, sys
from collections import Counter
from pathlib import Path
import numpy as np, yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.data_splits import create_data_splits, create_datasets
from src.data.audio_features import AudioFeatureExtractor, AudioFeatureConfig
from src.generation.typed import panels_to_pattern, pattern_to_panels, NUM_PATTERNS
from src.generation.radar_manifold import DIMS


# ---- L<->R mirror over the 15 patterns: swap L(0)<->R(3), keep D(1),U(2) => panel perm [3,1,2,0] ----
def _lr_mirror(idx: int) -> int:
    return int(panels_to_pattern(pattern_to_panels(idx)[[3, 1, 2, 0]]))
_MIRROR = np.array([_lr_mirror(i) for i in range(NUM_PATTERNS)])


def pstr(idx: int) -> str:
    b = pattern_to_panels(idx)
    return "".join(c if b[i] else "-" for i, c in enumerate("LDUR"))


def motif_str(canon) -> str:
    return " ".join(pstr(p) for p in canon)


def canon(window):
    """Canonicalize a window tuple under the L<->R mirror group (lex-min of {w, mirror(w)})."""
    w = tuple(int(x) for x in window)
    m = tuple(int(_MIRROR[x]) for x in window)
    return min(w, m)


def onset_tokens(chart_tensor) -> np.ndarray:
    arr = np.asarray(chart_tensor)
    active = arr != 0
    onset = active.any(1)
    return panels_to_pattern(active[onset]) if onset.any() else np.empty(0, np.int64)


def name_figure(canon) -> str:
    """Heuristic label for interpretability readout."""
    panels = [tuple(np.nonzero(pattern_to_panels(p))[0]) for p in canon]
    sizes = [len(p) for p in panels]
    singles = [p[0] for p in panels if len(p) == 1]
    if any(s >= 2 for s in sizes):
        return "jump/hands figure"
    if len(set(singles)) == 1:
        return "jack (repeat)"
    diffs = np.diff(singles)
    if len(singles) >= 3 and (np.all(diffs > 0) or np.all(diffs < 0)):
        return "staircase/sweep"
    if len(singles) >= 3 and np.all(np.abs(diffs) == np.abs(diffs[0])) and len(set(singles)) == 2:
        return "trill/alternation"
    if any((a == 0 and b == 3) or (a == 3 and b == 0) for a, b in zip(singles, singles[1:])):
        return "L<->R cross/candle"
    return "mixed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=int, nargs="+", default=[2, 3, 4, 6])
    ap.add_argument("--topn", type=int, default=120, help="codebook size per scale")
    args = ap.parse_args()
    rng = np.random.default_rng(42)

    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    tf, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                   use_metric_phase=True, use_highres_onset=True))
    tr, va, _ = create_datasets(train_files=tf, val_files=vf, test_files=[], audio_dir="data/",
                                max_sequence_length=msl, feature_extractor=ext, cache_dir='cache/samples_v3')
    seqs, radar = [], []
    for ds in (tr, va):
        for m in ds.valid_samples:
            ct = m.get("chart_tensor")
            if ct is None:
                continue
            seqs.append(onset_tokens(ct))
            radar.append(m["groove_radar"].to_vector().astype(float))
    radar = np.array(radar); N = len(seqs)
    print(f"loaded {N} charts. TOKEN=absolute which-panels over onsets; FOLD=L<->R mirror; "
          f"scales={args.scales}\n")

    # ---- per-scale codebook (top canonical motifs) + per-chart counts ----
    per_chart_counts = {W: [Counter() for _ in range(N)] for W in args.scales}
    global_counts = {W: Counter() for W in args.scales}
    for i, s in enumerate(seqs):
        for W in args.scales:
            if len(s) < W:
                continue
            for j in range(len(s) - W + 1):
                c = canon(s[j:j + W])
                per_chart_counts[W][i][c] += 1
                global_counts[W][c] += 1

    print("=" * 80)
    print("GATE 1 — INTERPRETABILITY: most frequent canonical motifs per scale")
    print("=" * 80)
    codebooks = {}
    for W in args.scales:
        vocab = [c for c, _ in global_counts[W].most_common(args.topn)]
        codebooks[W] = vocab
        tot = sum(global_counts[W].values())
        print(f"\n-- W={W}  (|codebook|={len(vocab)}, {tot} windows) --")
        for c, n in global_counts[W].most_common(8):
            print(f"   {100*n/tot:5.1f}%  [{motif_str(c):<29}]  {name_figure(c)}")

    # ---- per-chart multi-scale distribution (each scale block normalized; equal scale weight) ----
    blocks = []
    col_meta = []   # (W, canon) per column
    for W in args.scales:
        vocab = codebooks[W]
        vidx = {c: k for k, c in enumerate(vocab)}
        B = np.zeros((N, len(vocab)))
        for i in range(N):
            for c, n in per_chart_counts[W][i].items():
                if c in vidx:
                    B[i, vidx[c]] += n
        rs = B.sum(1, keepdims=True); rs[rs == 0] = 1
        blocks.append(B / rs)
        col_meta += [(W, c) for c in vocab]
    X = np.column_stack(blocks)                              # (N, total_codebook)
    print(f"\nper-chart multi-scale motif profile: {X.shape}")

    # ---- GATE 2: low-rank / manifold-like (PCA scree) ----
    def pca(M, k=12):
        mu = M.mean(0); Mc = M - mu
        sd = Mc.std(0); sd[sd == 0] = 1
        Z = Mc / sd
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        evr = (S ** 2) / (S ** 2).sum()
        return mu, sd, Vt[:k], evr[:k], (Z @ Vt[:k].T)       # loadings (k,P), scores (N,k)
    _, _, load, evr, scores = pca(X, k=12)
    cum = np.cumsum(evr)
    print("\n" + "=" * 80)
    print("GATE 2 — LOW-RANK (PCA scree on standardized motif profile)")
    print("=" * 80)
    print("   PC : " + " ".join(f"{i+1:>5}" for i in range(10)))
    print("  evr%: " + " ".join(f"{100*evr[i]:5.1f}" for i in range(10)))
    print("  cum%: " + " ".join(f"{100*cum[i]:5.1f}" for i in range(10)))
    n80 = int(np.argmax(cum >= 0.80) + 1) if cum[-1] >= 0.80 else f">{len(cum)}"
    print(f"   => ~{n80} PCs reach 80% variance (low-rank if small).")

    # ---- GATE 3: which motif-style PCs are ORTHOGONAL to the radar (the residual) ----
    print("\n" + "=" * 80)
    print("GATE 3 — RESIDUAL: |corr| of each top PC with the 5 radar dims (low on ALL = radar can't reach it)")
    print("=" * 80)
    print(f"   {'PC':>3}  " + " ".join(f"{d[:4]:>6}" for d in DIMS) + "   maxcorr  top motifs (|loading|)")
    for pc in range(6):
        cors = [abs(np.corrcoef(scores[:, pc], radar[:, d])[0, 1]) for d in range(5)]
        topcols = np.argsort(np.abs(load[pc]))[::-1][:3]
        tops = " | ".join(f"{motif_str(col_meta[c][1])}" for c in topcols)
        flag = "  <== RADAR-ORTHOGONAL" if max(cors) < 0.25 else ""
        print(f"   {pc+1:>3}  " + " ".join(f"{c:6.2f}" for c in cors) + f"   {max(cors):5.2f}   {tops}{flag}")

    # ---- GATE 3b: how many dims does the RADAR-ORTHOGONAL residual need? (the Phase-1 conditioning width) ----
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    Rfit = Ridge(alpha=1.0).fit(StandardScaler().fit_transform(radar), X)
    resid = X - Rfit.predict(StandardScaler().fit_transform(radar))   # motif usage the radar can't predict
    _, _, _, evr_r, scores_r = pca(resid, k=20)
    cum_r = np.cumsum(evr_r)
    nr80 = int(np.argmax(cum_r >= 0.80) + 1) if cum_r[-1] >= 0.80 else f">{len(cum_r)}"
    nr90 = int(np.argmax(cum_r >= 0.90) + 1) if cum_r[-1] >= 0.90 else f">{len(cum_r)}"
    print("\n" + "=" * 80)
    print("GATE 3b — RADAR-ORTHOGONAL RESIDUAL rank (motif conditioning width: condition on THIS, not PC1)")
    print("=" * 80)
    print("  resid evr%: " + " ".join(f"{100*evr_r[i]:4.1f}" for i in range(12)))
    print("  resid cum%: " + " ".join(f"{100*cum_r[i]:4.1f}" for i in range(12)))
    print(f"   => residual needs ~{nr80} dims for 80% / ~{nr90} for 90% -> the motif-style embedding width.")

    # ---- GATE 4: split-half stability of the top axes ----
    perm = rng.permutation(N); h1, h2 = perm[:N // 2], perm[N // 2:]
    _, _, l1, _, _ = pca(X[h1], k=6)
    _, _, l2, _, _ = pca(X[h2], k=6)
    print("\n" + "=" * 80)
    print("GATE 4 — STABILITY: top-PC loading-vector alignment across a random split-half")
    print("=" * 80)
    for pc in range(6):
        align = max(abs(np.dot(l1[pc] / np.linalg.norm(l1[pc]), l2[q] / np.linalg.norm(l2[q])))
                    for q in range(6))
        print(f"   PC{pc+1}: best |cos| with a half-2 axis = {align:.2f}"
              f"{'  stable' if align > 0.7 else '  (drifts)'}")


if __name__ == "__main__":
    main()

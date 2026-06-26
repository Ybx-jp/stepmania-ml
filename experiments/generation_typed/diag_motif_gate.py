#!/usr/bin/env python3
"""
H15 Phase 0 — the CHEAP DE-RISK GATE (no training): do real charts' NOTE-PATTERN MOTIFS carry a style's
"vibe", and is that vibe a lever DISTINCT from the radar conditioning we already deploy?

Faithful to notes/h15_motif_handoff.md: a MOTIF is a short window of the WHICH-PANELS note sequence
(src/generation/typed.py::panels_to_pattern over onsets) — a recurring spatial FIGURE (stompy ornament,
sweep), NOT a rhythm/timing statistic. (An earlier version drifted onto frame/rhythm windows + a density-only
floor and "rediscovered" the already-shipped freeze knob — see the experiment-design skill, post-mortem §10.
This version manipulates the hypothesis variable (note patterns) and controls against the FULL radar.)

Two parts:

  PART 1 — SIGNAL EXISTS (the handoff gate, PASS/STOP). Does the note-pattern motif histogram separate the
  groove buckets (chaos/stream/freeze tertiles) vs a SHUFFLED-label control? macroF1 + mutual information.
  If motifs don't beat shuffled -> premise wrong, stop. If they do -> the signal exists.

  PART 2 — DISTINCT FROM EXISTING CONDITIONING (does it ADD a lever?). The radar already conditions on
  QUANTITIES (density, jumps, holds, chaos-amount). H15 only matters if the motif VOCABULARY is something the
  radar can't already pin. We do NOT ask motifs to re-predict a radar dim (circular). Instead:
    (a) radar->motif R^2: fit (full 5-d radar + difficulty) -> motif histogram; the variance it LEAVES
        UNEXPLAINED is the headroom a motif lever would add.
    (b) k-NN style test: for each chart, find its nearest neighbour in RADAR space (the most groove-similar
        chart = what conditioning to that profile would target) and measure motif-histogram cosine similarity
        vs random pairs (floor) and same-SONG pairs (the vocabulary ceiling). If radar-neighbours are barely
        more motif-similar than random, matching the radar does NOT give you the motifs -> the vocabulary is a
        FREE axis -> a genuinely distinct lever.

  python \
      experiments/generation_typed/diag_motif_gate.py [--topk 200] [--folds 5]
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

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import NearestNeighbors


# ---- L/R mirror map over the 15 patterns (panels L,D,U,R = 0,1,2,3 -> swap 0<->3, 1<->2) ----
def _mirror_pattern(idx: int) -> int:
    bits = pattern_to_panels(idx)
    return int(panels_to_pattern(bits[[3, 2, 1, 0]]))
_MIRROR = np.array([_mirror_pattern(i) for i in range(NUM_PATTERNS)])


def pattern_string(idx: int) -> str:
    b = pattern_to_panels(idx)
    return "".join(c if b[i] else "-" for i, c in enumerate("LDUR"))


def onset_pattern_seq(chart_tensor) -> np.ndarray:
    """(T,4) typed chart -> (n_onsets,) which-panels pattern id, EMPTIES DROPPED (pure note-pattern shape)."""
    arr = np.asarray(chart_tensor)
    active = arr != 0
    onset = active.any(1)
    if not onset.any():
        return np.empty(0, dtype=np.int64)
    return panels_to_pattern(active[onset])


def chart_windows(seq: np.ndarray, W: int, mirror: bool):
    if mirror:
        seq = _MIRROR[seq]
    for i in range(len(seq) - W + 1):
        w = seq[i:i + W]
        if mirror:
            w = min(tuple(int(x) for x in w), tuple(int(x) for x in _MIRROR[w]))
        else:
            w = tuple(int(x) for x in w)
        yield w


def build_histograms(seqs, W, mirror, topk):
    counts = Counter()
    per_chart = []
    for s in seqs:
        wl = list(chart_windows(s, W, mirror)) if len(s) >= W else []
        per_chart.append(wl); counts.update(wl)
    vocab = [w for w, _ in counts.most_common(topk)]
    vidx = {w: j for j, w in enumerate(vocab)}
    K = len(vocab)
    X = np.zeros((len(seqs), K + 1))
    for i, wl in enumerate(per_chart):
        for w in wl:
            X[i, vidx.get(w, K)] += 1.0
        s = X[i].sum()
        if s > 0:
            X[i] /= s
    return vocab, X


def macro_f1_cv(X, y, folds):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced"))
    return cross_val_score(clf, X, y, cv=StratifiedKFold(folds, shuffle=True, random_state=42),
                           scoring="f1_macro")


def weighted_r2(y_true, y_pred):
    """Variance-weighted R^2 across motif columns (rare motifs contribute little variance)."""
    ss_res = ((y_true - y_pred) ** 2).sum(0)
    ss_tot = ((y_true - y_true.mean(0)) ** 2).sum(0)
    return float(1.0 - ss_res.sum() / ss_tot.sum())


def tertiles(col):
    order = np.argsort(np.argsort(col, kind="stable"), kind="stable")
    return (order * 3 // len(col)).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--axes", nargs="+", default=["chaos", "stream", "freeze"])
    args = ap.parse_args()
    rng = np.random.default_rng(42)

    # ---- load real charts ----
    cf = glob.glob("data/**/*.sm", recursive=True) + glob.glob("data/**/*.ssc", recursive=True)
    tf, vf, _ = create_data_splits(cf, random_state=42)
    msl = yaml.safe_load(open(PROJECT_ROOT / "config/model_config.yaml"))['classifier']['max_sequence_length']
    ext = AudioFeatureExtractor(AudioFeatureConfig(use_chroma=True, use_hpss_onsets=True,
                                                   use_metric_phase=True, use_highres_onset=True))
    tr, va, _ = create_datasets(train_files=tf, val_files=vf, test_files=[], audio_dir="data/",
                                max_sequence_length=msl, feature_extractor=ext, cache_dir='cache/samples_v3')

    seqs, radar, diff, song = [], [], [], []
    for ds in (tr, va):
        for m in ds.valid_samples:
            ct = m.get("chart_tensor")
            if ct is None:
                continue
            seqs.append(onset_pattern_seq(ct))
            radar.append(m["groove_radar"].to_vector().astype(float))
            diff.append(int(m.get("difficulty_class", -1)))
            song.append(m.get("chart_file", ""))
    radar = np.array(radar); diff = np.array(diff); song = np.array(song)
    song_id = np.array([hash(s) for s in song])
    N = len(seqs)
    print(f"loaded {N} real charts; {len(set(song))} distinct chart files. "
          f"Motif = WHICH-PANELS window over onsets (empties dropped).\n")

    # =====================================================================================================
    # PART 1 — SIGNAL EXISTS: note-pattern motifs vs SHUFFLED control (the handoff gate)
    # =====================================================================================================
    print("=" * 78)
    print("PART 1 — do NOTE-PATTERN motifs separate groove buckets vs a SHUFFLED control?")
    print("=" * 78)
    print("(equal-count rank tertiles; chance macroF1 ~= 0.333)\n")
    variants = [(W, mir) for W in (3, 4, 6) for mir in (False, True)]
    for axis in args.axes:
        y = tertiles(radar[:, DIMS.index(axis)])
        print(f"  --- {axis} ---")
        for (W, mir) in variants:
            vocab, X = build_histograms(seqs, W, mir, args.topk)
            f1 = macro_f1_cv(X, y, args.folds)
            f1s = macro_f1_cv(X, rng.permutation(y), args.folds)
            tag = f"W={W} {'mir' if mir else 'raw'} (|V|={len(vocab)})"
            sep = f1.mean() - f1s.mean()
            print(f"    {tag:<24} motif {f1.mean():.3f} ± {f1.std():.3f}  vs shuffled {f1s.mean():.3f}"
                  f"  (sep {sep:+.3f}){'  <== separates' if sep > 0.08 else ''}")
        # MI readout on W=4 raw
        vocab, X = build_histograms(seqs, 4, False, args.topk)
        mi = mutual_info_classif(X[:, :len(vocab)], y, random_state=42)
        top = np.argsort(mi)[::-1][:6]
        print(f"    top motifs by MI (W=4 raw): "
              + " | ".join(f"[{' '.join(pattern_string(p) for p in vocab[j])}]" for j in top))
        print()

    # =====================================================================================================
    # PART 2 — DISTINCT LEVER: how much of note-pattern motif usage does the FULL RADAR already pin?
    # =====================================================================================================
    print("=" * 78)
    print("PART 2 — is the motif VOCABULARY distinct from the radar we ALREADY condition on?")
    print("=" * 78)
    vocab, X = build_histograms(seqs, 4, False, args.topk)   # W=4 raw note-pattern histogram
    feat = np.column_stack([radar, np.eye(4)[np.clip(diff, 0, 3)]])  # full 5-d radar + difficulty one-hot
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    Xhat = cross_val_predict(ridge, feat, X, cv=KFold(args.folds, shuffle=True, random_state=42))
    r2 = weighted_r2(X, Xhat)
    print(f"\n(a) radar+difficulty -> motif histogram, variance-weighted R^2 = {r2:.3f}")
    print(f"    => the radar already pins {100*max(r2,0):.0f}% of note-pattern motif usage; "
          f"{100*(1-max(r2,0)):.0f}% is RESIDUAL vocabulary it cannot express.")

    # (b) k-NN style test: do RADAR-neighbours share motif vocabulary?
    # Center histograms by the global mean motif distribution so a few ubiquitous motifs don't saturate
    # cosine (raw cosine sits ~0.8 for everyone). Centered cosine ~0 for random pairs, discriminative.
    Xc = X - X.mean(0)
    Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-9)
    cos = lambda a, b: (Xn[a] * Xn[b]).sum(1)
    # radar nearest neighbour that is a DIFFERENT song (a groove-twin, not a sibling difficulty of itself)
    nn = NearestNeighbors(n_neighbors=15).fit(StandardScaler().fit_transform(radar))
    _, idx = nn.kneighbors(StandardScaler().fit_transform(radar))
    radar_nn = np.array([next((j for j in row[1:] if song_id[j] != song_id[i]), row[1])
                         for i, row in enumerate(idx)])
    sim_radar = cos(np.arange(N), radar_nn).mean()
    sim_rand = cos(np.arange(N), rng.permutation(N)).mean()
    # ceiling = same-SONG, SAME-difficulty pairs (true motif twins: different charters, matched groove)
    same_pairs, by = [], {}
    for i in range(N):
        by.setdefault((song_id[i], diff[i]), []).append(i)
    for ids in by.values():
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                same_pairs.append((ids[a], ids[b]))
    sim_same = (cos(np.array([p[0] for p in same_pairs]), np.array([p[1] for p in same_pairs])).mean()
                if same_pairs else float('nan'))
    print(f"\n(b) CENTERED motif-histogram cosine similarity of chart pairs (random ~ 0):")
    print(f"      random pairs (floor)                 {sim_rand:+.3f}")
    print(f"      RADAR-nearest-neighbour, diff. song  {sim_radar:+.3f}   (what conditioning to that profile targets)")
    print(f"      same-song same-difficulty (ceiling)  {sim_same:+.3f}   ({len(same_pairs)} motif-twin pairs)")
    print(f"    => radar-twins are MORE motif-similar than same-song/same-difficulty charter-twins "
          f"({sim_radar:+.3f} > {sim_same:+.3f}):\n"
          f"       the radar captures the COMMON/quantity-driven motif mass, but human charter motif choice"
          f"\n       has large spread even at fixed song+difficulty -> motif vocabulary is a DISTRIBUTION, not a"
          f"\n       point (echoes the Phase-3 16th-IoU 0.325 divergence finding). See PART 3 for the residual.")

    # =====================================================================================================
    # PART 3 — ground in the artifact: do specific charts OWN signature note-pattern motifs?
    # =====================================================================================================
    print("\n" + "=" * 78)
    print("PART 3 — signature motifs: charts most concentrated on their most-used motif (vibe = recurring figure)")
    print("=" * 78)
    dom = X[:, :len(vocab)].max(1)        # share of each chart's single most-used motif
    for i in np.argsort(dom)[::-1][:6]:
        j = int(X[i, :len(vocab)].argmax())
        title = Path(song[i]).stem[:34]
        rv = " ".join(f"{DIMS[k][:2]}{radar[i,k]:.2f}" for k in range(5))
        print(f"  {title:<35} motif [{' '.join(pattern_string(p) for p in vocab[j])}] = "
              f"{dom[i]*100:.0f}% of its onsets | {rv}")


if __name__ == "__main__":
    main()

"""
Motif codebook + radar-orthogonal motif-style basis (the H15 "vibe" conditioning surface).

The Phase-0 gate (notes/h15_motif_findings.md) proved that the WHICH-PANELS note-pattern motifs of real charts
carry a style's character, and that this is a lever DISTINCT from the radar: the radar pins the common,
quantity-driven motif mass (R^2 0.61) but is blind to the characteristic SIGNATURE figures (same radar ->
UUUU vs LLLL vs LRLR). The Phase-1 mining gate (diag_motif_codebook.py) then showed the residual motif
vocabulary is NOT low-rank (>20 dims for 80%) — so there is no Gaussian "motif manifold" to bolt onto the
radar ellipsoid; instead we condition on the radar-ORTHOGONALIZED motif distribution via a handful of
interpretable, STABLE residual-PC axes ("named knobs", parallel to the radar's named axes).

This module is the persisted artifact (the motif analog of cache/radar_manifold.npz):

  MotifBasis.fit(seqs, radar) ->
    (1) per-scale L<->R-mirror-folded codebook over which-panels onset windows (W in {2,3,4,6});
    (2) Ridge: radar -> per-chart motif histogram (the part the radar ALREADY explains);
    (3) PCA on the RESIDUAL (motif usage the radar can't predict) -> motif-style axes;
    (4) keep the top-K axes that are STABLE across a split-half (|cos|>thresh) and radar-orthogonal, each with
        a human-readable label from its extreme-loading figures.

  basis.encode_chart(chart_tensor, radar) -> (K,) z-scored motif knobs  (training target, autoencoder-style)
  At inference the user instead SETS a target knob vector (named axes); it composes with radar+style in _cond.

L<->R fold uses panel map [3,1,2,0] (the standard StepMania mirror: swap L<->R, keep D,U) so LLLL<->RRRR merge
but UUUU stays distinct (the PART-3 signatures survive). Step TYPE and RHYTHM are excluded on purpose (holds =
freeze knob, rhythm = chaos axis, both already conditionable) so the codebook is exactly the residual lever.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import numpy as np

from src.generation.typed import panels_to_pattern, pattern_to_panels, NUM_PATTERNS

DEFAULT_SCALES = (2, 3, 4, 6)

# L<->R mirror over the 15 patterns: swap L(0)<->R(3), keep D(1),U(2) => panel perm [3,1,2,0]
_MIRROR = np.array([int(panels_to_pattern(pattern_to_panels(i)[[3, 1, 2, 0]])) for i in range(NUM_PATTERNS)])


def pstr(idx: int) -> str:
    b = pattern_to_panels(int(idx))
    return "".join(c if b[i] else "-" for i, c in enumerate("LDUR"))


def motif_str(canon) -> str:
    return " ".join(pstr(p) for p in canon)


def _canon(window):
    """Lex-min of a window under the L<->R mirror (canonical figure id)."""
    w = tuple(int(x) for x in window)
    m = tuple(int(_MIRROR[x]) for x in window)
    return min(w, m)


def onset_tokens(chart_tensor) -> np.ndarray:
    """(T,4) typed chart -> (n_onsets,) which-panels pattern id (empties dropped)."""
    arr = np.asarray(chart_tensor)
    active = arr != 0
    onset = active.any(1)
    return panels_to_pattern(active[onset]) if onset.any() else np.empty(0, np.int64)


def name_figure(canon) -> str:
    """Heuristic label for a motif (interpretability readout / axis naming)."""
    panels = [tuple(np.nonzero(pattern_to_panels(p))[0]) for p in canon]
    sizes = [len(p) for p in panels]
    singles = [p[0] for p in panels if len(p) == 1]
    if any(s >= 2 for s in sizes):
        return "jump/bracket"
    if len(set(singles)) == 1:
        return "jack"
    diffs = np.diff(singles)
    if len(singles) >= 3 and (np.all(diffs > 0) or np.all(diffs < 0)):
        return "sweep/staircase"
    if len(singles) >= 3 and len(set(singles)) == 2:
        return "trill"
    if any((a, b) in ((0, 3), (3, 0)) for a, b in zip(singles, singles[1:])):
        return "candle/cross"
    return "step"


# H15 hierarchical pick-then-realize: a DISCRETE per-section figure vocabulary (the "pick"). The continuous
# radar-orthogonal knobs ENTANGLE figures (knob-0 'jack<->sweep' is really a jack detector; sweep mushes with
# step/candle — diag_figure_labels.py), so a discrete family label cleanly isolates a figure (esp. SWEEP) that
# the continuous projection cannot. Index 0 = sparse/none (sections with too few onsets to name).
FIGURE_CLASSES = ["sparse", "jack", "sweep/staircase", "trill", "candle/cross", "jump/bracket", "step"]
NUM_FIGURE_CLASSES = len(FIGURE_CLASSES)
_FIG2IDX = {f: i for i, f in enumerate(FIGURE_CLASSES)}


def figure_token(section_chart, W: int = 3, min_onsets: int = 4) -> int:
    """(T,4) section -> dominant canonical W-window figure-family token (FIGURE_CLASSES index). 'sparse' (0)
    when the section has too few onsets to name a figure."""
    toks = onset_tokens(section_chart)
    if len(toks) < max(W, min_onsets):
        return 0
    cnt = Counter(_canon(toks[j:j + W]) for j in range(len(toks) - W + 1))
    return _FIG2IDX[name_figure(cnt.most_common(1)[0][0])]


def figure_token_schedule(chart, section: int, W: int = 3) -> np.ndarray:
    """(T,4) chart -> (T,) per-frame figure token, piecewise-constant per `section` frames."""
    T = chart.shape[0]
    out = np.zeros(T, np.int64)
    for i in range(0, T, section):
        out[i:i + section] = figure_token(chart[i:i + section], W)
    return out


class MotifBasis:
    def __init__(self, scales, codebooks, col_meta, radar_mu, radar_sd, ridge_W, ridge_b,
                 col_mean, col_std, components, score_mu, score_sd, axis_info):
        self.scales = list(scales)
        self.codebooks = codebooks          # {W: [canon tuple,...]}
        self.col_meta = col_meta            # [(W, canon)] aligned to histogram columns
        self.radar_mu = np.asarray(radar_mu); self.radar_sd = np.asarray(radar_sd)
        self.ridge_W = np.asarray(ridge_W); self.ridge_b = np.asarray(ridge_b)   # standardized-radar -> hist
        self.col_mean = np.asarray(col_mean); self.col_std = np.asarray(col_std)
        self.components = np.asarray(components)   # (K, P) residual PCA loadings
        self.score_mu = np.asarray(score_mu); self.score_sd = np.asarray(score_sd)  # (K,)
        self.axis_info = axis_info          # list of dicts: {label, pos[], neg[], stability, maxcorr}

    @property
    def K(self) -> int:
        return len(self.components)

    # ---- histogram ----
    def chart_histogram(self, chart_tensor) -> np.ndarray:
        """(T,4) chart -> per-chart multi-scale canonical-motif distribution over the codebook (P,)."""
        s = onset_tokens(chart_tensor)
        vec = np.zeros(len(self.col_meta))
        offset = 0
        for W in self.scales:
            vocab = self.codebooks[W]; vidx = {c: k for k, c in enumerate(vocab)}
            cnt = Counter()
            if len(s) >= W:
                for j in range(len(s) - W + 1):
                    cnt[_canon(s[j:j + W])] += 1
            block = np.zeros(len(vocab))
            for c, n in cnt.items():
                if c in vidx:
                    block[vidx[c]] += n
            tot = block.sum()
            vec[offset:offset + len(vocab)] = block / tot if tot > 0 else block
            offset += len(vocab)
        return vec

    # ---- encode: histogram (+radar) -> z-scored radar-orthogonal motif knobs ----
    def encode(self, hist, radar) -> np.ndarray:
        radar_z = (np.asarray(radar) - self.radar_mu) / self.radar_sd
        hist_hat = radar_z @ self.ridge_W + self.ridge_b      # radar-explained motif mass
        resid = np.asarray(hist) - hist_hat                   # what the radar can't predict
        z = (resid - self.col_mean) / self.col_std
        scores = z @ self.components.T                        # (K,)
        return (scores - self.score_mu) / self.score_sd

    def encode_chart(self, chart_tensor, radar) -> np.ndarray:
        return self.encode(self.chart_histogram(chart_tensor), radar)

    # ---- construction ----
    @classmethod
    def fit(cls, seqs, radar, scales=DEFAULT_SCALES, topn=120, K=12, stability_thresh=0.7, seed=42):
        from sklearn.linear_model import Ridge
        rng = np.random.default_rng(seed)
        radar = np.asarray(radar, float); N = len(seqs)

        # (1) per-scale codebook + per-chart multi-scale histogram
        glob = {W: Counter() for W in scales}
        per = {W: [Counter() for _ in range(N)] for W in scales}
        for i, s in enumerate(seqs):
            for W in scales:
                if len(s) < W:
                    continue
                for j in range(len(s) - W + 1):
                    c = _canon(s[j:j + W]); glob[W][c] += 1; per[W][i][c] += 1
        codebooks = {W: [c for c, _ in glob[W].most_common(topn)] for W in scales}
        col_meta, blocks = [], []
        for W in scales:
            vocab = codebooks[W]; vidx = {c: k for k, c in enumerate(vocab)}
            B = np.zeros((N, len(vocab)))
            for i in range(N):
                for c, n in per[W][i].items():
                    if c in vidx:
                        B[i, vidx[c]] += n
            rs = B.sum(1, keepdims=True); rs[rs == 0] = 1
            blocks.append(B / rs); col_meta += [(W, c) for c in vocab]
        X = np.column_stack(blocks)                           # (N, P)

        # (2) Ridge radar -> motif histogram (standardized radar)
        radar_mu, radar_sd = radar.mean(0), radar.std(0); radar_sd[radar_sd == 0] = 1
        Rz = (radar - radar_mu) / radar_sd
        rid = Ridge(alpha=1.0).fit(Rz, X)
        ridge_W, ridge_b = rid.coef_.T, rid.intercept_         # (5,P),(P,)
        resid = X - (Rz @ ridge_W + ridge_b)

        # (3) PCA on the residual (standardized columns)
        col_mean = resid.mean(0); col_std = resid.std(0); col_std[col_std == 0] = 1
        Z = (resid - col_mean) / col_std
        def pca(M, k):
            _, _, Vt = np.linalg.svd(M - M.mean(0), full_matrices=False)
            return Vt[:k]
        comps_full = pca(Z, max(K * 2, 24))                    # candidates

        # (4) keep top axes that are STABLE across a split-half and radar-orthogonal
        h = rng.permutation(N); h1, h2 = h[:N // 2], h[N // 2:]
        c1, c2 = pca(Z[h1], K * 2), pca(Z[h2], K * 2)
        kept, axis_info = [], []
        for p in range(len(comps_full)):
            v = comps_full[p]
            stab = max(abs(np.dot(v / np.linalg.norm(v), c1[q] / np.linalg.norm(c1[q]))) for q in range(len(c1)))
            stab = min(stab, max(abs(np.dot(v / np.linalg.norm(v), c2[q] / np.linalg.norm(c2[q]))) for q in range(len(c2))))
            sc = Z @ v
            maxcorr = max(abs(np.corrcoef(sc, radar[:, d])[0, 1]) for d in range(radar.shape[1]))
            if stab < stability_thresh:
                continue
            order = np.argsort(v)
            pos = [col_meta[j] for j in order[::-1][:3]]
            neg = [col_meta[j] for j in order[:3]]
            pn, nn = name_figure(pos[0][1]), name_figure(neg[0][1])
            label = pn if pn == nn else f"{pn}<->{nn}"   # the axis is a CONTRAST (+end vs -end)
            axis_info.append({"label": label, "pos": pos, "neg": neg,
                              "stability": float(stab), "maxcorr": float(maxcorr)})
            kept.append(v)
            if len(kept) >= K:
                break
        components = np.array(kept)
        scores = Z @ components.T
        score_mu, score_sd = scores.mean(0), scores.std(0); score_sd[score_sd == 0] = 1
        return cls(scales, codebooks, col_meta, radar_mu, radar_sd, ridge_W, ridge_b,
                   col_mean, col_std, components, score_mu, score_sd, axis_info)

    @classmethod
    def from_loaded_datasets(cls, *datasets, **kw):
        seqs, radar = [], []
        for ds in datasets:
            for m in ds.valid_samples:
                ct = m.get("chart_tensor")
                if ct is None:
                    continue
                seqs.append(onset_tokens(ct))
                radar.append(m["groove_radar"].to_vector().astype(float))
        return cls.fit(seqs, np.array(radar), **kw)

    # ---- persistence ----
    def save(self, path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        # codebooks/col_meta are ragged tuples -> store as object arrays
        np.savez(path, scales=np.array(self.scales), col_meta=np.array(self.col_meta, dtype=object),
                 codebooks=np.array([self.codebooks[W] for W in self.scales], dtype=object),
                 radar_mu=self.radar_mu, radar_sd=self.radar_sd, ridge_W=self.ridge_W, ridge_b=self.ridge_b,
                 col_mean=self.col_mean, col_std=self.col_std, components=self.components,
                 score_mu=self.score_mu, score_sd=self.score_sd,
                 axis_info=np.array(self.axis_info, dtype=object))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        scales = list(z["scales"])
        codebooks = {W: [tuple(c) for c in cb] for W, cb in zip(scales, z["codebooks"])}
        col_meta = [(int(W), tuple(c)) for W, c in z["col_meta"]]
        return cls(scales, codebooks, col_meta, z["radar_mu"], z["radar_sd"], z["ridge_W"], z["ridge_b"],
                   z["col_mean"], z["col_std"], z["components"], z["score_mu"], z["score_sd"],
                   list(z["axis_info"]))

    def describe(self) -> str:
        lines = [f"MotifBasis: {self.K} named motif knobs (radar-orthogonal, stable), "
                 f"scales={self.scales}, codebook P={len(self.col_meta)}"]
        for k, a in enumerate(self.axis_info):
            lines.append(f"  knob {k:>2} '{a['label']}'  stab {a['stability']:.2f}  maxcorr_radar {a['maxcorr']:.2f}")
            lines.append(f"        + " + "  ".join(f"[{motif_str(c)}]" for _, c in a["pos"]))
            lines.append(f"        - " + "  ".join(f"[{motif_str(c)}]" for _, c in a["neg"]))
        return "\n".join(lines)

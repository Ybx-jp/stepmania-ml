"""
Manifold-aware groove-radar conditioning surface.

The 5 radar dims (stream, voltage, air, freeze, chaos) are NOT independent knobs: stream/voltage/air/chaos
form one correlated "intensity" cluster (r 0.71-0.92) and freeze is a near-orthogonal hold axis (r ~0.3).
So a radar POINT with one dim cranked while the others sit at the mean is OFF-MANIFOLD — the failure mode
behind the chaos-OOD bug. This surface lets a user steer a few NAMED axes (the "traversal dimensions") and
returns a coherent, ON-manifold 5-vector that feeds the EXISTING radar_proj + CFG conditioning (no retrain):

  spec (partial/loose) --(b) conditional-fill unspecified dims--> --(c) project onto the covariance
  ellipsoid--> coherent 5-vec.

See notes/radar_manifold_findings.md.

### Adjusting knob granularity/refinement (the one place to edit)
`LEVELS` maps a level NAME -> a quantile of the (per-difficulty) real distribution. Refine the knobs by
editing LEVELS (add 'vlow'/'modhigh'/... or change a quantile) — because the manifold keeps the raw
per-difficulty radar vectors, level changes take effect IMMEDIATELY with NO refit. A spec value may also be
given directly as a quantile (`q0.9`) or a raw radar value (`0.8`), so you can move from coarse named levels
-> fine quantiles -> continuous values without code changes. Refit (`RadarManifold.from_vectors(...).save`)
only when the underlying DATA changes.
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np

DIMS = ["stream", "voltage", "air", "freeze", "chaos"]   # order matches GrooveRadar.to_vector()
_IDX = {d: i for i, d in enumerate(DIMS)}

# --- KNOB GRANULARITY (edit here to refine) -------------------------------------------------------------
# level name -> quantile of the real per-difficulty distribution of that dim. Add/rename freely.
LEVELS: dict[str, float] = {
    "min": 0.02, "vlow": 0.10, "low": 0.15, "lowmod": 0.30,
    "mod": 0.50, "modhigh": 0.70, "high": 0.85, "vhigh": 0.95, "max": 0.98,
}
DEFAULT_PROJECT_QUANTILE = 0.90   # project off-manifold targets back to <= this pct of real Mahalanobis dist
# --------------------------------------------------------------------------------------------------------


class RadarManifold:
    """Fits a per-difficulty Gaussian (mean + covariance) over real groove-radar vectors and builds coherent,
    on-manifold targets from partial user specs. Keeps the raw vectors so quantiles/nearest recompute on the
    fly (LEVELS edits need no refit)."""

    def __init__(self, vectors: np.ndarray, difficulties: np.ndarray | None = None,
                 densities: np.ndarray | None = None,
                 levels: dict[str, float] | None = None,
                 project_quantile: float = DEFAULT_PROJECT_QUANTILE):
        self.vectors = np.asarray(vectors, dtype=float)
        self.difficulties = (np.asarray(difficulties) if difficulties is not None
                             else np.full(len(self.vectors), -1))
        # per-chart note density (frac of frames with a note) -> lets us derive a SOURCE-CHART-FREE density
        # target for a new song from (difficulty + style) instead of pinning to an existing chart.
        self.densities = np.asarray(densities, dtype=float) if densities is not None else None
        self.levels = dict(levels or LEVELS)
        self.project_quantile = project_quantile

    # ---- construction / persistence ----
    @classmethod
    def from_loaded_datasets(cls, *datasets, **kw) -> "RadarManifold":
        """Gather radar vectors + difficulty_class + note density from loaded Dataset objects (.valid_samples).
        Density = frac of frames with a note, from the cached chart_tensor (matches the exporter's real_density)."""
        V, D, DENS = [], [], []
        for ds in datasets:
            for m in ds.valid_samples:
                V.append(m["groove_radar"].to_vector().astype(float))
                D.append(int(m.get("difficulty_class", -1)))
                ct = m.get("chart_tensor")
                DENS.append(float((np.asarray(ct) != 0).any(1).mean()) if ct is not None else np.nan)
        DENS = np.array(DENS)
        return cls(np.array(V), np.array(D), densities=(None if np.isnan(DENS).all() else DENS), **kw)

    def save(self, path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        kw = dict(vectors=self.vectors, difficulties=self.difficulties, project_quantile=self.project_quantile)
        if self.densities is not None:
            kw["densities"] = self.densities
        np.savez(path, **kw)

    @classmethod
    def load(cls, path, **kw) -> "RadarManifold":
        z = np.load(path)
        return cls(z["vectors"], z["difficulties"],
                   densities=z["densities"] if "densities" in z else None,
                   project_quantile=float(z["project_quantile"]) if "project_quantile" in z
                   else DEFAULT_PROJECT_QUANTILE, **kw)

    # ---- per-difficulty fit (cached) ----
    def _bucket_mask(self, difficulty: int | None) -> np.ndarray:
        if difficulty is None or difficulty < 0:
            return np.ones(len(self.vectors), bool)
        sel = self.difficulties == difficulty
        return sel if sel.sum() >= 30 else np.ones(len(self.vectors), bool)   # fall back to all if too few

    def _bucket(self, difficulty: int | None) -> np.ndarray:
        return self.vectors[self._bucket_mask(difficulty)]

    def _fit(self, difficulty: int | None):
        Vb = self._bucket(difficulty)
        mu = Vb.mean(0)
        Sig = np.cov(Vb.T) + 1e-6 * np.eye(len(DIMS))
        Sinv = np.linalg.inv(Sig)
        real_d = np.sqrt(np.maximum(((Vb - mu) @ Sinv * (Vb - mu)).sum(1), 0.0))
        return Vb, mu, Sig, Sinv, real_d

    # ---- spec parsing (granularity-flexible) ----
    def resolve_value(self, dim: str, token, difficulty: int | None) -> float:
        """token -> raw radar value [0,1]. Accepts a LEVELS name, 'q<float>' (quantile), or a bare float."""
        Vb = self._bucket(difficulty)
        col = Vb[:, _IDX[dim]]
        if isinstance(token, (int, float)):
            return float(token)
        t = str(token).strip().lower()
        if t in self.levels:
            return float(np.quantile(col, self.levels[t]))
        m = re.fullmatch(r"q(0?\.\d+|\d?\.?\d+)", t)
        if m:
            return float(np.quantile(col, float(m.group(1))))
        try:
            return float(t)
        except ValueError:
            raise ValueError(f"unparseable radar level {token!r} for {dim} "
                             f"(use one of {list(self.levels)}, 'q0.9', or a 0-1 value)")

    @staticmethod
    def parse_spec(spec) -> dict[str, str]:
        """'stream=high,chaos=low' or {'stream':'high'} -> {dim: token} (validates dim names)."""
        if isinstance(spec, dict):
            d = {k: v for k, v in spec.items()}
        else:
            d = {}
            for part in str(spec).split(","):
                part = part.strip()
                if not part:
                    continue
                k, _, v = part.partition("=")
                d[k.strip()] = v.strip()
        bad = [k for k in d if k not in _IDX]
        if bad:
            raise ValueError(f"unknown radar dim(s) {bad}; valid: {DIMS}")
        return d

    # ---- the surface ----
    def build_target(self, spec, difficulty: int | None = None) -> tuple[np.ndarray, dict]:
        """Partial spec -> coherent on-manifold 5-vec. (b) conditional-fill free dims, (c) project to ellipsoid.
        Returns (vec5, info) where info has the filled vec, Mahalanobis distance + percentile, and projection."""
        spec = self.parse_spec(spec)
        Vb, mu, Sig, Sinv, real_d = self._fit(difficulty)
        fixed_idx = [_IDX[k] for k in spec]
        fixed_val = np.array([self.resolve_value(k, spec[k], difficulty) for k in spec])
        # (b) conditional fill E[free | fixed]
        x = mu.copy()
        if fixed_idx:
            x[fixed_idx] = fixed_val
            free = [i for i in range(len(DIMS)) if i not in fixed_idx]
            if free:
                Sfc = Sig[np.ix_(free, fixed_idx)]; Scc = Sig[np.ix_(fixed_idx, fixed_idx)]
                x[free] = mu[free] + Sfc @ np.linalg.solve(Scc, fixed_val - mu[fixed_idx])
        filled = x.copy()
        # (c) project onto the covariance ellipsoid (shrink along the Mahalanobis ray)
        d_raw = self._mahal(filled, mu, Sinv)
        max_d = float(np.quantile(real_d, self.project_quantile))
        if d_raw > max_d:
            x = mu + (max_d / d_raw) * (filled - mu)
        x = np.clip(x, 0.0, 1.0)
        info = {
            "spec": spec, "fixed": list(spec.keys()),
            "filled": filled, "target": x,
            "mahalanobis": d_raw, "max_mahalanobis": max_d, "projected": d_raw > max_d,
            "typicality_pct": float(100 * (real_d < d_raw).mean()),  # % of real LESS off-manifold than this
            "density": self.target_density(x, difficulty),           # source-chart-free density target (or None)
        }
        return x.astype(np.float32), info

    def target_density(self, radar_vec, difficulty: int | None = None) -> float | None:
        """SOURCE-CHART-FREE density: E[density | radar=radar_vec] under the real joint Gaussian for this
        difficulty bucket. Returns notes/frame, clipped to the bucket's observed range. None if no density data.
        This is how a brand-new song (no existing chart) gets its density target from difficulty + style."""
        if self.densities is None:
            return None
        mask = self._bucket_mask(difficulty)
        Vb = self.vectors[mask]; Db = self.densities[mask]
        ok = ~np.isnan(Db)
        Vb, Db = Vb[ok], Db[ok]
        if len(Db) < 30:
            return float(np.nanmean(self.densities))
        M = np.column_stack([Vb, Db])
        C = np.cov(M.T) + 1e-9 * np.eye(len(DIMS) + 1)
        mu = M.mean(0)
        dens = mu[-1] + C[-1, :-1] @ np.linalg.solve(C[:-1, :-1], np.asarray(radar_vec, float) - mu[:-1])
        return float(np.clip(dens, Db.min(), Db.max()))

    def nearest(self, vec, difficulty: int | None = None, k: int = 3):
        """k nearest real charts to `vec` in Mahalanobis metric -> list of (index_within_bucket, distance)."""
        Vb, mu, Sig, Sinv, _ = self._fit(difficulty)
        d = np.sqrt(np.maximum(((Vb - vec) @ Sinv * (Vb - vec)).sum(1), 0.0))
        order = np.argsort(d)[:k]
        return [(int(i), float(d[i])) for i in order]

    @staticmethod
    def _mahal(x, mu, Sinv) -> float:
        v = x - mu
        return float(np.sqrt(max(v @ Sinv @ v, 0.0)))

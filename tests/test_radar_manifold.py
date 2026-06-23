"""Manifold-aware groove-radar conditioning surface (src/generation/radar_manifold.py)."""
import numpy as np
import pytest

from src.generation.radar_manifold import RadarManifold, DIMS, LEVELS


def _synth(n=800, seed=0):
    # stream/voltage/air/chaos correlated (intensity cluster); freeze orthogonal -> mirrors the real manifold.
    rng = np.random.default_rng(seed)
    g = rng.normal(0.4, 0.12, n)                         # shared intensity factor
    V = np.stack([
        g + rng.normal(0, 0.03, n),                      # stream
        g + rng.normal(0, 0.04, n),                      # voltage
        g + rng.normal(0, 0.05, n),                      # air
        rng.normal(0.3, 0.15, n),                        # freeze (independent)
        g + rng.normal(0, 0.05, n),                      # chaos
    ], 1)
    return np.clip(V, 0, 1)


def test_parse_spec_and_levels():
    m = RadarManifold(_synth())
    assert m.parse_spec("stream=high, chaos=low") == {"stream": "high", "chaos": "low"}
    assert m.parse_spec({"air": "mod"}) == {"air": "mod"}
    with pytest.raises(ValueError):
        m.parse_spec("bogus=high")
    # granularity: a named level resolves to its quantile; 'high' > 'low'
    assert m.resolve_value("stream", "high", None) > m.resolve_value("stream", "low", None)
    # raw value and explicit quantile are both accepted
    assert m.resolve_value("stream", 0.7, None) == pytest.approx(0.7)
    q90 = m.resolve_value("stream", "q0.9", None)
    assert m.vectors[:, 0].min() <= q90 <= m.vectors[:, 0].max()


def test_build_target_fills_and_preserves():
    m = RadarManifold(_synth())
    vec, info = m.build_target("stream=high,chaos=low", difficulty=None)
    assert vec.shape == (5,) and ((vec >= 0) & (vec <= 1)).all()
    # the fill produces a full coherent vector; set dims are recorded
    assert set(info["fixed"]) == {"stream", "chaos"}
    # a free dim got filled from the conditional (not left at an arbitrary value)
    assert "filled" in info and info["filled"].shape == (5,)


def test_conditional_fill_respects_correlation():
    # intensity dims are correlated: asking for HIGH stream should pull the *filled* air/voltage ABOVE their
    # means (positive coupling), while the orthogonal freeze stays near its mean.
    m = RadarManifold(_synth())
    _, info = m.build_target("stream=high", difficulty=None)
    mu = m.vectors.mean(0); f = info["filled"]
    assert f[DIMS.index("air")] > mu[DIMS.index("air")]
    assert f[DIMS.index("voltage")] > mu[DIMS.index("voltage")]
    assert abs(f[DIMS.index("freeze")] - mu[DIMS.index("freeze")]) < 0.06   # ~unmoved


def test_projection_caps_off_manifold_targets():
    # a deliberately contradictory extreme (max stream, min chaos -- anti-correlated) must be pulled back to
    # within the projection radius; an on-manifold target must be left untouched.
    m = RadarManifold(_synth(), project_quantile=0.90)
    _, off = m.build_target("stream=max,chaos=min,air=min", difficulty=None)
    assert off["mahalanobis"] > off["max_mahalanobis"]      # the raw ask was off-manifold
    assert off["projected"]
    _, on = m.build_target("chaos=mod,stream=mod", difficulty=None)
    assert not on["projected"]


def test_save_load_roundtrip(tmp_path):
    m = RadarManifold(_synth())
    p = tmp_path / "manifold.npz"
    m.save(p)
    m2 = RadarManifold.load(p)
    v1, _ = m.build_target("air=high", None)
    v2, _ = m2.build_target("air=high", None)
    assert np.allclose(v1, v2)


def test_target_density_is_source_free_and_tracks_stream():
    # density is coupled to the intensity cluster, so a SOURCE-CHART-FREE density target derived from a
    # high-stream style should exceed a low-stream one -- no source chart involved.
    rng = np.random.default_rng(1)
    V = _synth(1000)
    dens = 0.15 + 0.5 * V[:, 0] + rng.normal(0, 0.01, len(V))   # density ~ stream (notes/frame)
    m = RadarManifold(V, densities=np.clip(dens, 0, 1))
    hi, _ = m.build_target("stream=high", None)
    lo, _ = m.build_target("stream=low", None)
    d_hi = m.target_density(hi); d_lo = m.target_density(lo)
    assert d_hi is not None and d_hi > d_lo
    # and it surfaces in build_target's info
    _, info = m.build_target("stream=high", None)
    assert info["density"] == pytest.approx(d_hi, abs=1e-6)
    # no density data -> None (gracefully source-density-free degrade)
    assert RadarManifold(V).target_density(hi) is None


def test_levels_are_editable_without_refit():
    # adding a finer level takes effect immediately (no refit) because raw vectors are retained.
    V = _synth()
    custom = dict(LEVELS, ultra=0.99)
    m = RadarManifold(V, levels=custom)
    assert m.resolve_value("stream", "ultra", None) >= m.resolve_value("stream", "vhigh", None)

"""Test 2: Audio feature alignment test"""

import os
import sys
import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.stepmania_parser import StepManiaParser
from src.data.audio_features import AudioFeatureExtractor, AudioFeatures


def _base_features(T=40):
    """Synthetic AudioFeatures with the original 23-dim set populated (no audio needed)."""
    rng = np.random.default_rng(0)
    return AudioFeatures(
        mfcc=rng.standard_normal((13, T)).astype(np.float32),
        onset_env=rng.random(T).astype(np.float32),
        onset_rate=rng.random(T).astype(np.float32),
        tempo=150.0,
        spectral_contrast=rng.standard_normal((7, T)).astype(np.float32),
        n_frames=T,
    )


def test_stage1_features_are_a_clean_suffix():
    # The new musical features must append AFTER the original 23, leaving dims 0..22 byte-identical
    # so existing 23-dim checkpoints/behavior are unaffected.
    T = 40
    base23 = _base_features(T).get_aligned_features()
    assert base23.shape == (T, 23)

    full = _base_features(T)
    rng = np.random.default_rng(1)
    full.chroma = rng.random((12, T)).astype(np.float32)
    full.perc_onset = rng.random(T).astype(np.float32)
    full.harm_onset = rng.random(T).astype(np.float32)
    full.metric_phase = AudioFeatureExtractor()._metric_phase(T)
    full41 = full.get_aligned_features()
    assert full41.shape == (T, 41)
    np.testing.assert_allclose(full41[:, :23], base23, rtol=0, atol=0)


def test_metric_phase_values():
    # beat-phase cycles every 4 frames (16th grid), measure-phase every 16; encoded as sin/cos.
    mp = AudioFeatureExtractor()._metric_phase(32)
    assert mp.shape == (32, 4)
    np.testing.assert_allclose(mp[0], [0.0, 1.0, 0.0, 1.0], atol=1e-6)  # t=0 = downbeat
    np.testing.assert_allclose(mp[0, :2], mp[4, :2], atol=1e-6)         # beat period 4
    np.testing.assert_allclose(mp[0, 2:], mp[16, 2:], atol=1e-6)        # measure period 16
    assert not np.allclose(mp[0, 2:], mp[4, 2:], atol=1e-6)             # measure != beat period


def test_default_config_unchanged_23dim():
    ext = AudioFeatureExtractor()
    assert not (ext.config.use_chroma or ext.config.use_hpss_onsets or ext.config.use_metric_phase)
    assert _base_features().get_aligned_features().shape[1] == 23

def test_audio_feature_alignment():
    """Test audio features align with chart and have correct properties"""
    # Relax the song-length gate so the tiny (~4s) fixture chart isn't filtered out;
    # the default window (75-130s) is for real songs, not the unit-test fixture.
    parser = StepManiaParser(min_song_length=1, max_song_length=100000)
    extractor = AudioFeatureExtractor()

    # Use same fixture as parser test
    chart_path = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_chart.sm')
    audio_dir = os.path.join(os.path.dirname(__file__), 'fixtures')

    # Ensure test audio exists
    audio_path = os.path.join(audio_dir, 'test_audio.wav')
    if not os.path.exists(audio_path):
        # Create test audio if needed
        import soundfile as sf
        sr = 22050
        duration = 10.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.1 * np.sin(2 * np.pi * 440 * t)
        sf.write(audio_path, audio, sr)

    # Parse chart
    result = parser.process_chart(chart_path)
    assert result is not None
    chart, chart_tensors = result

    # Extract audio features
    audio_features = extractor.extract_from_chart(audio_path, chart)
    assert audio_features is not None, "Audio feature extraction should succeed"

    chart_tensor = chart_tensors[0]
    audio_tensor = audio_features.get_aligned_features()

    # Assert: audio.shape[0] == chart.shape[0]
    assert audio_tensor.shape[0] == chart_tensor.shape[0], \
        f"Audio/chart alignment mismatch: {audio_tensor.shape[0]} != {chart_tensor.shape[0]}"

    # Assert: MFCC dim is exactly 13
    # Canonical audio feature dim is 23 (= MFCC 13 + onset_env 1 + onset_rate 1 + tempo 1
    # + spectral_contrast 7), per ExperimentConfig.audio_features_dim. Optional Stage-1
    # musical features (chroma/perc/harm/metric-phase) append after index 22 when enabled.
    assert audio_tensor.shape[1] == 23, f"Expected 23 audio features, got {audio_tensor.shape[1]}"

    # Assert: no NaNs or infs
    assert not np.any(np.isnan(audio_tensor)), "Audio features contain NaN values"
    assert not np.any(np.isinf(audio_tensor)), "Audio features contain Inf values"
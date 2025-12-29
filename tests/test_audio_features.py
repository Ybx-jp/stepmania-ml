"""Test 2: Audio feature alignment test"""

import os
import sys
import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.stepmania_parser import StepManiaParser
from src.data.audio_features import AudioFeatureExtractor

def test_audio_feature_alignment():
    """Test audio features align with chart and have correct properties"""
    parser = StepManiaParser()
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
    assert audio_tensor.shape[1] == 13, f"Expected 13 MFCC features, got {audio_tensor.shape[1]}"

    # Assert: no NaNs or infs
    assert not np.any(np.isnan(audio_tensor)), "Audio features contain NaN values"
    assert not np.any(np.isinf(audio_tensor)), "Audio features contain Inf values"
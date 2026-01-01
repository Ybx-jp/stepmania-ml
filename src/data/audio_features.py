"""
Audio feature extraction module aligned with chart tensors.
Extracts MFCC features synchronized with 16th note chart resolution.

Key features:
- MFCC extraction with hop_length aligned to chart timesteps
- Feature validation and normalization
- Integration with StepMania parser parameters
"""

import os
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from .stepmania_parser import StepManiaChart


@dataclass
class AudioFeatureConfig:
    """Configuration for audio feature extraction"""
    sample_rate: int = 22050
    n_mfcc: int = 13
    n_fft: int = 2048
    hop_length: int = 512  # Will be overridden by chart alignment
    n_mels: int = 128
    fmin: float = 0.0
    fmax: Optional[float] = None  # Will default to sr/2


@dataclass
class AudioFeatures:
    """Container for extracted audio features"""
    mfcc: np.ndarray  # Shape: (n_mfcc, n_frames)
    audio_duration: float
    sample_rate: int
    hop_length: int
    n_frames: int

    def get_aligned_features(self) -> np.ndarray:
        """Get features in model input format: (n_frames, n_mfcc)"""
        return self.mfcc.T


class AudioFeatureExtractor:
    """Extract audio features aligned with StepMania chart tensors"""

    def __init__(self, config: Optional[AudioFeatureConfig] = None):
        self.config = config if config is not None else AudioFeatureConfig()

    def extract_from_chart(self,
                          audio_file_path: str,
                          chart: StepManiaChart) -> Optional[AudioFeatures]:
        """
        Extract audio features aligned with chart timesteps.
        Uses chart-specific hop_length for perfect alignment.
        """
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        try:
            # Load audio with target sample rate
            audio, sr = librosa.load(
                audio_file_path,
                sr=self.config.sample_rate,
                offset=chart.offset,  # Apply chart offset
                duration=chart.song_length_seconds
            )

            # Use chart-aligned hop_length
            aligned_hop_length = chart.hop_length

            # Extract MFCC features with aligned hop_length
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=self.config.n_mfcc,
                n_fft=self.config.n_fft,
                hop_length=aligned_hop_length,
                n_mels=self.config.n_mels,
                fmin=self.config.fmin,
                fmax=self.config.fmax
            )

            # Align audio frames to match chart timesteps exactly
            expected_frames = chart.timesteps_total
            actual_frames = mfcc.shape[1]

            if expected_frames != actual_frames:
                mfcc = self._align_features(mfcc, expected_frames)

            return AudioFeatures(
                mfcc=mfcc,
                audio_duration=len(audio) / sr,
                sample_rate=sr,
                hop_length=aligned_hop_length,
                n_frames=mfcc.shape[1]
            )

        except Exception as e:
            raise RuntimeError(f"Error extracting features from {audio_file_path}: {e}") from e

    def extract_standalone(self,
                          audio_file_path: str,
                          sample_rate: int = 22050,
                          hop_length: int = 512,
                          offset: float = 0.0,
                          duration: Optional[float] = None) -> Optional[AudioFeatures]:
        """
        Extract audio features with manual parameters.
        Useful for exploration or when chart data is not available.
        """
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        try:
            # Load audio
            audio, sr = librosa.load(
                audio_file_path,
                sr=sample_rate,
                offset=offset,
                duration=duration
            )

            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=self.config.n_mfcc,
                n_fft=self.config.n_fft,
                hop_length=hop_length,
                n_mels=self.config.n_mels,
                fmin=self.config.fmin,
                fmax=self.config.fmax
            )

            return AudioFeatures(
                mfcc=mfcc,
                audio_duration=len(audio) / sr,
                sample_rate=sr,
                hop_length=hop_length,
                n_frames=mfcc.shape[1]
            )

        except Exception as e:
            raise RuntimeError(f"Error extracting features from {audio_file_path}: {e}") from e

    def _align_features(self, mfcc: np.ndarray, target_frames: int) -> np.ndarray:
        """Align feature dimensions to target frame count"""
        current_frames = mfcc.shape[1]

        if current_frames == target_frames:
            return mfcc

        elif current_frames > target_frames:
            # Truncate
            return mfcc[:, :target_frames]

        else:
            # Pad with zeros
            padding = target_frames - current_frames
            padded = np.pad(mfcc, ((0, 0), (0, padding)), mode='constant', constant_values=0)
            return padded

def create_feature_extractor(config: Optional[Dict] = None) -> AudioFeatureExtractor:
    """Factory function to create configured feature extractor"""
    if config is None:
        return AudioFeatureExtractor()

    feature_config = AudioFeatureConfig(**config)
    return AudioFeatureExtractor(feature_config)
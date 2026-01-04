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
    # Spectral contrast settings
    n_bands: int = 6  # Number of frequency bands for spectral contrast (produces n_bands + 1 features)
    # Onset detection settings
    use_onset_features: bool = True
    use_spectral_contrast: bool = True


@dataclass
class AudioFeatures:
    """Container for extracted audio features"""
    mfcc: np.ndarray  # Shape: (n_mfcc, n_frames)
    onset_env: Optional[np.ndarray] = None  # Shape: (n_frames,) onset strength envelope
    onset_rate: Optional[np.ndarray] = None  # Shape: (n_frames,) local onset density
    tempo: Optional[float] = None  # Estimated tempo in BPM
    spectral_contrast: Optional[np.ndarray] = None  # Shape: (n_bands+1, n_frames)
    audio_duration: float = 0.0
    sample_rate: int = 22050
    hop_length: int = 512
    n_frames: int = 0

    def get_aligned_features(self) -> np.ndarray:
        """
        Get features in model input format: (n_frames, n_features)

        Concatenates all available features:
        - MFCC: 13 features (timbral characteristics)
        - Onset envelope: 1 feature (rhythmic intensity, normalized)
        - Onset rate: 1 feature (local onset density)
        - Tempo: 1 feature (normalized BPM, repeated across frames)
        - Spectral contrast: 7 features (6 bands + 1 valley)

        Total: 23 features (if all enabled)
        """
        features = [self.mfcc.T]  # (n_frames, 13)

        if self.onset_env is not None:
            # Normalize onset envelope to 0-1 range and reshape
            onset_normalized = self.onset_env / (np.max(self.onset_env) + 1e-8)
            features.append(onset_normalized.reshape(-1, 1))  # (n_frames, 1)

        if self.onset_rate is not None:
            # Onset rate is already normalized during computation
            features.append(self.onset_rate.reshape(-1, 1))  # (n_frames, 1)

        if self.tempo is not None:
            # Normalize tempo to 0-1 range (assume 60-240 BPM range)
            tempo_normalized = (self.tempo - 60.0) / 180.0
            tempo_normalized = np.clip(tempo_normalized, 0.0, 1.0)
            tempo_feature = np.full((self.mfcc.shape[1], 1), tempo_normalized)
            features.append(tempo_feature)  # (n_frames, 1)

        if self.spectral_contrast is not None:
            features.append(self.spectral_contrast.T)  # (n_frames, 7)

        return np.concatenate(features, axis=1)


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

        Extracts:
        - MFCC: 13 features (timbral characteristics)
        - Onset envelope: 1 feature (rhythmic intensity)
        - Onset rate: 1 feature (local onset density)
        - Tempo: 1 feature (estimated BPM)
        - Spectral contrast: 7 features (timbral texture)
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

            # Extract onset and tempo features
            onset_env = None
            onset_rate = None
            tempo = None
            if self.config.use_onset_features:
                # Onset strength envelope (per-frame rhythmic intensity)
                onset_env = librosa.onset.onset_strength(
                    y=audio,
                    sr=sr,
                    hop_length=aligned_hop_length,
                    n_fft=self.config.n_fft
                )

                # Estimate tempo
                tempo_estimate = librosa.beat.tempo(
                    onset_envelope=onset_env,
                    sr=sr,
                    hop_length=aligned_hop_length
                )
                tempo = float(tempo_estimate[0]) if hasattr(tempo_estimate, '__len__') else float(tempo_estimate)

                # Compute onset rate (local density of onsets using rolling window)
                onset_rate = self._compute_onset_rate(onset_env, window_size=32)

            # Extract spectral contrast (timbral texture - peak vs valley in spectrum)
            spectral_contrast = None
            if self.config.use_spectral_contrast:
                spectral_contrast = librosa.feature.spectral_contrast(
                    y=audio,
                    sr=sr,
                    hop_length=aligned_hop_length,
                    n_fft=self.config.n_fft,
                    n_bands=self.config.n_bands,
                    fmin=max(self.config.fmin, 200.0)  # Minimum 200Hz for spectral contrast
                )

            # Align audio frames to match chart timesteps exactly
            expected_frames = chart.timesteps_total
            actual_frames = mfcc.shape[1]

            if expected_frames != actual_frames:
                mfcc = self._align_features(mfcc, expected_frames)
                if onset_env is not None:
                    onset_env = self._align_features_1d(onset_env, expected_frames)
                if onset_rate is not None:
                    onset_rate = self._align_features_1d(onset_rate, expected_frames)
                if spectral_contrast is not None:
                    spectral_contrast = self._align_features(spectral_contrast, expected_frames)

            return AudioFeatures(
                mfcc=mfcc,
                onset_env=onset_env,
                onset_rate=onset_rate,
                tempo=tempo,
                spectral_contrast=spectral_contrast,
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

    def _align_features(self, features: np.ndarray, target_frames: int) -> np.ndarray:
        """Align 2D feature dimensions to target frame count"""
        current_frames = features.shape[1]

        if current_frames == target_frames:
            return features

        elif current_frames > target_frames:
            # Truncate
            return features[:, :target_frames]

        else:
            # Pad with zeros
            padding = target_frames - current_frames
            padded = np.pad(features, ((0, 0), (0, padding)), mode='constant', constant_values=0)
            return padded

    def _align_features_1d(self, features: np.ndarray, target_frames: int) -> np.ndarray:
        """Align 1D feature dimensions to target frame count"""
        current_frames = len(features)

        if current_frames == target_frames:
            return features

        elif current_frames > target_frames:
            # Truncate
            return features[:target_frames]

        else:
            # Pad with zeros
            padding = target_frames - current_frames
            padded = np.pad(features, (0, padding), mode='constant', constant_values=0)
            return padded

    def _compute_onset_rate(self, onset_env: np.ndarray, window_size: int = 32) -> np.ndarray:
        """
        Compute local onset rate (density of onsets in a rolling window).

        Args:
            onset_env: Onset strength envelope
            window_size: Size of rolling window in frames

        Returns:
            Array of same length as onset_env with local onset density (0-1 normalized)
        """
        # Threshold onset envelope to detect onset events
        threshold = np.mean(onset_env) + 0.5 * np.std(onset_env)
        onset_events = (onset_env > threshold).astype(float)

        # Compute rolling sum using convolution
        kernel = np.ones(window_size) / window_size
        onset_rate = np.convolve(onset_events, kernel, mode='same')

        # Normalize to 0-1 range
        max_rate = onset_rate.max()
        if max_rate > 0:
            onset_rate = onset_rate / max_rate

        return onset_rate


def create_feature_extractor(config: Optional[Dict] = None) -> AudioFeatureExtractor:
    """Factory function to create configured feature extractor"""
    if config is None:
        return AudioFeatureExtractor()

    feature_config = AudioFeatureConfig(**config)
    return AudioFeatureExtractor(feature_config)
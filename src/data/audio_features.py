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
    # Musical-feature retrain (Stage 1, default OFF -> keeps the 23-dim path identical):
    use_chroma: bool = False          # 12-dim chromagram (per-pitch-class energy = melody/harmony)
    use_hpss_onsets: bool = False     # 2-dim: onset strength of percussive vs harmonic components
    use_metric_phase: bool = False    # 4-dim: sin/cos of beat-phase and measure-phase (from the 16th grid)
    timesteps_per_beat: int = 4       # chart resolution (16th notes); must match the parser
    beats_per_measure: int = 4


@dataclass
class AudioFeatures:
    """Container for extracted audio features"""
    mfcc: np.ndarray  # Shape: (n_mfcc, n_frames)
    onset_env: Optional[np.ndarray] = None  # Shape: (n_frames,) onset strength envelope
    onset_rate: Optional[np.ndarray] = None  # Shape: (n_frames,) local onset density
    tempo: Optional[float] = None  # Estimated tempo in BPM
    spectral_contrast: Optional[np.ndarray] = None  # Shape: (n_bands+1, n_frames)
    chroma: Optional[np.ndarray] = None        # Shape: (12, n_frames) per-pitch-class energy
    perc_onset: Optional[np.ndarray] = None    # Shape: (n_frames,) percussive onset strength (HPSS)
    harm_onset: Optional[np.ndarray] = None    # Shape: (n_frames,) harmonic onset strength (HPSS)
    metric_phase: Optional[np.ndarray] = None  # Shape: (n_frames, 4) sin/cos of beat & measure phase
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
        # Normalize MFCC features (z-score per feature, then clip)
        mfcc_normalized = self._safe_normalize(self.mfcc.T)
        features = [mfcc_normalized]  # (n_frames, 13)

        if self.onset_env is not None:
            # Normalize onset envelope to 0-1 range and reshape
            onset_max = np.max(self.onset_env)
            if onset_max > 1e-8:
                onset_normalized = self.onset_env / onset_max
            else:
                onset_normalized = np.zeros_like(self.onset_env)
            onset_normalized = self._safe_clip(onset_normalized.reshape(-1, 1))
            features.append(onset_normalized)  # (n_frames, 1)

        if self.onset_rate is not None:
            # Onset rate is already normalized during computation
            onset_rate_safe = self._safe_clip(self.onset_rate.reshape(-1, 1))
            features.append(onset_rate_safe)  # (n_frames, 1)

        if self.tempo is not None:
            # Normalize tempo to 0-1 range (assume 60-240 BPM range)
            tempo_normalized = (self.tempo - 60.0) / 180.0
            tempo_normalized = np.clip(tempo_normalized, 0.0, 1.0)
            tempo_feature = np.full((self.mfcc.shape[1], 1), tempo_normalized)
            features.append(tempo_feature)  # (n_frames, 1)

        if self.spectral_contrast is not None:
            # Normalize spectral contrast
            contrast_normalized = self._safe_normalize(self.spectral_contrast.T)
            features.append(contrast_normalized)  # (n_frames, 7)

        # --- Stage-1 musical features (appended AFTER the original 23 so dims 0..22 are unchanged) ---
        if self.chroma is not None:
            # per-pitch-class energy; z-score per class over the song (like MFCC)
            features.append(self._safe_normalize(self.chroma.T))  # (n_frames, 12)

        if self.perc_onset is not None:
            features.append(self._normalize_envelope(self.perc_onset))  # (n_frames, 1)
        if self.harm_onset is not None:
            features.append(self._normalize_envelope(self.harm_onset))  # (n_frames, 1)

        if self.metric_phase is not None:
            # already in [-1, 1] (sin/cos); pass through with a safety clip
            features.append(self._safe_clip(self.metric_phase))  # (n_frames, 4)

        result = np.concatenate(features, axis=1)

        # Final safety check for NaN/Inf
        result = self._safe_clip(result)

        return result

    def _safe_normalize(self, arr: np.ndarray, clip_range: float = 10.0) -> np.ndarray:
        """Safely normalize array with NaN/Inf handling."""
        # Replace NaN/Inf with zeros first
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Z-score normalization per feature (column)
        mean = np.mean(arr, axis=0, keepdims=True)
        std = np.std(arr, axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)  # Avoid division by zero

        normalized = (arr - mean) / std

        # Clip to reasonable range
        normalized = np.clip(normalized, -clip_range, clip_range)

        return normalized

    def _safe_clip(self, arr: np.ndarray, min_val: float = -100.0, max_val: float = 100.0) -> np.ndarray:
        """Clip array and replace NaN/Inf values."""
        arr = np.nan_to_num(arr, nan=0.0, posinf=max_val, neginf=min_val)
        return np.clip(arr, min_val, max_val)

    def _normalize_envelope(self, env: np.ndarray) -> np.ndarray:
        """Normalize a 1D strength envelope to 0-1 by its max, like onset_env -> (n_frames, 1)."""
        env = np.nan_to_num(env, nan=0.0, posinf=0.0, neginf=0.0)
        m = np.max(env)
        out = (env / m) if m > 1e-8 else np.zeros_like(env)
        return self._safe_clip(out.reshape(-1, 1))


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
            # Load full audio first, then slice - more robust than using offset parameter
            # (offset/seek fails on many MP3 files due to VBR encoding issues)
            audio_full, sr = librosa.load(
                audio_file_path,
                sr=self.config.sample_rate,
                offset=0,
                duration=None
            )

            # Apply offset and duration by slicing
            start_sample = int(chart.offset * sr) if chart.offset > 0 else 0
            end_sample = start_sample + int(chart.song_length_seconds * sr)

            # Ensure we don't exceed audio length
            start_sample = max(0, min(start_sample, len(audio_full)))
            end_sample = max(start_sample, min(end_sample, len(audio_full)))

            audio = audio_full[start_sample:end_sample]

            # Pad with silence if audio is too short
            expected_samples = int(chart.song_length_seconds * sr)
            if len(audio) < expected_samples:
                audio = np.pad(audio, (0, expected_samples - len(audio)), mode='constant')

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

            # --- Stage-1 musical features (chroma / HPSS onsets / metric phase) ---
            chroma = None
            if self.config.use_chroma:
                # chroma_stft (not _cqt): CQT constrains hop to a power of two; our hop is BPM-derived.
                # tuning=0.0 disables tuning estimation (its piptrack path segfaults in this env, and
                # tuning correction is negligible for pitch-class energy as a conditioning feature).
                chroma = librosa.feature.chroma_stft(
                    y=audio, sr=sr, hop_length=aligned_hop_length, n_fft=self.config.n_fft, tuning=0.0
                )  # (12, n_frames)

            perc_onset = None
            harm_onset = None
            if self.config.use_hpss_onsets:
                # split into harmonic (tuned) and percussive (drums), onset strength on each
                y_harm, y_perc = librosa.effects.hpss(audio)
                perc_onset = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=aligned_hop_length,
                                                          n_fft=self.config.n_fft)
                harm_onset = librosa.onset.onset_strength(y=y_harm, sr=sr, hop_length=aligned_hop_length,
                                                          n_fft=self.config.n_fft)

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
                if chroma is not None:
                    chroma = self._align_features(chroma, expected_frames)
                if perc_onset is not None:
                    perc_onset = self._align_features_1d(perc_onset, expected_frames)
                if harm_onset is not None:
                    harm_onset = self._align_features_1d(harm_onset, expected_frames)

            # metric phase is derived from the (BPM-aligned) frame index, not the audio
            metric_phase = self._metric_phase(mfcc.shape[1]) if self.config.use_metric_phase else None

            return AudioFeatures(
                mfcc=mfcc,
                onset_env=onset_env,
                onset_rate=onset_rate,
                tempo=tempo,
                spectral_contrast=spectral_contrast,
                chroma=chroma,
                perc_onset=perc_onset,
                harm_onset=harm_onset,
                metric_phase=metric_phase,
                audio_duration=len(audio) / sr,
                sample_rate=sr,
                hop_length=aligned_hop_length,
                n_frames=mfcc.shape[1]
            )

        except Exception as e:
            print(f"Warning: Error extracting features from {audio_file_path}: {e}")
            return None

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
            print(f"Warning: Error extracting features from {audio_file_path}: {e}")
            return None

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

    def _metric_phase(self, n_frames: int) -> np.ndarray:
        """Beat- and measure-phase as sin/cos, from the 16th-grid frame index -> (n_frames, 4).

        Frame t is the t-th 16th-note timestep (audio hop is BPM-aligned), so t % tpb is its
        position within a beat and t % (tpb*beats_per_measure) within a measure. Encoding each as
        (sin, cos) of its phase gives the model an explicit, cyclical sense of metric position
        (downbeat vs offbeat, bar boundaries) so syncopation can be metric-aware.
        """
        tpb = self.config.timesteps_per_beat
        tpm = tpb * self.config.beats_per_measure
        t = np.arange(n_frames)
        beat = 2.0 * np.pi * (t % tpb) / tpb
        meas = 2.0 * np.pi * (t % tpm) / tpm
        return np.stack([np.sin(beat), np.cos(beat), np.sin(meas), np.cos(meas)], axis=1).astype(np.float32)

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
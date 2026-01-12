"""
PyTorch dataset for StepMania chart difficulty classification.
Handles data loading, preprocessing, and tensor preparation following PyTorch patterns.

Responsibilities:
- Load and parse StepMania charts
- Extract aligned audio features
- Compute groove radar values for contrastive learning
- Apply joint padding/truncation
- Create attention masks
- Return consistent tensor format

Classification Target:
- Difficulty name (Beginner, Easy, Medium, Hard, Challenge) -> 4 classes
- Numeric difficulty (1-12+) retained as metadata for analysis
- Groove radar (5 values) for contrastive learning similarity
"""

import os
import pickle
from typing import Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

from .stepmania_parser import StepManiaParser
from .audio_features import AudioFeatureExtractor
from .groove_radar import GrooveRadarCalculator, GrooveRadar


# Standard StepMania difficulty names -> class indices (4 classes)
# Challenge folded into Hard due to rarity in dataset
DIFFICULTY_NAMES = ['Beginner', 'Easy', 'Medium', 'Hard']
DIFFICULTY_NAME_TO_IDX = {name: idx for idx, name in enumerate(DIFFICULTY_NAMES)}


def get_difficulty_class(difficulty_name: str) -> Optional[int]:
    """
    Map difficulty name to class index (0-3).

    Handles common StepMania difficulty name variants:
    - Beginner / Novice -> 0
    - Easy / Basic / Light -> 1
    - Medium / Another / Trick / Standard -> 2
    - Hard / Maniac / Heavy / Challenge / Expert / Oni / Edit -> 3

    Note: Challenge folded into Hard due to rarity in dataset.

    Args:
        difficulty_name: The difficulty name from the chart file (may have trailing colon)

    Returns:
        Class index (0-3) or None if name is unrecognized
    """
    # Strip whitespace and trailing colon from .sm file format
    name = difficulty_name.strip().rstrip(':').lower()

    # Beginner variants
    if name in ('beginner', 'novice'):
        return 0
    # Easy variants
    elif name in ('easy', 'basic', 'light'):
        return 1
    # Medium variants
    elif name in ('medium', 'another', 'trick', 'standard'):
        return 2
    # Hard variants (includes Challenge, folded due to rarity)
    elif name in ('hard', 'maniac', 'heavy', 'challenge', 'expert', 'oni', 'smaniac', 'edit'):
        return 3
    else:
        return None


class StepManiaDataset(Dataset):
    """Dataset for StepMania chart difficulty classification by difficulty name."""

    def __init__(self,
                 chart_files: List[str],
                 audio_dir: str,
                 max_sequence_length: int = 1200,
                 parser: Optional[StepManiaParser] = None,
                 feature_extractor: Optional[AudioFeatureExtractor] = None,
                 cache_dir: Optional[str] = None):
        """
        Initialize StepMania dataset.

        Args:
            chart_files: List of paths to .sm/.ssc chart files
            audio_dir: Directory containing audio files
            max_sequence_length: Maximum sequence length for padding/truncation
            parser: Optional StepMania parser instance
            feature_extractor: Optional audio feature extractor instance
            cache_dir: Optional directory for caching processed samples
        """
        self.chart_files = chart_files
        self.audio_dir = audio_dir
        self.max_sequence_length = max_sequence_length
        self.cache_dir = cache_dir

        # Initialize processors
        self.parser = parser if parser is not None else StepManiaParser()
        self.feature_extractor = (feature_extractor if feature_extractor is not None
                                else AudioFeatureExtractor())
        self.groove_radar_calculator = GrooveRadarCalculator()

        # Create sample metadata (like load_and_correct_labels in notebook)
        self.valid_samples = self._create_sample_metadata()

        print(f"Dataset initialized with {len(self.valid_samples)} valid samples")

    def __len__(self) -> int:
        """Return total number of valid samples"""
        return len(self.valid_samples)

    def get_data_info(self) -> Dict:
        """
        Get metadata about the dataset for logging/checkpointing.

        Returns:
            Dictionary with:
            - difficulty_distribution: counts per difficulty name class (0-4)
            - difficulty_names: list of class names ['Beginner', 'Easy', ...]
            - numeric_difficulty_distribution: counts per numeric difficulty (1-12+)
            - total_samples: total number of valid samples
            - chart_files: list of source chart files
        """
        from collections import Counter

        # Distribution by difficulty name class (0-3)
        class_counts = Counter(s['difficulty_class'] for s in self.valid_samples)
        class_distribution = {i: class_counts.get(i, 0) for i in range(4)}

        # Also track numeric difficulty distribution for analysis
        numeric_counts = Counter(s['difficulty_value'] for s in self.valid_samples)

        return {
            'difficulty_distribution': class_distribution,
            'difficulty_names': DIFFICULTY_NAMES,
            'numeric_difficulty_distribution': dict(sorted(numeric_counts.items())),
            'total_samples': len(self.valid_samples),
            'chart_files': self.chart_files
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and return a single training-ready sample.

        Returns:
            Dictionary with:
            - 'chart': (max_seq_len, 4) padded/truncated chart tensor
            - 'audio': (max_seq_len, audio_features_dim) padded/truncated audio features
            - 'mask': (max_seq_len,) attention mask (True = valid, False = padding)
            - 'length': original sequence length before padding
            - 'difficulty': scalar difficulty class (0-3 for CrossEntropy)
            - 'difficulty_value': numeric difficulty for analysis (not used in training)
            - 'groove_radar': (5,) normalized groove radar values [stream, voltage, air, freeze, chaos]

        Note: If audio extraction fails, returns a different valid sample to avoid crashing.
        """
        return self._get_sample_with_retry(idx, max_retries=10)

    def _get_sample_with_retry(self, idx: int, max_retries: int = 10) -> Dict[str, torch.Tensor]:
        """
        Try to load a sample, retrying with different indices if loading fails.

        Args:
            idx: Initial sample index
            max_retries: Maximum number of different samples to try

        Returns:
            Successfully loaded sample dictionary
        """
        import random

        tried_indices = set()
        current_idx = idx

        for attempt in range(max_retries):
            tried_indices.add(current_idx)
            sample_meta = self.valid_samples[current_idx]

            # Check cache first (stub for now)
            cached_sample = self._load_from_cache(current_idx)
            if cached_sample is not None:
                return cached_sample

            # Load chart tensor and audio features
            result = self._load_chart_and_audio(sample_meta)

            if result is not None:
                chart_tensor, audio_tensor, original_length = result

                # Apply joint padding/truncation to ensure alignment
                chart_padded, audio_padded, mask = self._apply_joint_padding_truncation(
                    chart_tensor, audio_tensor
                )

                # Get difficulty class (0-3)
                difficulty_class = sample_meta['difficulty_class']

                # Use pre-computed groove radar values
                groove_radar_vector = sample_meta['groove_radar'].to_vector()

                # Create final sample
                processed_sample = {
                    'chart': torch.from_numpy(chart_padded).float(),
                    'audio': torch.from_numpy(audio_padded).float(),
                    'mask': torch.from_numpy(mask).bool(),
                    'length': original_length,
                    'difficulty': torch.tensor(difficulty_class, dtype=torch.long),
                    'difficulty_value': torch.tensor(sample_meta['difficulty_value'], dtype=torch.long),
                    'groove_radar': torch.from_numpy(groove_radar_vector).float()
                }

                # Cache processed sample (stub for now)
                self._save_to_cache(current_idx, processed_sample)

                return processed_sample

            # Loading failed, try a different sample
            # Pick a random index that we haven't tried yet
            available_indices = [i for i in range(len(self.valid_samples)) if i not in tried_indices]
            if not available_indices:
                break
            current_idx = random.choice(available_indices)

        # All retries exhausted - raise an error
        raise RuntimeError(f"Failed to load any valid sample after {max_retries} attempts starting from index {idx}")

    def _create_sample_metadata(self) -> List[Dict]:
        """
        Parse chart files and create sample index.
        Similar to load_and_correct_labels in the notebook pattern.

        Each sample now has:
        - difficulty_class: 0-3 index for Beginner/Easy/Medium/Hard
        - difficulty_value: original numeric difficulty (for analysis)
        - difficulty_name: original difficulty name string
        - groove_radar: GrooveRadar object with 5 values for contrastive learning
        - hold_info: Hold arrow tracking data for freeze calculation
        """
        valid_samples = []
        skipped_unknown_difficulty = 0
        num_files = len(self.chart_files)

        print(f"Parsing {num_files} chart files...")
        for idx, chart_file in enumerate(self.chart_files):
            if idx % 50 == 0:
                print(f"  Parsing file {idx}/{num_files}...", end='\r')
            try:
                # Parse chart using StepManiaParser
                result = self.parser.process_chart(chart_file)
                if result is None:
                    continue

                chart, chart_tensors = result

                # Find corresponding audio file
                audio_file = self._find_audio_file(chart.audio_file, chart_file)
                if audio_file is None:
                    print(f"Audio file not found for {chart_file}")
                    continue

                # Compute average BPM for groove radar calculations
                avg_bpm = self.parser.compute_average_bpm(
                    chart.timing_events, chart.song_length_seconds
                )

                # Create sample metadata for each valid difficulty
                # note_data and chart_tensors are already filtered by parser.process_chart
                for note_data, chart_tensor in zip(chart.note_data, chart_tensors):
                    # Map difficulty name to class index (0-3)
                    difficulty_class = get_difficulty_class(note_data.difficulty_name)
                    if difficulty_class is None:
                        skipped_unknown_difficulty += 1
                        continue

                    # Use extended tensor conversion to get hold info
                    chart_tensor_ext, hold_info = self.parser.convert_to_tensor_extended(
                        chart, note_data
                    )

                    # Pre-compute groove radar during metadata creation
                    groove_radar = self.groove_radar_calculator.calculate(
                        chart_tensor=chart_tensor_ext,
                        hold_info=hold_info,
                        timing_events=chart.timing_events,
                        song_length_seconds=chart.song_length_seconds,
                        avg_bpm=avg_bpm
                    )

                    sample = {
                        'chart_file': chart_file,
                        'audio_file': audio_file,
                        'chart': chart,
                        'chart_tensor': chart_tensor_ext,
                        'hold_info': hold_info,
                        'groove_radar': groove_radar,  # Pre-computed GrooveRadar object
                        'difficulty_class': difficulty_class,  # 0-3 for training target
                        'difficulty_value': note_data.difficulty_value,  # numeric (for analysis)
                        'difficulty_name': note_data.difficulty_name  # original string
                    }
                    valid_samples.append(sample)

            except Exception as e:
                print(f"Error processing {chart_file}: {e}")
                continue

        print()  # Newline after progress
        if skipped_unknown_difficulty > 0:
            print(f"Skipped {skipped_unknown_difficulty} charts with unrecognized difficulty names")

        return valid_samples

    def _find_audio_file(self, audio_filename: str, chart_file: str) -> Optional[str]:
        """
        Find corresponding audio file in the same directory as the chart file.
        """
        if not audio_filename:
            return None

        # Look in the same directory as the chart file
        chart_dir = os.path.dirname(chart_file)
        audio_path = os.path.join(chart_dir, audio_filename)
        if os.path.exists(audio_path):
            return audio_path

        # Try common extensions in same directory
        base_name = os.path.splitext(audio_filename)[0]
        extensions = ['.ogg', '.mp3', '.wav', '.flac', '.m4a']

        for ext in extensions:
            audio_path = os.path.join(chart_dir, base_name + ext)
            if os.path.exists(audio_path):
                return audio_path

        return None

    def _load_chart_and_audio(self, sample_meta: Dict) -> Optional[tuple[np.ndarray, np.ndarray, int]]:
        """
        Load chart tensor and audio features for a sample.
        Similar to retrieve_image in the notebook pattern.

        Returns:
            Tuple of (chart_tensor, audio_tensor, original_length) or None if extraction fails.
        """
        try:
            # Extract audio features aligned with chart
            audio_features = self.feature_extractor.extract_from_chart(
                sample_meta['audio_file'],
                sample_meta['chart']
            )

            # Check if extraction failed
            if audio_features is None:
                return None

            # Get aligned tensors
            chart_tensor = sample_meta['chart_tensor']  # (timesteps, 4)
            audio_tensor = audio_features.get_aligned_features()  # (timesteps, audio_features_dim)

            # Check for NaN/Inf in audio tensor (should be handled by normalization, but double-check)
            if np.any(np.isnan(audio_tensor)) or np.any(np.isinf(audio_tensor)):
                print(f"Warning: NaN/Inf in audio features for {sample_meta['audio_file']}, skipping")
                return None

            # Ensure both tensors have same length (they should from alignment)
            if chart_tensor.shape[0] != audio_tensor.shape[0]:
                print(f"Warning: Chart/audio timestep mismatch: {chart_tensor.shape[0]} != {audio_tensor.shape[0]}")
                # Try to align by taking minimum length
                min_length = min(chart_tensor.shape[0], audio_tensor.shape[0])
                chart_tensor = chart_tensor[:min_length]
                audio_tensor = audio_tensor[:min_length]
            else:
                min_length = chart_tensor.shape[0]

            original_length = min_length

            return chart_tensor, audio_tensor, original_length

        except Exception as e:
            print(f"Warning: Error loading chart/audio for {sample_meta.get('audio_file', 'unknown')}: {e}")
            return None

    def _apply_joint_padding_truncation(self,
                                       chart_tensor: np.ndarray,
                                       audio_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Ensure both tensors are exactly max_sequence_length with masks.
        Following the padding/truncation pattern from the notebook.
        """
        current_length = chart_tensor.shape[0]

        if current_length >= self.max_sequence_length:
            # Truncate both tensors
            chart_padded = chart_tensor[:self.max_sequence_length]
            audio_padded = audio_tensor[:self.max_sequence_length]
            # All timesteps are valid (True = valid)
            mask = np.ones(self.max_sequence_length, dtype=bool)

        else:
            # Pad both tensors
            padding_length = self.max_sequence_length - current_length

            chart_padded = np.pad(
                chart_tensor,
                ((0, padding_length), (0, 0)),
                mode='constant',
                constant_values=0
            )

            audio_padded = np.pad(
                audio_tensor,
                ((0, padding_length), (0, 0)),
                mode='constant',
                constant_values=0
            )

            # Create attention mask: True for valid timesteps, False for padding
            mask = np.concatenate([
                np.ones(current_length, dtype=bool),
                np.zeros(padding_length, dtype=bool)
            ])

        return chart_padded, audio_padded, mask

    # Cache methods for avoiding repeated audio feature extraction
    def _get_cache_path(self, idx: int) -> str:
        """Get cache file path for sample."""
        if self.cache_dir is None:
            return ""
        return os.path.join(self.cache_dir, f"sample_{idx:06d}.pt")

    def _load_from_cache(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """Load cached sample if available."""
        cache_path = self._get_cache_path(idx)
        if cache_path and os.path.exists(cache_path):
            try:
                return torch.load(cache_path, weights_only=False)
            except Exception as e:
                # Cache file corrupted, will regenerate
                print(f"Warning: Failed to load cache {cache_path}: {e}")
                return None
        return None

    def _save_to_cache(self, idx: int, sample: Dict[str, torch.Tensor]):
        """Save sample to cache for faster loading next time."""
        if self.cache_dir is None:
            return

        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_path = self._get_cache_path(idx)
            torch.save(sample, cache_path)
        except Exception as e:
            # Non-fatal: just skip caching this sample
            print(f"Warning: Failed to cache sample {idx}: {e}")

    def warm_cache(self, show_progress: bool = True):
        """
        Pre-compute and cache all samples.

        Call this once before training to avoid I/O bottleneck during training.
        After warming, subsequent epochs load from cache (10-50x faster).

        Args:
            show_progress: Show progress bar during warming
        """
        if self.cache_dir is None:
            print("Warning: cache_dir not set, cannot warm cache")
            return

        from tqdm import tqdm

        print(f"Warming cache to {self.cache_dir}...")
        os.makedirs(self.cache_dir, exist_ok=True)

        iterator = range(len(self))
        if show_progress:
            iterator = tqdm(iterator, desc="Caching samples")

        cached = 0
        skipped = 0
        for idx in iterator:
            cache_path = self._get_cache_path(idx)
            if os.path.exists(cache_path):
                skipped += 1
                continue

            # This triggers full audio extraction and caches result
            try:
                _ = self[idx]
                cached += 1
            except Exception as e:
                print(f"Warning: Failed to cache sample {idx}: {e}")

        print(f"Cache warming complete: {cached} new, {skipped} existing")
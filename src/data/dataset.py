"""
PyTorch dataset for StepMania chart difficulty classification.
Handles data loading, preprocessing, and tensor preparation following PyTorch patterns.

Responsibilities:
- Load and parse StepMania charts
- Extract aligned audio features
- Apply joint padding/truncation
- Create attention masks
- Return consistent tensor format

Classification Target:
- Difficulty name (Beginner, Easy, Medium, Hard, Challenge) -> 5 classes
- Numeric difficulty (1-12+) retained as metadata for analysis
"""

import os
import pickle
from typing import Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

from .stepmania_parser import StepManiaParser
from .audio_features import AudioFeatureExtractor


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
            - 'difficulty': scalar difficulty class (0-4 for CrossEntropy)
            - 'difficulty_value': numeric difficulty for analysis (not used in training)
            - 'chart_stats': (8,) normalized chart statistics
        """
        sample_meta = self.valid_samples[idx]

        # Check cache first (stub for now)
        cached_sample = self._load_from_cache(idx)
        if cached_sample is not None:
            return cached_sample

        # Load chart tensor and audio features
        chart_tensor, audio_tensor, original_length = self._load_chart_and_audio(sample_meta)

        # Apply joint padding/truncation to ensure alignment
        chart_padded, audio_padded, mask = self._apply_joint_padding_truncation(
            chart_tensor, audio_tensor
        )

        # Get difficulty class (0-4)
        difficulty_class = sample_meta['difficulty_class']

        # Use pre-computed chart statistics
        chart_stats = sample_meta['chart_stats']

        # Create final sample
        processed_sample = {
            'chart': torch.from_numpy(chart_padded).float(),
            'audio': torch.from_numpy(audio_padded).float(),
            'mask': torch.from_numpy(mask).bool(),
            'length': original_length,
            'difficulty': torch.tensor(difficulty_class, dtype=torch.long),  # 0-4 for CrossEntropy
            'difficulty_value': torch.tensor(sample_meta['difficulty_value'], dtype=torch.long),  # Numeric for analysis
            'chart_stats': torch.from_numpy(chart_stats).float()  # (5,) difficulty features
        }

        # Cache processed sample (stub for now)
        self._save_to_cache(idx, processed_sample)

        return processed_sample

    def _create_sample_metadata(self) -> List[Dict]:
        """
        Parse chart files and create sample index.
        Similar to load_and_correct_labels in the notebook pattern.

        Each sample now has:
        - difficulty_class: 0-4 index for Beginner/Easy/Medium/Hard/Challenge
        - difficulty_value: original numeric difficulty (for analysis)
        - difficulty_name: original difficulty name string
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

                # Create sample metadata for each valid difficulty
                # note_data and chart_tensors are already filtered by parser.process_chart
                for note_data, chart_tensor in zip(chart.note_data, chart_tensors):
                    # Map difficulty name to class index (0-4)
                    difficulty_class = get_difficulty_class(note_data.difficulty_name)
                    if difficulty_class is None:
                        skipped_unknown_difficulty += 1
                        continue

                    # Pre-compute chart stats during metadata creation (avoid slow __getitem__ calls)
                    chart_stats = self._compute_chart_stats(chart_tensor, chart.song_length_seconds)

                    sample = {
                        'chart_file': chart_file,
                        'audio_file': audio_file,
                        'chart': chart,
                        'chart_tensor': chart_tensor,
                        'chart_stats': chart_stats,  # Pre-computed
                        'difficulty_class': difficulty_class,  # 0-4 for training target
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

    def _load_chart_and_audio(self, sample_meta: Dict) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Load chart tensor and audio features for a sample.
        Similar to retrieve_image in the notebook pattern.
        """
        # Extract audio features aligned with chart
        audio_features = self.feature_extractor.extract_from_chart(
            sample_meta['audio_file'],
            sample_meta['chart']
        )

        # Get aligned tensors
        chart_tensor = sample_meta['chart_tensor']  # (timesteps, 4)
        audio_tensor = audio_features.get_aligned_features()  # (timesteps, audio_features_dim)

        # Ensure both tensors have same length (they should from alignment)
        assert chart_tensor.shape[0] == audio_tensor.shape[0], \
            f"Chart/audio timestep mismatch after alignment: {chart_tensor.shape[0]} != {audio_tensor.shape[0]}"

        min_length = min(chart_tensor.shape[0], audio_tensor.shape[0])
        chart_tensor = chart_tensor[:min_length]
        audio_tensor = audio_tensor[:min_length]

        original_length = min_length

        return chart_tensor, audio_tensor, original_length

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

    def _compute_chart_stats(self, chart_tensor: np.ndarray, song_length_seconds: float) -> np.ndarray:
        """
        Compute difficulty-relevant statistics from chart tensor.

        Computes 8 features that correlate with chart difficulty:
        1. notes_per_second - total notes / song duration
        2. jump_ratio - fraction of timesteps with 2+ simultaneous notes
        3. max_stream_length - longest consecutive run of non-empty timesteps
        4. avg_gap - average gap between notes (in timesteps)
        5. peak_density - max notes in any 16-beat window
        6. crossover_ratio - patterns requiring hand crossing (L+D, L+U, R+D, R+U)
        7. consecutive_jump_ratio - jumps following jumps (pattern density)
        8. gap_variance - rhythm complexity (varied vs uniform gaps)

        Panel layout: L=0, D=1, U=2, R=3

        Args:
            chart_tensor: Binary chart encoding (timesteps, 4)
            song_length_seconds: Duration of the song in seconds

        Returns:
            numpy array of shape (8,) with computed statistics
        """
        # Notes per timestep (0-4)
        notes_per_timestep = chart_tensor.sum(axis=1)
        total_notes = notes_per_timestep.sum()
        num_timesteps = len(chart_tensor)

        # 1. Notes per second
        notes_per_second = total_notes / max(song_length_seconds, 1.0)

        # 2. Jump ratio (timesteps with 2+ notes)
        is_jump = notes_per_timestep >= 2
        jumps = is_jump.sum()
        jump_ratio = jumps / max(num_timesteps, 1)

        # 3. Max stream length (consecutive non-empty timesteps)
        has_note = notes_per_timestep > 0
        max_stream = 0
        current_stream = 0
        for has in has_note:
            if has:
                current_stream += 1
                max_stream = max(max_stream, current_stream)
            else:
                current_stream = 0

        # 4. Average gap between notes
        note_positions = np.where(has_note)[0]
        if len(note_positions) > 1:
            gaps = np.diff(note_positions)
            avg_gap = gaps.mean()
        else:
            gaps = np.array([num_timesteps])
            avg_gap = num_timesteps  # No notes or single note = max gap

        # 5. Peak density (max notes in any 16-beat window = 64 timesteps at 4 per beat)
        window_size = min(64, num_timesteps)
        if num_timesteps >= window_size:
            # Sliding window max
            peak_density = 0
            for i in range(num_timesteps - window_size + 1):
                window_notes = notes_per_timestep[i:i + window_size].sum()
                peak_density = max(peak_density, window_notes)
        else:
            peak_density = total_notes

        # 6. Crossover ratio - patterns requiring hand crossing
        # Panel layout: L=0, D=1, U=2, R=3
        # Crossovers: L+D (0,1), L+U (0,2), R+D (3,1), R+U (3,2)
        crossover_count = (
            ((chart_tensor[:, 0] > 0) & (chart_tensor[:, 1] > 0)).sum() +  # L+D
            ((chart_tensor[:, 0] > 0) & (chart_tensor[:, 2] > 0)).sum() +  # L+U
            ((chart_tensor[:, 3] > 0) & (chart_tensor[:, 1] > 0)).sum() +  # R+D
            ((chart_tensor[:, 3] > 0) & (chart_tensor[:, 2] > 0)).sum()    # R+U
        )
        crossover_ratio = crossover_count / max(total_notes, 1)

        # 7. Consecutive jump ratio - jumps following jumps (pattern density)
        if jumps > 0:
            consecutive_jumps = (is_jump[:-1] & is_jump[1:]).sum()
            consecutive_jump_ratio = consecutive_jumps / jumps
        else:
            consecutive_jump_ratio = 0.0

        # 8. Gap variance (rhythm complexity)
        if len(gaps) > 1:
            gap_std = gaps.std()
            gap_variance = min(gap_std / 10.0, 1.0)  # Normalize, std of 10 is high
        else:
            gap_variance = 0.0

        # Normalize features to roughly 0-1 range
        # Using reasonable expected ranges based on typical StepMania charts
        normalized_stats = np.array([
            min(notes_per_second / 12.0, 1.0),          # ~12 nps is very high
            jump_ratio,                                  # Already 0-1
            min(np.log1p(max_stream) / 6.0, 1.0),       # log(400)≈6, handles long streams
            1.0 - min(avg_gap / 50.0, 1.0),             # Invert: small gap = high difficulty
            min(np.log1p(peak_density) / 5.0, 1.0),     # log(150)≈5, handles dense sections
            min(crossover_ratio, 1.0),                   # Crossover patterns (0-1)
            consecutive_jump_ratio,                      # Already 0-1
            gap_variance,                                # Already normalized 0-1
        ], dtype=np.float32)

        return normalized_stats

    # Cache methods - stubs for later implementation
    def _get_cache_path(self, idx: int) -> str:
        """Get cache file path for sample - STUB"""
        if self.cache_dir is None:
            return ""
        return os.path.join(self.cache_dir, f"sample_{idx:06d}.pkl")

    def _load_from_cache(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """Load cached sample if available - STUB"""
        # TODO: Implement caching logic
        return None

    def _save_to_cache(self, idx: int, sample: Dict[str, torch.Tensor]):
        """Save sample to cache - STUB"""
        # TODO: Implement caching logic
        pass
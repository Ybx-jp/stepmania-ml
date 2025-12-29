"""
Data splitting utilities for StepMania chart dataset.
Handles train/validation/test splits and DataLoader creation.
"""

from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from ..data.dataset import StepManiaDataset
from ..data.stepmania_parser import StepManiaParser
from ..data.audio_features import AudioFeatureExtractor


def create_data_splits(chart_files: List[str],
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15,
                      random_state: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Create train/validation/test splits from chart files.

    Args:
        chart_files: List of paths to chart files
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_state: Random seed for reproducible splits

    Returns:
        Tuple of (train_files, val_files, test_files)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    # First split: train vs (val + test)
    train_files, temp_files = train_test_split(
        chart_files,
        test_size=(val_ratio + test_ratio),
        random_state=random_state
    )

    # Second split: val vs test
    val_files, test_files = train_test_split(
        temp_files,
        test_size=(test_ratio / (val_ratio + test_ratio)),
        random_state=random_state
    )

    print(f"Data splits created:")
    print(f"  Train: {len(train_files)} files ({train_ratio:.1%})")
    print(f"  Val:   {len(val_files)} files ({val_ratio:.1%})")
    print(f"  Test:  {len(test_files)} files ({test_ratio:.1%})")

    return train_files, val_files, test_files


def create_datasets(train_files: List[str],
                   val_files: List[str],
                   test_files: List[str],
                   audio_dir: str,
                   max_sequence_length: int = 1200,
                   parser: Optional[StepManiaParser] = None,
                   feature_extractor: Optional[AudioFeatureExtractor] = None,
                   cache_dir: Optional[str] = None) -> Tuple[StepManiaDataset, StepManiaDataset, StepManiaDataset]:
    """
    Create StepManiaDataset instances for train/val/test splits.

    Args:
        train_files: Training chart files
        val_files: Validation chart files
        test_files: Test chart files
        audio_dir: Directory containing audio files
        max_sequence_length: Maximum sequence length for padding
        parser: Optional shared parser instance
        feature_extractor: Optional shared feature extractor
        cache_dir: Optional cache directory

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Create shared processors if not provided
    if parser is None:
        parser = StepManiaParser()
    if feature_extractor is None:
        feature_extractor = AudioFeatureExtractor()

    # Create datasets
    train_dataset = StepManiaDataset(
        chart_files=train_files,
        audio_dir=audio_dir,
        max_sequence_length=max_sequence_length,
        parser=parser,
        feature_extractor=feature_extractor,
        cache_dir=f"{cache_dir}/train" if cache_dir else None
    )

    val_dataset = StepManiaDataset(
        chart_files=val_files,
        audio_dir=audio_dir,
        max_sequence_length=max_sequence_length,
        parser=parser,
        feature_extractor=feature_extractor,
        cache_dir=f"{cache_dir}/val" if cache_dir else None
    )

    test_dataset = StepManiaDataset(
        chart_files=test_files,
        audio_dir=audio_dir,
        max_sequence_length=max_sequence_length,
        parser=parser,
        feature_extractor=feature_extractor,
        cache_dir=f"{cache_dir}/test" if cache_dir else None
    )

    return train_dataset, val_dataset, test_dataset

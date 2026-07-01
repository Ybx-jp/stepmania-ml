"""
Data splitting utilities for StepMania chart dataset.
Handles train/validation/test splits and DataLoader creation.
"""

import glob
from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from ..data.dataset import StepManiaDataset
from ..data.stepmania_parser import StepManiaParser
from ..data.audio_features import AudioFeatureExtractor


def discover_chart_files(root: str = "data") -> List[str]:
    """All .sm + .ssc chart files under `root` (recursive) — the canonical probe/eval discovery idiom
    (was copy-pasted as `glob.glob(f"{root}/**/*.sm") + glob.glob(f"{root}/**/*.ssc")` in ~60 probes).

    NOTE: glob returns FILESYSTEM order, which is not stable across machines; the returned list is NOT sorted
    (deliberately — sorting would change which files land in val via create_data_splits below, altering every
    probe's cited results). Reproducibility of the SPLIT rests on create_data_splits(random_state=), not on order.
    """
    return (glob.glob(f"{root}/**/*.sm", recursive=True)
            + glob.glob(f"{root}/**/*.ssc", recursive=True))


def split_chart_files(root: str = "data", random_state: int = 42,
                      **split_kwargs) -> Tuple[List[str], List[str], List[str]]:
    """discover_chart_files(root) + create_data_splits(random_state) -> (train, val, test).

    The one-liner for the ubiquitous probe idiom `cf = glob(...); tf, vf, _ = create_data_splits(cf,
    random_state=42)`. Drop-in for both `tf, vf, _ = split_chart_files()` and `_, vf, _ = split_chart_files()`.
    Bit-identical to the inline form on a given machine (same glob, same seed). Extra ratio/stratify args pass
    through to create_data_splits.
    """
    return create_data_splits(discover_chart_files(root), random_state=random_state, **split_kwargs)


def create_data_splits(chart_files: List[str],
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15,
                      random_state: int = 42,
                      stratify_labels: Optional[List] = None) -> Tuple[List[str], List[str], List[str]]:
    """
    Create train/validation/test splits from chart files.

    Args:
        chart_files: List of paths to chart files
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_state: Random seed for reproducible splits
        stratify_labels: Optional list of labels for stratified splitting.
            Must be same length as chart_files. When provided, splits
            maintain class proportions across train/val/test sets.

    Returns:
        Tuple of (train_files, val_files, test_files)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    # First split: train vs (val + test)
    train_files, temp_files = train_test_split(
        chart_files,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=stratify_labels
    )

    # For second split, extract corresponding stratify labels if provided
    if stratify_labels is not None:
        # Build index map to get labels for temp_files
        file_to_label = dict(zip(chart_files, stratify_labels))
        temp_labels = [file_to_label[f] for f in temp_files]
    else:
        temp_labels = None

    # Second split: val vs test
    val_files, test_files = train_test_split(
        temp_files,
        test_size=(test_ratio / (val_ratio + test_ratio)),
        random_state=random_state,
        stratify=temp_labels
    )

    print(f"Data splits created{' (stratified)' if stratify_labels is not None else ''}:")
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
                   cache_dir: Optional[str] = None,
                   data_config: Optional[Dict] = None) -> Tuple[StepManiaDataset, StepManiaDataset, StepManiaDataset]:
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
        data_config: Optional data config dict (from data_config.yaml['data']['stepmania'])

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Create shared processors if not provided
    if parser is None:
        parser = StepManiaParser(config=data_config)
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

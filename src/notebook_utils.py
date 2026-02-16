"""
Notebook utilities for simplified experiment setup and data loading.

Provides one-line functions to replace 100+ lines of boilerplate in notebooks.

Key functions:
- setup_data_splits(): Create train/val/test datasets with caching
- load_experiment(): Load model + config from checkpoint
- setup_contrastive_experiment(): Full contrastive experiment setup
- quick_train(): Quick training with sensible defaults

Examples:
    # Simple data loading
    >>> train, val, test = setup_data_splits()

    # Load a trained model
    >>> model, config, checkpoint = load_experiment('classifier_baseline')

    # Full contrastive experiment setup
    >>> exp = setup_contrastive_experiment('contrastive_experiment_b')
    >>> model, trainer = exp['model'], exp['trainer']
    >>> history = trainer.fit(epochs=20)
"""

import torch
import yaml
import glob
from pathlib import Path
from typing import Tuple, Dict, Optional, Union, List
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Import from our modules
from .config.paths import (PROJECT_ROOT, DATA_DIR, CHECKPOINT_DIR, CONFIG_DIR, CACHE_DIR,
                           get_checkpoint_path, list_experiments)
from .data import StepManiaDataset, ContrastiveTripletDataset, create_contrastive_dataset
from .models.classifier import LateFusionClassifier
from .training import Trainer, ContrastiveTrainer


def setup_data_splits(
    data_dir: Union[str, Path] = None,
    cache_dir: Union[str, Path] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_seq_len: int = 8192,
    warm_cache: bool = True,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[StepManiaDataset, StepManiaDataset, StepManiaDataset]:
    """
    Create train/val/test datasets with automatic caching in one line.

    Uses sensible defaults from project structure. This replaces ~50 lines
    of dataset creation boilerplate in notebooks.

    Args:
        data_dir: Directory containing chart files (default: PROJECT_ROOT/data)
        cache_dir: Cache directory (default: PROJECT_ROOT/cache/samples)
        train_ratio: Proportion for training (default: 0.7)
        val_ratio: Proportion for validation (default: 0.15)
        test_ratio: Proportion for testing (default: 0.15)
        max_seq_len: Maximum sequence length (default: 8192)
        warm_cache: Whether to warm cache (default: True)
        seed: Random seed for splitting (default: 42)
        verbose: Whether to print progress (default: True)

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)

    Examples:
        >>> # Use defaults
        >>> train, val, test = setup_data_splits()

        >>> # Custom data directory
        >>> train, val, test = setup_data_splits(data_dir='./my_data')

        >>> # Skip cache warming for speed
        >>> train, val, test = setup_data_splits(warm_cache=False)
    """
    # Use defaults if not specified
    if data_dir is None:
        data_dir = DATA_DIR
    if cache_dir is None:
        cache_dir = CACHE_DIR / 'samples'

    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir)

    if verbose:
        print(f"Loading data from: {data_dir}")
        print(f"Cache directory: {cache_dir}")

    # Find all chart files
    chart_patterns = ['**/*.sm', '**/*.ssc']
    chart_files = []
    for pattern in chart_patterns:
        chart_files.extend(glob.glob(str(data_dir / pattern), recursive=True))

    if len(chart_files) == 0:
        raise ValueError(f"No chart files found in {data_dir}")

    if verbose:
        print(f"Found {len(chart_files)} chart files")

    # Split files
    temp_ratio = val_ratio + test_ratio
    train_files, temp_files = train_test_split(
        chart_files, test_size=temp_ratio, random_state=seed
    )
    val_files, test_files = train_test_split(
        temp_files, test_size=(test_ratio / temp_ratio), random_state=seed
    )

    if verbose:
        print(f"\nChart file splits:")
        print(f"  Train: {len(train_files)}")
        print(f"  Val: {len(val_files)}")
        print(f"  Test: {len(test_files)}")

    # Create datasets
    train_dataset = StepManiaDataset(
        chart_files=train_files,
        audio_dir=str(data_dir),
        max_sequence_length=max_seq_len,
        cache_dir=str(cache_dir / 'train')
    )

    val_dataset = StepManiaDataset(
        chart_files=val_files,
        audio_dir=str(data_dir),
        max_sequence_length=max_seq_len,
        cache_dir=str(cache_dir / 'val')
    )

    test_dataset = StepManiaDataset(
        chart_files=test_files,
        audio_dir=str(data_dir),
        max_sequence_length=max_seq_len,
        cache_dir=str(cache_dir / 'test')
    )

    if verbose:
        print(f"\nDataset sizes:")
        print(f"  Train: {len(train_dataset)} valid samples")
        print(f"  Val: {len(val_dataset)} valid samples")
        print(f"  Test: {len(test_dataset)} valid samples")

    # Warm cache if requested
    if warm_cache:
        if verbose:
            print("\nWarming dataset cache (this may take a few minutes on first run)...")
        train_dataset.warm_cache(show_progress=verbose)
        val_dataset.warm_cache(show_progress=verbose)
        test_dataset.warm_cache(show_progress=verbose)
        if verbose:
            print("Cache warming complete!")

    return train_dataset, val_dataset, test_dataset


def load_experiment(
    experiment_name: str,
    checkpoint_name: str = 'best_val_loss.pt',
    device: str = 'auto',
    load_config_from_checkpoint: bool = True
) -> Dict[str, any]:
    """
    Load a trained model and its configuration from checkpoint.

    Automatically finds the checkpoint, loads the model, and returns
    everything needed for inference or continued training.

    Args:
        experiment_name: Name of experiment (e.g., 'classifier_baseline', 'contrastive')
        checkpoint_name: Checkpoint file (default: 'best_val_loss.pt')
        device: Device ('auto', 'cuda', 'cpu')
        load_config_from_checkpoint: Load config from checkpoint vs config file

    Returns:
        Dictionary with keys:
        - 'model': Loaded model
        - 'config': Configuration dict
        - 'checkpoint': Full checkpoint dict
        - 'device': torch.device

    Examples:
        >>> # Load best checkpoint
        >>> exp = load_experiment('classifier_baseline')
        >>> model = exp['model']

        >>> # Load last checkpoint
        >>> exp = load_experiment('contrastive', checkpoint_name='last.pt')

        >>> # Use for inference
        >>> exp = load_experiment('classifier_baseline')
        >>> with torch.no_grad():
        ...     output = exp['model'](audio, chart, mask)
    """
    # Auto-detect device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    # Find checkpoint
    checkpoint_path = get_checkpoint_path(experiment_name, checkpoint_name)
    if checkpoint_path is None:
        raise ValueError(f"Checkpoint not found for experiment '{experiment_name}' with name '{checkpoint_name}'")

    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config
    if load_config_from_checkpoint and 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Try to find config file
        config_path = CONFIG_DIR / f"{experiment_name}.yaml"
        if not config_path.exists():
            config_path = CONFIG_DIR / 'model_config.yaml'

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    # Create model
    model = LateFusionClassifier(config if isinstance(config, dict) else config.get('classifier', config))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")

    return {
        'model': model,
        'config': config,
        'checkpoint': checkpoint,
        'device': device
    }


def setup_contrastive_experiment(
    experiment_config: Union[str, dict],
    pretrained_checkpoint: str = None,
    data_dir: Union[str, Path] = None,
    cache_dir: Union[str, Path] = None,
    checkpoint_dir: Union[str, Path] = None,
    device: str = 'auto',
    batch_size: int = None,
    num_workers: int = 4,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Complete contrastive experiment setup in one call.

    This replaces ~300 lines of boilerplate from notebooks with a single function call.
    Creates datasets, loads pretrained model, creates contrastive dataset, dataloaders,
    and initializes trainer with all the right callbacks.

    Args:
        experiment_config: Config name (e.g., 'contrastive_experiment_b') or dict
        pretrained_checkpoint: Name of pretrained classifier checkpoint
        data_dir: Data directory (default: PROJECT_ROOT/data)
        cache_dir: Cache directory (default: PROJECT_ROOT/cache)
        checkpoint_dir: Where to save checkpoints (default: checkpoints/contrastive)
        device: Device ('auto', 'cuda', 'cpu')
        batch_size: Batch size (uses config if None)
        num_workers: DataLoader workers (default: 4)
        verbose: Whether to print progress

    Returns:
        Dictionary with keys:
        - 'model': Model with projection head
        - 'trainer': ContrastiveTrainer instance
        - 'train_loader': Training dataloader (triplets)
        - 'val_loader': Validation dataloader
        - 'config': Full configuration
        - 'optimizer': Optimizer instance

    Examples:
        >>> # Minimal usage
        >>> exp = setup_contrastive_experiment('contrastive_experiment_b')
        >>> history = exp['trainer'].fit(epochs=20)

        >>> # With custom pretrained model
        >>> exp = setup_contrastive_experiment(
        ...     'contrastive_experiment_b',
        ...     pretrained_checkpoint='my_classifier/best_val_loss.pt'
        ... )

        >>> # Access components
        >>> model = exp['model']
        >>> trainer = exp['trainer']
        >>> train_loader = exp['train_loader']
    """
    # Auto-detect device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    # Use defaults
    if data_dir is None:
        data_dir = DATA_DIR
    if cache_dir is None:
        cache_dir = CACHE_DIR
    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR / 'contrastive'

    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    if isinstance(experiment_config, str):
        config_path = CONFIG_DIR / f"{experiment_config}.yaml"
        if not config_path.exists():
            raise ValueError(f"Config not found: {config_path}")

        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)

        if verbose:
            print(f"Loaded config: {config_path}")
    else:
        full_config = experiment_config

    classifier_config = full_config.get('classifier', {})
    training_config = full_config.get('training', {})
    contrastive_config = full_config.get('contrastive', {})

    # Override batch_size if specified
    if batch_size is not None:
        training_config['batch_size'] = batch_size

    # Create datasets (reuse setup_data_splits)
    if verbose:
        print("\n" + "="*60)
        print("Setting up datasets...")
        print("="*60)

    train_dataset, val_dataset, _ = setup_data_splits(
        data_dir=data_dir,
        cache_dir=cache_dir / 'samples',
        max_seq_len=classifier_config.get('max_sequence_length', 8192),
        warm_cache=True,
        verbose=verbose
    )

    # Create contrastive dataset
    if verbose:
        print("\n" + "="*60)
        print("Creating contrastive triplet dataset...")
        print("="*60)

    contrastive_train_dataset = create_contrastive_dataset(
        base_dataset=train_dataset,
        radar_weights=contrastive_config.get('radar_weights', [1.0] * 5),
        positive_percentile=contrastive_config.get('positive_percentile', 15.0),
        negative_percentile=contrastive_config.get('negative_percentile', 85.0),
        precompute=contrastive_config.get('precompute_triplets', True),
        resample=contrastive_config.get('resample_epoch', False)
    )

    # Create dataloaders
    pin_memory = device.type == 'cuda'
    batch_size = training_config['batch_size']

    train_loader = DataLoader(
        contrastive_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None
    )

    if verbose:
        print(f"\nDataLoaders created: train={len(train_loader)} batches, val={len(val_loader)} batches")

    # Create model with projection head
    model_config = classifier_config.copy()
    model_config['use_projection_head'] = True
    model_config['use_groove_radar'] = True

    if verbose:
        print("\n" + "="*60)
        print("Creating model...")
        print("="*60)

    model = LateFusionClassifier(model_config)

    # Load pretrained weights if specified
    if pretrained_checkpoint:
        if verbose:
            print(f"\nLoading pretrained weights from: {pretrained_checkpoint}")

        checkpoint_path = get_checkpoint_path(pretrained_checkpoint, 'best_val_loss.pt')
        if checkpoint_path is None:
            checkpoint_path = Path(pretrained_checkpoint)

        model = LateFusionClassifier.from_pretrained(
            checkpoint_path=str(checkpoint_path),
            config=model_config,
            freeze_backbone=False,
            device=str(device)
        )

    model = model.to(device)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config.get('weight_decay', 0.01)
    )

    # Merge configs for trainer
    trainer_config = {
        **training_config,
        **contrastive_config,
        'num_classes': classifier_config.get('num_classes', 4)
    }

    # Create trainer with legacy parameter support
    if verbose:
        print("\n" + "="*60)
        print("Creating ContrastiveTrainer...")
        print("="*60)

    trainer = ContrastiveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        config=trainer_config,
        checkpoint_dir=str(checkpoint_dir),
        device=device,
        warmup_epochs=training_config.get('warmup_epochs', 0),
        warmup_cls_weight=training_config.get('warmup_cls_weight', 0.0),
        finetune_cls_weight=training_config.get('finetune_cls_weight', None),
        selective_unfreeze=training_config.get('selective_unfreeze', None)
    )

    if verbose:
        print("\n" + "="*60)
        print("✓ Contrastive experiment setup complete!")
        print("="*60)
        print("\nReady to train! Use: exp['trainer'].fit(epochs=20)")

    return {
        'model': model,
        'trainer': trainer,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'config': full_config,
        'optimizer': optimizer,
        'device': device
    }


# Re-export useful functions from other modules
__all__ = [
    'setup_data_splits',
    'load_experiment',
    'setup_contrastive_experiment',
    'list_experiments',
    'get_checkpoint_path',
    'PROJECT_ROOT',
    'DATA_DIR',
    'CHECKPOINT_DIR',
    'CONFIG_DIR',
    'CACHE_DIR',
]

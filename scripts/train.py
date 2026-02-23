#!/usr/bin/env python3
"""
Training script for StepMania difficulty classification.

Usage:
    python scripts/train.py --config config/model_config.yaml --data_dir path/to/stepmania/data --audio_dir path/to/audio
    python scripts/train.py --config config/model_config.yaml --data_dir data/ --audio_dir data/ --model_type mlp_baseline
"""

import warnings
import os

# Suppress noisy warnings from audio libraries
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')
os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'

import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import glob
import sys
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import LateFusionClassifier, MLPBaseline, PooledFeatureBaseline
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.training.trainer import Trainer
from src.data.dataset import get_difficulty_class


def generate_checkpoint_subdir(base_dir: str, model_name: str = "classifier") -> str:
    """
    Generate timestamped checkpoint subdirectory.

    Args:
        base_dir: Base checkpoint directory (e.g., "checkpoints")
        model_name: Model type name (default: "classifier")

    Returns:
        Full path to timestamped subdirectory (e.g., "checkpoints/classifier_01_02_14-30")
    """
    timestamp = datetime.now().strftime("%m_%d_%H-%M")
    subdir_name = f"{model_name}_{timestamp}"
    return os.path.join(base_dir, subdir_name)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train StepMania difficulty classifier')

    parser.add_argument('--config', type=str, required=True,
                       help='Path to model config YAML file')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to StepMania data directory')
    parser.add_argument('--audio_dir', type=str, required=True,
                       help='Path to audio files directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Base directory for checkpoints')
    parser.add_argument('--checkpoint_subdir', type=str, default=None,
                       help='Specific checkpoint subdirectory name (auto-generated if not provided)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pretrained checkpoint to load')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze encoder/fusion/backbone when loading pretrained')
    parser.add_argument('--model_type', type=str, default='classifier',
                       choices=['classifier', 'mlp_baseline', 'pooled_baseline'],
                       help='Model type to train (default: classifier)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def extract_difficulty_labels(chart_files):
    """
    Quick pre-parse of chart files to extract difficulty labels for stratification.

    Reads the first difficulty name from each chart file without full parsing.

    Args:
        chart_files: List of chart file paths

    Returns:
        List of difficulty class indices (one per file), or None if extraction fails
    """
    from src.data.stepmania_parser import StepManiaParser
    parser = StepManiaParser()
    labels = []

    for f in chart_files:
        try:
            chart = parser.parse_file(f)
            if chart and chart.note_data:
                # Use the first valid difficulty class found
                label = None
                for nd in chart.note_data:
                    dc = get_difficulty_class(nd.difficulty_name)
                    if dc is not None:
                        label = dc
                        break
                labels.append(label if label is not None else 0)
            else:
                labels.append(0)
        except Exception:
            labels.append(0)

    return labels


def create_data_loaders(data_dir: str, audio_dir: str, config: dict, num_workers: int, data_config: dict = None):
    """Create datasets and data loaders using existing utilities."""
    # Find all chart files
    chart_files = glob.glob(f"{data_dir}/**/*.sm", recursive=True)
    chart_files += glob.glob(f"{data_dir}/**/*.ssc", recursive=True)

    print(f"Found {len(chart_files)} chart files")

    # Extract difficulty labels for stratified splitting
    print("Extracting difficulty labels for stratified splits...")
    stratify_labels = extract_difficulty_labels(chart_files)

    # Create train/val/test splits (stratified by difficulty)
    train_files, val_files, test_files = create_data_splits(
        chart_files, stratify_labels=stratify_labels
    )

    # Extract stepmania config for parser
    stepmania_config = data_config.get('data', {}).get('stepmania', {}) if data_config else None

    # Create datasets using existing utility
    training_config = config.get('training', {})
    max_seq_len = config['classifier']['max_sequence_length']
    cache_dir = training_config.get('cache_dir', 'cache/samples')

    train_dataset, val_dataset, test_dataset = create_datasets(
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
        audio_dir=audio_dir,
        max_sequence_length=max_seq_len,
        data_config=stepmania_config,
        cache_dir=cache_dir
    )

    # Warm cache to precompute all audio features (one-time cost on first run)
    print("\nWarming dataset cache (this may take 5-10 minutes on first run)...")
    print("Training dataset:")
    train_dataset.warm_cache(show_progress=True)
    print("\nValidation dataset:")
    val_dataset.warm_cache(show_progress=True)
    print("Cache warming complete!\n")

    # Create data loaders with optimized settings
    batch_size = config['training']['batch_size']

    # DataLoader optimization settings
    pin_memory = torch.cuda.is_available()
    persistent_workers = training_config.get('persistent_workers', True) and num_workers > 0
    prefetch_factor = training_config.get('prefetch_factor', 4) if num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=True  # Consistent batch sizes for training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )

    return train_loader, val_loader


def create_model(config: dict, model_type: str = 'classifier'):
    """
    Create model from config and model type.

    Args:
        config: Full configuration dictionary
        model_type: One of 'classifier', 'mlp_baseline', 'pooled_baseline'

    Returns:
        Instantiated model
    """
    if model_type == 'classifier':
        return LateFusionClassifier(config['classifier'])
    elif model_type == 'mlp_baseline':
        return MLPBaseline(config.get('baseline', config['classifier']))
    elif model_type == 'pooled_baseline':
        return PooledFeatureBaseline(config.get('baseline', config['classifier']))
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_optimizer(model, config: dict):
    """Create optimizer from config."""
    optimizer_name = config['training']['optimizer'].lower()

    if optimizer_name == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    elif optimizer_name == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate']
        )
    elif optimizer_name == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=0.9,
            weight_decay=config['training']['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def main():
    """Main training function."""
    args = parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Load data configuration
    data_config_path = os.path.join(os.path.dirname(args.config), 'data_config.yaml')
    data_config = load_config(data_config_path) if os.path.exists(data_config_path) else None
    if data_config:
        print(f"Loaded data config from {data_config_path}")

    # MLflow experiment tracking
    try:
        import mlflow
        import mlflow.pytorch
        mlflow_available = True
        mlflow.set_experiment("stepmania-difficulty-classifier")
    except ImportError:
        mlflow_available = False
        print("MLflow not installed, skipping experiment tracking")

    # Create datasets and data loaders
    print("Creating datasets...")
    train_loader, val_loader = create_data_loaders(
        args.data_dir, args.audio_dir, config, args.num_workers, data_config
    )

    # Create model and optimizer
    print(f"Creating model (type: {args.model_type})...")
    if args.pretrained and args.model_type == 'classifier':
        print(f"Loading pretrained weights from {args.pretrained}")
        model = LateFusionClassifier.from_pretrained(
            checkpoint_path=args.pretrained,
            config=config['classifier'],
            freeze_backbone=args.freeze_backbone,
            device='cpu'  # Will be moved to device by trainer
        )
    else:
        model = create_model(config, args.model_type)

    optimizer = create_optimizer(model, config)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Determine checkpoint directory
    if args.checkpoint_subdir:
        checkpoint_dir = os.path.join(args.checkpoint_dir, args.checkpoint_subdir)
    elif args.resume:
        checkpoint_dir = os.path.dirname(args.resume)
    else:
        checkpoint_dir = generate_checkpoint_subdir(args.checkpoint_dir, model_name=args.model_type)

    print(f"Checkpoint directory: {checkpoint_dir}")

    # Override num_epochs from CLI if provided
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
        print(f"Overriding num_epochs to {args.epochs}")

    # Run training (optionally within MLflow context)
    def run_training():
        # Log params to MLflow
        if mlflow_available:
            mlflow.log_params({
                'model_type': args.model_type,
                'seed': args.seed,
                'batch_size': config['training']['batch_size'],
                'learning_rate': config['training']['learning_rate'],
                'optimizer': config['training']['optimizer'],
                'weight_decay': config['training'].get('weight_decay', 0),
                'num_epochs': config['training'].get('num_epochs', 100),
                'num_classes': config['training'].get('num_classes', 4),
                'use_class_weights': config['training'].get('use_class_weights', False),
                'max_sequence_length': config['classifier']['max_sequence_length'],
                'use_amp': config['training'].get('use_amp', True),
                'accumulation_steps': config['training'].get('accumulation_steps', 1),
            })

        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config['training'],
            checkpoint_dir=checkpoint_dir,
            mlflow_logging=mlflow_available,
        )

        # Resume from checkpoint if specified
        if args.resume:
            print(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)

        # Start training
        print("Starting training...")
        history = trainer.fit()

        # Get best val loss from checkpoint callback
        best_val_loss = min(history['val_loss']) if history['val_loss'] else float('inf')

        # Log best model to MLflow
        if mlflow_available:
            best_checkpoint = os.path.join(checkpoint_dir, 'best_val_loss.pt')
            if os.path.exists(best_checkpoint):
                mlflow.pytorch.log_model(model, "best_model")
            mlflow.log_metrics({
                'best_val_loss': best_val_loss,
            })

        print("Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Checkpoints saved to: {checkpoint_dir}")

        return history

    if mlflow_available:
        with mlflow.start_run():
            run_training()
    else:
        run_training()


if __name__ == '__main__':
    main()

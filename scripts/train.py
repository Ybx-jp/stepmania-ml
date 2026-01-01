#!/usr/bin/env python3
"""
Training script for StepMania difficulty classification.

Usage:
    python scripts/train.py --config config/model_config.yaml --data_dir path/to/stepmania/data
"""

import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import glob
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import LateFusionClassifier
from src.utils.data_splits import create_data_splits, create_datasets
from src.training.trainer import Trainer


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
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_data_loaders(data_dir: str, audio_dir: str, config: dict, num_workers: int):
    """Create datasets and data loaders using existing utilities."""
    # Find all chart files
    chart_files = glob.glob(f"{data_dir}/**/*.sm", recursive=True)
    chart_files += glob.glob(f"{data_dir}/**/*.ssc", recursive=True)

    print(f"Found {len(chart_files)} chart files")

    # Create train/val/test splits
    train_files, val_files, test_files = create_data_splits(chart_files)

    # Create datasets using existing utility
    max_seq_len = config['classifier']['max_sequence_length']
    train_dataset, val_dataset, test_dataset = create_datasets(
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
        audio_dir=audio_dir,
        max_sequence_length=max_seq_len
    )

    # Create data loaders
    batch_size = config['training']['batch_size']
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader


def create_model_and_optimizer(config: dict):
    """Create model and optimizer from config."""
    # Create model
    model = LateFusionClassifier(config['classifier'])

    # Create optimizer
    optimizer_name = config['training']['optimizer'].lower()

    if optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate']
        )
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=0.9,
            weight_decay=config['training']['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return model, optimizer


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Create datasets and data loaders
    print("Creating datasets...")
    train_loader, val_loader = create_data_loaders(
        args.data_dir, args.audio_dir, config, args.num_workers
    )

    # Create model and optimizer
    print("Creating model...")
    model, optimizer = create_model_and_optimizer(config)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        config=config['training'],
        checkpoint_dir=args.checkpoint_dir
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Start training
    print("Starting training...")
    history = trainer.fit()

    print("Training completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")


if __name__ == '__main__':
    main()
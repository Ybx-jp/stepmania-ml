#!/usr/bin/env python3
"""
Run the ordinal vs classification 2x2 experiment.

Runs 4 variants: {standard, contrastive} x {classification, ordinal}
with identical hyperparameters, seeds, and data splits.

Usage:
    # Run all 4 variants
    python experiments/ordinal_vs_classification/run_experiment.py \
        --data_dir data/ --audio_dir data/

    # Run a single variant
    python experiments/ordinal_vs_classification/run_experiment.py \
        --data_dir data/ --audio_dir data/ --variant standard_ordinal

    # Override epochs
    python experiments/ordinal_vs_classification/run_experiment.py \
        --data_dir data/ --audio_dir data/ --epochs 10

    # Multi-seed runs for statistical significance
    python experiments/ordinal_vs_classification/run_experiment.py \
        --data_dir data/ --audio_dir data/ --seeds 42,123,456
"""

import warnings
import os

warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')
os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'

import argparse
import yaml
import torch
import torch.optim as optim
import glob
import sys
from pathlib import Path

# Add experiment dir first (so local config.py wins), then project root
EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.insert(1, str(PROJECT_ROOT))

from config import ExperimentConfig, VARIANTS, VariantConfig
from src.models import LateFusionClassifier
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.training import Trainer, ContrastiveTrainer
from src.training.callbacks import CheckpointCallback, LRSchedulerCallback
from src.data import create_contrastive_dataset
from src.data.dataset import get_difficulty_class
from src.data.stepmania_parser import StepManiaParser


def parse_args():
    parser = argparse.ArgumentParser(description='Ordinal vs classification experiment')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to StepMania data directory')
    parser.add_argument('--audio_dir', type=str, required=True,
                        help='Path to audio files directory')
    parser.add_argument('--variant', type=str, default=None,
                        choices=[v.name for v in VARIANTS],
                        help='Run only this variant (default: run all)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (used when --seeds is not set)')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated seeds for multi-seed runs (e.g. 42,123,456)')
    return parser.parse_args()


def extract_difficulty_labels(chart_files):
    """Quick pre-parse for stratification labels."""
    parser = StepManiaParser()
    labels = []
    for f in chart_files:
        try:
            chart = parser.parse_file(f)
            if chart and chart.note_data:
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


def load_model_config():
    """Load the shared model config."""
    config_path = PROJECT_ROOT / "config" / "model_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data_config():
    """Load data config."""
    config_path = PROJECT_ROOT / "config" / "data_config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return None


def setup_data(model_config, data_dir, audio_dir, exp_config):
    """Create shared data splits and datasets (same across all variants)."""
    chart_files = glob.glob(f"{data_dir}/**/*.sm", recursive=True)
    chart_files += glob.glob(f"{data_dir}/**/*.ssc", recursive=True)
    print(f"Found {len(chart_files)} chart files")

    stratify_labels = extract_difficulty_labels(chart_files)

    train_files, val_files, test_files = create_data_splits(
        chart_files, stratify_labels=stratify_labels,
        random_state=exp_config.seed
    )

    data_config = load_data_config()
    stepmania_config = data_config.get('data', {}).get('stepmania', {}) if data_config else None
    max_seq_len = model_config['classifier']['max_sequence_length']

    train_dataset, val_dataset, test_dataset = create_datasets(
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
        audio_dir=audio_dir,
        max_sequence_length=max_seq_len,
        data_config=stepmania_config,
        cache_dir='cache/samples'
    )

    # Warm cache once (shared across variants)
    print("\nWarming dataset cache...")
    train_dataset.warm_cache(show_progress=True)
    val_dataset.warm_cache(show_progress=True)
    print("Cache warming complete!\n")

    return train_dataset, val_dataset, test_dataset


def create_data_loaders(train_dataset, val_dataset, exp_config, is_contrastive=False):
    """Create DataLoaders (contrastive wraps the base dataset)."""
    batch_size = exp_config.batch_size
    num_workers = exp_config.num_workers
    pin_memory = torch.cuda.is_available()

    if is_contrastive:
        contrastive_dataset = create_contrastive_dataset(
            base_dataset=train_dataset,
            positive_percentile=exp_config.positive_percentile,
            negative_percentile=exp_config.negative_percentile,
        )
        train_loader = torch.utils.data.DataLoader(
            contrastive_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def run_variant(variant: VariantConfig, model_config: dict,
                train_dataset, val_dataset, exp_config: ExperimentConfig,
                seed_suffix: str = ""):
    """Run a single experiment variant.

    Args:
        seed_suffix: Optional suffix for checkpoint dir (e.g. "/seed_123")
                     to keep multi-seed checkpoints separate.
    """
    print(f"\n{'='*70}")
    print(f"  VARIANT: {variant.name}  (seed={exp_config.seed})")
    print(f"  head_type={variant.head_type}, contrastive={variant.use_contrastive}")
    print(f"{'='*70}\n")

    # Reset seed for each variant
    set_seed(exp_config.seed)

    # Set head_type and ordinal mode in model config
    classifier_config = dict(model_config['classifier'])
    classifier_config['head_type'] = variant.head_type
    classifier_config['ordinal_multi_output'] = variant.ordinal_multi_output

    # Create model
    model = LateFusionClassifier(classifier_config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dataset, val_dataset, exp_config,
        is_contrastive=variant.use_contrastive
    )

    # Checkpoint directory (append seed suffix for multi-seed runs)
    checkpoint_dir = str(PROJECT_ROOT / exp_config.checkpoint_base / (variant.checkpoint_subdir + seed_suffix))

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=exp_config.learning_rate,
        weight_decay=exp_config.weight_decay,
    )

    # MLflow tracking
    try:
        import mlflow
        mlflow.set_experiment("ordinal-vs-classification")
        mlflow_available = True
    except ImportError:
        mlflow_available = False

    def train():
        if mlflow_available:
            import mlflow
            mlflow.log_params({
                'variant': variant.name,
                'head_type': variant.head_type,
                'use_contrastive': str(variant.use_contrastive),
                'seed': str(exp_config.seed),
                'num_epochs': str(exp_config.num_epochs),
                'batch_size': str(exp_config.batch_size),
                'learning_rate': str(exp_config.learning_rate),
            })

        # Create trainer with val_acc checkpointing (comparable across head types,
        # unlike val_loss which uses different scales for CE vs BCE)
        training_config = (exp_config.to_contrastive_config() if variant.use_contrastive
                          else exp_config.to_training_config())

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=exp_config.scheduler_factor,
            patience=exp_config.scheduler_patience,
        )
        callbacks = [
            CheckpointCallback(
                checkpoint_dir=checkpoint_dir,
                monitor='val_accuracy', mode='max',
                best_checkpoint_name='best_val_loss.pt',  # keep filename for compare.py
            ),
            LRSchedulerCallback(scheduler=scheduler, monitor='val_loss'),
        ]

        if variant.use_contrastive:
            trainer = ContrastiveTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                config=training_config,
                checkpoint_dir=checkpoint_dir,
                callbacks=callbacks,
                mlflow_logging=mlflow_available,
            )
        else:
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                config=training_config,
                checkpoint_dir=checkpoint_dir,
                callbacks=callbacks,
                mlflow_logging=mlflow_available,
            )

        history = trainer.fit(epochs=exp_config.num_epochs)

        best_val_loss = min(history['val_loss']) if history.get('val_loss') else float('inf')
        best_val_acc = max(history['val_acc']) if history.get('val_acc') else 0.0

        print(f"\n  {variant.name} complete:")
        print(f"    Best val_loss: {best_val_loss:.4f}")
        print(f"    Best val_acc:  {best_val_acc:.4f}")
        print(f"    Checkpoint:    {checkpoint_dir}")

        if mlflow_available:
            import mlflow
            mlflow.log_metrics({
                'best_val_loss': best_val_loss,
                'best_val_accuracy': best_val_acc,
            })

        return history

    run_name = f"{variant.experiment_tag}/seed_{exp_config.seed}"
    if mlflow_available:
        import mlflow
        with mlflow.start_run(run_name=run_name):
            return train()
    else:
        return train()


def main():
    args = parse_args()

    # Determine seed list
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
    else:
        seeds = [args.seed]

    model_config = load_model_config()

    # Filter variants if specific one requested
    variants = VARIANTS
    if args.variant:
        variants = [v for v in VARIANTS if v.name == args.variant]

    multi_seed = len(seeds) > 1
    if multi_seed:
        print(f"Multi-seed run: {seeds}")

    for seed_idx, seed in enumerate(seeds):
        exp_config = ExperimentConfig(seed=seed)
        if args.epochs is not None:
            exp_config.num_epochs = args.epochs

        set_seed(seed)

        # Seed suffix for checkpoint dirs (only needed for multi-seed)
        seed_suffix = f"/seed_{seed}" if multi_seed else ""

        if multi_seed:
            print(f"\n{'#'*70}")
            print(f"  SEED {seed_idx+1}/{len(seeds)}: {seed}")
            print(f"{'#'*70}")

        # Re-create data splits per seed (different seed = different splits)
        print("Setting up data...")
        train_dataset, val_dataset, test_dataset = setup_data(
            model_config, args.data_dir, args.audio_dir, exp_config
        )

        # Run each variant
        for variant in variants:
            run_variant(
                variant, model_config,
                train_dataset, val_dataset, exp_config,
                seed_suffix=seed_suffix,
            )

    # Summary
    print(f"\n{'='*70}")
    print("  ALL VARIANTS COMPLETE")
    print(f"{'='*70}")
    if multi_seed:
        print(f"\nSeeds: {seeds}")
        print("Next: run compare.py --seeds ... to aggregate results.")
    else:
        print("\nNext: run compare.py to evaluate and compare all checkpoints.")

    for variant in variants:
        if multi_seed:
            for seed in seeds:
                ckpt = PROJECT_ROOT / exp_config.checkpoint_base / (variant.checkpoint_subdir + f"/seed_{seed}")
                print(f"  {variant.name} (seed={seed}): {ckpt}")
        else:
            ckpt = PROJECT_ROOT / exp_config.checkpoint_base / variant.checkpoint_subdir
            print(f"  {variant.name}: {ckpt}")


if __name__ == '__main__':
    main()

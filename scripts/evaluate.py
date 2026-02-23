#!/usr/bin/env python3
"""
Evaluation script for StepMania difficulty classification.

Loads a trained checkpoint and generates evaluation artifacts:
- Confusion matrix (normalized + raw counts)
- Per-class precision/recall/F1
- Error analysis (adjacent vs distant misclassifications)
- Confidence distribution
- 3-tier metrics (primary: macro F1, secondary: per-class, business: accuracy)

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/classifier_01_03_21-38/best_val_loss.pt \
        --config config/model_config.yaml --data_dir data/ --audio_dir data/

    python scripts/evaluate.py --checkpoint checkpoints/classifier_01_03_21-38/best_val_loss.pt \
        --config config/model_config.yaml --data_dir data/ --audio_dir data/ \
        --baseline_checkpoint checkpoints/mlp_baseline_01_03/best_val_loss.pt
"""

import warnings
import os

warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')
os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'

import argparse
import yaml
import torch
import glob
import sys
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import LateFusionClassifier, MLPBaseline, PooledFeatureBaseline
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.evaluation import compute_metrics, load_and_evaluate, DIFFICULTY_NAMES
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_error_analysis,
    plot_confidence_distribution,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate StepMania difficulty classifier')

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to model config YAML file')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to StepMania data directory')
    parser.add_argument('--audio_dir', type=str, required=True,
                       help='Path to audio files directory')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Directory for evaluation outputs')
    parser.add_argument('--model_type', type=str, default='classifier',
                       choices=['classifier', 'mlp_baseline', 'pooled_baseline'],
                       help='Model type (default: classifier)')
    parser.add_argument('--baseline_checkpoint', type=str, default=None,
                       help='Path to baseline checkpoint for comparison')
    parser.add_argument('--baseline_model_type', type=str, default='mlp_baseline',
                       choices=['classifier', 'mlp_baseline', 'pooled_baseline'],
                       help='Baseline model type (default: mlp_baseline)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (must match training seed for same splits)')

    return parser.parse_args()


def load_model(checkpoint_path, config, model_type='classifier'):
    """Load model from checkpoint."""
    model_config = config.get('baseline', config['classifier'])

    if model_type == 'classifier':
        model = LateFusionClassifier(config['classifier'])
    elif model_type == 'mlp_baseline':
        model = MLPBaseline(model_config)
    elif model_type == 'pooled_baseline':
        model = PooledFeatureBaseline(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})")

    return model


def main():
    args = parse_args()
    set_seed(args.seed)

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load data config
    data_config_path = os.path.join(os.path.dirname(args.config), 'data_config.yaml')
    data_config = None
    if os.path.exists(data_config_path):
        with open(data_config_path, 'r') as f:
            data_config = yaml.safe_load(f)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find chart files and create splits (same seed = same splits as training)
    chart_files = glob.glob(f"{args.data_dir}/**/*.sm", recursive=True)
    chart_files += glob.glob(f"{args.data_dir}/**/*.ssc", recursive=True)
    print(f"Found {len(chart_files)} chart files")

    train_files, val_files, test_files = create_data_splits(chart_files)

    # Create test dataset
    stepmania_config = data_config.get('data', {}).get('stepmania', {}) if data_config else None
    max_seq_len = config['classifier']['max_sequence_length']
    training_config = config.get('training', {})
    cache_dir = training_config.get('cache_dir', 'cache/samples')

    _, _, test_dataset = create_datasets(
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
        audio_dir=args.audio_dir,
        max_sequence_length=max_seq_len,
        data_config=stepmania_config,
        cache_dir=cache_dir,
    )

    print(f"Test set: {len(test_dataset)} samples")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Load and evaluate primary model
    print(f"\nEvaluating {args.model_type} model...")
    model = load_model(args.checkpoint, config, args.model_type)
    metrics, y_true, y_pred, y_probs = load_and_evaluate(model, test_loader)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nPrimary metric (Macro F1): {metrics['primary']['macro_f1']:.4f}")
    print(f"Business metric (Accuracy): {metrics['business']['accuracy']:.4f}")
    print(f"\nPer-class metrics:")
    for name, m in metrics['secondary'].items():
        print(f"  {name:>10s}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")
    print(f"\nError analysis:")
    ea = metrics['error_analysis']
    print(f"  Total errors: {ea['total_errors']}")
    print(f"  Adjacent (+-1 class): {ea['adjacent_errors']}")
    print(f"  Distant (>1 class):   {ea['distant_errors']}")

    # Generate plots
    print("\nGenerating evaluation plots...")

    # Confusion matrix (normalized)
    fig_ax = plot_confusion_matrix(y_true, y_pred, normalize=True,
                                    title='Confusion Matrix (Normalized)')
    fig = fig_ax.get_figure() if hasattr(fig_ax, 'get_figure') else fig_ax
    fig.savefig(output_dir / 'confusion_matrix_normalized.png', dpi=150, bbox_inches='tight')

    # Confusion matrix (raw counts)
    fig_ax = plot_confusion_matrix(y_true, y_pred, normalize=False,
                                    title='Confusion Matrix (Raw Counts)')
    fig = fig_ax.get_figure() if hasattr(fig_ax, 'get_figure') else fig_ax
    fig.savefig(output_dir / 'confusion_matrix_raw.png', dpi=150, bbox_inches='tight')

    # Per-class metrics
    fig = plot_per_class_metrics(y_true, y_pred)
    fig.savefig(output_dir / 'per_class_metrics.png', dpi=150, bbox_inches='tight')

    # Error analysis
    fig = plot_error_analysis(y_true, y_pred)
    fig.savefig(output_dir / 'error_analysis.png', dpi=150, bbox_inches='tight')

    # Confidence distribution
    fig = plot_confidence_distribution(y_probs, y_true, y_pred)
    fig.savefig(output_dir / 'confidence_distribution.png', dpi=150, bbox_inches='tight')

    print(f"Plots saved to {output_dir}/")

    # Baseline comparison
    if args.baseline_checkpoint and os.path.exists(args.baseline_checkpoint):
        print(f"\nComparing to baseline ({args.baseline_model_type})...")
        baseline_model = load_model(args.baseline_checkpoint, config, args.baseline_model_type)
        baseline_metrics, _, _, _ = load_and_evaluate(baseline_model, test_loader)

        bl_f1 = baseline_metrics['primary']['macro_f1']
        model_f1 = metrics['primary']['macro_f1']
        improvement = ((model_f1 - bl_f1) / bl_f1 * 100) if bl_f1 > 0 else float('inf')

        print(f"\n  Baseline Macro F1:  {bl_f1:.4f}")
        print(f"  Model Macro F1:     {model_f1:.4f}")
        print(f"  Improvement:        {improvement:+.1f}%")

    # Log to MLflow if available
    try:
        import mlflow
        mlflow.set_experiment("stepmania-difficulty-classifier")
        with mlflow.start_run(run_name=f"eval_{args.model_type}"):
            mlflow.log_params({
                'model_type': args.model_type,
                'checkpoint': args.checkpoint,
                'test_samples': len(test_dataset),
            })
            mlflow.log_metrics({
                'test_macro_f1': metrics['primary']['macro_f1'],
                'test_accuracy': metrics['business']['accuracy'],
                'test_adjacent_errors': ea['adjacent_errors'],
                'test_distant_errors': ea['distant_errors'],
            })
            # Log plot artifacts
            for png_file in output_dir.glob('*.png'):
                mlflow.log_artifact(str(png_file))
    except ImportError:
        pass

    import matplotlib.pyplot as plt
    plt.close('all')

    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()

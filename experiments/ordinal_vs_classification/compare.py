#!/usr/bin/env python3
"""
Compare all 4 variants of the ordinal vs classification experiment.

Loads best checkpoints, evaluates on the held-out test set, and produces:
- Comparison table with ordinal metrics
- Confusion matrices (2x2 grid)
- Adjacent error rate bar chart
- Go/no-go verdict against the 15% threshold

Usage:
    python experiments/ordinal_vs_classification/compare.py \
        --data_dir data/ --audio_dir data/

    # Custom output directory
    python experiments/ordinal_vs_classification/compare.py \
        --data_dir data/ --audio_dir data/ --output_dir outputs/ordinal_exp
"""

import warnings
import os

warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')
os.environ['AUDIOREAD_LOG_LEVEL'] = 'ERROR'

import argparse
import yaml
import torch
import numpy as np
import glob
import sys
from pathlib import Path

# Add experiment dir first (so local config.py wins), then project root
EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.insert(1, str(PROJECT_ROOT))

from config import ExperimentConfig, VARIANTS
from metrics import compute_ordinal_metrics, format_results_table, DIFFICULTY_NAMES
from src.models import LateFusionClassifier
from src.utils.reproducibility import set_seed
from src.utils.data_splits import create_data_splits, create_datasets
from src.evaluation import load_and_evaluate
from src.data.dataset import get_difficulty_class
from src.data.stepmania_parser import StepManiaParser


def parse_args():
    parser = argparse.ArgumentParser(description='Compare ordinal vs classification results')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--audio_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/ordinal_experiment')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_variant_model(variant, model_config, exp_config):
    """Load best checkpoint for a variant, falling back to last.pt."""
    checkpoint_dir = PROJECT_ROOT / exp_config.checkpoint_base / variant.checkpoint_subdir
    checkpoint_path = checkpoint_dir / "best_val_loss.pt"

    if not checkpoint_path.exists():
        # Fall back to last.pt
        checkpoint_path = checkpoint_dir / "last.pt"

    if not checkpoint_path.exists():
        print(f"  WARNING: No checkpoint found in {checkpoint_dir}")
        return None

    # Set head_type before creating model
    classifier_config = dict(model_config['classifier'])
    classifier_config['head_type'] = variant.head_type

    model = LateFusionClassifier(classifier_config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', '?')
    print(f"  Loaded {variant.name} from epoch {epoch}")
    return model


def extract_difficulty_labels(chart_files):
    """Quick pre-parse for stratification labels (must match run_experiment.py)."""
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


def create_test_loader(model_config, data_dir, audio_dir, seed=42):
    """Create test DataLoader with same stratified splits as training."""
    chart_files = glob.glob(f"{data_dir}/**/*.sm", recursive=True)
    chart_files += glob.glob(f"{data_dir}/**/*.ssc", recursive=True)

    # Must use stratified split identical to run_experiment.py
    stratify_labels = extract_difficulty_labels(chart_files)
    train_files, val_files, test_files = create_data_splits(
        chart_files, stratify_labels=stratify_labels, random_state=seed
    )

    data_config_path = PROJECT_ROOT / "config" / "data_config.yaml"
    data_config = None
    if data_config_path.exists():
        with open(data_config_path, 'r') as f:
            data_config = yaml.safe_load(f)

    stepmania_config = data_config.get('data', {}).get('stepmania', {}) if data_config else None
    max_seq_len = model_config['classifier']['max_sequence_length']

    _, _, test_dataset = create_datasets(
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
        audio_dir=audio_dir,
        max_sequence_length=max_seq_len,
        data_config=stepmania_config,
        cache_dir='cache/samples',
    )

    print(f"Test set: {len(test_dataset)} samples")

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )
    return test_loader


def plot_comparison(all_results, all_y_true, all_y_pred, output_dir):
    """Generate comparison visualizations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_names = list(all_results.keys())

    # --- 1. Confusion matrix grid (2x2) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Confusion Matrices: Ordinal vs Classification', fontsize=14, fontweight='bold')

    # Layout: rows = {standard, contrastive}, cols = {classification, ordinal}
    layout = [
        ('standard_classification', 'Standard + Classification'),
        ('standard_ordinal', 'Standard + Ordinal'),
        ('contrastive_classification', 'Contrastive + Classification'),
        ('contrastive_ordinal', 'Contrastive + Ordinal'),
    ]

    for idx, (vname, title) in enumerate(layout):
        ax = axes[idx // 2, idx % 2]
        if vname in all_y_true:
            cm = confusion_matrix(all_y_true[vname], all_y_pred[vname], normalize='true')
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_xticks(range(4))
            ax.set_yticks(range(4))
            ax.set_xticklabels(DIFFICULTY_NAMES, fontsize=8, rotation=45)
            ax.set_yticklabels(DIFFICULTY_NAMES, fontsize=8)
            for i in range(4):
                for j in range(4):
                    color = 'white' if cm[i, j] > 0.5 else 'black'
                    ax.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center',
                            color=color, fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No checkpoint', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(title, fontsize=11, color='gray')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / 'confusion_matrices_2x2.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- 2. Adjacent misclassification rate bar chart ---
    fig, ax = plt.subplots(figsize=(10, 6))

    names = []
    adj_rates = []
    colors = []
    threshold = 0.15

    for vname in ['standard_classification', 'standard_ordinal',
                  'contrastive_classification', 'contrastive_ordinal']:
        if vname in all_results:
            rate = all_results[vname]['primary']['adjacent_misclass_rate']
            names.append(vname.replace('_', '\n'))
            adj_rates.append(rate)
            colors.append('#2ecc71' if rate < threshold else '#e74c3c')

    bars = ax.bar(names, adj_rates, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold ({threshold:.0%})')
    ax.set_ylabel('Adjacent Misclassification Rate')
    ax.set_title('Adjacent Misclassification Rate by Variant')
    ax.set_ylim(0, max(max(adj_rates) * 1.3, threshold * 1.5) if adj_rates else 0.3)
    ax.legend()

    for bar, rate in zip(bars, adj_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{rate:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    fig.tight_layout()
    fig.savefig(output_dir / 'adjacent_misclass_rate.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- 3. Multi-metric comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Metric Comparison Across Variants', fontsize=13, fontweight='bold')

    metric_groups = [
        ('Accuracy', [all_results[v]['business']['accuracy'] for v in variant_names if v in all_results]),
        ('Macro F1', [all_results[v]['secondary']['macro_f1'] for v in variant_names if v in all_results]),
        ('Mean Abs Error', [all_results[v]['business']['mean_absolute_error'] for v in variant_names if v in all_results]),
    ]
    present_names = [v.replace('_', '\n') for v in variant_names if v in all_results]

    for ax, (metric_name, values) in zip(axes, metric_groups):
        bar_colors = ['#3498db' if 'classification' in n else '#e67e22' for n in present_names]
        bars = ax.bar(present_names, values, color=bar_colors, edgecolor='black', linewidth=0.5)
        ax.set_title(metric_name)
        ax.set_ylim(0, max(values) * 1.2 if values else 1.0)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_dir / 'metric_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Plots saved to {output_dir}/")


def main():
    args = parse_args()
    set_seed(args.seed)
    exp_config = ExperimentConfig(seed=args.seed)

    # Load model config
    config_path = PROJECT_ROOT / "config" / "model_config.yaml"
    with open(config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    # Create test loader
    print("Creating test dataset...")
    test_loader = create_test_loader(model_config, args.data_dir, args.audio_dir, args.seed)

    # Evaluate each variant
    print("\nLoading and evaluating checkpoints...")
    all_results = {}
    all_y_true = {}
    all_y_pred = {}

    for variant in VARIANTS:
        print(f"\n--- {variant.name} ---")
        model = load_variant_model(variant, model_config, exp_config)
        if model is None:
            continue

        _, y_true, y_pred, _ = load_and_evaluate(model, test_loader)
        ordinal_metrics = compute_ordinal_metrics(y_true, y_pred)

        all_results[variant.name] = ordinal_metrics
        all_y_true[variant.name] = y_true
        all_y_pred[variant.name] = y_pred

    if not all_results:
        print("\nNo checkpoints found. Run run_experiment.py first.")
        return

    # Print comparison table
    print("\n" + "=" * 80)
    print("  EXPERIMENT RESULTS: Ordinal vs Classification Head")
    print("=" * 80)
    print(format_results_table(all_results))

    # Hypothesis verdict
    print("\n  HYPOTHESIS VERDICT")
    print("  " + "-" * 40)
    threshold = exp_config.adjacent_misclass_threshold

    for vname, metrics in all_results.items():
        rate = metrics['primary']['adjacent_misclass_rate']
        passed = rate < threshold
        status = "PASS" if passed else "FAIL"
        print(f"  {vname:<35s} {rate:.1%} [{status}]")

    # Compare ordinal vs classification (standard training)
    std_cls = all_results.get('standard_classification')
    std_ord = all_results.get('standard_ordinal')
    if std_cls and std_ord:
        cls_rate = std_cls['primary']['adjacent_misclass_rate']
        ord_rate = std_ord['primary']['adjacent_misclass_rate']
        delta = cls_rate - ord_rate
        print(f"\n  Standard training delta (cls - ord): {delta:+.1%}")
        if delta > 0:
            print(f"  -> Ordinal head REDUCES adjacent errors by {delta:.1%} (absolute)")
        else:
            print(f"  -> Classification head has FEWER adjacent errors by {-delta:.1%}")

    # Compare ordinal vs classification (contrastive training)
    ctr_cls = all_results.get('contrastive_classification')
    ctr_ord = all_results.get('contrastive_ordinal')
    if ctr_cls and ctr_ord:
        cls_rate = ctr_cls['primary']['adjacent_misclass_rate']
        ord_rate = ctr_ord['primary']['adjacent_misclass_rate']
        delta = cls_rate - ord_rate
        print(f"  Contrastive training delta (cls - ord): {delta:+.1%}")
        if delta > 0:
            print(f"  -> Ordinal head REDUCES adjacent errors by {delta:.1%} (absolute)")
        else:
            print(f"  -> Classification head has FEWER adjacent errors by {-delta:.1%}")

    # Generate plots
    print("\nGenerating comparison plots...")
    plot_comparison(all_results, all_y_true, all_y_pred, args.output_dir)

    # Log to MLflow
    try:
        import mlflow
        mlflow.set_experiment("ordinal-vs-classification")
        with mlflow.start_run(run_name="comparison"):
            for vname, metrics in all_results.items():
                mlflow.log_metrics({
                    f'{vname}/adj_misclass_rate': metrics['primary']['adjacent_misclass_rate'],
                    f'{vname}/accuracy': metrics['business']['accuracy'],
                    f'{vname}/macro_f1': metrics['secondary']['macro_f1'],
                    f'{vname}/mae': metrics['business']['mean_absolute_error'],
                })
            for png_file in Path(args.output_dir).glob('*.png'):
                mlflow.log_artifact(str(png_file))
    except ImportError:
        pass

    print("\nComparison complete!")


if __name__ == '__main__':
    main()

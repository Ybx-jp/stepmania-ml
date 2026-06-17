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

    # Multi-seed aggregation (mean +/- std across seeds)
    python experiments/ordinal_vs_classification/compare.py \
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
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (used when --seeds is not set)')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated seeds for multi-seed aggregation (e.g. 42,123,456)')
    return parser.parse_args()


def load_variant_model(variant, model_config, exp_config, seed_suffix=""):
    """Load best checkpoint for a variant, falling back to last.pt."""
    checkpoint_dir = PROJECT_ROOT / exp_config.checkpoint_base / (variant.checkpoint_subdir + seed_suffix)
    checkpoint_path = checkpoint_dir / "best_val_loss.pt"

    if not checkpoint_path.exists():
        # Fall back to last.pt
        checkpoint_path = checkpoint_dir / "last.pt"

    if not checkpoint_path.exists():
        print(f"  WARNING: No checkpoint found in {checkpoint_dir}")
        return None

    # Set head_type and ordinal mode before creating model
    classifier_config = dict(model_config['classifier'])
    classifier_config['head_type'] = variant.head_type
    classifier_config['ordinal_multi_output'] = variant.ordinal_multi_output

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

    # --- 1. Confusion matrix grid (2x3) ---
    # Layout: rows = {standard, contrastive}, cols = {classification, ordinal, ordinal_multi}
    layout = [
        ('standard_classification', 'Standard + Classification'),
        ('standard_ordinal', 'Standard + Ordinal'),
        ('standard_ordinal_multi', 'Standard + Ordinal Multi'),
        ('contrastive_classification', 'Contrastive + Classification'),
        ('contrastive_ordinal', 'Contrastive + Ordinal'),
        ('contrastive_ordinal_multi', 'Contrastive + Ordinal Multi'),
    ]

    # Filter to only variants that have results
    present_layout = [(v, t) for v, t in layout if v in all_y_true]
    n_present = len(present_layout)

    if n_present <= 4:
        nrows, ncols = 2, 2
    else:
        nrows, ncols = 2, 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    fig.suptitle('Confusion Matrices: Ordinal vs Classification', fontsize=14, fontweight='bold')
    axes_flat = axes.flatten()

    for idx, (vname, title) in enumerate(layout):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        if vname in all_y_true:
            cm = confusion_matrix(all_y_true[vname], all_y_pred[vname], normalize='true')
            ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
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

    # Hide unused axes
    for idx in range(len(layout), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / 'confusion_matrices_2x2.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- 2. Adjacent misclassification rate bar chart ---
    fig, ax = plt.subplots(figsize=(10, 6))

    names = []
    adj_rates = []
    colors = []
    threshold = 0.15

    for vname in ['standard_classification', 'standard_ordinal', 'standard_ordinal_multi',
                  'contrastive_classification', 'contrastive_ordinal', 'contrastive_ordinal_multi']:
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


def aggregate_multi_seed(per_seed_results):
    """Aggregate metrics across seeds: compute mean and std.

    Args:
        per_seed_results: dict of seed -> {variant_name -> ordinal_metrics}

    Returns:
        Dict of variant_name -> {metric: {'mean': ..., 'std': ...}}
    """
    from collections import defaultdict

    # Collect metric values per variant across seeds
    variant_metrics = defaultdict(lambda: defaultdict(list))

    for seed, results in per_seed_results.items():
        for vname, metrics in results.items():
            variant_metrics[vname]['adjacent_misclass_rate'].append(
                metrics['primary']['adjacent_misclass_rate'])
            variant_metrics[vname]['accuracy'].append(
                metrics['business']['accuracy'])
            variant_metrics[vname]['macro_f1'].append(
                metrics['secondary']['macro_f1'])
            variant_metrics[vname]['mean_absolute_error'].append(
                metrics['business']['mean_absolute_error'])

    # Compute mean/std
    aggregated = {}
    for vname, metrics_dict in variant_metrics.items():
        aggregated[vname] = {}
        for metric_name, values in metrics_dict.items():
            arr = np.array(values)
            aggregated[vname][metric_name] = {
                'mean': float(arr.mean()),
                'std': float(arr.std()),
                'values': values,
            }

    return aggregated


def format_multi_seed_table(aggregated):
    """Format aggregated multi-seed results as a readable table."""
    header = (
        f"{'Variant':<30s} | {'Adj.Misclass%':>13s} | {'Accuracy':>13s} | "
        f"{'Macro F1':>13s} | {'MAE':>13s}"
    )
    separator = "-" * len(header)

    lines = [separator, header, separator]

    for vname, metrics in aggregated.items():
        adj = metrics['adjacent_misclass_rate']
        acc = metrics['accuracy']
        f1 = metrics['macro_f1']
        mae = metrics['mean_absolute_error']

        lines.append(
            f"{vname:<30s} | "
            f"{adj['mean']:>5.1%} +/- {adj['std']:>4.1%} | "
            f"{acc['mean']:>5.4f} +/- {acc['std']:.4f} | "
            f"{f1['mean']:>5.4f} +/- {f1['std']:.4f} | "
            f"{mae['mean']:>5.3f} +/- {mae['std']:.3f}"
        )

    lines.append(separator)
    return "\n".join(lines)


def run_single_seed_comparison(seed, model_config, args):
    """Evaluate all variants for a single seed. Returns (results, y_true, y_pred)."""
    exp_config = ExperimentConfig(seed=seed)
    set_seed(seed)

    multi_seed = args.seeds is not None
    seed_suffix = f"/seed_{seed}" if multi_seed else ""

    # Create test loader with this seed's splits
    test_loader = create_test_loader(model_config, args.data_dir, args.audio_dir, seed)

    results = {}
    y_true_all = {}
    y_pred_all = {}

    for variant in VARIANTS:
        print(f"  {variant.name} (seed={seed})...")
        model = load_variant_model(variant, model_config, exp_config, seed_suffix)
        if model is None:
            continue

        _, y_true, y_pred, _ = load_and_evaluate(model, test_loader)
        ordinal_metrics = compute_ordinal_metrics(y_true, y_pred)

        results[variant.name] = ordinal_metrics
        y_true_all[variant.name] = y_true
        y_pred_all[variant.name] = y_pred

    return results, y_true_all, y_pred_all


def main():
    args = parse_args()

    # Determine seed list
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
    else:
        seeds = [args.seed]

    multi_seed = len(seeds) > 1

    # Load model config
    config_path = PROJECT_ROOT / "config" / "model_config.yaml"
    with open(config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    print("Loading and evaluating checkpoints...")

    if multi_seed:
        # --- Multi-seed mode: aggregate across seeds ---
        print(f"Multi-seed comparison: {seeds}\n")
        per_seed_results = {}

        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            results, _, _ = run_single_seed_comparison(seed, model_config, args)
            if results:
                per_seed_results[seed] = results

        if not per_seed_results:
            print("\nNo checkpoints found. Run run_experiment.py --seeds ... first.")
            return

        aggregated = aggregate_multi_seed(per_seed_results)

        print("\n" + "=" * 95)
        print(f"  EXPERIMENT RESULTS: Ordinal vs Classification (aggregated over {len(per_seed_results)} seeds)")
        print("=" * 95)
        print(format_multi_seed_table(aggregated))

        # Hypothesis verdict with uncertainty
        exp_config = ExperimentConfig()
        threshold = exp_config.adjacent_misclass_threshold

        print("\n  HYPOTHESIS VERDICT (mean +/- std)")
        print("  " + "-" * 55)
        for vname, metrics in aggregated.items():
            adj = metrics['adjacent_misclass_rate']
            passed = adj['mean'] < threshold
            status = "PASS" if passed else "FAIL"
            print(f"  {vname:<35s} {adj['mean']:.1%} +/- {adj['std']:.1%}  [{status}]")

        # Compare ordinal vs classification
        for prefix, label in [('standard', 'Standard'), ('contrastive', 'Contrastive')]:
            cls_key = f'{prefix}_classification'
            ord_key = f'{prefix}_ordinal'
            if cls_key in aggregated and ord_key in aggregated:
                cls_mean = aggregated[cls_key]['adjacent_misclass_rate']['mean']
                ord_mean = aggregated[ord_key]['adjacent_misclass_rate']['mean']
                delta = cls_mean - ord_mean
                print(f"\n  {label} delta (cls - ord): {delta:+.1%}")
                if delta > 0:
                    print(f"  -> Ordinal head REDUCES adjacent errors by {delta:.1%}")
                else:
                    print(f"  -> Classification head has FEWER adjacent errors by {-delta:.1%}")

        # Use the last seed's predictions for confusion matrix plots
        last_seed = seeds[-1]
        _, last_y_true, last_y_pred = run_single_seed_comparison(last_seed, model_config, args)
        print(f"\nGenerating plots (confusion matrices from seed {last_seed})...")
        # Build single-seed results for plot_comparison
        plot_results = per_seed_results.get(last_seed, {})
        plot_comparison(plot_results, last_y_true, last_y_pred, args.output_dir)

    else:
        # --- Single-seed mode: original behavior ---
        seed = seeds[0]
        print(f"\nCreating test dataset (seed={seed})...")
        all_results, all_y_true, all_y_pred = run_single_seed_comparison(
            seed, model_config, args)

        if not all_results:
            print("\nNo checkpoints found. Run run_experiment.py first.")
            return

        print("\n" + "=" * 80)
        print("  EXPERIMENT RESULTS: Ordinal vs Classification Head")
        print("=" * 80)
        print(format_results_table(all_results))

        # Hypothesis verdict
        exp_config = ExperimentConfig(seed=seed)
        threshold = exp_config.adjacent_misclass_threshold

        print("\n  HYPOTHESIS VERDICT")
        print("  " + "-" * 40)
        for vname, metrics in all_results.items():
            rate = metrics['primary']['adjacent_misclass_rate']
            passed = rate < threshold
            status = "PASS" if passed else "FAIL"
            print(f"  {vname:<35s} {rate:.1%} [{status}]")

        # Compare ordinal vs classification
        for prefix, label in [('standard', 'Standard'), ('contrastive', 'Contrastive')]:
            cls_key = f'{prefix}_classification'
            ord_key = f'{prefix}_ordinal'
            cls_m = all_results.get(cls_key)
            ord_m = all_results.get(ord_key)
            if cls_m and ord_m:
                cls_rate = cls_m['primary']['adjacent_misclass_rate']
                ord_rate = ord_m['primary']['adjacent_misclass_rate']
                delta = cls_rate - ord_rate
                print(f"\n  {label} training delta (cls - ord): {delta:+.1%}")
                if delta > 0:
                    print(f"  -> Ordinal head REDUCES adjacent errors by {delta:.1%} (absolute)")
                else:
                    print(f"  -> Classification head has FEWER adjacent errors by {-delta:.1%}")

        print("\nGenerating comparison plots...")
        plot_comparison(all_results, all_y_true, all_y_pred, args.output_dir)

    # Log to MLflow
    try:
        import mlflow
        mlflow.set_experiment("ordinal-vs-classification")
        run_name = f"comparison-{len(seeds)}seeds" if multi_seed else "comparison"
        with mlflow.start_run(run_name=run_name):
            if multi_seed:
                for vname, metrics in aggregated.items():
                    mlflow.log_metrics({
                        f'{vname}/adj_misclass_mean': metrics['adjacent_misclass_rate']['mean'],
                        f'{vname}/adj_misclass_std': metrics['adjacent_misclass_rate']['std'],
                        f'{vname}/accuracy_mean': metrics['accuracy']['mean'],
                        f'{vname}/accuracy_std': metrics['accuracy']['std'],
                        f'{vname}/macro_f1_mean': metrics['macro_f1']['mean'],
                        f'{vname}/macro_f1_std': metrics['macro_f1']['std'],
                    })
            else:
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

"""
Ordinal-aware evaluation metrics for the head_type comparison experiment.

Key metric: adjacent misclassification rate = off-by-one errors / total samples.
Success criterion: < 15%.
"""

import numpy as np
from typing import Dict
from sklearn.metrics import f1_score, accuracy_score


DIFFICULTY_NAMES = ['Beginner', 'Easy', 'Medium', 'Hard']


def compute_ordinal_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Compute ordinal-aware metrics for difficulty classification.

    Args:
        y_true: True class labels (N,), values 0..3
        y_pred: Predicted class labels (N,), values 0..3

    Returns:
        Dictionary with all metrics organized by tier:
        - primary: adjacent_misclass_rate (hypothesis metric)
        - secondary: macro_f1, per_class_accuracy, confusion breakdown
        - business: accuracy, mean_absolute_error
    """
    n = len(y_true)
    errors = y_pred != y_true
    distances = np.abs(y_pred.astype(int) - y_true.astype(int))

    # Primary: adjacent misclassification rate (off-by-one / total)
    off_by_one = int((distances == 1).sum())
    adjacent_misclass_rate = off_by_one / n if n > 0 else 0.0

    # Error breakdown
    off_by_two_plus = int((distances >= 2).sum())
    total_errors = int(errors.sum())

    # Secondary: standard classification metrics
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    # Per-class accuracy
    per_class_acc = {}
    for cls_idx, name in enumerate(DIFFICULTY_NAMES):
        mask = y_true == cls_idx
        if mask.sum() > 0:
            per_class_acc[name] = float((y_pred[mask] == cls_idx).mean())
        else:
            per_class_acc[name] = 0.0

    # Business: mean absolute error (average class distance)
    mae = float(distances.mean()) if n > 0 else 0.0

    return {
        'primary': {
            'adjacent_misclass_rate': adjacent_misclass_rate,
            'off_by_one_count': off_by_one,
        },
        'secondary': {
            'macro_f1': macro_f1,
            'per_class_accuracy': per_class_acc,
            'total_errors': total_errors,
            'off_by_two_plus': off_by_two_plus,
        },
        'business': {
            'accuracy': accuracy,
            'mean_absolute_error': mae,
        },
    }


def format_results_table(results: Dict[str, Dict]) -> str:
    """
    Format comparison results as a readable table.

    Args:
        results: Dict mapping variant name -> compute_ordinal_metrics() output

    Returns:
        Formatted string table
    """
    header = (
        f"{'Variant':<30s} | {'Adj.Misclass%':>13s} | {'Accuracy':>8s} | "
        f"{'Macro F1':>8s} | {'MAE':>5s} | {'Pass?':>5s}"
    )
    separator = "-" * len(header)

    lines = [separator, header, separator]

    for name, m in results.items():
        adj_rate = m['primary']['adjacent_misclass_rate']
        acc = m['business']['accuracy']
        f1 = m['secondary']['macro_f1']
        mae = m['business']['mean_absolute_error']
        passed = adj_rate < 0.15

        lines.append(
            f"{name:<30s} | {adj_rate:>12.1%} | {acc:>8.4f} | "
            f"{f1:>8.4f} | {mae:>5.3f} | {'YES' if passed else 'NO':>5s}"
        )

    lines.append(separator)
    return "\n".join(lines)

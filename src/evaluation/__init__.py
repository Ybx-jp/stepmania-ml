"""
Evaluation utilities for StepMania difficulty classification.

Provides functions for model evaluation, metrics computation, and
comparison to baselines.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import classification_report, f1_score, accuracy_score
from typing import Dict, Optional, Tuple
from tqdm import tqdm


# Standard difficulty class names
DIFFICULTY_NAMES = ['Beginner', 'Easy', 'Medium', 'Hard']


def compute_metrics(y_true, y_pred, y_probs=None) -> Dict:
    """
    Compute 3-tier evaluation metrics.

    Tier 1 (Primary): Macro F1 - handles class imbalance
    Tier 2 (Secondary): Per-class precision, recall, F1
    Tier 3 (Business): Overall accuracy

    Args:
        y_true: True labels (numpy array)
        y_pred: Predicted labels (numpy array)
        y_probs: Optional prediction probabilities (numpy array, N x num_classes)

    Returns:
        Dictionary with all metrics organized by tier
    """
    # Tier 1: Primary metric
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    # Tier 2: Per-class metrics
    report = classification_report(
        y_true, y_pred,
        target_names=DIFFICULTY_NAMES,
        output_dict=True,
        zero_division=0
    )

    per_class = {}
    for name in DIFFICULTY_NAMES:
        if name in report:
            per_class[name] = {
                'precision': report[name]['precision'],
                'recall': report[name]['recall'],
                'f1': report[name]['f1-score'],
                'support': report[name]['support'],
            }

    # Tier 3: Business metric
    accuracy = accuracy_score(y_true, y_pred)

    # Error analysis: adjacent vs distant
    errors = y_pred != y_true
    if errors.sum() > 0:
        error_distances = np.abs(y_pred[errors].astype(int) - y_true[errors].astype(int))
        adjacent_errors = int((error_distances == 1).sum())
        distant_errors = int((error_distances > 1).sum())
    else:
        adjacent_errors = 0
        distant_errors = 0

    # Confidence stats if probabilities available
    confidence_stats = None
    if y_probs is not None:
        max_probs = y_probs.max(axis=1)
        correct_mask = y_pred == y_true
        confidence_stats = {
            'mean_confidence': float(max_probs.mean()),
            'correct_confidence': float(max_probs[correct_mask].mean()) if correct_mask.sum() > 0 else 0.0,
            'incorrect_confidence': float(max_probs[~correct_mask].mean()) if (~correct_mask).sum() > 0 else 0.0,
        }

    return {
        'primary': {'macro_f1': macro_f1},
        'secondary': per_class,
        'business': {'accuracy': accuracy},
        'error_analysis': {
            'total_errors': int(errors.sum()),
            'adjacent_errors': adjacent_errors,
            'distant_errors': distant_errors,
        },
        'confidence': confidence_stats,
    }


def load_and_evaluate(model: nn.Module,
                      test_loader: DataLoader,
                      device: torch.device = None,
                      use_amp: bool = True) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run single-pass evaluation on a test set.

    Args:
        model: Trained model (already loaded with weights)
        test_loader: Test DataLoader
        device: Device to evaluate on (auto-detected if None)
        use_amp: Whether to use automatic mixed precision

    Returns:
        Tuple of (metrics_dict, y_true, y_pred, y_probs)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            audio = batch['audio'].to(device)
            chart = batch['chart'].to(device)
            mask = batch['mask'].to(device)
            targets = batch['difficulty'].to(device)

            groove_radar = batch.get('groove_radar', None)
            if groove_radar is not None:
                groove_radar = groove_radar.to(device)

            with autocast(enabled=use_amp and device.type == 'cuda'):
                logits = model(audio, chart, mask, groove_radar=groove_radar)

            # Handle dict output (contrastive mode)
            if isinstance(logits, dict):
                logits = logits['logits']

            # Ordinal model: cumulative logits → class predictions & probabilities
            if hasattr(model, 'head_type') and model.head_type == 'ordinal':
                preds = model.predict_class_from_logits(logits)
                # Convert cumulative logits to class probabilities via sigmoid + diff
                cumulative_probs = torch.sigmoid(logits)  # P(Y > k)
                ones = torch.ones(logits.shape[0], 1, device=logits.device)
                zeros = torch.zeros(logits.shape[0], 1, device=logits.device)
                extended = torch.cat([ones, cumulative_probs, zeros], dim=1)
                probs = torch.clamp(extended[:, :-1] - extended[:, 1:], min=1e-8)
            else:
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)

    metrics = compute_metrics(y_true, y_pred, y_probs)

    return metrics, y_true, y_pred, y_probs

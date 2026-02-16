"""
Reusable plotting functions for StepMania chart generator.

Provides consistent visualization across notebooks with minimal boilerplate.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from typing import List, Dict, Optional, Union
import torch


# Difficulty names for labels
DIFFICULTY_NAMES = ['Beginner', 'Easy', 'Medium', 'Hard']


def plot_confusion_matrix(y_true, y_pred,
                          labels: List[str] = None,
                          normalize: bool = True,
                          title: str = 'Confusion Matrix',
                          figsize: tuple = (8, 6),
                          cmap: str = 'Blues',
                          ax=None):
    """
    Plot confusion matrix with optional normalization.

    Args:
        y_true: True labels (numpy array or tensor)
        y_pred: Predicted labels (numpy array or tensor)
        labels: Class labels (default: DIFFICULTY_NAMES)
        normalize: Whether to normalize by row (default: True)
        title: Plot title
        figsize: Figure size (width, height)
        cmap: Colormap name
        ax: Matplotlib axes (creates new figure if None)

    Returns:
        Matplotlib axes object

    Examples:
        >>> plot_confusion_matrix(y_true, y_pred)
        >>> plot_confusion_matrix(y_true, y_pred, normalize=False, title='Raw Counts')
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    if labels is None:
        labels = DIFFICULTY_NAMES

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=labels, yticklabels=labels,
                ax=ax, cbar_kws={'label': 'Proportion' if normalize else 'Count'})

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

    return ax


def plot_training_curves(history: Dict[str, List[float]],
                         metrics: List[str] = None,
                         figsize: tuple = (14, 5),
                         title: str = 'Training History'):
    """
    Plot training and validation curves.

    Args:
        history: Dictionary with keys like 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        metrics: List of metric names to plot (default: ['loss', 'acc'])
        figsize: Figure size (width, height)
        title: Overall title

    Returns:
        Matplotlib figure object

    Examples:
        >>> plot_training_curves(trainer.history)
        >>> plot_training_curves(history, metrics=['loss', 'acc', 'cls_loss'])
    """
    if metrics is None:
        metrics = ['loss', 'acc']

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    epochs = range(1, len(history.get(f'train_{metrics[0]}', [])) + 1)

    for ax, metric in zip(axes, metrics):
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'

        if train_key in history:
            ax.plot(epochs, history[train_key], 'b-', label='Train', linewidth=2)
        if val_key in history:
            ax.plot(epochs, history[val_key], 'r-', label='Validation', linewidth=2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    return fig


def plot_per_class_metrics(y_true, y_pred,
                           labels: List[str] = None,
                           figsize: tuple = (12, 6)):
    """
    Plot per-class precision, recall, and F1 scores.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels (default: DIFFICULTY_NAMES)
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    if labels is None:
        labels = DIFFICULTY_NAMES

    # Get classification report as dict
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)

    # Extract metrics
    metrics_data = {'precision': [], 'recall': [], 'f1-score': []}
    for label in labels:
        if label in report:
            metrics_data['precision'].append(report[label]['precision'])
            metrics_data['recall'].append(report[label]['recall'])
            metrics_data['f1-score'].append(report[label]['f1-score'])

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(labels))
    width = 0.25

    ax.bar(x - width, metrics_data['precision'], width, label='Precision', alpha=0.8)
    ax.bar(x, metrics_data['recall'], width, label='Recall', alpha=0.8)
    ax.bar(x + width, metrics_data['f1-score'], width, label='F1-Score', alpha=0.8)

    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    return fig


def plot_error_analysis(y_true, y_pred,
                       labels: List[str] = None,
                       figsize: tuple = (10, 6)):
    """
    Analyze prediction errors: adjacent vs distant misclassifications.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels (default: DIFFICULTY_NAMES)
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    if labels is None:
        labels = DIFFICULTY_NAMES

    # Compute error distances
    errors = y_pred != y_true
    error_distances = np.abs(y_pred[errors] - y_true[errors])

    # Categorize errors
    adjacent_errors = (error_distances == 1).sum()
    distant_errors = (error_distances > 1).sum()
    total_errors = errors.sum()
    correct = (~errors).sum()

    # Create pie chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Overall accuracy pie
    ax1.pie([correct, total_errors], labels=['Correct', 'Errors'],
            autopct='%1.1f%%', startangle=90, colors=['#2ecc71', '#e74c3c'])
    ax1.set_title(f'Overall Accuracy: {100*correct/(correct+total_errors):.1f}%')

    # Error breakdown pie
    if total_errors > 0:
        ax2.pie([adjacent_errors, distant_errors],
                labels=['Adjacent (±1)', 'Distant (>1)'],
                autopct='%1.1f%%', startangle=90,
                colors=['#f39c12', '#c0392b'])
        ax2.set_title(f'Error Breakdown (n={total_errors})')
    else:
        ax2.text(0.5, 0.5, 'No errors!', ha='center', va='center',
                transform=ax2.transAxes, fontsize=16)
        ax2.set_title('Error Breakdown')

    plt.tight_layout()
    return fig


def plot_confidence_distribution(probs, y_true, y_pred,
                                 labels: List[str] = None,
                                 figsize: tuple = (12, 5)):
    """
    Plot confidence distribution for correct vs incorrect predictions.

    Args:
        probs: Prediction probabilities (N x num_classes)
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels (default: DIFFICULTY_NAMES)
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    # Convert to numpy if needed
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # Get max probabilities (confidence)
    confidences = probs.max(axis=1)

    # Split by correct/incorrect
    correct_mask = y_pred == y_true
    correct_conf = confidences[correct_mask]
    incorrect_conf = confidences[~correct_mask]

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    ax1.hist(correct_conf, bins=30, alpha=0.6, label='Correct', color='green', edgecolor='black')
    ax1.hist(incorrect_conf, bins=30, alpha=0.6, label='Incorrect', color='red', edgecolor='black')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Count')
    ax1.set_title('Confidence Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2.boxplot([correct_conf, incorrect_conf], labels=['Correct', 'Incorrect'])
    ax2.set_ylabel('Confidence')
    ax2.set_title('Confidence by Correctness')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add mean lines
    ax2.axhline(correct_conf.mean(), color='green', linestyle='--', alpha=0.7)
    ax2.axhline(incorrect_conf.mean(), color='red', linestyle='--', alpha=0.7)

    plt.tight_layout()
    return fig


def plot_embedding_space(embeddings, labels,
                         method: str = 'tsne',
                         color_by: str = 'difficulty',
                         labels_names: List[str] = None,
                         figsize: tuple = (10, 8),
                         title: str = None):
    """
    Visualize embedding space using dimensionality reduction.

    Args:
        embeddings: Embedding vectors (N x D)
        labels: Labels for coloring points (N,)
        method: Dimensionality reduction method ('tsne' or 'pca')
        color_by: What to color by ('difficulty' or 'continuous')
        labels_names: Names for discrete labels
        figsize: Figure size
        title: Plot title

    Returns:
        Matplotlib figure object
    """
    # Convert to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    if labels_names is None:
        labels_names = DIFFICULTY_NAMES

    # Dimensionality reduction
    if method.lower() == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) // 3))
        title_method = 't-SNE'
    else:  # pca
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        title_method = 'PCA'

    embeddings_2d = reducer.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    if color_by == 'difficulty':
        # Discrete coloring by difficulty class
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                           c=labels, cmap='viridis', alpha=0.6, s=20)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Difficulty Class')
        if labels_names:
            cbar.set_ticks(range(len(labels_names)))
            cbar.set_ticklabels(labels_names)
    else:
        # Continuous coloring
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                           c=labels, cmap='plasma', alpha=0.6, s=20)
        plt.colorbar(scatter, ax=ax, label=color_by)

    ax.set_xlabel(f'{title_method} Component 1')
    ax.set_ylabel(f'{title_method} Component 2')

    if title is None:
        title = f'Embedding Space ({title_method})'
    ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_triplet_margins(anchor_pos_dists, anchor_neg_dists,
                         target_margin: float = 2.0,
                         figsize: tuple = (14, 5)):
    """
    Plot triplet margin distributions for contrastive learning analysis.

    Args:
        anchor_pos_dists: Distances between anchor and positive (N,)
        anchor_neg_dists: Distances between anchor and negative (N,)
        target_margin: Target margin value
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    # Convert to numpy if needed
    if isinstance(anchor_pos_dists, torch.Tensor):
        anchor_pos_dists = anchor_pos_dists.cpu().numpy()
    if isinstance(anchor_neg_dists, torch.Tensor):
        anchor_neg_dists = anchor_neg_dists.cpu().numpy()

    # Compute margins
    margins = anchor_neg_dists - anchor_pos_dists

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Distance distributions
    ax = axes[0]
    ax.hist(anchor_pos_dists, bins=50, alpha=0.5, label='Anchor-Positive', color='green', edgecolor='black')
    ax.hist(anchor_neg_dists, bins=50, alpha=0.5, label='Anchor-Negative', color='red', edgecolor='black')
    ax.set_xlabel('Embedding Distance')
    ax.set_ylabel('Count')
    ax.set_title('Triplet Distance Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Margin distribution
    ax = axes[1]
    ax.hist(margins, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero margin')
    ax.axvline(target_margin, color='green', linestyle='--', linewidth=2,
               label=f'Target margin ({target_margin})')
    ax.set_xlabel('Margin (AN - AP distance)')
    ax.set_ylabel('Count')
    ax.set_title('Triplet Margin Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add statistics
    satisfied = (margins > 0).sum()
    total = len(margins)
    ax.text(0.02, 0.98, f'Positive margin: {satisfied}/{total} ({100*satisfied/total:.1f}%)',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig

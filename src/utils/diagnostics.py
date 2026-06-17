"""
Training diagnostics for contrastive learning.

Tracks:
- Per-module gradient norms
- Embedding space drift
- Triplet margin statistics
- Loss component evolution
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class TrainingDiagnostics:
    """
    Comprehensive diagnostics tracker for contrastive training.

    Tracks gradient flow, embedding drift, and triplet statistics
    to help debug representation learning issues.
    """

    def __init__(self, save_dir: str, enabled: bool = True):
        """
        Initialize diagnostics tracker.

        Args:
            save_dir: Directory to save diagnostic outputs
            enabled: If False, all methods become no-ops (for production)
        """
        self.enabled = enabled
        if not enabled:
            return

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Time-series tracking
        self.history = defaultdict(list)

        # Embedding checkpoints for drift analysis
        self.embedding_checkpoints = {}

        # Current epoch reference embeddings
        self.reference_embeddings = None
        self.reference_labels = None

        print(f"Training diagnostics enabled, saving to {self.save_dir}")

    def log_gradient_norms(self, epoch: int, norms: Dict[str, float]):
        """
        Log gradient norms for each module.

        Args:
            epoch: Current epoch number
            norms: Dict mapping module name to gradient norm
        """
        if not self.enabled:
            return

        self.history['epoch'].append(epoch)
        for module_name, norm in norms.items():
            key = f'grad_norm_{module_name}'
            self.history[key].append(norm)

    def log_triplet_stats(self, epoch: int,
                         ap_dist: float,
                         an_dist: float,
                         margin: float):
        """
        Log triplet loss statistics.

        Args:
            epoch: Current epoch number
            ap_dist: Mean anchor-positive distance
            an_dist: Mean anchor-negative distance
            margin: Mean margin (an_dist - ap_dist)
        """
        if not self.enabled:
            return

        self.history['triplet_ap_dist'].append(ap_dist)
        self.history['triplet_an_dist'].append(an_dist)
        self.history['triplet_margin'].append(margin)

    def save_embeddings(self, epoch: int,
                       embeddings: np.ndarray,
                       labels: np.ndarray):
        """
        Save embedding checkpoint for drift analysis.

        Args:
            epoch: Current epoch number
            embeddings: Embedding vectors (N, D)
            labels: Class labels (N,)
        """
        if not self.enabled:
            return

        checkpoint_path = self.save_dir / f'embeddings_epoch_{epoch}.npz'
        np.savez_compressed(
            checkpoint_path,
            embeddings=embeddings,
            labels=labels,
            epoch=epoch
        )

        self.embedding_checkpoints[epoch] = checkpoint_path

        # Update reference for drift computation
        if epoch == 0 or self.reference_embeddings is None:
            self.reference_embeddings = embeddings
            self.reference_labels = labels

    def compute_embedding_drift(self, epoch: int,
                               embeddings: np.ndarray,
                               labels: np.ndarray) -> Dict[str, float]:
        """
        Compute embedding drift relative to initial epoch.

        Args:
            epoch: Current epoch number
            embeddings: Current embedding vectors (N, D)
            labels: Class labels (N,)

        Returns:
            Dict with drift metrics:
            - mean_cosine_similarity: Average cosine similarity with epoch 0
            - mean_l2_distance: Average L2 distance from epoch 0
            - per_class_drift: Mean drift per difficulty class
        """
        if not self.enabled or self.reference_embeddings is None:
            return {}

        # Normalize embeddings for cosine similarity
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        reference_norm = self.reference_embeddings / (np.linalg.norm(self.reference_embeddings, axis=1, keepdims=True) + 1e-8)

        # Cosine similarity (higher = less drift)
        cosine_sim = np.sum(embeddings_norm * reference_norm, axis=1)
        mean_cosine_sim = np.mean(cosine_sim)

        # L2 distance (lower = less drift)
        l2_dist = np.linalg.norm(embeddings - self.reference_embeddings, axis=1)
        mean_l2_dist = np.mean(l2_dist)

        # Per-class drift
        per_class_drift = {}
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            if np.sum(mask) > 0:
                class_l2 = np.mean(l2_dist[mask])
                per_class_drift[int(label)] = float(class_l2)

        drift_metrics = {
            'mean_cosine_similarity': float(mean_cosine_sim),
            'mean_l2_distance': float(mean_l2_dist),
            'per_class_drift': per_class_drift
        }

        # Log to history
        self.history['embedding_cosine_sim'].append(mean_cosine_sim)
        self.history['embedding_l2_dist'].append(mean_l2_dist)

        return drift_metrics

    def plot_training_curves(self, output_path: Optional[str] = None):
        """
        Generate comprehensive training curve plots.

        Args:
            output_path: Path to save plot (default: save_dir/training_curves.png)
        """
        if not self.enabled or len(self.history['epoch']) == 0:
            return

        if output_path is None:
            output_path = self.save_dir / 'training_curves.png'

        # Create subplot grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        epochs = self.history['epoch']

        # 1. Gradient norms by module
        ax = axes[0]
        grad_keys = [k for k in self.history.keys() if k.startswith('grad_norm_')]
        for key in grad_keys:
            module_name = key.replace('grad_norm_', '')
            ax.plot(epochs, self.history[key], label=module_name, marker='o', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norms by Module')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # 2. Triplet distances
        ax = axes[1]
        if 'triplet_ap_dist' in self.history:
            ax.plot(epochs, self.history['triplet_ap_dist'],
                   label='Anchor-Positive', marker='o', markersize=3, color='green')
            ax.plot(epochs, self.history['triplet_an_dist'],
                   label='Anchor-Negative', marker='o', markersize=3, color='red')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Distance')
        ax.set_title('Triplet Distances')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Triplet margin
        ax = axes[2]
        if 'triplet_margin' in self.history:
            ax.plot(epochs, self.history['triplet_margin'],
                   marker='o', markersize=3, color='purple')
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero margin')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Margin (AN - AP)')
        ax.set_title('Triplet Margin Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Embedding drift (cosine similarity)
        ax = axes[3]
        if 'embedding_cosine_sim' in self.history:
            ax.plot(epochs, self.history['embedding_cosine_sim'],
                   marker='o', markersize=3, color='blue')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cosine Similarity with Epoch 0')
        ax.set_title('Embedding Drift (Cosine Similarity)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

        # 5. Embedding drift (L2 distance)
        ax = axes[4]
        if 'embedding_l2_dist' in self.history:
            ax.plot(epochs, self.history['embedding_l2_dist'],
                   marker='o', markersize=3, color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('L2 Distance from Epoch 0')
        ax.set_title('Embedding Drift (L2 Distance)')
        ax.grid(True, alpha=0.3)

        # 6. Gradient norm ratios (backbone vs classifier)
        ax = axes[5]
        if 'grad_norm_backbone' in self.history and 'grad_norm_classifier_head' in self.history:
            backbone_norms = np.array(self.history['grad_norm_backbone'])
            classifier_norms = np.array(self.history['grad_norm_classifier_head'])
            ratio = backbone_norms / (classifier_norms + 1e-8)
            ax.plot(epochs, ratio, marker='o', markersize=3, color='purple')
            ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Equal ratio')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Backbone / Classifier Gradient Ratio')
        ax.set_title('Relative Gradient Flow')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {output_path}")
        plt.close()

    def export_summary(self, output_path: Optional[str] = None) -> Dict:
        """
        Export diagnostic summary as JSON.

        Args:
            output_path: Path to save JSON (default: save_dir/diagnostics_summary.json)

        Returns:
            Dict with diagnostic summary
        """
        if not self.enabled:
            return {}

        if output_path is None:
            output_path = self.save_dir / 'diagnostics_summary.json'

        # Compute summary statistics
        summary = {
            'total_epochs': len(self.history.get('epoch', [])),
            'gradient_norms': {},
            'triplet_stats': {},
            'embedding_drift': {}
        }

        # Gradient norm statistics
        grad_keys = [k for k in self.history.keys() if k.startswith('grad_norm_')]
        for key in grad_keys:
            module_name = key.replace('grad_norm_', '')
            values = self.history[key]
            if len(values) > 0:
                summary['gradient_norms'][module_name] = {
                    'initial': float(values[0]),
                    'final': float(values[-1]),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }

        # Triplet statistics
        if 'triplet_margin' in self.history and len(self.history['triplet_margin']) > 0:
            summary['triplet_stats'] = {
                'initial_margin': float(self.history['triplet_margin'][0]),
                'final_margin': float(self.history['triplet_margin'][-1]),
                'mean_margin': float(np.mean(self.history['triplet_margin'])),
                'margin_improvement': float(self.history['triplet_margin'][-1] - self.history['triplet_margin'][0])
            }

        # Embedding drift
        if 'embedding_l2_dist' in self.history and len(self.history['embedding_l2_dist']) > 0:
            summary['embedding_drift'] = {
                'final_cosine_similarity': float(self.history['embedding_cosine_sim'][-1]) if 'embedding_cosine_sim' in self.history else None,
                'final_l2_distance': float(self.history['embedding_l2_dist'][-1]),
                'total_drift': float(self.history['embedding_l2_dist'][-1])
            }

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Saved diagnostics summary to {output_path}")
        return summary

    def log_training_phase(self, epoch: int, phase: str, message: str):
        """
        Log training phase transitions (e.g., warmup → fine-tune).

        Args:
            epoch: Current epoch
            phase: Phase name ('warmup', 'finetune', etc.)
            message: Descriptive message
        """
        if not self.enabled:
            return

        log_path = self.save_dir / 'training_log.txt'
        with open(log_path, 'a') as f:
            f.write(f"Epoch {epoch} - [{phase.upper()}] {message}\n")

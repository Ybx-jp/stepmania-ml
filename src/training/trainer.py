"""
Basic trainer class for StepMania difficulty classification.

Implements minimal training loop with:
- ReduceLROnPlateau scheduler monitoring val_loss
- Simple checkpointing (best_val_loss.pt and last.pt)
- CrossEntropy loss for 0-9 indexed difficulty classes
"""

import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from typing import Dict, Optional
from tqdm import tqdm

from src.models.components.heads import OrdinalRegressionHead
from src.losses.ordinal import encode_ordinal_targets


class Trainer:
    """Minimal trainer for StepMania difficulty classification."""

    def __init__(self,
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 optimizer: torch.optim.Optimizer,
                 config: Dict,
                 checkpoint_dir: str = "checkpoints",
                 device: Optional[torch.device] = None):
        """
        Initialize trainer.

        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer instance
            config: Training configuration
            checkpoint_dir: Full path to checkpoint directory (may include timestamped subdirectory)
            device: Device to use for training (auto-detected if None)
        """
        # Device configuration
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        print(f"Using device: {self.device}")

        self.model = model.to(self.device)  # Move model to device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.config = config
        self.checkpoint_dir = checkpoint_dir

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Detect head type from model
        self.head_type = getattr(model, 'head_type', 'classification')
        self.num_classes = config.get('num_classes')  # Difficulty levels

        # Compute class weights if enabled (before creating criterion)
        class_weights = None
        if config.get('use_class_weights', False) and self.head_type == 'classification':
            class_weights = self._compute_class_weights()
            if class_weights is not None:
                print(f"Using class weights: {class_weights.tolist()}")

        # Loss function - depends on head type
        if self.head_type == 'ordinal':
            self.criterion = nn.BCEWithLogitsLoss()
            print("Using ordinal regression with BCEWithLogitsLoss")
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            print("Using classification with CrossEntropyLoss")

        # Scheduler - ReduceLROnPlateau monitoring val_loss
        self.scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.get('factor', 0.5),
            patience=config.get('patience', 5)        )

        # Checkpointing
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        # Metrics
        self.val_accuracy = MulticlassAccuracy(
            num_classes=10,
            average="micro"
        ).to(self.device)

        self.val_confusion = MulticlassConfusionMatrix(
            num_classes=10
        ).to(self.device)

        # Collect data info from datasets for checkpoint logging
        self.data_info = self._collect_data_info()
        self.chart_stats_summary = self._compute_chart_stats_summary()

    def _compute_chart_stats_summary(self) -> Optional[Dict]:
        """Compute mean chart statistics per difficulty level."""
        # Check if dataset has chart_stats
        sample = self.train_loader.dataset[0]
        if 'chart_stats' not in sample:
            return None

        # Collect stats by difficulty
        from collections import defaultdict
        stats_by_difficulty = defaultdict(list)

        for i in range(len(self.train_loader.dataset)):
            sample = self.train_loader.dataset[i]
            difficulty = sample['difficulty'].item() + 1  # Convert 0-indexed to 1-indexed
            chart_stats = sample['chart_stats'].numpy()
            stats_by_difficulty[difficulty].append(chart_stats)

        # Compute mean per difficulty
        import numpy as np
        summary = {}
        for difficulty in range(1, 11):
            if difficulty in stats_by_difficulty:
                stats_array = np.array(stats_by_difficulty[difficulty])
                summary[difficulty] = {
                    'mean': stats_array.mean(axis=0).tolist(),
                    'std': stats_array.std(axis=0).tolist(),
                    'count': len(stats_array)
                }

        print(f"Computed chart_stats_summary for {len(summary)} difficulty levels")
        return summary

    def _collect_data_info(self) -> Dict:
        """Collect data info from train/val datasets for checkpoint logging."""
        data_info = {}

        # Get info from train dataset if available
        if hasattr(self.train_loader.dataset, 'get_data_info'):
            train_info = self.train_loader.dataset.get_data_info()
            data_info['train'] = {
                'difficulty_distribution': train_info['difficulty_distribution'],
                'total_samples': train_info['total_samples']
            }

        # Get info from val dataset if available
        if hasattr(self.val_loader.dataset, 'get_data_info'):
            val_info = self.val_loader.dataset.get_data_info()
            data_info['val'] = {
                'difficulty_distribution': val_info['difficulty_distribution'],
                'total_samples': val_info['total_samples']
            }

        return data_info

    def _compute_class_weights(self) -> Optional[torch.Tensor]:
        """
        Compute inverse-frequency class weights from training data distribution.

        Uses the formula: weight[c] = total_samples / (num_classes * class_count[c])
        This gives higher weight to underrepresented classes.

        Returns:
            Tensor of shape (num_classes,) with class weights, or None if data unavailable
        """
        if not hasattr(self.train_loader.dataset, 'get_data_info'):
            print("Warning: Dataset doesn't support get_data_info, skipping class weights")
            return None

        data_info = self.train_loader.dataset.get_data_info()
        distribution = data_info.get('difficulty_distribution', {})
        total_samples = data_info.get('total_samples', 0)

        if total_samples == 0:
            return None

        num_classes = 10  # Difficulty levels 1-10 (indexed 0-9)
        weights = []

        for class_idx in range(num_classes):
            # Distribution uses 1-10 keys, but we need 0-9 indices
            count = distribution.get(class_idx + 1, 0)
            if count > 0:
                # Inverse frequency weighting
                weight = total_samples / (num_classes * count)
            else:
                # For classes with no samples, use a reasonable default
                weight = 1.0
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float32).to(self.device)

    def _log_ordinal_score_stats(self):
        """Log ordinal score distribution and error metrics for debugging."""
        self.model.eval()
        with torch.no_grad():
            # Sample one batch
            sample_batch = next(iter(self.train_loader))
            audio = sample_batch['audio'].to(self.device)
            chart = sample_batch['chart'].to(self.device)
            mask = sample_batch['mask'].to(self.device)
            targets = sample_batch['difficulty'].to(self.device)
            chart_stats = sample_batch.get('chart_stats')
            if chart_stats is not None:
                chart_stats = chart_stats.to(self.device)

            # Get pooled features
            reps = self.model.get_feature_representations(audio, chart, mask)
            pooled = reps['pooled_features']

            # Add chart stats if enabled
            if self.model.use_chart_stats and chart_stats is not None:
                stats_features = self.model.stats_mlp(chart_stats)
                pooled = torch.cat([pooled, stats_features], dim=-1)

            # Get score from ordinal head
            score = self.model.classifier_head.get_score(pooled)
            thresholds = self.model.classifier_head.thresholds

            # Get predictions
            logits = self.model.classifier_head(pooled)
            predictions = self.model.predict_class_from_logits(logits)

            # Compute MAE and off-by-1 rate
            errors = (predictions - targets).abs().float()
            mae = errors.mean().item()
            off_by_1 = (errors <= 1).float().mean().item() * 100  # percentage
            median_error = errors.median().item()

            print(f"  Ordinal score: mean={score.mean().item():.3f}, std={score.std().item():.3f}, "
                  f"min={score.min().item():.3f}, max={score.max().item():.3f}")
            print(f"  Thresholds: [{thresholds[0].item():.2f}, ..., {thresholds[-1].item():.2f}]")
            print(f"  MAE (classes): {mae:.3f}, Median: {median_error:.3f}, Off-by-1 or better: {off_by_1:.1f}%")

        self.model.train()

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(self.train_loader, desc=f"Train Epoch {self.current_epoch}")

        for batch in progress_bar:
            # Extract batch data and move to device
            audio = batch['audio'].to(self.device)  # (B, L, 13)
            chart = batch['chart'].to(self.device)  # (B, L, 4)
            mask = batch['mask'].to(self.device)    # (B, L)
            targets = batch['difficulty'].to(self.device)  # (B,) with values 0-9

            # Extract chart_stats if available
            chart_stats = batch.get('chart_stats', None)
            if chart_stats is not None:
                chart_stats = chart_stats.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(audio, chart, mask, chart_stats=chart_stats)

            # Compute loss (different for ordinal vs classification)
            if self.head_type == 'ordinal':
                ordinal_targets = encode_ordinal_targets(
                    targets, self.num_classes
                )
                loss = self.criterion(logits, ordinal_targets)
            else:
                loss = self.criterion(logits, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.get('gradient_clip_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip_norm']
                )

            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            predictions = self.model.predict_class_from_logits(logits)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

            # Update progress bar
            current_acc = correct / total
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{current_acc:.4f}"
            })

        # Debug: log ordinal score distribution once per epoch
        if self.head_type == 'ordinal':
            self._log_ordinal_score_stats()

        return {
            'train_loss': total_loss / len(self.train_loader),
            'train_acc': correct / total
        }

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        self.val_accuracy.reset()
        self.val_confusion.reset()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f"Val Epoch {self.current_epoch}")

            for batch in progress_bar:
                # Extract batch data and move to device
                audio = batch['audio'].to(self.device)
                chart = batch['chart'].to(self.device)
                mask = batch['mask'].to(self.device)
                targets = batch['difficulty'].to(self.device)  # 0-9 indexed

                # Extract chart_stats if available
                chart_stats = batch.get('chart_stats', None)
                if chart_stats is not None:
                    chart_stats = chart_stats.to(self.device)

                # Forward pass
                logits = self.model(audio, chart, mask, chart_stats=chart_stats)

                # Compute loss (different for ordinal vs classification)
                if self.head_type == 'ordinal':
                    ordinal_targets = encode_ordinal_targets(
                        targets, self.num_classes
                    )
                    loss = self.criterion(logits, ordinal_targets)
                else:
                    loss = self.criterion(logits, targets)

                # Track metrics
                total_loss += loss.item()
                predictions = self.model.predict_class_from_logits(logits)
                correct += (predictions == targets).sum().item()
                total += targets.size(0)

                # Update progress bar
                current_acc = correct / total
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{current_acc:.4f}"
                })

                self.val_accuracy.update(predictions, targets)
                self.val_confusion.update(predictions, targets)

            avg_loss = total_loss / len(self.val_loader)

        return {
            'val_loss': avg_loss,
            'val_acc': self.val_accuracy.compute().item(),
            "confusion_matrix": self.val_confusion.compute().cpu()
        }

    def fit(self) -> Dict[str, list]:
        """Main training loop."""
        num_epochs = self.config.get('num_epochs', 100)
        start_epoch = self.current_epoch  # 0 for fresh start, or loaded epoch for resume

        print(f"Starting training for {num_epochs} epochs (from epoch {start_epoch + 1})")
        print(f"Checkpoints will be saved to: {self.checkpoint_dir}")

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch + 1

            # Train epoch
            train_metrics = self.train_epoch()

            # Validation epoch
            val_metrics = self.validate_epoch()

            # Update history
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['train_acc'].append(train_metrics['train_acc'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_acc'].append(val_metrics['val_acc'])

            # Print epoch summary
            print(f"Epoch {self.current_epoch}:")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}, Train Acc: {train_metrics['train_acc']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}, Val Acc: {val_metrics['val_acc']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Update scheduler
            self.scheduler.step(val_metrics['val_loss'])

            # Save best model
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_best_checkpoint(val_metrics)
                print(f"  New best validation loss: {self.best_val_loss:.4f}")

            # Save last checkpoint with history
            self.save_last_checkpoint(epoch, {**train_metrics, **val_metrics}, self.history)

        print("Training completed!")
        return self.history

    def save_best_checkpoint(self, metrics: Dict[str, float]):
        """Save best model checkpoint with history."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'history': self.history,
            'data_info': self.data_info,
            'chart_stats_summary': self.chart_stats_summary
        }

        path = os.path.join(self.checkpoint_dir, 'best_val_loss.pt')
        torch.save(checkpoint, path)

    def save_last_checkpoint(self, epoch: int, metrics: Dict[str, float], history: Dict = None):
        """Save most recent checkpoint with training history."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'history': history,
            'data_info': self.data_info,
            'chart_stats_summary': self.chart_stats_summary
        }

        path = os.path.join(self.checkpoint_dir, 'last.pt')
        torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and resume training."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.current_epoch = checkpoint['epoch']

        # Restore training history if available
        if 'history' in checkpoint and checkpoint['history'] is not None:
            self.history = checkpoint['history']
            print(f"Restored history with {len(self.history['train_loss'])} epochs")

        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        return checkpoint['metrics']
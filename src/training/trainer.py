"""
Basic trainer class for StepMania difficulty classification.

Implements minimal training loop with:
- ReduceLROnPlateau scheduler monitoring val_loss
- Simple checkpointing (best_val_loss.pt and last.pt)
- CrossEntropy loss for difficulty name classes (0-4: Beginner, Easy, Medium, Hard, Challenge)
- Mixed precision training (AMP) for faster training
- Gradient accumulation for larger effective batch sizes
"""

import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from typing import Dict, Optional
from tqdm import tqdm

# Difficulty name constants for display (Challenge folded into Hard)
DIFFICULTY_NAMES = ['Beginner', 'Easy', 'Medium', 'Hard']


class Trainer:
    """Minimal trainer for StepMania difficulty classification."""

    def __init__(self,
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 optimizer: torch.optim.Optimizer,
                 config: Dict,
                 checkpoint_dir: str = "checkpoints",
                 device: Optional[torch.device] = None,
                 use_amp: Optional[bool] = None,
                 accumulation_steps: int = 1):
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
            use_amp: Enable automatic mixed precision (default: True if CUDA available)
            accumulation_steps: Number of batches to accumulate gradients over
        """
        # Device configuration
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        print(f"Using device: {self.device}")

        # Mixed precision training (AMP)
        if use_amp is None:
            use_amp = config.get('use_amp', self.device.type == 'cuda')
        self.use_amp = use_amp and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            print("Using mixed precision training (AMP)")

        # Gradient accumulation
        self.accumulation_steps = config.get('accumulation_steps', accumulation_steps)
        if self.accumulation_steps > 1:
            print(f"Using gradient accumulation: {self.accumulation_steps} steps")

        self.model = model.to(self.device)  # Move model to device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.config = config
        self.checkpoint_dir = checkpoint_dir

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Classification settings
        self.num_classes = config.get('num_classes', 5)  # 5 difficulty name classes
        self.head_type = 'classification'  # Always use classification head

        # Compute class weights if enabled (before creating criterion)
        class_weights = None
        if config.get('use_class_weights', False):
            class_weights = self._compute_class_weights()
            if class_weights is not None:
                print(f"Using class weights: {class_weights.tolist()}")

        # Loss function - CrossEntropy for classification
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using classification with CrossEntropyLoss ({self.num_classes} classes: {DIFFICULTY_NAMES})")

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
            num_classes=self.num_classes,
            average="micro"
        ).to(self.device)

        self.val_confusion = MulticlassConfusionMatrix(
            num_classes=self.num_classes
        ).to(self.device)

        # Collect data info from datasets for checkpoint logging
        self.data_info = self._collect_data_info()
        self.chart_stats_summary = self._compute_chart_stats_summary()

    def _compute_chart_stats_summary(self) -> Optional[Dict]:
        """Compute mean chart statistics per difficulty class (0-4)."""
        # Access pre-computed chart_stats from dataset's valid_samples metadata
        # This avoids calling __getitem__ which would load audio for every sample
        dataset = self.train_loader.dataset
        if not hasattr(dataset, 'valid_samples') or len(dataset.valid_samples) == 0:
            return None

        # Check if chart_stats are available in metadata
        if 'chart_stats' not in dataset.valid_samples[0]:
            return None

        # Collect stats by difficulty class (0-4) from metadata
        from collections import defaultdict
        stats_by_class = defaultdict(list)

        for sample_meta in dataset.valid_samples:
            difficulty_class = sample_meta['difficulty_class']
            chart_stats = sample_meta['chart_stats']
            stats_by_class[difficulty_class].append(chart_stats)

        # Compute mean per difficulty class
        import numpy as np
        summary = {}
        for class_idx in range(self.num_classes):
            if class_idx in stats_by_class:
                stats_array = np.array(stats_by_class[class_idx])
                summary[class_idx] = {
                    'mean': stats_array.mean(axis=0).tolist(),
                    'std': stats_array.std(axis=0).tolist(),
                    'count': len(stats_array)
                }

        print(f"Computed chart_stats_summary for {len(summary)} difficulty classes: {DIFFICULTY_NAMES[:len(summary)]}")
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

        weights = []

        for class_idx in range(self.num_classes):
            # Distribution uses 0-indexed class indices
            count = distribution.get(class_idx, 0)
            if count > 0:
                # Inverse frequency weighting
                weight = total_samples / (self.num_classes * count)
            else:
                # For classes with no samples, use a reasonable default
                weight = 1.0
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float32).to(self.device)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with AMP and gradient accumulation support."""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(self.train_loader, desc=f"Train Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            # Extract batch data and move to device
            audio = batch['audio'].to(self.device)  # (B, L, 13)
            chart = batch['chart'].to(self.device)  # (B, L, 4)
            mask = batch['mask'].to(self.device)    # (B, L)
            targets = batch['difficulty'].to(self.device)  # (B,) with values 0-9

            # Extract chart_stats if available
            chart_stats = batch.get('chart_stats', None)
            if chart_stats is not None:
                chart_stats = chart_stats.to(self.device)

            # Forward pass with AMP
            with autocast(enabled=self.use_amp):
                logits = self.model(audio, chart, mask, chart_stats=chart_stats)
                loss = self.criterion(logits, targets)
                # Scale loss for gradient accumulation
                if self.accumulation_steps > 1:
                    loss = loss / self.accumulation_steps

            # Backward pass with AMP
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step (every accumulation_steps or at end of epoch)
            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                if self.scaler:
                    # Unscale before clipping
                    self.scaler.unscale_(self.optimizer)

                # Gradient clipping
                if self.config.get('gradient_clip_norm'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip_norm']
                    )

                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # Track metrics (use unscaled loss for logging)
            actual_loss = loss.item() * self.accumulation_steps if self.accumulation_steps > 1 else loss.item()
            total_loss += actual_loss
            predictions = self.model.predict_class_from_logits(logits)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

            # Update progress bar
            current_acc = correct / total
            progress_bar.set_postfix({
                'loss': f"{actual_loss:.4f}",
                'acc': f"{current_acc:.4f}"
            })

        return {
            'train_loss': total_loss / len(self.train_loader),
            'train_acc': correct / total
        }

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch with AMP support."""
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

                # Forward pass with AMP
                with autocast(enabled=self.use_amp):
                    logits = self.model(audio, chart, mask, chart_stats=chart_stats)
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
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'history': self.history,
            'data_info': self.data_info,
            'chart_stats_summary': self.chart_stats_summary,
            'use_amp': self.use_amp,
            'accumulation_steps': self.accumulation_steps
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
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'history': history,
            'data_info': self.data_info,
            'chart_stats_summary': self.chart_stats_summary,
            'use_amp': self.use_amp,
            'accumulation_steps': self.accumulation_steps
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

        # Restore scaler state if available and using AMP
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Restore training history if available
        if 'history' in checkpoint and checkpoint['history'] is not None:
            self.history = checkpoint['history']
            print(f"Restored history with {len(self.history['train_loss'])} epochs")

        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        return checkpoint['metrics']
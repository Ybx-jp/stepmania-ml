"""
Basic trainer class for StepMania difficulty classification.

Implements minimal training loop with:
- ReduceLROnPlateau scheduler monitoring val_loss
- Simple checkpointing (best_val_loss.pt and last.pt)
- CrossEntropy loss for difficulty name classes (0-4: Beginner, Easy, Medium, Hard, Challenge)
"""

import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from typing import Dict, Optional
from tqdm import tqdm

# Difficulty name constants for display (Challenge folded into Hard)
DIFFICULTY_NAMES = ['Beginner', 'Easy', 'Medium', 'Hard']
SOURCE_NAMES = ['community', 'official']


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

        # Classification settings
        self.num_classes = config.get('num_classes', 4)  # 4 difficulty name classes
        self.num_source_classes = 2  # community vs official
        self.head_type = 'classification'  # Always use classification head

        # Compute class weights if enabled (before creating criterion)
        difficulty_weights = None
        source_weights = None
        if config.get('use_class_weights', False):
            difficulty_weights = self._compute_class_weights('difficulty')
            source_weights = self._compute_class_weights('source')
            if difficulty_weights is not None:
                print(f"Using difficulty class weights: {difficulty_weights.tolist()}")
            if source_weights is not None:
                print(f"Using source class weights: {source_weights.tolist()}")

        # Loss functions - CrossEntropy for both classification tasks
        self.difficulty_criterion = nn.CrossEntropyLoss(weight=difficulty_weights)
        self.source_criterion = nn.CrossEntropyLoss(weight=source_weights)
        self.source_loss_weight = config.get('source_loss_weight', 1.0)

        print(f"Using 2-head classification:")
        print(f"  Difficulty: CrossEntropyLoss ({self.num_classes} classes: {DIFFICULTY_NAMES})")
        print(f"  Source: CrossEntropyLoss ({self.num_source_classes} classes: {SOURCE_NAMES})")
        print(f"  Loss = L_difficulty + {self.source_loss_weight} * L_source")

        # Scheduler - ReduceLROnPlateau monitoring val_loss
        self.scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.get('factor', 0.5),
            patience=config.get('patience', 5)        )

        # Checkpointing
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        self.history = {
            'train_loss': [], 'train_loss_difficulty': [], 'train_loss_source': [],
            'train_acc_difficulty': [], 'train_acc_source': [],
            'val_loss': [], 'val_loss_difficulty': [], 'val_loss_source': [],
            'val_acc_difficulty': [], 'val_acc_source': []
        }

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

    def _compute_class_weights(self, task: str = 'difficulty') -> Optional[torch.Tensor]:
        """
        Compute inverse-frequency class weights from training data distribution.

        Uses the formula: weight[c] = total_samples / (num_classes * class_count[c])
        This gives higher weight to underrepresented classes.

        Args:
            task: Which task to compute weights for ('difficulty' or 'source')

        Returns:
            Tensor of shape (num_classes,) with class weights, or None if data unavailable
        """
        if not hasattr(self.train_loader.dataset, 'get_data_info'):
            print(f"Warning: Dataset doesn't support get_data_info, skipping {task} class weights")
            return None

        data_info = self.train_loader.dataset.get_data_info()
        total_samples = data_info.get('total_samples', 0)

        if total_samples == 0:
            return None

        # Get distribution and num_classes based on task
        if task == 'difficulty':
            distribution = data_info.get('difficulty_distribution', {})
            num_classes = self.num_classes
        elif task == 'source':
            distribution = data_info.get('source_distribution', {})
            num_classes = self.num_source_classes
        else:
            print(f"Warning: Unknown task '{task}', skipping class weights")
            return None

        weights = []

        for class_idx in range(num_classes):
            # Distribution uses 0-indexed class indices
            count = distribution.get(class_idx, 0)
            if count > 0:
                # Inverse frequency weighting
                weight = total_samples / (num_classes * count)
            else:
                # For classes with no samples, use a reasonable default
                weight = 1.0
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float32).to(self.device)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with dual-head loss."""
        self.model.train()

        total_loss = 0.0
        total_loss_difficulty = 0.0
        total_loss_source = 0.0
        correct_difficulty = 0
        correct_source = 0
        total = 0

        progress_bar = tqdm(self.train_loader, desc=f"Train Epoch {self.current_epoch}")

        for batch in progress_bar:
            # Extract batch data and move to device
            audio = batch['audio'].to(self.device)  # (B, L, 23)
            chart = batch['chart'].to(self.device)  # (B, L, 4)
            mask = batch['mask'].to(self.device)    # (B, L)
            targets_difficulty = batch['difficulty'].to(self.device)  # (B,) with values 0-3
            targets_source = batch['source'].to(self.device)  # (B,) with values 0-1

            # Extract chart_stats if available
            chart_stats = batch.get('chart_stats', None)
            if chart_stats is not None:
                chart_stats = chart_stats.to(self.device)

            # Forward pass - returns dict with 'difficulty' and 'source' logits
            self.optimizer.zero_grad()
            outputs = self.model(audio, chart, mask, chart_stats=chart_stats)

            # Compute dual loss: L = L_difficulty + lambda * L_source
            loss_difficulty = self.difficulty_criterion(outputs['difficulty'], targets_difficulty)
            loss_source = self.source_criterion(outputs['source'], targets_source)
            loss = loss_difficulty + self.source_loss_weight * loss_source

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.get('gradient_clip_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip_norm']
                )

            self.optimizer.step()

            # Track metrics for both tasks
            total_loss += loss.item()
            total_loss_difficulty += loss_difficulty.item()
            total_loss_source += loss_source.item()

            preds_difficulty = self.model.predict_class_from_logits(outputs, task='difficulty')
            preds_source = self.model.predict_class_from_logits(outputs, task='source')

            correct_difficulty += (preds_difficulty == targets_difficulty).sum().item()
            correct_source += (preds_source == targets_source).sum().item()
            total += targets_difficulty.size(0)

            # Update progress bar
            acc_diff = correct_difficulty / total
            acc_src = correct_source / total
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'diff_acc': f"{acc_diff:.4f}",
                'src_acc': f"{acc_src:.4f}"
            })

        num_batches = len(self.train_loader)
        return {
            'train_loss': total_loss / num_batches,
            'train_loss_difficulty': total_loss_difficulty / num_batches,
            'train_loss_source': total_loss_source / num_batches,
            'train_acc_difficulty': correct_difficulty / total,
            'train_acc_source': correct_source / total
        }

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch with dual-head loss."""
        self.model.eval()
        self.val_accuracy.reset()
        self.val_confusion.reset()

        total_loss = 0.0
        total_loss_difficulty = 0.0
        total_loss_source = 0.0
        correct_difficulty = 0
        correct_source = 0
        total = 0

        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f"Val Epoch {self.current_epoch}")

            for batch in progress_bar:
                # Extract batch data and move to device
                audio = batch['audio'].to(self.device)
                chart = batch['chart'].to(self.device)
                mask = batch['mask'].to(self.device)
                targets_difficulty = batch['difficulty'].to(self.device)  # 0-3 indexed
                targets_source = batch['source'].to(self.device)  # 0-1 indexed

                # Extract chart_stats if available
                chart_stats = batch.get('chart_stats', None)
                if chart_stats is not None:
                    chart_stats = chart_stats.to(self.device)

                # Forward pass - returns dict with 'difficulty' and 'source' logits
                outputs = self.model(audio, chart, mask, chart_stats=chart_stats)

                # Compute dual loss: L = L_difficulty + lambda * L_source
                loss_difficulty = self.difficulty_criterion(outputs['difficulty'], targets_difficulty)
                loss_source = self.source_criterion(outputs['source'], targets_source)
                loss = loss_difficulty + self.source_loss_weight * loss_source

                # Track metrics for both tasks
                total_loss += loss.item()
                total_loss_difficulty += loss_difficulty.item()
                total_loss_source += loss_source.item()

                preds_difficulty = self.model.predict_class_from_logits(outputs, task='difficulty')
                preds_source = self.model.predict_class_from_logits(outputs, task='source')

                correct_difficulty += (preds_difficulty == targets_difficulty).sum().item()
                correct_source += (preds_source == targets_source).sum().item()
                total += targets_difficulty.size(0)

                # Update progress bar
                acc_diff = correct_difficulty / total
                acc_src = correct_source / total
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'diff_acc': f"{acc_diff:.4f}",
                    'src_acc': f"{acc_src:.4f}"
                })

                # Update metrics (for difficulty only, to maintain confusion matrix)
                self.val_accuracy.update(preds_difficulty, targets_difficulty)
                self.val_confusion.update(preds_difficulty, targets_difficulty)

        num_batches = len(self.val_loader)
        return {
            'val_loss': total_loss / num_batches,
            'val_loss_difficulty': total_loss_difficulty / num_batches,
            'val_loss_source': total_loss_source / num_batches,
            'val_acc_difficulty': correct_difficulty / total,
            'val_acc_source': correct_source / total,
            'confusion_matrix': self.val_confusion.compute().cpu()
        }

    def fit(self) -> Dict[str, list]:
        """Main training loop with dual-head tracking."""
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

            # Update history for both tasks
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['train_loss_difficulty'].append(train_metrics['train_loss_difficulty'])
            self.history['train_loss_source'].append(train_metrics['train_loss_source'])
            self.history['train_acc_difficulty'].append(train_metrics['train_acc_difficulty'])
            self.history['train_acc_source'].append(train_metrics['train_acc_source'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_loss_difficulty'].append(val_metrics['val_loss_difficulty'])
            self.history['val_loss_source'].append(val_metrics['val_loss_source'])
            self.history['val_acc_difficulty'].append(val_metrics['val_acc_difficulty'])
            self.history['val_acc_source'].append(val_metrics['val_acc_source'])

            # Print epoch summary
            print(f"Epoch {self.current_epoch}:")
            print(f"  Train - Loss: {train_metrics['train_loss']:.4f} (diff: {train_metrics['train_loss_difficulty']:.4f}, src: {train_metrics['train_loss_source']:.4f})")
            print(f"        - Acc:  diff={train_metrics['train_acc_difficulty']:.4f}, src={train_metrics['train_acc_source']:.4f}")
            print(f"  Val   - Loss: {val_metrics['val_loss']:.4f} (diff: {val_metrics['val_loss_difficulty']:.4f}, src: {val_metrics['val_loss_source']:.4f})")
            print(f"        - Acc:  diff={val_metrics['val_acc_difficulty']:.4f}, src={val_metrics['val_acc_source']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Update scheduler (still monitors total val_loss)
            self.scheduler.step(val_metrics['val_loss'])

            # Save best model (based on total val_loss)
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
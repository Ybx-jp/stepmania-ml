"""
Basic trainer class for StepMania difficulty classification.

Implements minimal training loop with:
- CrossEntropy loss for difficulty name classes (Beginner, Easy, Medium, Hard)
- Automatic checkpointing and learning rate scheduling via callbacks
- Mixed precision training (AMP) for faster training (inherited from BaseTrainer)
- Gradient accumulation for larger effective batch sizes (inherited from BaseTrainer)
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from typing import Dict, Optional, List
from tqdm import tqdm

from .base_trainer import BaseTrainer
from .callbacks import CheckpointCallback, LRSchedulerCallback

# Difficulty name constants for display
DIFFICULTY_NAMES = ['Beginner', 'Easy', 'Medium', 'Hard']


class Trainer(BaseTrainer):
    """
    Trainer for StepMania difficulty classification.

    Inherits shared training infrastructure from BaseTrainer and adds
    classification-specific logic.
    """

    def __init__(self,
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 optimizer: torch.optim.Optimizer,
                 config: Dict,
                 checkpoint_dir: str = "checkpoints",
                 device: Optional[torch.device] = None,
                 use_amp: Optional[bool] = None,
                 accumulation_steps: int = 1,
                 callbacks: Optional[List] = None):
        """
        Initialize trainer.

        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer instance
            config: Training configuration
            checkpoint_dir: Full path to checkpoint directory
            device: Device to use for training (auto-detected if None)
            use_amp: Enable automatic mixed precision (default: True if CUDA available)
            accumulation_steps: Number of batches to accumulate gradients over
            callbacks: List of callback objects (if None, creates default callbacks)
        """
        # Get max grad norm from config
        max_grad_norm = config.get('gradient_clip_norm', 1.0)

        # Initialize base trainer
        super().__init__(
            model=model,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp,
            accumulation_steps=accumulation_steps,
            max_grad_norm=max_grad_norm,
            callbacks=callbacks
        )

        # Store data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.checkpoint_dir = checkpoint_dir

        # Classification settings
        self.num_classes = config.get('num_classes', 4)  # 4 difficulty classes
        self.head_type = 'classification'  # Always use classification head

        # Compute class weights if enabled
        class_weights = None
        if config.get('use_class_weights', False):
            class_weights = self._compute_class_weights(train_loader.dataset)
            if class_weights is not None:
                print(f"Using class weights: {class_weights.tolist()}")

        # Loss function - CrossEntropy for classification
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using classification with CrossEntropyLoss ({self.num_classes} classes: {DIFFICULTY_NAMES})")

        # Scheduler - ReduceLROnPlateau monitoring val_loss
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.get('factor', 0.5),
            patience=config.get('patience', 5)
        )

        # Add default callbacks if none provided
        if callbacks is None:
            self.callbacks = [
                CheckpointCallback(checkpoint_dir=checkpoint_dir, monitor='val_loss', mode='min'),
                LRSchedulerCallback(scheduler=scheduler, monitor='val_loss')
            ]
        else:
            # Callbacks provided, just store scheduler for manual use if needed
            self.scheduler = scheduler

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
        self.groove_radar_summary = self._compute_groove_radar_summary()

    def _compute_groove_radar_summary(self) -> Optional[Dict]:
        """Compute mean groove radar values per difficulty class."""
        dataset = self.train_loader.dataset
        if not hasattr(dataset, 'valid_samples') or len(dataset.valid_samples) == 0:
            return None

        # Check if groove_radar is available in metadata
        if 'groove_radar' not in dataset.valid_samples[0]:
            return None

        # Collect groove radar by difficulty class
        from collections import defaultdict
        import numpy as np
        radar_by_class = defaultdict(list)

        for sample_meta in dataset.valid_samples:
            difficulty_class = sample_meta['difficulty_class']
            groove_radar = sample_meta['groove_radar']
            # Convert GrooveRadar object to vector if needed
            if hasattr(groove_radar, 'to_vector'):
                groove_radar = groove_radar.to_vector()
            radar_by_class[difficulty_class].append(groove_radar)

        # Compute mean per difficulty class
        summary = {}
        for class_idx in range(self.num_classes):
            if class_idx in radar_by_class:
                radar_array = np.array(radar_by_class[class_idx])
                summary[class_idx] = {
                    'mean': radar_array.mean(axis=0).tolist(),
                    'std': radar_array.std(axis=0).tolist(),
                    'count': len(radar_array)
                }

        print(f"Computed groove_radar_summary for {len(summary)} difficulty classes")
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

    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch with AMP and gradient accumulation support.

        Args:
            train_loader: Training data loader (provided by BaseTrainer.fit())
            epoch: Current epoch number

        Returns:
            Dictionary with 'loss' and 'accuracy' keys
        """
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Train Epoch {epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            # Extract batch data and move to device
            audio = batch['audio'].to(self.device)
            chart = batch['chart'].to(self.device)
            mask = batch['mask'].to(self.device)
            targets = batch['difficulty'].to(self.device)

            # Extract groove_radar if available
            groove_radar = batch.get('groove_radar', None)
            if groove_radar is not None:
                groove_radar = groove_radar.to(self.device)

            # Forward pass with AMP
            with autocast(enabled=self.use_amp):
                logits = self.model(audio, chart, mask, groove_radar=groove_radar)
                loss = self.criterion(logits, targets)

            # Backward pass (handled by BaseTrainer with gradient accumulation)
            self._optimizer_step(loss, batch_idx)

            # Track metrics
            with torch.no_grad():
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

        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / total
        }

    def validate(self, val_loader) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader (provided by BaseTrainer.fit())

        Returns:
            Dictionary with 'loss', 'accuracy', and 'confusion_matrix' keys
        """
        self.model.eval()
        self.val_accuracy.reset()
        self.val_confusion.reset()

        total_loss = 0.0

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Val Epoch {self.current_epoch}")

            for batch in progress_bar:
                # Extract batch data and move to device
                audio = batch['audio'].to(self.device)
                chart = batch['chart'].to(self.device)
                mask = batch['mask'].to(self.device)
                targets = batch['difficulty'].to(self.device)

                # Extract groove_radar if available
                groove_radar = batch.get('groove_radar', None)
                if groove_radar is not None:
                    groove_radar = groove_radar.to(self.device)

                # Forward pass with AMP
                with autocast(enabled=self.use_amp):
                    logits = self.model(audio, chart, mask, groove_radar=groove_radar)
                    loss = self.criterion(logits, targets)

                # Track metrics
                total_loss += loss.item()
                predictions = self.model.predict_class_from_logits(logits)

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}"
                })

                self.val_accuracy.update(predictions, targets)
                self.val_confusion.update(predictions, targets)

        return {
            'loss': total_loss / len(val_loader),
            'accuracy': self.val_accuracy.compute().item(),
            'confusion_matrix': self.val_confusion.compute().cpu()
        }

    def fit(self, epochs: Optional[int] = None, start_epoch: int = 1) -> Dict[str, list]:
        """
        Main training loop.

        Args:
            epochs: Number of epochs to train (uses config if not specified)
            start_epoch: Starting epoch number (for resuming training)

        Returns:
            Training history dictionary
        """
        if epochs is None:
            epochs = self.config.get('num_epochs', 100)

        print(f"Starting training for {epochs} epochs (from epoch {start_epoch})")
        print(f"Checkpoints will be saved to: {self.checkpoint_dir}")

        # Use BaseTrainer's fit method with callbacks
        return super().fit(self.train_loader, self.val_loader, epochs, start_epoch)

    def save_checkpoint(self, filepath: str, epoch: int, **extra_state):
        """
        Save checkpoint with additional trainer-specific state.

        Extends BaseTrainer's save_checkpoint to include data_info and groove_radar_summary.
        """
        # Add trainer-specific state
        extra_state.update({
            'data_info': self.data_info,
            'groove_radar_summary': self.groove_radar_summary,
        })

        # Call parent method
        super().save_checkpoint(filepath, epoch, **extra_state)

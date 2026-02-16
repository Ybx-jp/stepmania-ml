"""
Base trainer class with shared functionality for all training modes.

This module provides the BaseTrainer abstract class that encapsulates:
- Device and AMP (Automatic Mixed Precision) configuration
- Gradient accumulation and clipping
- Class weight computation from imbalanced datasets
- Checkpoint saving and loading
- Callback system for extensibility
- Training loop orchestration

Subclasses should implement:
- train_epoch(): Training logic for one epoch
- validate(): Validation logic
"""

import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from typing import Dict, List, Optional, Any
from pathlib import Path
from tqdm import tqdm


class BaseTrainer:
    """
    Abstract base trainer with shared functionality for all training modes.

    This class provides common training infrastructure including device management,
    mixed precision training, gradient accumulation, checkpointing, and a callback
    system for extensibility.

    Subclasses must implement:
    - train_epoch(train_loader, epoch): Training logic for one epoch
    - validate(val_loader): Validation logic
    """

    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: Optional[torch.device] = None,
                 use_amp: Optional[bool] = None,
                 accumulation_steps: int = 1,
                 max_grad_norm: float = 1.0,
                 callbacks: Optional[List] = None):
        """
        Initialize base trainer.

        Args:
            model: The model to train
            optimizer: Optimizer instance
            device: Device to use for training (auto-detected if None)
            use_amp: Enable automatic mixed precision (default: True if CUDA available)
            accumulation_steps: Number of batches to accumulate gradients over
            max_grad_norm: Maximum gradient norm for clipping
            callbacks: List of callback objects for extensibility
        """
        # Device configuration
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        print(f"Using device: {self.device}")

        # Mixed precision training (AMP)
        if use_amp is None:
            use_amp = self.device.type == 'cuda'
        self.use_amp = use_amp and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            print("Using mixed precision training (AMP)")

        # Gradient accumulation
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        if self.accumulation_steps > 1:
            print(f"Using gradient accumulation: {self.accumulation_steps} steps")

        # Model and optimizer
        self.model = model.to(self.device)
        self.optimizer = optimizer

        # Callbacks for extensibility
        self.callbacks = callbacks or []

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

        # Current epoch (for callbacks)
        self.current_epoch = 0

    def _compute_class_weights(self, train_dataset) -> torch.Tensor:
        """
        Compute inverse frequency class weights from dataset.

        Helps handle class imbalance by giving higher weights to less frequent classes.

        Args:
            train_dataset: Dataset with .valid_samples attribute containing difficulty_class

        Returns:
            Tensor of class weights normalized to sum to num_classes
        """
        difficulties = [s['difficulty_class'] for s in train_dataset.valid_samples]
        counts = torch.bincount(torch.tensor(difficulties))
        weights = 1.0 / counts.float()
        weights = weights / weights.sum() * len(weights)  # Normalize
        return weights.to(self.device)

    def _optimizer_step(self, loss: torch.Tensor, step_idx: int):
        """
        Perform optimizer step with gradient accumulation and clipping.

        Handles both AMP and non-AMP training modes with proper gradient scaling,
        accumulation, and clipping.

        Args:
            loss: Loss value for the current batch
            step_idx: Current step index (for determining when to step optimizer)
        """
        # Scale loss for gradient accumulation
        loss = loss / self.accumulation_steps

        if self.use_amp:
            # AMP backward pass
            self.scaler.scale(loss).backward()

            # Only step optimizer every accumulation_steps
            if (step_idx + 1) % self.accumulation_steps == 0:
                # Unscale gradients before clipping
                self.scaler.unscale_(self.optimizer)

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # Step optimizer and scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            # Standard backward pass
            loss.backward()

            # Only step optimizer every accumulation_steps
            if (step_idx + 1) % self.accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # Step optimizer
                self.optimizer.step()
                self.optimizer.zero_grad()

    def save_checkpoint(self, filepath: str, epoch: int, **extra_state):
        """
        Save model and optimizer state to checkpoint.

        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch number
            **extra_state: Additional state to save (e.g., scheduler state)
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            **extra_state
        }

        # Ensure parent directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str, load_optimizer: bool = True) -> Dict[str, Any]:
        """
        Load checkpoint and optionally restore optimizer state.

        Args:
            filepath: Path to checkpoint file
            load_optimizer: Whether to restore optimizer state

        Returns:
            Loaded checkpoint dictionary
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Optionally load optimizer state
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore history
        if 'history' in checkpoint:
            self.history = checkpoint['history']

        # Restore epoch
        if 'epoch' in checkpoint:
            self.current_epoch = checkpoint['epoch']

        return checkpoint

    def _call_callbacks(self, event: str, **kwargs):
        """
        Call all callbacks for a given event.

        Args:
            event: Event name (e.g., 'on_epoch_start', 'on_batch_end')
            **kwargs: Arguments to pass to the callback method
        """
        for callback in self.callbacks:
            if hasattr(callback, event):
                getattr(callback, event)(self, **kwargs)

    # Abstract methods to be implemented by subclasses
    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary with training metrics (must include 'loss' and 'accuracy')
        """
        raise NotImplementedError("Subclasses must implement train_epoch()")

    def validate(self, val_loader) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary with validation metrics (must include 'loss' and 'accuracy')
        """
        raise NotImplementedError("Subclasses must implement validate()")

    def fit(self, train_loader, val_loader, epochs: int, start_epoch: int = 1) -> Dict[str, List]:
        """
        Main training loop with callback hooks.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            start_epoch: Starting epoch number (for resuming training)

        Returns:
            Training history dictionary
        """
        # Notify callbacks that training is starting
        self._call_callbacks('on_train_start')

        for epoch in range(start_epoch, start_epoch + epochs):
            self.current_epoch = epoch

            # Notify callbacks that epoch is starting
            self._call_callbacks('on_epoch_start', epoch=epoch)

            # Train and validate
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])

            # Notify callbacks that epoch has ended
            self._call_callbacks('on_epoch_end', epoch=epoch,
                                train_metrics=train_metrics, val_metrics=val_metrics)

        # Notify callbacks that training has ended
        self._call_callbacks('on_train_end')

        return self.history

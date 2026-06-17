"""
Callback system for extensible training.

Provides a callback interface for hooking into training events, allowing
custom behavior to be added without modifying trainer code.

Key callbacks:
- CheckpointCallback: Save best/last checkpoints based on metrics
- WarmupScheduleCallback: Handle warmup phase with module freezing
- DiagnosticsCallback: Track training diagnostics
- LRSchedulerCallback: Handle learning rate scheduling
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from typing import Optional, List, Dict, Any


class Callback:
    """
    Base callback class for training events.

    All callbacks should inherit from this class and override
    the methods for events they want to handle.
    """

    def on_train_start(self, trainer):
        """Called at the start of training."""
        pass

    def on_train_end(self, trainer):
        """Called at the end of training."""
        pass

    def on_epoch_start(self, trainer, epoch: int):
        """
        Called at the start of each epoch.

        Args:
            trainer: The trainer instance
            epoch: Current epoch number
        """
        pass

    def on_epoch_end(self, trainer, epoch: int, train_metrics: Dict[str, float],
                     val_metrics: Dict[str, float]):
        """
        Called at the end of each epoch.

        Args:
            trainer: The trainer instance
            epoch: Current epoch number
            train_metrics: Training metrics for this epoch
            val_metrics: Validation metrics for this epoch
        """
        pass

    def on_batch_start(self, trainer, batch: Any, batch_idx: int):
        """
        Called at the start of each training batch.

        Args:
            trainer: The trainer instance
            batch: The current batch
            batch_idx: Current batch index
        """
        pass

    def on_batch_end(self, trainer, batch: Any, batch_idx: int, loss: float):
        """
        Called at the end of each training batch.

        Args:
            trainer: The trainer instance
            batch: The current batch
            batch_idx: Current batch index
            loss: Loss value for this batch
        """
        pass


class CheckpointCallback(Callback):
    """
    Save best and last checkpoints based on a monitored metric.

    Saves two checkpoints:
    - last.pt: Always saved after each epoch
    - best_val_loss.pt (or custom name): Saved when monitored metric improves
    """

    def __init__(self,
                 checkpoint_dir: str,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 best_checkpoint_name: str = 'best_val_loss.pt',
                 save_last: bool = True):
        """
        Initialize checkpoint callback.

        Args:
            checkpoint_dir: Directory to save checkpoints
            monitor: Metric to monitor (e.g., 'val_loss', 'val_acc')
            mode: 'min' or 'max' - whether lower or higher metric is better
            best_checkpoint_name: Filename for best checkpoint
            save_last: Whether to save last checkpoint after each epoch
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.mode = mode
        self.best_checkpoint_name = best_checkpoint_name
        self.save_last = save_last

        # Initialize best value
        self.best_value = float('inf') if mode == 'min' else float('-inf')

        print(f"CheckpointCallback: Monitoring '{monitor}' ({'minimize' if mode == 'min' else 'maximize'})")
        print(f"  Checkpoints will be saved to: {self.checkpoint_dir}")

    def on_epoch_end(self, trainer, epoch, train_metrics, val_metrics):
        """Save checkpoints after each epoch."""
        # Save last checkpoint
        if self.save_last:
            last_path = self.checkpoint_dir / 'last.pt'
            trainer.save_checkpoint(str(last_path), epoch)

        # Get monitored metric value
        metric_key = self.monitor.replace('val_', '').replace('train_', '')
        if 'val_' in self.monitor:
            value = val_metrics.get(metric_key)
        else:
            value = train_metrics.get(metric_key)

        if value is None:
            print(f"Warning: Monitored metric '{self.monitor}' not found in metrics")
            return

        # Check if this is the best value
        is_better = (value < self.best_value) if self.mode == 'min' else (value > self.best_value)

        if is_better:
            self.best_value = value
            best_path = self.checkpoint_dir / self.best_checkpoint_name
            trainer.save_checkpoint(str(best_path), epoch)
            print(f"  New best {self.monitor}: {value:.4f} (saved to {self.best_checkpoint_name})")


class WarmupScheduleCallback(Callback):
    """
    Handle warmup phase with module freezing and loss weight transitions.

    Useful for contrastive learning experiments where you want to:
    1. Freeze certain modules during warmup (e.g., classifier_head)
    2. Use different loss weights during warmup vs fine-tuning
    3. Automatically transition after N epochs
    """

    def __init__(self,
                 warmup_epochs: int,
                 freeze_modules: Optional[List[str]] = None,
                 warmup_loss_weights: Optional[Dict[str, float]] = None,
                 finetune_loss_weights: Optional[Dict[str, float]] = None):
        """
        Initialize warmup schedule callback.

        Args:
            warmup_epochs: Number of epochs to run warmup phase
            freeze_modules: List of module names to freeze during warmup
            warmup_loss_weights: Loss weights during warmup (e.g., {'classification': 0.0, 'contrastive': 1.0})
            finetune_loss_weights: Loss weights after warmup (e.g., {'classification': 0.3, 'contrastive': 2.0})
        """
        self.warmup_epochs = warmup_epochs
        self.freeze_modules = freeze_modules or []
        self.warmup_weights = warmup_loss_weights or {}
        self.finetune_weights = finetune_loss_weights or {}
        self.is_warmup = True
        self.frozen_params = []

        if warmup_epochs > 0:
            print(f"WarmupScheduleCallback: {warmup_epochs} warmup epochs")
            if freeze_modules:
                print(f"  Will freeze modules: {freeze_modules}")
            if warmup_loss_weights:
                print(f"  Warmup loss weights: {warmup_loss_weights}")
            if finetune_loss_weights:
                print(f"  Fine-tune loss weights: {finetune_loss_weights}")

    def on_train_start(self, trainer):
        """Freeze specified modules at the start of training."""
        if self.warmup_epochs > 0 and self.freeze_modules:
            print(f"\nFreezing modules for warmup: {self.freeze_modules}")

            for name, module in trainer.model.named_children():
                if name in self.freeze_modules:
                    # Freeze all parameters in this module
                    for param in module.parameters():
                        param.requires_grad = False
                        self.frozen_params.append((name, param))

            # Count trainable parameters
            trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in trainer.model.parameters())
            print(f"  Trainable parameters: {trainable:,} / {total:,}")

            # Set warmup loss weights
            if self.warmup_weights:
                for key, value in self.warmup_weights.items():
                    weight_attr = f'{key}_weight'
                    if hasattr(trainer, weight_attr):
                        setattr(trainer, weight_attr, value)

    def on_epoch_start(self, trainer, epoch):
        """Transition from warmup to fine-tuning after N epochs."""
        if epoch == self.warmup_epochs + 1 and self.is_warmup:
            print(f"\n{'='*60}")
            print(f"Transitioning from warmup to fine-tuning at epoch {epoch}")
            print(f"{'='*60}")

            self.is_warmup = False

            # Unfreeze modules
            if self.freeze_modules:
                print(f"Unfreezing modules: {self.freeze_modules}")
                for name, module in trainer.model.named_children():
                    if name in self.freeze_modules:
                        for param in module.parameters():
                            param.requires_grad = True

                # Count trainable parameters
                trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in trainer.model.parameters())
                print(f"  Trainable parameters: {trainable:,} / {total:,}")

            # Update loss weights for fine-tuning
            if self.finetune_weights:
                print(f"Updating loss weights: {self.finetune_weights}")
                for key, value in self.finetune_weights.items():
                    weight_attr = f'{key}_weight'
                    if hasattr(trainer, weight_attr):
                        setattr(trainer, weight_attr, value)
                        print(f"  {weight_attr}: {value}")


class DiagnosticsCallback(Callback):
    """
    Track training diagnostics like gradient norms and embedding drift.

    Integrates with TrainingDiagnostics utility to log and plot diagnostic information.
    """

    def __init__(self,
                 log_dir: str,
                 log_every_n_epochs: int = 5,
                 track_gradients: bool = True,
                 track_embeddings: bool = True):
        """
        Initialize diagnostics callback.

        Args:
            log_dir: Directory to save diagnostic plots and data
            log_every_n_epochs: How often to save diagnostic plots
            track_gradients: Whether to track gradient norms
            track_embeddings: Whether to track embedding drift
        """
        from ..utils.diagnostics import TrainingDiagnostics

        self.diagnostics = TrainingDiagnostics(log_dir)
        self.log_every_n_epochs = log_every_n_epochs
        self.track_gradients = track_gradients
        self.track_embeddings = track_embeddings

        print(f"DiagnosticsCallback: Saving diagnostics to {log_dir}")
        print(f"  Tracking gradients: {track_gradients}")
        print(f"  Tracking embeddings: {track_embeddings}")
        print(f"  Saving plots every {log_every_n_epochs} epochs")

    def on_batch_end(self, trainer, batch, batch_idx, loss):
        """Log gradients after each batch."""
        if self.track_gradients:
            self.diagnostics.log_gradients(trainer.model)

    def on_epoch_end(self, trainer, epoch, train_metrics, val_metrics):
        """Save diagnostic plots every N epochs."""
        if epoch % self.log_every_n_epochs == 0:
            self.diagnostics.plot_training_curves()
            print(f"  Saved diagnostic plots")

    def on_train_end(self, trainer):
        """Save final diagnostics and generate summary."""
        self.diagnostics.plot_training_curves()
        self.diagnostics.save_summary()
        print(f"\nFinal diagnostics saved to {self.diagnostics.log_dir}")


class LRSchedulerCallback(Callback):
    """
    Handle learning rate scheduling.

    Supports both step-based schedulers (e.g., StepLR) and metric-based
    schedulers (e.g., ReduceLROnPlateau).
    """

    def __init__(self,
                 scheduler,
                 monitor: str = 'val_loss',
                 step_on_batch: bool = False):
        """
        Initialize LR scheduler callback.

        Args:
            scheduler: PyTorch learning rate scheduler instance
            monitor: Metric to monitor (for ReduceLROnPlateau)
            step_on_batch: Whether to step scheduler after each batch (for OneCycleLR, etc.)
        """
        self.scheduler = scheduler
        self.monitor = monitor
        self.step_on_batch = step_on_batch

        scheduler_name = scheduler.__class__.__name__
        print(f"LRSchedulerCallback: Using {scheduler_name}")
        if isinstance(scheduler, ReduceLROnPlateau):
            print(f"  Monitoring: {monitor}")

    def on_batch_end(self, trainer, batch, batch_idx, loss):
        """Step scheduler after each batch if needed."""
        if self.step_on_batch:
            self.scheduler.step()

    def on_epoch_end(self, trainer, epoch, train_metrics, val_metrics):
        """Step scheduler after each epoch."""
        if not self.step_on_batch:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                # Need metric value for ReduceLROnPlateau
                metric_key = self.monitor.replace('val_', '').replace('train_', '')
                if 'val_' in self.monitor:
                    metric = val_metrics.get(metric_key)
                else:
                    metric = train_metrics.get(metric_key)

                if metric is not None:
                    self.scheduler.step(metric)
                else:
                    print(f"Warning: Monitored metric '{self.monitor}' not found")
            else:
                # Regular step for epoch-based schedulers
                self.scheduler.step()

            # Log current learning rate
            current_lr = self.scheduler.optimizer.param_groups[0]['lr']
            print(f"  LR: {current_lr:.6f}")


class SelectiveUnfreezeCallback(Callback):
    """
    Selectively freeze/unfreeze modules during training.

    Useful for fine-tuning where you want to train only certain parts of the model.
    This is an alternative to WarmupScheduleCallback for Experiment B style training.
    """

    def __init__(self, trainable_modules: List[str]):
        """
        Initialize selective unfreeze callback.

        Args:
            trainable_modules: List of module names to keep trainable (all others frozen)
        """
        self.trainable_modules = trainable_modules

        print(f"SelectiveUnfreezeCallback: Only training modules: {trainable_modules}")

    def on_train_start(self, trainer):
        """Freeze all modules except those in trainable_modules list."""
        # First, freeze everything
        for param in trainer.model.parameters():
            param.requires_grad = False

        # Then unfreeze specified modules
        for name, module in trainer.model.named_children():
            if name in self.trainable_modules:
                for param in module.parameters():
                    param.requires_grad = True

        # Count trainable parameters
        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in trainer.model.parameters())
        print(f"  Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

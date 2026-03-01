"""
Contrastive trainer for multi-task learning with groove radar similarity.

Handles:
- Triplet batches (anchor, positive, negative)
- Multi-task loss: classification + contrastive
- Groove radar-based adaptive margins
- Mixed precision training (AMP) inherited from BaseTrainer
- Gradient accumulation inherited from BaseTrainer
- Extensibility via callbacks (warmup, diagnostics, etc.)
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from typing import Dict, Optional, List
from tqdm import tqdm

from .base_trainer import BaseTrainer
from .callbacks import (CheckpointCallback, LRSchedulerCallback,
                        WarmupScheduleCallback, SelectiveUnfreezeCallback,
                        DiagnosticsCallback)
from ..losses.contrastive import create_contrastive_loss
from ..losses.ordinal import encode_ordinal_targets
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Difficulty name constants for display
DIFFICULTY_NAMES = ['Beginner', 'Easy', 'Medium', 'Hard']


class ContrastiveTrainer(BaseTrainer):
    """
    Trainer for multi-task learning with contrastive loss.

    Inherits shared training infrastructure from BaseTrainer and adds
    contrastive learning with triplet batches.

    Handles triplet batches and combines:
    - Difficulty classification loss (CrossEntropy on anchor)
    - Contrastive loss (TripletMarginLoss with groove radar margins)
    """

    def __init__(self,
                 model: nn.Module,
                 train_loader,  # ContrastiveTripletDataset loader
                 val_loader,    # Standard StepManiaDataset loader for validation
                 optimizer: torch.optim.Optimizer,
                 config: Dict,
                 checkpoint_dir: str = "checkpoints",
                 device: Optional[torch.device] = None,
                 use_amp: Optional[bool] = None,
                 accumulation_steps: int = 1,
                 callbacks: Optional[List] = None,
                 # Legacy parameters for backwards compatibility
                 warmup_epochs: int = 0,
                 warmup_cls_weight: float = 0.0,
                 finetune_cls_weight: Optional[float] = None,
                 selective_unfreeze: Optional[List[str]] = None):
        """
        Initialize contrastive trainer.

        Args:
            model: Model with projection head enabled
            train_loader: DataLoader for ContrastiveTripletDataset (triplets)
            val_loader: DataLoader for StepManiaDataset (single samples for validation)
            optimizer: Optimizer instance
            config: Training configuration with contrastive settings
            checkpoint_dir: Path to checkpoint directory
            device: Device to use for training
            use_amp: Enable automatic mixed precision (default: True if CUDA available)
            accumulation_steps: Number of batches to accumulate gradients over
            callbacks: List of callback objects (if None, creates default callbacks)
            warmup_epochs: (Legacy) Number of epochs for warmup phase
            warmup_cls_weight: (Legacy) Classification weight during warmup
            finetune_cls_weight: (Legacy) Classification weight after warmup
            selective_unfreeze: (Legacy) Module names to unfreeze
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
        self.num_classes = config.get('num_classes', 4)
        self.head_type = getattr(model, 'head_type', 'classification')
        self._encode_ordinal_targets = encode_ordinal_targets

        # Access base dataset through contrastive wrapper
        dataset = train_loader.dataset
        if hasattr(dataset, 'base_dataset'):
            dataset = dataset.base_dataset

        if self.head_type == 'ordinal':
            # Ordinal regression: BCEWithLogitsLoss on cumulative logits
            # No pos_weight — asymmetric weighting causes middle-class collapse
            self.classification_criterion = nn.BCEWithLogitsLoss()
            print(f"Using ordinal regression with BCEWithLogitsLoss ({self.num_classes} classes: {DIFFICULTY_NAMES})")
        else:
            # Standard classification: CrossEntropyLoss
            class_weights = None
            if config.get('use_class_weights', False):
                class_weights = self._compute_class_weights(dataset)
                if class_weights is not None:
                    print(f"Using class weights: {class_weights.tolist()}")
            self.classification_criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Contrastive loss
        contrastive_type = config.get('contrastive_loss', 'triplet_radar')
        self.contrastive_criterion = create_contrastive_loss(
            loss_type=contrastive_type,
            base_margin=config.get('triplet_margin', 1.0),
            margin_scale=config.get('margin_scale', 0.5),
            temperature=config.get('infonce_temperature', 0.07)
        )

        # Loss weights
        self.classification_weight = config.get('classification_weight', 1.0)
        self.contrastive_weight = config.get('contrastive_weight', 0.5)

        print(f"Using multi-task loss: {self.classification_weight}*classification + "
              f"{self.contrastive_weight}*contrastive ({contrastive_type})")

        # Scheduler
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.get('factor', 0.5),
            patience=config.get('patience', 5)
        )

        # Create callbacks if none provided (including legacy parameter support)
        if callbacks is None:
            callbacks = [
                CheckpointCallback(checkpoint_dir=checkpoint_dir, monitor='val_loss', mode='min'),
                LRSchedulerCallback(scheduler=scheduler, monitor='val_loss')
            ]

            # Handle legacy warmup parameters
            if warmup_epochs > 0:
                if finetune_cls_weight is None:
                    finetune_cls_weight = self.classification_weight

                warmup_callback = WarmupScheduleCallback(
                    warmup_epochs=warmup_epochs,
                    freeze_modules=['classifier_head'],
                    warmup_loss_weights={
                        'classification': warmup_cls_weight,
                        'contrastive': self.contrastive_weight
                    },
                    finetune_loss_weights={
                        'classification': finetune_cls_weight,
                        'contrastive': self.contrastive_weight
                    }
                )
                callbacks.append(warmup_callback)

            # Handle legacy selective unfreeze
            if selective_unfreeze:
                selective_callback = SelectiveUnfreezeCallback(trainable_modules=selective_unfreeze)
                callbacks.append(selective_callback)

            # Add diagnostics callback if enabled in config
            diagnostics_config = config.get('diagnostics', {})
            if diagnostics_config.get('enabled', False):
                diagnostics_callback = DiagnosticsCallback(
                    log_dir=f"{checkpoint_dir}/diagnostics",
                    log_every_n_epochs=diagnostics_config.get('save_embeddings_every', 5),
                    track_gradients=diagnostics_config.get('log_gradients', False),
                    track_embeddings=diagnostics_config.get('track_embeddings', False)
                )
                callbacks.append(diagnostics_callback)

            self.callbacks = callbacks
        else:
            self.scheduler = scheduler

        # Metrics
        self.val_accuracy = MulticlassAccuracy(
            num_classes=self.num_classes,
            average="micro"
        ).to(self.device)

        self.val_confusion = MulticlassConfusionMatrix(
            num_classes=self.num_classes
        ).to(self.device)

        # Update history to track component losses
        self.history['train_cls_loss'] = []
        self.history['train_contrastive_loss'] = []

    def _compute_ordinal_pos_weight(self, dataset) -> Optional[torch.Tensor]:
        """
        Compute per-threshold pos_weight for BCEWithLogitsLoss from class distribution.

        For threshold k: pos_weight[k] = count(label <= k) / count(label > k)
        """
        try:
            if hasattr(dataset, 'valid_samples'):
                labels = [s['difficulty_class'] for s in dataset.valid_samples]
            else:
                return None

            import numpy as np
            labels = np.array(labels)
            num_thresholds = self.num_classes - 1
            pos_weights = []

            for k in range(num_thresholds):
                n_positive = (labels > k).sum()
                n_negative = (labels <= k).sum()
                if n_positive > 0:
                    pos_weights.append(n_negative / n_positive)
                else:
                    pos_weights.append(1.0)

            return torch.tensor(pos_weights, dtype=torch.float32)
        except Exception as e:
            print(f"Warning: Could not compute ordinal pos_weight: {e}")
            return None

    def _extract_triplet_batch(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Extract and organize triplet batch data."""
        result = {}

        # Anchor data
        result['anchor_audio'] = batch['anchor_audio'].to(self.device)
        result['anchor_chart'] = batch['anchor_chart'].to(self.device)
        result['anchor_mask'] = batch['anchor_mask'].to(self.device)
        result['anchor_difficulty'] = batch['anchor_difficulty'].to(self.device)
        result['anchor_groove_radar'] = batch['anchor_groove_radar'].to(self.device)

        # Positive data
        result['positive_audio'] = batch['positive_audio'].to(self.device)
        result['positive_chart'] = batch['positive_chart'].to(self.device)
        result['positive_mask'] = batch['positive_mask'].to(self.device)
        result['positive_groove_radar'] = batch['positive_groove_radar'].to(self.device)

        # Negative data
        result['negative_audio'] = batch['negative_audio'].to(self.device)
        result['negative_chart'] = batch['negative_chart'].to(self.device)
        result['negative_mask'] = batch['negative_mask'].to(self.device)
        result['negative_groove_radar'] = batch['negative_groove_radar'].to(self.device)

        return result

    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch with multi-task loss, AMP, and gradient accumulation.

        Args:
            train_loader: Training data loader (provided by BaseTrainer.fit())
            epoch: Current epoch number

        Returns:
            Dictionary with 'loss', 'accuracy', and component losses
        """
        self.model.train()

        total_loss = 0.0
        total_cls_loss = 0.0
        total_contrastive_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Train Epoch {epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            # Extract triplet batch
            data = self._extract_triplet_batch(batch)

            # Forward pass for all three samples with AMP
            with autocast(enabled=self.use_amp):
                anchor_out = self.model(
                    data['anchor_audio'], data['anchor_chart'], data['anchor_mask'],
                    groove_radar=data['anchor_groove_radar'], return_embeddings=True
                )
                positive_out = self.model(
                    data['positive_audio'], data['positive_chart'], data['positive_mask'],
                    groove_radar=data['positive_groove_radar'], return_embeddings=True
                )
                negative_out = self.model(
                    data['negative_audio'], data['negative_chart'], data['negative_mask'],
                    groove_radar=data['negative_groove_radar'], return_embeddings=True
                )

                # Classification loss (on anchor only)
                if self.head_type == 'ordinal':
                    ordinal_targets = self._encode_ordinal_targets(
                        data['anchor_difficulty'], self.num_classes
                    )
                    cls_loss = self.classification_criterion(
                        anchor_out['logits'], ordinal_targets
                    )
                else:
                    cls_loss = self.classification_criterion(
                        anchor_out['logits'], data['anchor_difficulty']
                    )

                # Contrastive loss
                contrastive_loss = self.contrastive_criterion(
                    anchor_out['embeddings'],
                    positive_out['embeddings'],
                    negative_out['embeddings'],
                    data['anchor_groove_radar'],
                    data['negative_groove_radar']
                )

                # Combined loss
                loss = (self.classification_weight * cls_loss +
                        self.contrastive_weight * contrastive_loss)

            # Backward pass (handled by BaseTrainer with gradient accumulation)
            self._optimizer_step(loss, batch_idx)

            # Track metrics
            with torch.no_grad():
                total_loss += loss.item()
                total_cls_loss += cls_loss.item()
                total_contrastive_loss += contrastive_loss.item()

                predictions = self.model.predict_class_from_logits(anchor_out['logits'])
                correct += (predictions == data['anchor_difficulty']).sum().item()
                total += data['anchor_difficulty'].size(0)

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'cls': f"{cls_loss.item():.4f}",
                'ctr': f"{contrastive_loss.item():.4f}",
                'acc': f"{correct/total:.4f}"
            })

        n_batches = len(train_loader)
        metrics = {
            'loss': total_loss / n_batches,
            'accuracy': correct / total,
            'cls_loss': total_cls_loss / n_batches,
            'contrastive_loss': total_contrastive_loss / n_batches,
        }

        # Update component loss history
        self.history['train_cls_loss'].append(metrics['cls_loss'])
        self.history['train_contrastive_loss'].append(metrics['contrastive_loss'])

        return metrics

    def validate(self, val_loader) -> Dict[str, float]:
        """
        Validate the model (classification only, no triplets).

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
                # Standard single-sample batch
                audio = batch['audio'].to(self.device)
                chart = batch['chart'].to(self.device)
                mask = batch['mask'].to(self.device)
                targets = batch['difficulty'].to(self.device)
                groove_radar = batch.get('groove_radar')
                if groove_radar is not None:
                    groove_radar = groove_radar.to(self.device)

                # Forward pass with AMP (classification only)
                with autocast(enabled=self.use_amp):
                    output = self.model(audio, chart, mask, groove_radar=groove_radar)

                    # Handle dict output (model may return embeddings)
                    if isinstance(output, dict):
                        logits = output['logits']
                    else:
                        logits = output

                    # Compute loss
                    if self.head_type == 'ordinal':
                        ordinal_targets = self._encode_ordinal_targets(targets, self.num_classes)
                        loss = self.classification_criterion(logits, ordinal_targets)
                    else:
                        loss = self.classification_criterion(logits, targets)

                # Track metrics
                total_loss += loss.item()
                predictions = self.model.predict_class_from_logits(logits)

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

        print(f"Starting contrastive training for {epochs} epochs (from epoch {start_epoch})")
        print(f"Checkpoints will be saved to: {self.checkpoint_dir}")

        # Use BaseTrainer's fit method with callbacks
        return super().fit(self.train_loader, self.val_loader, epochs, start_epoch)

    def save_checkpoint(self, filepath: str, epoch: int, **extra_state):
        """
        Save checkpoint with additional contrastive trainer-specific state.

        Extends BaseTrainer's save_checkpoint to include contrastive config.
        """
        # Add trainer-specific state
        extra_state.update({
            'config': self.config,
            'classification_weight': self.classification_weight,
            'contrastive_weight': self.contrastive_weight,
        })

        # Call parent method
        super().save_checkpoint(filepath, epoch, **extra_state)

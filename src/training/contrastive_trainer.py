"""
Contrastive trainer for multi-task learning with groove radar similarity.

Extends the base Trainer to handle:
- Triplet batches (anchor, positive, negative)
- Multi-task loss: classification + contrastive
- Groove radar-based adaptive margins
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

from ..losses.contrastive import TripletMarginLossWithRadar, create_contrastive_loss


# Difficulty name constants for display
DIFFICULTY_NAMES = ['Beginner', 'Easy', 'Medium', 'Hard']


class ContrastiveTrainer:
    """
    Trainer for multi-task learning with contrastive loss.

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
                 accumulation_steps: int = 1):
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

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.config = config
        self.checkpoint_dir = checkpoint_dir

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Classification settings
        self.num_classes = config.get('num_classes', 4)

        # Compute class weights if enabled
        class_weights = None
        if config.get('use_class_weights', False):
            class_weights = self._compute_class_weights()
            if class_weights is not None:
                print(f"Using class weights: {class_weights.tolist()}")

        # Classification loss
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
        self.scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.get('factor', 0.5),
            patience=config.get('patience', 5)
        )

        # Checkpointing
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        self.history = {
            'train_loss': [], 'train_cls_loss': [], 'train_contrastive_loss': [],
            'train_acc': [], 'val_loss': [], 'val_acc': []
        }

        # Metrics
        self.val_accuracy = MulticlassAccuracy(
            num_classes=self.num_classes,
            average="micro"
        ).to(self.device)

        self.val_confusion = MulticlassConfusionMatrix(
            num_classes=self.num_classes
        ).to(self.device)

    def _compute_class_weights(self) -> Optional[torch.Tensor]:
        """Compute inverse-frequency class weights from training data."""
        # Access base dataset through contrastive wrapper
        dataset = self.train_loader.dataset
        if hasattr(dataset, 'base_dataset'):
            dataset = dataset.base_dataset

        if not hasattr(dataset, 'get_data_info'):
            return None

        data_info = dataset.get_data_info()
        distribution = data_info.get('difficulty_distribution', {})
        total_samples = data_info.get('total_samples', 0)

        if total_samples == 0:
            return None

        weights = []
        for class_idx in range(self.num_classes):
            count = distribution.get(class_idx, 0)
            if count > 0:
                weight = total_samples / (self.num_classes * count)
            else:
                weight = 1.0
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float32).to(self.device)

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

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with multi-task loss, AMP, and gradient accumulation."""
        self.model.train()

        total_loss = 0.0
        total_cls_loss = 0.0
        total_contrastive_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(self.train_loader, desc=f"Train Epoch {self.current_epoch}")

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
            scale = self.accumulation_steps if self.accumulation_steps > 1 else 1
            actual_loss = loss.item() * scale
            actual_cls_loss = cls_loss.item()
            actual_contrastive_loss = contrastive_loss.item()

            total_loss += actual_loss
            total_cls_loss += actual_cls_loss
            total_contrastive_loss += actual_contrastive_loss

            predictions = self.model.predict_class_from_logits(anchor_out['logits'])
            correct += (predictions == data['anchor_difficulty']).sum().item()
            total += data['anchor_difficulty'].size(0)

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{actual_loss:.4f}",
                'cls': f"{actual_cls_loss:.4f}",
                'ctr': f"{actual_contrastive_loss:.4f}",
                'acc': f"{correct/total:.4f}"
            })

        n_batches = len(self.train_loader)
        return {
            'train_loss': total_loss / n_batches,
            'train_cls_loss': total_cls_loss / n_batches,
            'train_contrastive_loss': total_contrastive_loss / n_batches,
            'train_acc': correct / total
        }

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch (classification only, no triplets) with AMP."""
        self.model.eval()
        self.val_accuracy.reset()
        self.val_confusion.reset()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f"Val Epoch {self.current_epoch}")

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
                    logits = self.model(audio, chart, mask, groove_radar=groove_radar)

                    # Handle dict output
                    if isinstance(logits, dict):
                        logits = logits['logits']

                    # Compute loss
                    loss = self.classification_criterion(logits, targets)

                # Track metrics
                total_loss += loss.item()
                predictions = self.model.predict_class_from_logits(logits)
                correct += (predictions == targets).sum().item()
                total += targets.size(0)

                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{correct/total:.4f}"
                })

                self.val_accuracy.update(predictions, targets)
                self.val_confusion.update(predictions, targets)

        return {
            'val_loss': total_loss / len(self.val_loader),
            'val_acc': self.val_accuracy.compute().item(),
            'confusion_matrix': self.val_confusion.compute().cpu()
        }

    def fit(self) -> Dict[str, list]:
        """Main training loop."""
        num_epochs = self.config.get('num_epochs', 100)
        start_epoch = self.current_epoch

        print(f"Starting contrastive training for {num_epochs} epochs (from epoch {start_epoch + 1})")
        print(f"Checkpoints will be saved to: {self.checkpoint_dir}")

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch + 1

            # Resample triplets if configured
            if hasattr(self.train_loader.dataset, 'resample'):
                if self.train_loader.dataset.resample_epoch and epoch > start_epoch:
                    self.train_loader.dataset.resample()

            # Train epoch
            train_metrics = self.train_epoch()

            # Validation epoch
            val_metrics = self.validate_epoch()

            # Update history
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['train_cls_loss'].append(train_metrics['train_cls_loss'])
            self.history['train_contrastive_loss'].append(train_metrics['train_contrastive_loss'])
            self.history['train_acc'].append(train_metrics['train_acc'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_acc'].append(val_metrics['val_acc'])

            # Print epoch summary
            print(f"Epoch {self.current_epoch}:")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f} "
                  f"(cls: {train_metrics['train_cls_loss']:.4f}, "
                  f"ctr: {train_metrics['train_contrastive_loss']:.4f})")
            print(f"  Train Acc: {train_metrics['train_acc']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}, Val Acc: {val_metrics['val_acc']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Update scheduler
            self.scheduler.step(val_metrics['val_loss'])

            # Save best model
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint('best_val_loss.pt', {**train_metrics, **val_metrics})
                print(f"  New best validation loss: {self.best_val_loss:.4f}")

            # Save last checkpoint
            self.save_checkpoint('last.pt', {**train_metrics, **val_metrics})

        print("Contrastive training completed!")
        return self.history

    def save_checkpoint(self, filename: str, metrics: Dict[str, float]):
        """Save checkpoint with scaler state for AMP."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'history': self.history,
            'config': self.config,
            'use_amp': self.use_amp,
            'accumulation_steps': self.accumulation_steps
        }

        path = os.path.join(self.checkpoint_dir, filename)
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

        if 'history' in checkpoint and checkpoint['history'] is not None:
            self.history = checkpoint['history']
            print(f"Restored history with {len(self.history['train_loss'])} epochs")

        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        return checkpoint.get('metrics', {})

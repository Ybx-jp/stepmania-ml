"""
Experiment configuration for ordinal vs classification head comparison.

6-way comparison: {standard, contrastive} x {classification, ordinal, ordinal_multi}

Hypothesis: switching head_type from classification to ordinal will improve
adjacent-class accuracy because difficulty levels are ordered.

Success criterion: adjacent misclassification rate < 15% of total samples.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class VariantConfig:
    """Configuration for a single experiment variant."""
    name: str
    head_type: str  # "classification" or "ordinal"
    use_contrastive: bool
    experiment_tag: str  # MLflow run name
    ordinal_multi_output: bool = False  # Use multi-output ordinal head

    @property
    def checkpoint_subdir(self) -> str:
        return f"ordinal_exp/{self.name}"


# The 6 experiment variants
VARIANTS: List[VariantConfig] = [
    VariantConfig(
        name="standard_classification",
        head_type="classification",
        use_contrastive=False,
        experiment_tag="ordinal-exp/std-cls",
    ),
    VariantConfig(
        name="standard_ordinal",
        head_type="ordinal",
        use_contrastive=False,
        experiment_tag="ordinal-exp/std-ord",
    ),
    VariantConfig(
        name="standard_ordinal_multi",
        head_type="ordinal",
        use_contrastive=False,
        ordinal_multi_output=True,
        experiment_tag="ordinal-exp/std-ord-multi",
    ),
    VariantConfig(
        name="contrastive_classification",
        head_type="classification",
        use_contrastive=True,
        experiment_tag="ordinal-exp/ctr-cls",
    ),
    VariantConfig(
        name="contrastive_ordinal",
        head_type="ordinal",
        use_contrastive=True,
        experiment_tag="ordinal-exp/ctr-ord",
    ),
    VariantConfig(
        name="contrastive_ordinal_multi",
        head_type="ordinal",
        use_contrastive=True,
        ordinal_multi_output=True,
        experiment_tag="ordinal-exp/ctr-ord-multi",
    ),
]


@dataclass
class ExperimentConfig:
    """Shared experiment settings (identical across all 4 variants)."""

    # Reproducibility
    seed: int = 42

    # Training (locked across variants for fair comparison)
    num_epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    optimizer: str = "adamw"
    use_class_weights: bool = True
    use_amp: bool = True
    accumulation_steps: int = 2
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 5
    num_workers: int = 4

    # Scheduler
    scheduler: str = "reduce_on_plateau"
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5

    # Contrastive settings (only used when use_contrastive=True)
    contrastive_loss: str = "triplet_radar"
    triplet_margin: float = 1.0
    margin_scale: float = 0.5
    classification_weight: float = 0.8
    contrastive_weight: float = 1.0
    positive_percentile: float = 20.0
    negative_percentile: float = 80.0
    same_difficulty_only: bool = True

    # Paths (relative to project root)
    config_path: str = "config/model_config.yaml"
    data_config_path: str = "config/data_config.yaml"
    checkpoint_base: str = "checkpoints"

    # Success criterion
    adjacent_misclass_threshold: float = 0.15  # 15% of total samples

    def to_training_config(self) -> Dict:
        """Convert to training config dict compatible with Trainer/ContrastiveTrainer."""
        return {
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'optimizer': self.optimizer,
            'use_class_weights': self.use_class_weights,
            'use_amp': self.use_amp,
            'accumulation_steps': self.accumulation_steps,
            'gradient_clip_norm': self.gradient_clip_norm,
            'early_stopping_patience': self.early_stopping_patience,
            'num_workers': self.num_workers,
            'num_classes': 4,
            'patience': self.scheduler_patience,
            'factor': self.scheduler_factor,
            'cache_dir': 'cache/samples',
        }

    def to_contrastive_config(self) -> Dict:
        """Convert to contrastive config dict."""
        config = self.to_training_config()
        config.update({
            'contrastive_loss': self.contrastive_loss,
            'triplet_margin': self.triplet_margin,
            'margin_scale': self.margin_scale,
            'classification_weight': self.classification_weight,
            'contrastive_weight': self.contrastive_weight,
        })
        return config

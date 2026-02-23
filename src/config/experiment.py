"""
Experiment configuration dataclass.

Consolidates all settings from YAML configs into a typed dataclass
with validation and MLflow integration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

import yaml


@dataclass
class ExperimentConfig:
    """
    Consolidated experiment configuration.

    Groups all hyperparameters and settings needed for a training run
    into a single validated object.
    """

    # Experiment identity
    experiment_name: str = "stepmania-difficulty-classifier"
    model_type: str = "classifier"  # classifier, mlp_baseline, pooled_baseline
    seed: int = 42

    # Model architecture
    audio_features_dim: int = 23
    chart_sequence_dim: int = 4
    max_sequence_length: int = 1440
    fusion_dim: int = 256
    num_classes: int = 4
    backbone_blocks: int = 4
    pooling_type: str = "mean_max"
    fusion_type: str = "late"
    classifier_hidden_dim: int = 64
    classifier_dropout: float = 0.2
    backbone_dropout: float = 0.4
    use_groove_radar: bool = True

    # Training hyperparameters
    batch_size: int = 128
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    optimizer: str = "adamw"
    num_epochs: int = 5
    early_stopping_patience: int = 5
    gradient_clip_norm: float = 1.0
    use_class_weights: bool = True
    use_amp: bool = True
    accumulation_steps: int = 2

    # Scheduler
    scheduler: str = "reduce_on_plateau"
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5

    # Data
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    num_workers: int = 4
    cache_dir: str = "cache/samples"

    # Paths
    checkpoint_dir: str = "checkpoints"
    data_dir: str = ""
    audio_dir: str = ""
    config_path: str = ""

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """
        Load configuration from a YAML file.

        Reads the YAML config and maps nested keys (classifier.*, training.*)
        into the flat dataclass fields.

        Args:
            path: Path to YAML config file

        Returns:
            Populated ExperimentConfig instance
        """
        with open(path, 'r') as f:
            raw = yaml.safe_load(f)

        classifier = raw.get('classifier', {})
        training = raw.get('training', {})

        kwargs = {'config_path': str(path)}

        # Map classifier config
        field_map_classifier = {
            'audio_features_dim': 'audio_features_dim',
            'chart_sequence_dim': 'chart_sequence_dim',
            'max_sequence_length': 'max_sequence_length',
            'fusion_dim': 'fusion_dim',
            'num_classes': 'num_classes',
            'backbone_blocks': 'backbone_blocks',
            'pooling_type': 'pooling_type',
            'fusion_type': 'fusion_type',
            'classifier_hidden_dim': 'classifier_hidden_dim',
            'classifier_dropout': 'classifier_dropout',
            'backbone_dropout': 'backbone_dropout',
            'use_groove_radar': 'use_groove_radar',
        }
        for yaml_key, field_name in field_map_classifier.items():
            if yaml_key in classifier:
                kwargs[field_name] = classifier[yaml_key]

        # Map training config
        field_map_training = {
            'batch_size': 'batch_size',
            'learning_rate': 'learning_rate',
            'weight_decay': 'weight_decay',
            'optimizer': 'optimizer',
            'num_epochs': 'num_epochs',
            'early_stopping_patience': 'early_stopping_patience',
            'gradient_clip_norm': 'gradient_clip_norm',
            'use_class_weights': 'use_class_weights',
            'use_amp': 'use_amp',
            'accumulation_steps': 'accumulation_steps',
            'num_workers': 'num_workers',
            'cache_dir': 'cache_dir',
        }
        for yaml_key, field_name in field_map_training.items():
            if yaml_key in training:
                kwargs[field_name] = training[yaml_key]

        # Scheduler
        if 'scheduler' in training:
            kwargs['scheduler'] = training['scheduler']
        if 'patience' in training:
            kwargs['scheduler_patience'] = training['patience']
        if 'factor' in training:
            kwargs['scheduler_factor'] = training['factor']

        return cls(**kwargs)

    def validate(self):
        """
        Validate configuration values.

        Raises:
            ValueError: If any configuration values are invalid
        """
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if not (0.99 < self.train_ratio + self.val_ratio + self.test_ratio < 1.01):
            raise ValueError("Split ratios must sum to 1.0")
        if self.optimizer not in ('adam', 'adamw', 'sgd'):
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        if self.model_type not in ('classifier', 'mlp_baseline', 'pooled_baseline'):
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def to_mlflow_params(self) -> Dict[str, str]:
        """
        Convert config to a flat dict for mlflow.log_params().

        Returns:
            Dictionary of string key-value pairs for MLflow
        """
        return {
            'experiment_name': self.experiment_name,
            'model_type': self.model_type,
            'seed': str(self.seed),
            'audio_features_dim': str(self.audio_features_dim),
            'chart_sequence_dim': str(self.chart_sequence_dim),
            'max_sequence_length': str(self.max_sequence_length),
            'fusion_dim': str(self.fusion_dim),
            'num_classes': str(self.num_classes),
            'backbone_blocks': str(self.backbone_blocks),
            'pooling_type': self.pooling_type,
            'fusion_type': self.fusion_type,
            'use_groove_radar': str(self.use_groove_radar),
            'batch_size': str(self.batch_size),
            'learning_rate': str(self.learning_rate),
            'weight_decay': str(self.weight_decay),
            'optimizer': self.optimizer,
            'num_epochs': str(self.num_epochs),
            'early_stopping_patience': str(self.early_stopping_patience),
            'use_class_weights': str(self.use_class_weights),
            'use_amp': str(self.use_amp),
            'accumulation_steps': str(self.accumulation_steps),
            'scheduler': self.scheduler,
        }

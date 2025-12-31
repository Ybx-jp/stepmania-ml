"""
Simple MLP baseline model for StepMania difficulty classification.

Provides a simple baseline to compare against the more sophisticated conv-based models.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .components.pooling import MaskedGlobalPool, MaskedAttentionPool, MaskedMeanMaxPool
from .components.heads import ClassificationHead


class MLPBaseline(nn.Module):
    """Simple MLP baseline for difficulty classification."""

    def __init__(self, config: Dict):
        """
        Initialize MLP baseline model.

        Args:
            config: Model configuration dictionary with:
                {
                    'audio_features_dim': int,        # Audio input dimension (13)
                    'chart_sequence_dim': int,        # Chart input dimension (4)
                    'hidden_dims': List[int],         # Hidden layer dimensions
                    'num_classes': int,               # Number of output classes (10)
                    'dropout': float,                 # Dropout probability
                    'pooling_type': str               # 'attention', 'mean_max', 'global'
                }
        """
        super().__init__()

        self.config = config

        # Input dimensions
        audio_dim = config['audio_features_dim']        # 13
        chart_dim = config['chart_sequence_dim']        # 4
        input_dim = audio_dim + chart_dim               # 17 (concatenated features)

        # Pooling strategy for variable-length sequences
        pooling_type = config.get('pooling_type', 'global')
        if pooling_type == 'attention':
            self.pooling = MaskedAttentionPool(
                input_dim=input_dim,
                hidden_dim=config.get('attention_hidden_dim', 64)
            )
            pooled_dim = input_dim
        elif pooling_type == 'mean_max':
            self.pooling = MaskedMeanMaxPool()
            pooled_dim = input_dim * 2  # Concatenated mean + max
        elif pooling_type == 'global':
            self.pooling = MaskedGlobalPool()
            pooled_dim = input_dim
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")

        # MLP layers
        hidden_dims = config.get('hidden_dims', [256, 128])
        dropout = config.get('dropout', 0.2)

        layers = []
        prev_dim = pooled_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            ])
            prev_dim = hidden_dim

        self.feature_layers = nn.Sequential(*layers)

        # Classification head
        self.classifier_head = ClassificationHead(
            input_dim=prev_dim,
            num_classes=config['num_classes'],
            dropout=config.get('classifier_dropout', dropout)
        )

    def forward(self,
                audio: torch.Tensor,
                chart: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP baseline.

        Args:
            audio: Audio features (B, L, audio_features_dim)
            chart: Chart sequences (B, L, chart_sequence_dim)
            mask: Attention mask (B, L) where 1 = valid, 0 = padding

        Returns:
            Classification logits (B, num_classes)
        """
        # Simple concatenation of audio and chart features
        combined_features = torch.cat([audio, chart], dim=-1)  # (B, L, 17)

        # Pool to fixed-size representation
        pooled_features = self.pooling(combined_features, mask)  # (B, pooled_dim)

        # Pass through MLP layers
        features = self.feature_layers(pooled_features)        # (B, last_hidden_dim)

        # Final classification
        logits = self.classifier_head(features)                # (B, num_classes)

        return logits

    @classmethod
    def from_config_file(cls, config_path: str):
        """
        Create baseline model from configuration file.

        Args:
            config_path: Path to YAML config file

        Returns:
            Initialized baseline model instance
        """
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Extract baseline config or use classifier config as fallback
        baseline_config = config.get('baseline', config.get('classifier', config))
        return cls(baseline_config)


class SimpleConcatBaseline(nn.Module):
    """Even simpler baseline that just flattens and concatenates everything."""

    def __init__(self, config: Dict):
        """
        Initialize simple concatenation baseline.

        Args:
            config: Model configuration dictionary
        """
        super().__init__()

        # Input dimensions
        audio_dim = config['audio_features_dim']        # 13
        chart_dim = config['chart_sequence_dim']        # 4
        max_length = config['max_sequence_length']      # 960

        # Flattened input dimension
        flattened_dim = (audio_dim + chart_dim) * max_length  # 17 * 960 = 16320

        # Simple MLP on flattened features
        hidden_dim = config.get('hidden_dim', 512)
        dropout = config.get('dropout', 0.3)

        self.model = nn.Sequential(
            nn.Linear(flattened_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, config['num_classes'])
        )

    def forward(self,
                audio: torch.Tensor,
                chart: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with simple concatenation + flattening.

        Args:
            audio: Audio features (B, L, audio_features_dim)
            chart: Chart sequences (B, L, chart_sequence_dim)
            mask: Attention mask (B, L) - ignored in this simple baseline

        Returns:
            Classification logits (B, num_classes)
        """
        batch_size = audio.size(0)

        # Concatenate features
        combined = torch.cat([audio, chart], dim=-1)  # (B, L, 17)

        # Flatten everything (ignoring masks - not ideal but simple)
        flattened = combined.view(batch_size, -1)     # (B, 17*L)

        # Pass through MLP
        return self.model(flattened)


class PooledFeatureBaseline(nn.Module):
    """Baseline using simple pooled statistical features."""

    def __init__(self, config: Dict):
        """
        Initialize pooled feature baseline.

        Args:
            config: Model configuration dictionary
        """
        super().__init__()

        # Feature dimensions
        audio_dim = config['audio_features_dim']        # 13
        chart_dim = config['chart_sequence_dim']        # 4

        # Statistical features: mean, std, min, max per channel
        # For both audio and chart features
        stat_features_per_channel = 4  # mean, std, min, max
        audio_stat_dim = audio_dim * stat_features_per_channel     # 13 * 4 = 52
        chart_stat_dim = chart_dim * stat_features_per_channel     # 4 * 4 = 16
        total_stat_dim = audio_stat_dim + chart_stat_dim           # 68

        # Simple MLP on statistical features
        hidden_dim = config.get('hidden_dim', 256)
        dropout = config.get('dropout', 0.2)

        self.model = nn.Sequential(
            nn.Linear(total_stat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, config['num_classes'])
        )

    def forward(self,
                audio: torch.Tensor,
                chart: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using statistical features.

        Args:
            audio: Audio features (B, L, audio_features_dim)
            chart: Chart sequences (B, L, chart_sequence_dim)
            mask: Attention mask (B, L)

        Returns:
            Classification logits (B, num_classes)
        """
        # Apply mask to zero out padding
        mask_expanded = mask.unsqueeze(-1).float()
        audio_masked = audio * mask_expanded
        chart_masked = chart * mask_expanded

        # Compute sequence lengths for proper statistics
        seq_lengths = mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)

        # Statistical features for audio
        audio_mean = audio_masked.sum(dim=1) / seq_lengths
        audio_std = torch.sqrt(((audio_masked - audio_mean.unsqueeze(1)) ** 2 * mask_expanded).sum(dim=1) / seq_lengths)
        audio_min = audio_masked.masked_fill(~mask_expanded.bool(), float('inf')).min(dim=1)[0]
        audio_max = audio_masked.max(dim=1)[0]

        # Statistical features for chart
        chart_mean = chart_masked.sum(dim=1) / seq_lengths
        chart_std = torch.sqrt(((chart_masked - chart_mean.unsqueeze(1)) ** 2 * mask_expanded).sum(dim=1) / seq_lengths)
        chart_min = chart_masked.masked_fill(~mask_expanded.bool(), float('inf')).min(dim=1)[0]
        chart_max = chart_masked.max(dim=1)[0]

        # Concatenate all statistical features
        features = torch.cat([
            audio_mean, audio_std, audio_min, audio_max,
            chart_mean, chart_std, chart_min, chart_max
        ], dim=1)

        # Pass through MLP
        return self.model(features)
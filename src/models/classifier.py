"""
Complete late-fusion classifier for StepMania difficulty prediction.

Architecture:
- Separate encoders for audio and chart features
- Late fusion after temporal encoding
- Mask-aware pooling for variable sequence lengths
- Classification head for difficulty prediction
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Union

from .components.encoders import AudioEncoder, ChartEncoder
from .components.fusion import LateFusionModule
from .components.backbone import Conv1DBackbone
from .components.pooling import MaskedAttentionPool, MaskedMeanMaxPool
from .components.heads import ClassificationHead, RegressionHead


class LateFusionClassifier(nn.Module):
    """Complete classifier with late fusion architecture."""

    def __init__(self, config: Dict):
        """
        Initialize late fusion classifier.

        Args:
            config: Model configuration dictionary with the following structure:
                {
                    'audio_features_dim': int,        # Audio input dimension (13)
                    'chart_sequence_dim': int,        # Chart input dimension (4)
                    'max_sequence_length': int,       # Maximum sequence length
                    'audio_encoder': {
                        'hidden_dim': int,            # Audio encoder hidden dimension
                        'num_layers': int,            # Number of ResBlocks
                        'dropout': float              # Dropout probability
                    },
                    'chart_encoder': {
                        'embedding_dim': int,         # Chart embedding dimension
                        'hidden_dim': int,            # Chart encoder hidden dimension
                        'num_layers': int,            # Number of ResBlocks
                        'dropout': float              # Dropout probability
                    },
                    'fusion_dim': int,                # Fusion output dimension
                    'num_classes': int,               # Number of output classes (10)
                    'classifier_dropout': float,      # Classifier head dropout
                    'fusion_type': str,               # 'late', 'gated', 'additive'
                    'pooling_type': str               # 'attention', 'mean_max', 'global'
                }
        """
        super().__init__()

        self.config = config

        # Separate encoders for each modality
        self.audio_encoder = AudioEncoder(
            input_dim=config['audio_features_dim'],
            hidden_dim=config['audio_encoder']['hidden_dim']
        )

        self.chart_encoder = ChartEncoder(
            input_dim=config['chart_sequence_dim'],
            embedding_dim=config['chart_encoder'].get('embedding_dim', 64),
            hidden_dim=config['chart_encoder']['hidden_dim']        )

        # Late fusion module
        fusion_type = config.get('fusion_type', 'late')
        if fusion_type == 'late':
            self.fusion_module = LateFusionModule(
                audio_dim=config['audio_encoder']['hidden_dim'],
                chart_dim=config['chart_encoder']['hidden_dim'],
                fusion_dim=config['fusion_dim'],
                dropout=config.get('fusion_dropout', 0.1)
            )
        elif fusion_type == 'gated':
            self.fusion_module = GatedFusionModule(
                audio_dim=config['audio_encoder']['hidden_dim'],
                chart_dim=config['chart_encoder']['hidden_dim'],
                fusion_dim=config['fusion_dim'],
                dropout=config.get('fusion_dropout', 0.1)
            )
        elif fusion_type == 'additive':
            self.fusion_module = AdditiveFusionModule(
                audio_dim=config['audio_encoder']['hidden_dim'],
                chart_dim=config['chart_encoder']['hidden_dim'],
                fusion_dim=config['fusion_dim'],
                dropout=config.get('fusion_dropout', 0.1)
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # Conv backbone for temporal reasoning
        self.backbone = Conv1DBackbone(
            input_dim=config['fusion_dim'],
            hidden_dim=config['fusion_dim'],
            num_blocks=config.get('backbone_blocks', 3),
            dropout=config.get('backbone_dropout', 0.1)
        )

        # Mask-aware pooling
        pooling_type = config.get('pooling_type', 'attention')
        if pooling_type == 'attention':
            self.pooling = MaskedAttentionPool(
                input_dim=config['fusion_dim'],
                hidden_dim=config.get('attention_hidden_dim', 128),
                dropout=config.get('attention_dropout', 0.1)
            )
            pooled_dim = config['fusion_dim']
        elif pooling_type == 'mean_max':
            self.pooling = MaskedMeanMaxPool()
            pooled_dim = config['fusion_dim'] * 2  # Concatenated mean + max
        elif pooling_type == 'global':
            self.pooling = MaskedGlobalPool()
            pooled_dim = config['fusion_dim']
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")

        # Chart statistics branch (optional)
        self.use_chart_stats = config.get('use_chart_stats', False)
        if self.use_chart_stats:
            stats_dim = config.get('chart_stats_dim', 5)
            stats_hidden = config.get('stats_hidden_dim', 32)
            self.stats_mlp = nn.Sequential(
                nn.Linear(stats_dim, stats_hidden),
                nn.ReLU(),
                nn.Linear(stats_hidden, stats_hidden),
                nn.Dropout(p=config.get('stats_dropout', 0.5))
            )
            pooled_dim += stats_hidden  # Expand classifier input dim

        # Classification head
        self.classifier_head = ClassificationHead(
            input_dim=pooled_dim,
            num_classes=config['num_classes'],
            hidden_dim=config.get('classifier_hidden_dim', None),
            dropout=config['classifier_dropout']
        )

    def forward(self,
                audio: torch.Tensor,
                chart: torch.Tensor,
                mask: torch.Tensor,
                chart_stats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the complete model.

        Args:
            audio: Audio features (B, L, audio_features_dim)
            chart: Chart sequences (B, L, chart_sequence_dim)
            mask: Attention mask (B, L) where 1 = valid, 0 = padding
            chart_stats: Optional chart statistics (B, stats_dim) for difficulty features

        Returns:
            Classification logits (B, num_classes)
        """
        # Separate encoding for each modality
        audio_encoded = self.audio_encoder(audio, mask)      # (B, L, audio_hidden_dim)
        chart_encoded = self.chart_encoder(chart, mask)      # (B, L, chart_hidden_dim)

        # Late fusion after encoding
        fused_features = self.fusion_module(
            audio_encoded, chart_encoded, mask
        )  # (B, L, fusion_dim)

        # Conv backbone for temporal reasoning
        processed_features = self.backbone(fused_features, mask)  # (B, L, fusion_dim)

        # Mask-aware pooling to handle variable sequence lengths
        pooled_features = self.pooling(processed_features, mask)  # (B, pooled_dim)

        # Concatenate chart statistics if enabled
        if self.use_chart_stats and chart_stats is not None:
            stats_features = self.stats_mlp(chart_stats)  # (B, stats_hidden)
            pooled_features = torch.cat([pooled_features, stats_features], dim=-1)

        # Final classification
        logits = self.classifier_head(pooled_features)       # (B, num_classes)

        return logits

    def get_feature_representations(self,
                                    audio: torch.Tensor,
                                    chart: torch.Tensor,
                                    mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get intermediate feature representations for analysis.

        Args:
            audio: Audio features (B, L, audio_features_dim)
            chart: Chart sequences (B, L, chart_sequence_dim)
            mask: Attention mask (B, L)

        Returns:
            Dictionary of intermediate representations
        """
        # Encode each modality
        audio_encoded = self.audio_encoder(audio, mask)
        chart_encoded = self.chart_encoder(chart, mask)

        # Fuse features
        fused_features = self.fusion_module(audio_encoded, chart_encoded, mask)

        # Process through backbone
        processed_features = self.backbone(fused_features, mask)

        # Pool features
        pooled_features = self.pooling(processed_features, mask)

        return {
            'audio_encoded': audio_encoded,
            'chart_encoded': chart_encoded,
            'fused_features': fused_features,
            'processed_features': processed_features,
            'pooled_features': pooled_features
        }

    @classmethod
    def from_config_file(cls, config_path: str):
        """
        Create model from configuration file.

        Args:
            config_path: Path to YAML config file

        Returns:
            Initialized model instance
        """
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Extract classifier config
        classifier_config = config.get('classifier', config)
        return cls(classifier_config)

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
from .components.fusion import LateFusionModule, GatedFusionModule, AdditiveFusionModule
from .components.pooling import MaskedAttentionPool, MaskedMeanMaxPool, MaskedGlobalPool
from .components.heads import ClassificationHead, RegressionHead, DualHead


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
            hidden_dim=config['audio_encoder']['hidden_dim'],
            num_res_blocks=config['audio_encoder']['num_layers'],
            dropout=config['audio_encoder']['dropout']
        )

        self.chart_encoder = ChartEncoder(
            input_dim=config['chart_sequence_dim'],
            embedding_dim=config['chart_encoder'].get('embedding_dim', 64),
            hidden_dim=config['chart_encoder']['hidden_dim'],
            num_res_blocks=config['chart_encoder']['num_layers'],
            dropout=config['chart_encoder']['dropout']
        )

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
                mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete model.

        Args:
            audio: Audio features (B, L, audio_features_dim)
            chart: Chart sequences (B, L, chart_sequence_dim)
            mask: Attention mask (B, L) where 1 = valid, 0 = padding

        Returns:
            Classification logits (B, num_classes)
        """
        # Separate encoding for each modality
        audio_encoded = self.audio_encoder(audio, mask)      # (B, L, audio_hidden_dim)
        chart_encoded = self.chart_encoder(chart, mask)      # (B, L, chart_hidden_dim)

        # Late fusion after temporal encoding
        fused_features = self.fusion_module(
            audio_encoded, chart_encoded, mask
        )  # (B, L, fusion_dim)

        # Mask-aware pooling to handle variable sequence lengths
        pooled_features = self.pooling(fused_features, mask)  # (B, pooled_dim)

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

        # Pool features
        pooled_features = self.pooling(fused_features, mask)

        return {
            'audio_encoded': audio_encoded,
            'chart_encoded': chart_encoded,
            'fused_features': fused_features,
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


class DualTaskClassifier(LateFusionClassifier):
    """Classifier with both classification and regression heads."""

    def __init__(self, config: Dict):
        """
        Initialize dual-task classifier.

        Same config as LateFusionClassifier, but with dual head instead of single classifier.
        """
        # Initialize parent class but replace the head
        super().__init__(config)

        # Replace single classification head with dual head
        pooling_type = config.get('pooling_type', 'attention')
        if pooling_type == 'mean_max':
            pooled_dim = config['fusion_dim'] * 2
        else:
            pooled_dim = config['fusion_dim']

        self.classifier_head = DualHead(
            input_dim=pooled_dim,
            num_classes=config['num_classes'],
            regression_dim=config.get('regression_dim', 1),
            shared_hidden_dim=config.get('shared_hidden_dim', None),
            clf_hidden_dim=config.get('classifier_hidden_dim', None),
            reg_hidden_dim=config.get('regression_hidden_dim', None),
            dropout=config['classifier_dropout']
        )

    def forward(self,
                audio: torch.Tensor,
                chart: torch.Tensor,
                mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both classification and regression outputs.

        Returns:
            Tuple of (classification_logits, regression_outputs)
        """
        # Same encoding and fusion as parent
        audio_encoded = self.audio_encoder(audio, mask)
        chart_encoded = self.chart_encoder(chart, mask)
        fused_features = self.fusion_module(audio_encoded, chart_encoded, mask)
        pooled_features = self.pooling(fused_features, mask)

        # Dual head outputs
        return self.classifier_head(pooled_features)
"""
Complete late-fusion classifier for StepMania difficulty prediction.

Architecture:
- Separate encoders for audio and chart features
- Late fusion after temporal encoding
- Mask-aware pooling for variable sequence lengths
- Classification head for difficulty prediction
- Optional projection head for contrastive learning
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Union

from .components.encoders import AudioEncoder, ChartEncoder
from .components.fusion import LateFusionModule
from .components.backbone import Conv1DBackbone
from .components.pooling import MaskedAttentionPool, MaskedMeanMaxPool
from .components.heads import ClassificationHead, RegressionHead, OrdinalRegressionHead


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

        # Groove radar branch (optional, replaces chart_stats)
        # Uses 5 groove radar values: stream, voltage, air, freeze, chaos
        self.use_groove_radar = config.get('use_groove_radar', False)
        if self.use_groove_radar:
            radar_dim = 5  # Fixed: stream, voltage, air, freeze, chaos
            radar_hidden = config.get('radar_hidden_dim', 32)
            self.radar_mlp = nn.Sequential(
                nn.Linear(radar_dim, radar_hidden),
                nn.ReLU(),
                nn.Linear(radar_hidden, radar_hidden),
                nn.Dropout(p=config.get('radar_dropout', 0.3))
            )
            pooled_dim += radar_hidden  # Expand classifier input dim

        # Store pooled_dim for projection head
        self._pooled_dim = pooled_dim

        # Projection head for contrastive learning (optional)
        self.use_projection_head = config.get('use_projection_head', False)
        if self.use_projection_head:
            projection_dim = config.get('projection_dim', 128)
            self.projection_head = nn.Sequential(
                nn.Linear(pooled_dim, projection_dim),
                nn.ReLU(),
                nn.Linear(projection_dim, projection_dim)
            )

        # Classification/regression head (swappable)
        self.head_type = config.get('head_type', 'classification')
        if self.head_type == 'classification':
            self.classifier_head = ClassificationHead(
                input_dim=pooled_dim,
                num_classes=config['num_classes'],
                hidden_dim=config.get('classifier_hidden_dim', None),
                dropout=config['classifier_dropout']
            )
        elif self.head_type == 'ordinal':
            self.classifier_head = OrdinalRegressionHead(
                input_dim=pooled_dim,
                num_classes=config['num_classes'],
                hidden_dim=config.get('classifier_hidden_dim', None),
                dropout=config['classifier_dropout']
            )
        else:
            raise ValueError(f"Unknown head_type: {self.head_type}")

    def forward(self,
                audio: torch.Tensor,
                chart: torch.Tensor,
                mask: torch.Tensor,
                groove_radar: Optional[torch.Tensor] = None,
                return_embeddings: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the complete model.

        Args:
            audio: Audio features (B, L, audio_features_dim)
            chart: Chart sequences (B, L, chart_sequence_dim)
            mask: Attention mask (B, L) where 1 = valid, 0 = padding
            groove_radar: Optional groove radar values (B, 5) for classification features
            return_embeddings: If True, return dict with logits and embeddings for contrastive learning

        Returns:
            If return_embeddings=False: Classification logits (B, num_classes)
            If return_embeddings=True: Dict with 'logits' and 'embeddings' keys
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

        # Concatenate groove radar features if enabled
        if self.use_groove_radar and groove_radar is not None:
            radar_features = self.radar_mlp(groove_radar)  # (B, radar_hidden)
            pooled_features = torch.cat([pooled_features, radar_features], dim=-1)

        # Final classification/ordinal regression
        logits = self.classifier_head(pooled_features)
        # Note: For 'classification' head: (B, num_classes) class logits
        #       For 'ordinal' head: (B, num_classes-1) cumulative logits

        if return_embeddings and self.use_projection_head:
            embeddings = self.projection_head(pooled_features)
            return {
                'logits': logits,
                'embeddings': embeddings
            }

        return logits

    def predict_class_from_logits(self, logits: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Get class predictions from forward() output.

        Args:
            logits: Output from forward() - can be tensor or dict with 'logits' key

        Returns:
            Predicted class indices (B,) with values 0..num_classes-1
        """
        # Handle dict output from contrastive mode
        if isinstance(logits, dict):
            logits = logits['logits']

        if self.head_type == 'ordinal':
            return OrdinalRegressionHead.logits_to_class(logits)
        else:
            return logits.argmax(dim=1)

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

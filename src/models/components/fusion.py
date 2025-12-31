"""
Late fusion modules for combining encoded audio and chart features.

Fusion strategies:
- LateFusionModule: Simple concatenation + projection fusion
- CrossModalFusion: More advanced cross-modal interaction (future)
"""

import torch
import torch.nn as nn
from typing import Optional


class LateFusionModule(nn.Module):
    """Late fusion of encoded audio and chart features via concatenation + projection."""

    def __init__(self,
                 audio_dim: int,
                 chart_dim: int,
                 fusion_dim: int,
                 dropout: float = 0.1,
                 use_norm: bool = True):
        """
        Initialize late fusion module.

        Args:
            audio_dim: Dimension of encoded audio features
            chart_dim: Dimension of encoded chart features
            fusion_dim: Output dimension after fusion
            dropout: Dropout probability
            use_norm: Whether to use layer normalization
        """
        super().__init__()

        # Fusion projection: concat(audio, chart) → fusion_dim
        self.fusion_proj = nn.Linear(audio_dim + chart_dim, fusion_dim)

        # Normalization and regularization
        self.norm = nn.LayerNorm(fusion_dim) if use_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self,
                audio_features: torch.Tensor,
                chart_features: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Fuse audio and chart features via late fusion.

        Args:
            audio_features: Encoded audio features (B, L, audio_dim)
            chart_features: Encoded chart features (B, L, chart_dim)
            mask: Attention mask (B, L) - not used but kept for interface consistency

        Returns:
            Fused features (B, L, fusion_dim)
        """
        # Concatenate along feature dimension
        fused = torch.cat([audio_features, chart_features], dim=-1)  # (B, L, audio_dim + chart_dim)

        # Project to fusion dimension
        fused = self.fusion_proj(fused)           # (B, L, fusion_dim)
        fused = self.norm(fused)
        fused = self.activation(fused)
        fused = self.dropout(fused)

        return fused


class GatedFusionModule(nn.Module):
    """Gated fusion with learnable weighting between modalities."""

    def __init__(self,
                 audio_dim: int,
                 chart_dim: int,
                 fusion_dim: int,
                 dropout: float = 0.1):
        """
        Initialize gated fusion module.

        Args:
            audio_dim: Dimension of encoded audio features
            chart_dim: Dimension of encoded chart features
            fusion_dim: Output dimension after fusion
            dropout: Dropout probability
        """
        super().__init__()

        # Project each modality to fusion dimension
        self.audio_proj = nn.Linear(audio_dim, fusion_dim)
        self.chart_proj = nn.Linear(chart_dim, fusion_dim)

        # Gating mechanism to learn importance weights
        self.gate_audio = nn.Linear(audio_dim + chart_dim, fusion_dim)
        self.gate_chart = nn.Linear(audio_dim + chart_dim, fusion_dim)

        self.norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self,
                audio_features: torch.Tensor,
                chart_features: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Fuse audio and chart features with gating.

        Args:
            audio_features: Encoded audio features (B, L, audio_dim)
            chart_features: Encoded chart features (B, L, chart_dim)
            mask: Attention mask (B, L)

        Returns:
            Gated fused features (B, L, fusion_dim)
        """
        # Project each modality
        audio_proj = self.audio_proj(audio_features)    # (B, L, fusion_dim)
        chart_proj = self.chart_proj(chart_features)    # (B, L, fusion_dim)

        # Compute gating weights based on both modalities
        concat_features = torch.cat([audio_features, chart_features], dim=-1)
        gate_audio = torch.sigmoid(self.gate_audio(concat_features))  # (B, L, fusion_dim)
        gate_chart = torch.sigmoid(self.gate_chart(concat_features))  # (B, L, fusion_dim)

        # Gated fusion
        fused = gate_audio * audio_proj + gate_chart * chart_proj

        fused = self.norm(fused)
        fused = self.dropout(fused)

        return fused


class AdditiveFusionModule(nn.Module):
    """Simple additive fusion after dimension alignment."""

    def __init__(self,
                 audio_dim: int,
                 chart_dim: int,
                 fusion_dim: int,
                 dropout: float = 0.1):
        """
        Initialize additive fusion module.

        Args:
            audio_dim: Dimension of encoded audio features
            chart_dim: Dimension of encoded chart features
            fusion_dim: Output dimension after fusion
            dropout: Dropout probability
        """
        super().__init__()

        # Project to same dimension for addition
        self.audio_proj = nn.Linear(audio_dim, fusion_dim)
        self.chart_proj = nn.Linear(chart_dim, fusion_dim)

        self.norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self,
                audio_features: torch.Tensor,
                chart_features: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Fuse audio and chart features additively.

        Args:
            audio_features: Encoded audio features (B, L, audio_dim)
            chart_features: Encoded chart features (B, L, chart_dim)
            mask: Attention mask (B, L)

        Returns:
            Additively fused features (B, L, fusion_dim)
        """
        # Project to same dimension
        audio_proj = self.audio_proj(audio_features)    # (B, L, fusion_dim)
        chart_proj = self.chart_proj(chart_features)    # (B, L, fusion_dim)

        # Additive fusion
        fused = audio_proj + chart_proj

        fused = self.norm(fused)
        fused = self.activation(fused)
        fused = self.dropout(fused)

        return fused
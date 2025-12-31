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

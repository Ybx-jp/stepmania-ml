"""
Simple encoders for audio and chart sequences.

Encoders just map inputs to embedding space - no temporal reasoning.
Temporal processing moved to backbone.

Encoders:
- AudioEncoder: Maps MFCC audio features to hidden space
- ChartEncoder: Maps chart step sequences to hidden space via embedding
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .conv_blocks import Conv1DBlock, ResidualBlock1D, DownsampleBlock1D


class AudioEncoder(nn.Module):
    """Simple audio encoder: maps MFCC features to hidden space."""

    def __init__(self,
                 input_dim: int = 13,
                 hidden_dim: int = 256):
        """
        Args:
            input_dim: Input feature dimension (13 for MFCC)
            hidden_dim: Hidden dimension for output
        """
        super().__init__()

        self.net = nn.Sequential(
            Conv1DBlock(input_dim, hidden_dim, kernel_size=3),
            Conv1DBlock(hidden_dim, hidden_dim, kernel_size=3)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass - simple embedding from input to hidden space.

        Args:
            x: Input tensor (B, L, input_dim)
            mask: Attention mask (B, L) - unused but kept for interface consistency

        Returns:
            Encoded features (B, L, hidden_dim)
        """
        # Convert to conv1d format: (B, L, D) → (B, D, L)
        x = x.transpose(1, 2)

        # Simple conv processing
        x = self.net(x)  # (B, hidden_dim, L)

        # Convert back to sequence format: (B, D, L) → (B, L, D)
        return x.transpose(1, 2)


class ChartEncoder(nn.Module):
    """Simple chart encoder: embedding + conv blocks."""

    def __init__(self,
                 input_dim: int = 4,
                 embedding_dim: int = 64,
                 hidden_dim: int = 256):
        """
        Args:
            input_dim: Input dimension (4 for Left/Down/Up/Right panels)
            embedding_dim: Chart step embedding dimension
            hidden_dim: Hidden dimension for output
        """
        super().__init__()

        # Chart step embedding: (B, L, 4) → (B, L, embedding_dim)
        self.chart_embedding = nn.Linear(input_dim, embedding_dim)

        # Simple conv processing
        self.net = nn.Sequential(
            Conv1DBlock(embedding_dim, hidden_dim, kernel_size=3),
            Conv1DBlock(hidden_dim, hidden_dim, kernel_size=3)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass - simple embedding + conv processing.

        Args:
            x: Chart tensor (B, L, 4) - binary step encoding
            mask: Attention mask (B, L) - unused but kept for interface consistency

        Returns:
            Encoded chart features (B, L, hidden_dim)
        """
        # Embed chart steps first
        x = self.chart_embedding(x)  # (B, L, embedding_dim)

        # Convert to conv1d format: (B, L, D) → (B, D, L)
        x = x.transpose(1, 2)

        # Simple conv processing
        x = self.net(x)  # (B, hidden_dim, L)

        # Convert back to sequence format: (B, D, L) → (B, L, D)
        return x.transpose(1, 2)


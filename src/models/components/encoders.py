"""
Minimal pattern encoders for audio and chart sequences.

Architecture pattern: Conv → ResBlock → Downsample → 2-3 ResBlocks → Upsample → Skip Connection

Encoders:
- AudioEncoder: Processes MFCC audio features
- ChartEncoder: Processes chart step sequences with embedding
"""

import torch
import torch.nn as nn
from typing import Optional

from .conv_blocks import Conv1DBlock, ResidualBlock1D, DownsampleBlock1D


class AudioEncoder(nn.Module):
    """Minimal 1D encoder for audio features following the pattern:
    Conv → ResBlock → Downsample → ResBlocks → Upsample + Skip Connection"""

    def __init__(self,
                 input_dim: int = 13,
                 hidden_dim: int = 256,
                 num_res_blocks: int = 3,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        """
        Args:
            input_dim: Input feature dimension (13 for MFCC)
            hidden_dim: Hidden dimension throughout the encoder
            num_res_blocks: Number of ResBlocks in downsampled space (2-3)
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()

        # Initial conv: (B, input_dim, L) → (B, hidden_dim, L)
        self.input_conv = Conv1DBlock(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            dropout=dropout,
            activation=activation
        )

        # Pre-downsample residual block
        self.pre_downsample_res = ResidualBlock1D(
            channels=hidden_dim,
            dropout=dropout,
            activation=activation
        )

        # Downsample with stride 2: (B, hidden_dim, L) → (B, hidden_dim, L//2)
        self.downsample = DownsampleBlock1D(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            stride=2,
            kernel_size=3,
            dropout=dropout,
            activation=activation
        )

        # 2-3 ResBlocks in downsampled space
        self.res_blocks = nn.ModuleList([
            ResidualBlock1D(
                channels=hidden_dim,
                dropout=dropout,
                activation=activation
            )
            for _ in range(num_res_blocks)
        ])

        # Upsample back to original length: (B, hidden_dim, L//2) → (B, hidden_dim, L)
        self.upsample = nn.ConvTranspose1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=2,
            stride=2
        )

        # Skip connection projection (identity if dimensions match)
        self.skip_proj = nn.Identity()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with minimal encoder pattern.

        Args:
            x: Input tensor (B, L, input_dim)
            mask: Attention mask (B, L) tracking valid timesteps. Not applied to individual
                  convolution operations, but propagated across temporal resolution changes
                  to preserve alignment for downstream fusion and pooling.

        Returns:
            Encoded features (B, L, hidden_dim)
        """

        B, L, _ = x.shape

        # Convert to conv1d format: (B, L, D) → (B, D, L)
        x = x.transpose(1, 2)

        # Initial conv + pre-downsample ResBlock
        x = self.input_conv(x)  # (B, hidden_dim, L)
        skip = self.pre_downsample_res(x)  # Store skip connection

        # Downsample → ResBlocks → Upsample
        x = self.downsample(skip)  # (B, hidden_dim, L//2)
        if mask is not None:
            mask_ds = downsample_mask(mask, stride=2)

        for res_block in self.res_blocks:
            x = res_block(x)  # (B, hidden_dim, L//2)

        x = self.upsample(x)  # (B, hidden_dim, L)
        x = x[:, :, :L]  # crop to original length
        if mask is not None:
            mask = upsample_mask(mask_ds, target_len=L)

        # Residual skip connection from pre-downsample
        x = x + self.skip_proj(skip)  # (B, hidden_dim, L)

        # Convert back to sequence format: (B, D, L) → (B, L, D)
        return x.transpose(1, 2)


class ChartEncoder(nn.Module):
    """Minimal 1D encoder for chart sequences with same pattern as audio encoder."""

    def __init__(self,
                 input_dim: int = 4,
                 embedding_dim: int = 64,
                 hidden_dim: int = 256,
                 num_res_blocks: int = 2,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        """
        Args:
            input_dim: Input dimension (4 for Left/Down/Up/Right panels)
            embedding_dim: Chart step embedding dimension
            hidden_dim: Hidden dimension throughout the encoder
            num_res_blocks: Number of ResBlocks in downsampled space (2-3)
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()

        # Chart step embedding: (B, L, 4) → (B, L, embedding_dim)
        self.chart_embedding = nn.Linear(input_dim, embedding_dim)

        # Same minimal pattern as audio encoder
        self.input_conv = Conv1DBlock(
            in_channels=embedding_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            dropout=dropout,
            activation=activation
        )

        self.pre_downsample_res = ResidualBlock1D(
            channels=hidden_dim,
            dropout=dropout,
            activation=activation
        )

        self.downsample = DownsampleBlock1D(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            stride=2,
            kernel_size=3,
            dropout=dropout,
            activation=activation
        )

        self.res_blocks = nn.ModuleList([
            ResidualBlock1D(
                channels=hidden_dim,
                dropout=dropout,
                activation=activation
            )
            for _ in range(num_res_blocks)
        ])

        self.upsample = nn.ConvTranspose1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=2,
            stride=2
        )

        self.skip_proj = nn.Identity()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with chart embedding + minimal encoder pattern.

        Args:
            x: Chart tensor (B, L, 4) - binary step encoding
            mask: Attention mask (B, L) - not used in conv operations

        Returns:
            Encoded chart features (B, L, hidden_dim)
        """
        # Embed chart steps first
        x = self.chart_embedding(x)  # (B, L, embedding_dim)

        # Convert to conv1d format: (B, L, D) → (B, D, L)
        x = x.transpose(1, 2)

        # Same minimal pattern as audio encoder
        x = self.input_conv(x)  # (B, hidden_dim, L)
        skip = self.pre_downsample_res(x)  # Store skip connection

        x = self.downsample(skip)  # (B, hidden_dim, L//2)
        if mask is not None:
            mask_ds = downsample_mask(mask, stride=2)

        for res_block in self.res_blocks:
            x = res_block(x)  # (B, hidden_dim, L//2)

        x = self.upsample(x)  # (B, hidden_dim, L)
        if mask is not None:
            mask = upsample_mask(mask_ds, target_len=L)

        # Residual skip connection
        x = x + self.skip_proj(skip)  # (B, hidden_dim, L)

        # Convert back to sequence format: (B, D, L) → (B, L, D)
        return x.transpose(1, 2)


def downsample_mask(mask, stride):
    # mask: (B, L) → (B, L//stride)
    return mask.unfold(1, stride, stride).max(dim=-1).values


def upsample_mask(mask, target_len):
    # simple nearest-neighbor upsample
    return mask.repeat_interleave(2, dim=1)[:, :target_len]

"""
Conv1D backbone for temporal modeling after fusion.

Implements the minimal pattern: Conv → ResBlock → Downsample → ResBlocks → Upsample + Skip Connection
All temporal reasoning happens here, not in encoders.
"""

import torch
import torch.nn as nn
from typing import Optional

from .conv_blocks import Conv1DBlock, ResidualBlock1D, DownsampleBlock1D


class Conv1DBackbone(nn.Module):
    """1D convolutional backbone for temporal feature processing."""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_blocks: int = 3,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        """
        Initialize conv backbone.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for processing
            num_blocks: Number of residual blocks
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()

        # Enforce dimension matching - no automatic projection
        assert input_dim == hidden_dim, \
            f"Expected input_dim == hidden_dim, got {input_dim} vs {hidden_dim}"

        # Initial conv
        self.input_conv = Conv1DBlock(
            in_channels=hidden_dim,
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

        # Downsample with stride 2
        self.downsample = DownsampleBlock1D(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            stride=2,
            kernel_size=3,
            dropout=dropout,
            activation=activation
        )

        # ResBlocks in downsampled space
        self.res_blocks = nn.ModuleList([
            ResidualBlock1D(
                channels=hidden_dim,
                dropout=dropout,
                activation=activation
            )
            for _ in range(num_blocks)
        ])

        # Upsample back to original length
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
        Forward pass implementing minimal pattern with temporal reasoning.

        Args:
            x: Input features (B, L, input_dim)
            mask: Optional attention mask (B, L) - tracks valid positions through downsample/upsample

        Returns:
            Processed features (B, L, hidden_dim)
        """
        B, L, _ = x.shape

        # Convert to conv1d format: (B, L, D) → (B, D, L)
        x = x.transpose(1, 2)

        # Initial conv + pre-downsample ResBlock
        x = self.input_conv(x)                    # (B, hidden_dim, L)
        skip = self.pre_downsample_res(x)         # Store skip connection

        # Downsample → ResBlocks → Upsample
        x = self.downsample(skip)                 # (B, hidden_dim, L//2)
        if mask is not None:
            mask_ds = downsample_mask(mask, stride=2)

        for res_block in self.res_blocks:
            x = res_block(x)                      # (B, hidden_dim, L//2)

        x = self.upsample(x)                      # (B, hidden_dim, L)
        x = x[:, :, :L]                          # Crop to original length
        if mask is not None:
            mask = upsample_mask(mask_ds, target_len=L)

        # Residual skip connection from pre-downsample
        x = x + self.skip_proj(skip)              # (B, hidden_dim, L)

        # Convert back to sequence format: (B, D, L) → (B, L, D)
        return x.transpose(1, 2)


def downsample_mask(mask, stride):
    """Downsample mask by taking max over stride windows."""
    # mask: (B, L) → (B, L//stride)
    return mask.unfold(1, stride, stride).max(dim=-1).values


def upsample_mask(mask, target_len):
    """Upsample mask using nearest-neighbor interpolation."""
    # simple nearest-neighbor upsample
    return mask.repeat_interleave(2, dim=1)[:, :target_len]
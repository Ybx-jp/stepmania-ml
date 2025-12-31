"""
Modular 1D convolutional building blocks for StepMania chart processing.

Components:
- Conv1DBlock: Basic conv1d + norm + activation
- ResidualBlock1D: 1D residual block with skip connections
- DownsampleBlock1D: Downsampling conv with stride
"""

import torch
import torch.nn as nn
from typing import Optional


class Conv1DBlock(nn.Module):
    """Basic 1D convolution block with normalization and activation."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: Optional[int] = None,
                 dilation: int = 1,
                 dropout: float = 0.0,
                 activation: str = 'relu',
                 use_norm: bool = True):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Padding (auto if None)
            dilation: Dilation factor
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', 'none')
            use_norm: Whether to use batch normalization
        """
        super().__init__()

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=not use_norm
        )

        self.norm = nn.BatchNorm1d(out_channels) if use_norm else nn.Identity()

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'none':
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: conv -> norm -> activation -> dropout"""
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class ResidualBlock1D(nn.Module):
    """1D residual block with skip connections."""

    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 dilation: int = 1,
                 dropout: float = 0.0,
                 activation: str = 'relu'):
        """
        Args:
            channels: Number of channels (input and output)
            kernel_size: Convolution kernel size
            dilation: Dilation factor
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()

        self.conv1 = Conv1DBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            dropout=dropout,
            activation=activation
        )

        self.conv2 = Conv1DBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            dropout=0.0,  # No dropout on second conv
            activation='none'  # Activation after skip connection
        )

        if activation == 'relu':
            self.final_activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.final_activation = nn.GELU()
        else:
            self.final_activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection: x + conv2(conv1(x))"""
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        x = self.final_activation(x)
        return x


class DownsampleBlock1D(nn.Module):
    """Downsampling convolutional block with stride."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 2,
                 kernel_size: int = 3,
                 dropout: float = 0.0,
                 activation: str = 'relu'):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            stride: Downsampling stride
            kernel_size: Convolution kernel size
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()

        self.conv = Conv1DBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            activation=activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: strided convolution for downsampling"""
        return self.conv(x)
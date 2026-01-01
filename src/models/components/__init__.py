"""
Modular components for StepMania neural network architectures.

Components:
- conv_blocks: Basic 1D convolutional building blocks
- encoders: Audio and chart sequence encoders with minimal pattern
- fusion: Late fusion modules for combining modalities
- pooling: Mask-aware pooling operations
- heads: Classification and regression heads
"""

from .conv_blocks import Conv1DBlock, ResidualBlock1D, DownsampleBlock1D
from .encoders import AudioEncoder, ChartEncoder
from .fusion import LateFusionModule
from .pooling import MaskedAttentionPool, MaskedMeanMaxPool
from .heads import ClassificationHead, RegressionHead

__all__ = [
    # Conv blocks
    'Conv1DBlock',
    'ResidualBlock1D',
    'DownsampleBlock1D',

    # Encoders
    'AudioEncoder',
    'ChartEncoder',

    # Fusion
    'LateFusionModule',

    # Pooling
    'MaskedAttentionPool',
    'MaskedMeanMaxPool',

    # Heads
    'ClassificationHead',
    'RegressionHead',
]
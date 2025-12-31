"""
Classification and regression heads for final predictions.

Heads:
- ClassificationHead: Multi-class difficulty classification
- RegressionHead: Continuous difficulty score regression
"""

import torch
import torch.nn as nn
from typing import Optional


class ClassificationHead(nn.Module):
    """Classification head for difficulty prediction."""

    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 hidden_dim: Optional[int] = None,
                 dropout: float = 0.2,
                 use_norm: bool = True):
        """
        Initialize classification head.

        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes (10 for difficulty levels 1-10)
            hidden_dim: Optional hidden layer dimension (None for direct classification)
            dropout: Dropout probability
            use_norm: Whether to use layer normalization
        """
        super().__init__()

        layers = []

        if hidden_dim is not None:
            # Two-layer head: input → hidden → output
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_norm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, num_classes)
            ])
        else:
            # Single-layer head: input → output
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(input_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.

        Args:
            x: Input features (B, input_dim)

        Returns:
            Class logits (B, num_classes)
        """
        return self.classifier(x)


class RegressionHead(nn.Module):
    """Regression head for continuous difficulty scores."""

    def __init__(self,
                 input_dim: int,
                 output_dim: int = 1,
                 hidden_dim: Optional[int] = None,
                 dropout: float = 0.2,
                 use_norm: bool = True,
                 output_activation: Optional[str] = None):
        """
        Initialize regression head.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension (1 for single difficulty score)
            hidden_dim: Optional hidden layer dimension
            dropout: Dropout probability
            use_norm: Whether to use layer normalization
            output_activation: Optional output activation ('sigmoid', 'tanh', None)
        """
        super().__init__()

        layers = []

        if hidden_dim is not None:
            # Two-layer head: input → hidden → output
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_norm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, output_dim)
            ])
        else:
            # Single-layer head: input → output
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(input_dim, output_dim))

        # Optional output activation
        if output_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif output_activation == 'tanh':
            layers.append(nn.Tanh())
        elif output_activation is not None:
            raise ValueError(f"Unknown output activation: {output_activation}")

        self.regressor = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for regression.

        Args:
            x: Input features (B, input_dim)

        Returns:
            Regression outputs (B, output_dim)
        """
        return self.regressor(x)

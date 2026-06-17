"""
Classification and regression heads for final predictions.

Heads:
- ClassificationHead: Multi-class difficulty classification
- OrdinalRegressionHead: Ordinal regression with scalar or multi-output mode
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


class OrdinalRegressionHead(nn.Module):
    """
    Ordinal regression head using cumulative link model.

    Supports two modes controlled by `multi_output`:

    - multi_output=False (default, proportional odds):
      Projects features to a single scalar, then computes K-1 cumulative logits
      as `score - threshold_k`. Maximally constrained: all thresholds share
      the same feature projection.

    - multi_output=True (independent cumulative logits):
      Projects features directly to K-1 logits via a linear layer. Each threshold
      gets its own learned feature combination, removing the scalar bottleneck.
      Ordinal structure is still enforced by the BCE target encoding
      (targets are monotone: [1,1,...,0,0]).

    Training uses BCEWithLogitsLoss on cumulative predictions.
    Target encoding: for true class k, targets are [1,1,...,1,0,0,...,0]
                     where first k positions are 1 (Y > 0, Y > 1, ..., Y > k-1)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.2,
        use_norm: bool = True,
        multi_output: bool = False,
    ):
        """
        Initialize ordinal regression head.

        Args:
            input_dim: Input feature dimension
            num_classes: Number of ordinal classes
            hidden_dim: Optional hidden layer dimension
            dropout: Dropout probability
            use_norm: Whether to use layer normalization
            multi_output: If True, use independent logits per threshold instead
                          of a shared scalar projection + learnable thresholds.
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.multi_output = multi_output

        # Output dimension: 1 for scalar mode, K-1 for multi-output mode
        out_dim = self.num_thresholds if multi_output else 1

        # Feature projection
        layers = []
        if hidden_dim is not None:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_norm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, out_dim)
            ])
        else:
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(input_dim, out_dim))

        self.feature_proj = nn.Sequential(*layers)

        # Initialize final linear layer with small weights
        self._init_final_layer()

        # Learnable thresholds only needed in scalar mode
        if not multi_output:
            # Initialize tightly around 0 to match xavier_uniform_(gain=0.1)
            initial_thresholds = torch.linspace(-0.5, 0.5, self.num_thresholds)
            self.thresholds = nn.Parameter(initial_thresholds)

    def _init_final_layer(self):
        """Initialize the final linear layer with small weights."""
        for module in reversed(list(self.feature_proj.modules())):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                break

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ordinal regression.

        Args:
            x: Input features (B, input_dim)

        Returns:
            Cumulative logits (B, num_thresholds) for BCEWithLogitsLoss
        """
        if self.multi_output:
            # Independent logits per threshold — no scalar bottleneck
            return self.feature_proj(x)  # (B, K-1)
        else:
            # Proportional odds: scalar score - thresholds
            score = self.feature_proj(x)  # (B, 1)
            return score - self.thresholds  # (B, K-1)

    @staticmethod
    def logits_to_class(cumulative_logits: torch.Tensor) -> torch.Tensor:
        """
        Convert cumulative logits to class predictions.

        Args:
            cumulative_logits: Output of forward(), shape (B, K-1)

        Returns:
            Predicted class indices (B,) with values 0..K-1
        """
        cumulative_probs = torch.sigmoid(cumulative_logits)  # P(Y > k)

        batch_size = cumulative_logits.shape[0]
        device = cumulative_logits.device
        ones = torch.ones(batch_size, 1, device=device)
        zeros = torch.zeros(batch_size, 1, device=device)

        # [1, P(Y>1), P(Y>2), ..., P(Y>K-1), 0]
        extended = torch.cat([ones, cumulative_probs, zeros], dim=1)

        # Class probs = differences
        class_probs = extended[:, :-1] - extended[:, 1:]

        return class_probs.argmax(dim=1)

    def get_score(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_proj(x)


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

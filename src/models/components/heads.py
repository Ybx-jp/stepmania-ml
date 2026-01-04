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


class OrdinalRegressionHead(nn.Module):
    """
    Ordinal regression head using cumulative link model (proportional odds).

    For K classes (difficulties 1-10), predicts K-1 cumulative logits:
    logit(P(Y > 1)), logit(P(Y > 2)), ..., logit(P(Y > K-1))

    Training uses BCEWithLogitsLoss on these cumulative predictions.
    Target encoding: for true class k, targets are [1,1,...,1,0,0,...,0]
                     where first k-1 positions are 1 (Y > 1, Y > 2, ..., Y > k-1)

    This naturally captures ordinal structure - adjacent classes share
    decision boundaries, reducing off-by-1 errors.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.2,
        use_norm: bool = True,
    ):
        """
        Initialize ordinal regression head.

        Args:
            input_dim: Input feature dimension
            num_classes: Number of ordinal classes (10 for difficulty 1-10)
            hidden_dim: Optional hidden layer dimension
            dropout: Dropout probability
            use_norm: Whether to use layer normalization
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1

        # Feature projection to scalar (shared across all thresholds)
        layers = []
        if hidden_dim is not None:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_norm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, 1)  # Project to scalar
            ])
        else:
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(input_dim, 1))  # Project to scalar

        self.feature_proj = nn.Sequential(*layers)

        # Initialize final linear layer with small weights to prevent extreme outputs
        self._init_final_layer()

        # Learnable thresholds (biases) for each cumulative probability
        # Initialize with evenly spaced values centered at 0
        initial_thresholds = torch.linspace(-1.5, 1.5, self.num_thresholds)
        self.thresholds = nn.Parameter(initial_thresholds)

    def _init_final_layer(self):
        """Initialize the final linear layer with small weights."""
        # Find the last Linear layer
        for module in reversed(list(self.feature_proj.modules())):
            if isinstance(module, nn.Linear):
                # Small initialization to keep outputs near 0 initially
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
            logit[k] = logit(P(Y > k+1)) for k in 0..K-2
        """
        # Project features to scalar score
        score = self.feature_proj(x)  # (B, 1)

        # Compute cumulative logits: score - threshold_k for each k
        # Higher score = higher difficulty = more thresholds exceeded
        cumulative_logits = score - self.thresholds  # (B, num_thresholds)

        return cumulative_logits

    # def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Get class probabilities from cumulative logits.
    #
    #     Args:
    #         x: Input features (B, input_dim)
    #
    #     Returns:
    #         Class probabilities (B, num_classes)
    #     """
    #     cumulative_logits = self.forward(x)
    #     cumulative_probs = torch.sigmoid(cumulative_logits)  # P(Y > k)
    #
    #     # Convert to class probabilities
    #     # P(Y = 1) = 1 - P(Y > 1)
    #     # P(Y = k) = P(Y > k-1) - P(Y > k)
    #     # P(Y = K) = P(Y > K-1)
    #     ones = torch.ones(x.shape[0], 1, device=x.device)
    #     zeros = torch.zeros(x.shape[0], 1, device=x.device)
    #
    #     # Prepend 1 and append 0: [1, P(Y>1), P(Y>2), ..., P(Y>K-1), 0]
    #     extended = torch.cat([ones, cumulative_probs, zeros], dim=1)
    #
    #     # Class probs = differences: P(Y=k) = extended[k-1] - extended[k]
    #     class_probs = extended[:, :-1] - extended[:, 1:]
    #
    #     return torch.clamp(class_probs, min=1e-8)
    #
    # def predict_class(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Predict class labels directly.
    #
    #     Args:
    #         x: Input features (B, input_dim)
    #
    #     Returns:
    #         Predicted class indices (B,) with values 0..K-1
    #     """
    #     class_probs = self.predict_probs(x)
    #     return class_probs.argmax(dim=1)

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

        # Convert to class probabilities
        batch_size = cumulative_logits.shape[0]
        device = cumulative_logits.device
        ones = torch.ones(batch_size, 1, device=device)
        zeros = torch.zeros(batch_size, 1, device=device)

        # [1, P(Y>1), P(Y>2), ..., P(Y>K-1), 0]
        extended = torch.cat([ones, cumulative_probs, zeros], dim=1)

        # Class probs = differences
        class_probs = extended[:, :-1] - extended[:, 1:]

        return class_probs.argmax(dim=1)

    # @staticmethod
    # def encode_ordinal_targets(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    #     """
    #     Encode class labels as ordinal targets for BCEWithLogitsLoss.
    #
    #     Args:
    #         targets: Class indices (B,) with values 0..K-1
    #         num_classes: Total number of classes K
    #
    #     Returns:
    #         Ordinal targets (B, K-1) where target[b,k] = 1 if targets[b] > k
    #     """
    #     batch_size = targets.shape[0]
    #     num_thresholds = num_classes - 1
    #
    #     # Create threshold indices [0, 1, 2, ..., K-2]
    #     thresholds = torch.arange(num_thresholds, device=targets.device)
    #
    #     # targets[b] > k means class is at least k+1 (0-indexed)
    #     # Expand for broadcasting: targets (B,1) > thresholds (K-1,)
    #     ordinal_targets = (targets.unsqueeze(1) > thresholds).float()
    #
    #     return ordinal_targets
    #
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

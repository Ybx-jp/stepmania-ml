"""
Mask-aware pooling operations for variable-length sequences.

Pooling strategies:
- MaskedMeanMaxPool: Masked mean + max pooling
- MaskedAttentionPool: Attention-based pooling with mask awareness
"""

import torch
import torch.nn as nn
from typing import Optional


class MaskedMeanMaxPool(nn.Module):
    """Masked mean + max pooling for variable-length sequences."""

    def __init__(self):
        """Initialize masked mean+max pooling module."""
        super().__init__()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply masked mean and max pooling.

        Args:
            x: Input tensor (B, L, D)
            mask: Attention mask (B, L) where 1 = valid, 0 = padding

        Returns:
            Pooled features (B, 2*D) - concatenated [mean_pooled, max_pooled]
        """
        # Expand mask for broadcasting: (B, L) → (B, L, 1)
        mask_expanded = mask.unsqueeze(-1).float()

        # Masked mean pooling
        masked_x = x * mask_expanded              # Zero out padding positions
        seq_lengths = mask.sum(dim=1, keepdim=True).float()  # (B, 1)
        seq_lengths = seq_lengths.clamp(min=1.0)  # Avoid division by zero
        mean_pooled = masked_x.sum(dim=1) / seq_lengths  # (B, D)

        # Masked max pooling
        # Set padding positions to -inf before max pooling
        masked_x_max = x.masked_fill(~mask_expanded.bool(), float('-inf'))
        max_pooled = masked_x_max.max(dim=1)[0]   # (B, D)

        # Concatenate mean and max features
        return torch.cat([mean_pooled, max_pooled], dim=1)  # (B, 2*D)


class MaskedAttentionPool(nn.Module):
    """Attention-based pooling with mask awareness."""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 dropout: float = 0.1):
        """
        Initialize attention pooling module.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for attention computation
            dropout: Dropout probability in attention network
        """
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply attention-based pooling with mask.

        Args:
            x: Input tensor (B, L, D)
            mask: Attention mask (B, L) where 1 = valid, 0 = padding

        Returns:
            Attention-pooled features (B, D)
        """
        # Compute attention weights for each position
        attn_logits = self.attention(x).squeeze(-1)  # (B, L)

        # Mask attention logits: set padding positions to -inf
        attn_logits = attn_logits.masked_fill(~mask.bool(), float('-inf'))

        # Compute attention weights via softmax
        attn_weights = torch.softmax(attn_logits, dim=1)  # (B, L)

        # Handle case where entire sequence is masked (shouldn't happen in practice)
        # If all weights are -inf, softmax will be nan, so we replace with uniform
        all_masked = torch.all(~mask.bool(), dim=1, keepdim=True)  # (B, 1)
        uniform_weights = mask.float() / mask.sum(dim=1, keepdim=True).clamp(min=1)
        attn_weights = torch.where(all_masked, uniform_weights, attn_weights)

        # Apply attention weights to input features
        pooled = (x * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, D)

        return pooled

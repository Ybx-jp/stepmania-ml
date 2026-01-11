"""
Contrastive loss functions for groove radar-based similarity learning.

Provides:
- TripletMarginLossWithRadar: Triplet loss with groove radar-weighted margins
- InfoNCELoss: InfoNCE/NT-Xent contrastive loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TripletMarginLossWithRadar(nn.Module):
    """
    Triplet margin loss with groove radar-weighted adaptive margins.

    The margin is proportional to the groove radar distance between
    anchor and negative, encouraging the model to push negatives
    further away when they have very different groove characteristics.

    Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
    where margin = base_margin + margin_scale * ||radar_anchor - radar_negative||
    """

    def __init__(self,
                 base_margin: float = 1.0,
                 margin_scale: float = 0.5,
                 p: int = 2,
                 reduction: str = 'mean'):
        """
        Initialize triplet loss with radar-weighted margins.

        Args:
            base_margin: Base margin value
            margin_scale: Scale factor for radar distance contribution to margin
            p: Norm degree for distance computation (1 or 2)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.base_margin = base_margin
        self.margin_scale = margin_scale
        self.p = p
        self.reduction = reduction

    def forward(self,
                anchor_emb: torch.Tensor,
                positive_emb: torch.Tensor,
                negative_emb: torch.Tensor,
                anchor_radar: torch.Tensor,
                negative_radar: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss with adaptive margins.

        Args:
            anchor_emb: (B, D) anchor embeddings from projection head
            positive_emb: (B, D) positive embeddings
            negative_emb: (B, D) negative embeddings
            anchor_radar: (B, 5) anchor groove radar vectors (normalized)
            negative_radar: (B, 5) negative groove radar vectors (normalized)

        Returns:
            Scalar loss value (or per-sample if reduction='none')
        """
        # Compute adaptive margin based on groove radar distance
        radar_dist = torch.norm(anchor_radar - negative_radar, p=2, dim=1)
        margins = self.base_margin + self.margin_scale * radar_dist

        # Compute embedding distances
        dist_pos = torch.norm(anchor_emb - positive_emb, p=self.p, dim=1)
        dist_neg = torch.norm(anchor_emb - negative_emb, p=self.p, dim=1)

        # Triplet loss with per-sample margins
        losses = F.relu(dist_pos - dist_neg + margins)

        # Apply reduction
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses


class TripletMarginLoss(nn.Module):
    """
    Standard triplet margin loss without radar weighting.

    Provided for comparison and ablation studies.
    """

    def __init__(self,
                 margin: float = 1.0,
                 p: int = 2,
                 reduction: str = 'mean'):
        """
        Initialize standard triplet loss.

        Args:
            margin: Margin value
            p: Norm degree for distance computation
            reduction: Reduction method
        """
        super().__init__()
        self.margin = margin
        self.p = p
        self.reduction = reduction
        self._triplet_loss = nn.TripletMarginLoss(
            margin=margin, p=p, reduction=reduction
        )

    def forward(self,
                anchor_emb: torch.Tensor,
                positive_emb: torch.Tensor,
                negative_emb: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """
        Compute standard triplet loss.

        Args:
            anchor_emb: (B, D) anchor embeddings
            positive_emb: (B, D) positive embeddings
            negative_emb: (B, D) negative embeddings
            **kwargs: Ignored (for API compatibility)

        Returns:
            Scalar loss value
        """
        return self._triplet_loss(anchor_emb, positive_emb, negative_emb)


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss (used in SimCLR, CLIP, etc.).

    Treats each positive pair against all other samples in the batch as negatives.
    Uses cosine similarity and temperature scaling.
    """

    def __init__(self,
                 temperature: float = 0.07,
                 normalize: bool = True):
        """
        Initialize InfoNCE loss.

        Args:
            temperature: Temperature for scaling similarities
            normalize: Whether to L2-normalize embeddings
        """
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize

    def forward(self,
                anchor_emb: torch.Tensor,
                positive_emb: torch.Tensor,
                negative_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Can be used in two modes:
        1. With explicit negatives: (anchor, positive, negative) triplets
        2. Batch-wise: anchor and positive, using other batch samples as negatives

        Args:
            anchor_emb: (B, D) anchor embeddings
            positive_emb: (B, D) positive embeddings
            negative_emb: Optional (B, D) explicit negative embeddings

        Returns:
            Scalar loss value
        """
        if self.normalize:
            anchor_emb = F.normalize(anchor_emb, p=2, dim=1)
            positive_emb = F.normalize(positive_emb, p=2, dim=1)
            if negative_emb is not None:
                negative_emb = F.normalize(negative_emb, p=2, dim=1)

        batch_size = anchor_emb.size(0)

        if negative_emb is not None:
            # Explicit triplet mode
            # Positive similarity
            pos_sim = (anchor_emb * positive_emb).sum(dim=1) / self.temperature

            # Negative similarity
            neg_sim = (anchor_emb * negative_emb).sum(dim=1) / self.temperature

            # Stack and compute softmax loss (positive should have index 0)
            logits = torch.stack([pos_sim, neg_sim], dim=1)  # (B, 2)
            labels = torch.zeros(batch_size, dtype=torch.long, device=anchor_emb.device)

            return F.cross_entropy(logits, labels)

        else:
            # Batch-wise mode: use all batch samples as negatives
            # Compute similarity matrix between anchors and all embeddings
            all_embeddings = torch.cat([positive_emb, anchor_emb], dim=0)  # (2B, D)

            sim_matrix = torch.matmul(anchor_emb, all_embeddings.T) / self.temperature
            # sim_matrix: (B, 2B) where first B columns are positives

            # Labels: positive is at index i for anchor i
            labels = torch.arange(batch_size, device=anchor_emb.device)

            return F.cross_entropy(sim_matrix, labels)


class NTXentLoss(nn.Module):
    """
    NT-Xent loss (Normalized Temperature-scaled Cross Entropy).

    Symmetric version of InfoNCE used in SimCLR.
    Computes loss for both (anchor->positive) and (positive->anchor) directions.
    """

    def __init__(self,
                 temperature: float = 0.5,
                 normalize: bool = True):
        """
        Initialize NT-Xent loss.

        Args:
            temperature: Temperature for scaling
            normalize: Whether to L2-normalize embeddings
        """
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize

    def forward(self,
                z_i: torch.Tensor,
                z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss for a batch of positive pairs.

        Args:
            z_i: (B, D) first view embeddings
            z_j: (B, D) second view embeddings (positive pairs)

        Returns:
            Scalar loss value
        """
        batch_size = z_i.size(0)

        if self.normalize:
            z_i = F.normalize(z_i, p=2, dim=1)
            z_j = F.normalize(z_j, p=2, dim=1)

        # Concatenate both views
        z = torch.cat([z_i, z_j], dim=0)  # (2B, D)

        # Compute similarity matrix
        sim_matrix = torch.matmul(z, z.T) / self.temperature  # (2B, 2B)

        # Mask out self-similarities
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

        # Create labels: positive pair for z_i[k] is z_j[k] at index (k + batch_size)
        labels_i = torch.arange(batch_size, device=z.device) + batch_size
        labels_j = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels_i, labels_j], dim=0)

        # Compute loss
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


def create_contrastive_loss(loss_type: str = 'triplet',
                            **kwargs) -> nn.Module:
    """
    Factory function to create contrastive loss.

    Args:
        loss_type: Type of loss ('triplet', 'triplet_radar', 'infonce', 'ntxent')
        **kwargs: Loss-specific parameters

    Returns:
        Loss module
    """
    if loss_type == 'triplet':
        return TripletMarginLoss(
            margin=kwargs.get('margin', 1.0),
            p=kwargs.get('p', 2)
        )
    elif loss_type == 'triplet_radar':
        return TripletMarginLossWithRadar(
            base_margin=kwargs.get('base_margin', 1.0),
            margin_scale=kwargs.get('margin_scale', 0.5),
            p=kwargs.get('p', 2)
        )
    elif loss_type == 'infonce':
        return InfoNCELoss(
            temperature=kwargs.get('temperature', 0.07)
        )
    elif loss_type == 'ntxent':
        return NTXentLoss(
            temperature=kwargs.get('temperature', 0.5)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

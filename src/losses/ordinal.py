import torch


def encode_ordinal_targets(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Encode 0-indexed class labels as ordinal targets for BCEWithLogitsLoss.

    Args:
        labels: Class indices (B,) with values 0..K-1 (0-indexed)
        num_classes: Total number of classes K

    Returns:
        Ordinal targets (B, K-1) where target[b,k] = 1 if labels[b] > k
    """
    # Labels are always 0-indexed (0 to num_classes-1)
    # Convert to 1-indexed for threshold comparison
    labels_1indexed = labels + 1

    # Thresholds represent "class >= threshold"
    # For K=10: thresholds = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    thresholds = torch.arange(2, num_classes + 1, device=labels.device)

    # ordinal_targets[k] = 1 if label >= threshold[k]
    return (labels_1indexed.unsqueeze(1) >= thresholds).float()

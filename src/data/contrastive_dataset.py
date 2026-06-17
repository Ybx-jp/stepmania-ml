"""
Contrastive triplet dataset wrapper for StepMania charts.

Wraps the base StepManiaDataset to serve triplets (anchor, positive, negative)
for contrastive learning based on groove radar similarity.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import StepManiaDataset
from .groove_radar import GrooveRadar
from .similarity import GrooveRadarSimilarity, TripletSelector


class ContrastiveTripletDataset(Dataset):
    """
    Wrapper dataset that serves triplets (anchor, positive, negative)
    based on groove radar similarity.

    Supports multi-task learning by returning:
    - Triplet samples for contrastive loss
    - Difficulty labels for classification loss
    """

    def __init__(self,
                 base_dataset: StepManiaDataset,
                 triplet_selector: Optional[TripletSelector] = None,
                 similarity_weights: Optional[np.ndarray] = None,
                 positive_percentile: float = 20.0,
                 negative_percentile: float = 80.0,
                 precompute_triplets: bool = True,
                 resample_epoch: bool = False,
                 max_triplets_per_anchor: int = 1):
        """
        Initialize contrastive triplet dataset.

        Args:
            base_dataset: Underlying StepManiaDataset instance
            triplet_selector: Optional pre-configured TripletSelector
            similarity_weights: Weights for groove radar dimensions
                               [stream, voltage, air, freeze, chaos]
            positive_percentile: Percentile threshold for positive pairs
            negative_percentile: Percentile threshold for negative pairs
            precompute_triplets: If True, compute all triplets upfront (static)
            resample_epoch: If True, resample triplets each epoch (call resample())
            max_triplets_per_anchor: Max triplets per anchor sample
        """
        self.base_dataset = base_dataset
        self.precompute_triplets = precompute_triplets
        self.resample_epoch = resample_epoch
        self.max_triplets_per_anchor = max_triplets_per_anchor

        # Extract groove radars from base dataset
        self.groove_radars = self._extract_groove_radars()

        # Extract difficulty classes
        self.difficulty_classes = [
            s['difficulty_class'] for s in base_dataset.valid_samples
        ]

        # Create or use provided triplet selector
        if triplet_selector is not None:
            self.triplet_selector = triplet_selector
        else:
            similarity_fn = GrooveRadarSimilarity(weights=similarity_weights)
            self.triplet_selector = TripletSelector(
                similarity_fn=similarity_fn,
                positive_percentile=positive_percentile,
                negative_percentile=negative_percentile,
                same_difficulty_only=False,  # Any difficulty (per plan)
                hard_mining=True
            )

        # Fit triplet selector if not already fitted
        if not self.triplet_selector._fitted:
            print("Fitting triplet selector...")
            self.triplet_selector.fit(self.groove_radars, self.difficulty_classes)

        # Print statistics
        stats = self.triplet_selector.get_statistics(
            self.groove_radars, self.difficulty_classes
        )
        print(f"Triplet selection stats: {stats['has_both']}/{stats['total_samples']} "
              f"({stats['valid_anchor_ratio']:.1%}) samples can form valid triplets")

        # Precompute triplets if requested
        self.triplets: Optional[List[Tuple[int, int, int]]] = None
        if precompute_triplets:
            self._precompute_triplets()

    def _extract_groove_radars(self) -> List[GrooveRadar]:
        """Extract groove radar objects from all samples in base dataset."""
        return [s['groove_radar'] for s in self.base_dataset.valid_samples]

    def _precompute_triplets(self):
        """Precompute all valid triplets."""
        print("Precomputing triplets...")
        self.triplets = self.triplet_selector.select_all_triplets(
            self.groove_radars,
            self.difficulty_classes,
            max_triplets_per_anchor=self.max_triplets_per_anchor
        )
        print(f"Precomputed {len(self.triplets)} triplets")

    def resample(self):
        """
        Resample triplets.

        Call between epochs if resample_epoch=True for variety.
        """
        if self.precompute_triplets:
            self._precompute_triplets()

    def __len__(self) -> int:
        """Return number of triplets (or base dataset size if not precomputed)."""
        if self.triplets is not None:
            return len(self.triplets)
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Return triplet sample with all data needed for multi-task learning.

        Returns:
            Dictionary with:
            - 'anchor_*': All fields from base dataset for anchor sample
            - 'positive_*': All fields for positive sample
            - 'negative_*': All fields for negative sample
            - 'anchor_groove_radar': (5,) normalized groove radar vector
            - 'positive_groove_radar': (5,) groove radar vector
            - 'negative_groove_radar': (5,) groove radar vector
        """
        # Get triplet indices
        if self.triplets is not None:
            anchor_idx, positive_idx, negative_idx = self.triplets[idx]
        else:
            # Dynamic selection (slower but more varied)
            anchor_idx = idx
            result = self.triplet_selector.select_triplet(
                anchor_idx, self.groove_radars, self.difficulty_classes
            )
            if result is None:
                # Fallback: use random indices (not ideal but prevents crash)
                positive_idx = np.random.randint(len(self.base_dataset))
                negative_idx = np.random.randint(len(self.base_dataset))
                while positive_idx == anchor_idx:
                    positive_idx = np.random.randint(len(self.base_dataset))
                while negative_idx == anchor_idx or negative_idx == positive_idx:
                    negative_idx = np.random.randint(len(self.base_dataset))
            else:
                positive_idx, negative_idx = result

        # Get base samples
        anchor = self.base_dataset[anchor_idx]
        positive = self.base_dataset[positive_idx]
        negative = self.base_dataset[negative_idx]

        # Build result with prefixed keys
        result = {}

        # Anchor fields
        for key, value in anchor.items():
            result[f'anchor_{key}'] = value

        # Positive fields
        for key, value in positive.items():
            result[f'positive_{key}'] = value

        # Negative fields
        for key, value in negative.items():
            result[f'negative_{key}'] = value

        return result

    def get_triplet_statistics(self) -> Dict:
        """
        Get statistics about the triplet dataset.

        Returns:
            Dictionary with triplet dataset statistics
        """
        base_stats = self.triplet_selector.get_statistics(
            self.groove_radars, self.difficulty_classes
        )

        return {
            **base_stats,
            'num_triplets': len(self.triplets) if self.triplets else 0,
            'base_dataset_size': len(self.base_dataset),
            'precomputed': self.triplets is not None,
            'resample_epoch': self.resample_epoch
        }


def create_contrastive_dataset(
    base_dataset: StepManiaDataset,
    radar_weights: Optional[List[float]] = None,
    positive_percentile: float = 20.0,
    negative_percentile: float = 80.0,
    precompute: bool = True,
    resample: bool = False
) -> ContrastiveTripletDataset:
    """
    Convenience function to create a contrastive triplet dataset.

    Args:
        base_dataset: StepManiaDataset instance
        radar_weights: Weights for [stream, voltage, air, freeze, chaos]
        positive_percentile: Percentile for positive threshold
        negative_percentile: Percentile for negative threshold
        precompute: Whether to precompute triplets
        resample: Whether to resample triplets each epoch

    Returns:
        ContrastiveTripletDataset instance
    """
    weights = np.array(radar_weights) if radar_weights else None

    return ContrastiveTripletDataset(
        base_dataset=base_dataset,
        similarity_weights=weights,
        positive_percentile=positive_percentile,
        negative_percentile=negative_percentile,
        precompute_triplets=precompute,
        resample_epoch=resample
    )

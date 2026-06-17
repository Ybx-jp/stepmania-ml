"""
Groove radar similarity computation and triplet selection for contrastive learning.

Provides:
- GrooveRadarSimilarity: Compute weighted distance between groove radar vectors
- TripletSelector: Select positive/negative pairs based on groove radar similarity
"""

from typing import List, Tuple, Optional, Dict
import numpy as np

from .groove_radar import GrooveRadar


class GrooveRadarSimilarity:
    """
    Compute similarity between charts based on groove radar vectors.

    Supports weighted distance computation with multiple metrics.
    """

    def __init__(self,
                 weights: Optional[np.ndarray] = None,
                 metric: str = 'euclidean'):
        """
        Initialize similarity calculator.

        Args:
            weights: Per-dimension weights for radar values
                     [stream, voltage, air, freeze, chaos]
                     Default: equal weights, normalized to sum to 1
            metric: Distance metric ('euclidean', 'cosine', 'manhattan')
        """
        if weights is not None:
            self.weights = np.array(weights, dtype=np.float32)
            self.weights = self.weights / self.weights.sum()  # Normalize
        else:
            self.weights = np.ones(5, dtype=np.float32) / 5.0

        self.metric = metric

    def distance(self, radar1: GrooveRadar, radar2: GrooveRadar) -> float:
        """
        Compute weighted distance between two groove radars.

        Args:
            radar1: First groove radar
            radar2: Second groove radar

        Returns:
            Distance value (lower = more similar)
        """
        v1 = radar1.to_vector() * np.sqrt(self.weights)
        v2 = radar2.to_vector() * np.sqrt(self.weights)

        if self.metric == 'euclidean':
            return float(np.linalg.norm(v1 - v2))
        elif self.metric == 'cosine':
            dot = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 < 1e-8 or norm2 < 1e-8:
                return 1.0  # Maximum distance for zero vectors
            return float(1.0 - dot / (norm1 * norm2))
        elif self.metric == 'manhattan':
            return float(np.abs(v1 - v2).sum())
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def similarity(self, radar1: GrooveRadar, radar2: GrooveRadar,
                   temperature: float = 0.5) -> float:
        """
        Convert distance to similarity score.

        Uses exponential decay: sim = exp(-dist / temperature)

        Args:
            radar1: First groove radar
            radar2: Second groove radar
            temperature: Controls decay rate (higher = slower decay)

        Returns:
            Similarity score in [0, 1] where 1 = identical
        """
        dist = self.distance(radar1, radar2)
        return float(np.exp(-dist / temperature))

    def distance_matrix(self, radars: List[GrooveRadar]) -> np.ndarray:
        """
        Compute pairwise distance matrix for a list of groove radars.

        Args:
            radars: List of GrooveRadar objects

        Returns:
            (N, N) distance matrix
        """
        n = len(radars)
        matrix = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            for j in range(i + 1, n):
                dist = self.distance(radars[i], radars[j])
                matrix[i, j] = dist
                matrix[j, i] = dist

        return matrix


class TripletSelector:
    """
    Select triplets (anchor, positive, negative) based on groove radar similarity.

    Uses percentile-based thresholds computed from the dataset distribution.
    Supports hard mining strategies.
    """

    def __init__(self,
                 similarity_fn: GrooveRadarSimilarity,
                 positive_percentile: float = 20.0,
                 negative_percentile: float = 80.0,
                 same_difficulty_only: bool = False,
                 hard_mining: bool = True):
        """
        Initialize triplet selector.

        Args:
            similarity_fn: GrooveRadarSimilarity instance for distance computation
            positive_percentile: Charts within this percentile of distances
                                 are considered "positive" (similar)
            negative_percentile: Charts beyond this percentile of distances
                                 are considered "negative" (dissimilar)
            same_difficulty_only: If True, only match within same difficulty class
            hard_mining: If True, prefer hardest positives and negatives
        """
        self.similarity_fn = similarity_fn
        self.positive_percentile = positive_percentile
        self.negative_percentile = negative_percentile
        self.same_difficulty_only = same_difficulty_only
        self.hard_mining = hard_mining

        # Computed from dataset
        self.positive_threshold: Optional[float] = None
        self.negative_threshold: Optional[float] = None
        self._fitted = False

    def fit(self, groove_radars: List[GrooveRadar],
            difficulty_classes: Optional[List[int]] = None,
            sample_size: int = 1000) -> 'TripletSelector':
        """
        Compute thresholds from dataset distribution.

        Samples pairwise distances and computes percentile thresholds.

        Args:
            groove_radars: List of GrooveRadar objects from dataset
            difficulty_classes: Optional list of difficulty class indices
            sample_size: Number of samples for threshold estimation

        Returns:
            self (for method chaining)
        """
        n = len(groove_radars)
        if n < 2:
            raise ValueError("Need at least 2 samples to fit thresholds")

        # Sample pairs for threshold estimation
        n_pairs = min(sample_size * sample_size // 2, n * (n - 1) // 2)
        n_samples = min(n, sample_size)
        indices = np.random.choice(n, n_samples, replace=False)

        distances = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx_i, idx_j = indices[i], indices[j]

                # Skip if same_difficulty_only and difficulties differ
                if self.same_difficulty_only and difficulty_classes is not None:
                    if difficulty_classes[idx_i] != difficulty_classes[idx_j]:
                        continue

                dist = self.similarity_fn.distance(
                    groove_radars[idx_i],
                    groove_radars[idx_j]
                )
                distances.append(dist)

        if len(distances) < 10:
            raise ValueError("Not enough valid pairs for threshold estimation")

        distances = np.array(distances)
        self.positive_threshold = float(np.percentile(distances, self.positive_percentile))
        self.negative_threshold = float(np.percentile(distances, self.negative_percentile))
        self._fitted = True

        print(f"TripletSelector fitted: positive_threshold={self.positive_threshold:.4f}, "
              f"negative_threshold={self.negative_threshold:.4f}")

        return self

    def select_triplet(self,
                       anchor_idx: int,
                       groove_radars: List[GrooveRadar],
                       difficulty_classes: Optional[List[int]] = None,
                       exclude_indices: Optional[set] = None) -> Optional[Tuple[int, int]]:
        """
        Select positive and negative indices for an anchor.

        Args:
            anchor_idx: Index of anchor sample
            groove_radars: List of all GrooveRadar objects
            difficulty_classes: Optional list of difficulty class indices
            exclude_indices: Set of indices to exclude from selection

        Returns:
            Tuple of (positive_idx, negative_idx) or None if no valid triplet
        """
        if not self._fitted:
            raise RuntimeError("TripletSelector not fitted. Call fit() first.")

        if exclude_indices is None:
            exclude_indices = set()

        anchor_radar = groove_radars[anchor_idx]
        anchor_difficulty = difficulty_classes[anchor_idx] if difficulty_classes else None

        positives = []  # (idx, distance)
        negatives = []  # (idx, distance)

        for i, radar in enumerate(groove_radars):
            if i == anchor_idx or i in exclude_indices:
                continue

            # Same difficulty filter
            if self.same_difficulty_only and difficulty_classes is not None:
                if difficulty_classes[i] != anchor_difficulty:
                    continue

            dist = self.similarity_fn.distance(anchor_radar, radar)

            if dist <= self.positive_threshold:
                positives.append((i, dist))
            elif dist >= self.negative_threshold:
                negatives.append((i, dist))

        # Cannot form valid triplet
        if not positives or not negatives:
            return None

        if self.hard_mining:
            # Hardest positive: largest distance within positive range
            positives.sort(key=lambda x: x[1], reverse=True)
            positive_idx = positives[0][0]

            # Hardest negative: smallest distance within negative range
            negatives.sort(key=lambda x: x[1])
            negative_idx = negatives[0][0]
        else:
            # Random selection
            positive_idx = positives[np.random.randint(len(positives))][0]
            negative_idx = negatives[np.random.randint(len(negatives))][0]

        return positive_idx, negative_idx

    def select_all_triplets(self,
                            groove_radars: List[GrooveRadar],
                            difficulty_classes: Optional[List[int]] = None,
                            max_triplets_per_anchor: int = 1) -> List[Tuple[int, int, int]]:
        """
        Select triplets for all samples in the dataset.

        Args:
            groove_radars: List of all GrooveRadar objects
            difficulty_classes: Optional list of difficulty class indices
            max_triplets_per_anchor: Maximum triplets per anchor sample

        Returns:
            List of (anchor_idx, positive_idx, negative_idx) tuples
        """
        triplets = []

        for anchor_idx in range(len(groove_radars)):
            for _ in range(max_triplets_per_anchor):
                result = self.select_triplet(
                    anchor_idx, groove_radars, difficulty_classes
                )
                if result is not None:
                    positive_idx, negative_idx = result
                    triplets.append((anchor_idx, positive_idx, negative_idx))

        return triplets

    def get_statistics(self, groove_radars: List[GrooveRadar],
                       difficulty_classes: Optional[List[int]] = None) -> Dict:
        """
        Get statistics about triplet selection.

        Args:
            groove_radars: List of all GrooveRadar objects
            difficulty_classes: Optional list of difficulty class indices

        Returns:
            Dictionary with statistics about positive/negative availability
        """
        n = len(groove_radars)
        has_positive = 0
        has_negative = 0
        has_both = 0

        for anchor_idx in range(n):
            anchor_radar = groove_radars[anchor_idx]
            anchor_difficulty = difficulty_classes[anchor_idx] if difficulty_classes else None

            found_positive = False
            found_negative = False

            for i, radar in enumerate(groove_radars):
                if i == anchor_idx:
                    continue

                if self.same_difficulty_only and difficulty_classes is not None:
                    if difficulty_classes[i] != anchor_difficulty:
                        continue

                dist = self.similarity_fn.distance(anchor_radar, radar)

                if dist <= self.positive_threshold:
                    found_positive = True
                elif dist >= self.negative_threshold:
                    found_negative = True

                if found_positive and found_negative:
                    break

            if found_positive:
                has_positive += 1
            if found_negative:
                has_negative += 1
            if found_positive and found_negative:
                has_both += 1

        return {
            'total_samples': n,
            'has_positive': has_positive,
            'has_negative': has_negative,
            'has_both': has_both,
            'valid_anchor_ratio': has_both / n if n > 0 else 0,
            'positive_threshold': self.positive_threshold,
            'negative_threshold': self.negative_threshold
        }

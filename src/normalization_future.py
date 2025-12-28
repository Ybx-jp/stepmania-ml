def normalize_features(self, features: AudioFeatures,
                       method: str = 'standardize') -> AudioFeatures:
    """
    Normalize audio features for model training.

    Args:
        features: AudioFeatures object
        method: 'standardize' (z-score) or 'minmax'
    """
    mfcc = features.mfcc.copy()

    if method == 'standardize':
        # Z-score normalization across time dimension
        mean = np.mean(mfcc, axis=1, keepdims=True)
        std = np.std(mfcc, axis=1, keepdims=True)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        mfcc = (mfcc - mean) / std

    elif method == 'minmax':
        # Min-max normalization to [0, 1]
        min_vals = np.min(mfcc, axis=1, keepdims=True)
        max_vals = np.max(mfcc, axis=1, keepdims=True)
        range_vals = max_vals - min_vals
        range_vals = np.where(range_vals == 0, 1, range_vals)  # Avoid division by zero
        mfcc = (mfcc - min_vals) / range_vals

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return AudioFeatures(
        mfcc=mfcc,
        audio_duration=features.audio_duration,
        sample_rate=features.sample_rate,
        hop_length=features.hop_length,
        n_frames=features.n_frames
    )
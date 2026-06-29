"""Test 3: Dataset sample contract test + Test 4: Batch smoke test"""

import os
import sys
import torch

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from torch.utils.data import DataLoader
from src.data.dataset import StepManiaDataset
from src.data.stepmania_parser import StepManiaParser

# The default parser filters charts shorter than 75s (a real-song gate); the unit-test
# fixture is ~4s, so inject a parser with a relaxed window or the dataset comes up empty.
def _fixture_parser():
    return StepManiaParser(min_song_length=1, max_song_length=100000)

def test_dataset_sample_contract():
    """Test dataset[0] returns correct contract"""
    chart_path = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_chart.sm')
    audio_dir = os.path.join(os.path.dirname(__file__), 'fixtures')

    # Ensure test audio exists
    audio_path = os.path.join(audio_dir, 'test_audio.wav')
    if not os.path.exists(audio_path):
        import numpy as np
        import soundfile as sf
        sr = 22050
        duration = 10.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.1 * np.sin(2 * np.pi * 440 * t)
        sf.write(audio_path, audio, sr)

    dataset = StepManiaDataset(
        chart_files=[chart_path],
        audio_dir=audio_dir,
        parser=_fixture_parser(),
        max_sequence_length=1200
    )

    assert len(dataset) > 0, "Dataset should have at least one sample"

    # Get sample
    sample = dataset[0]

    # Assert: the required keys exist. The contract is "at least these"; the dataset also
    # carries additive metadata (e.g. groove_radar, difficulty_value) that downstream code
    # ignores when unused, so use a subset check rather than exact equality.
    required_keys = {'chart', 'audio', 'mask', 'difficulty', 'length'}
    assert required_keys.issubset(sample.keys()), \
        f"Missing required keys {required_keys - set(sample.keys())}; got {set(sample.keys())}"

    # Assert: dtypes are correct
    assert sample['chart'].dtype == torch.float32, f"chart dtype: expected float32, got {sample['chart'].dtype}"
    assert sample['audio'].dtype == torch.float32, f"audio dtype: expected float32, got {sample['audio'].dtype}"
    assert sample['mask'].dtype == torch.bool, f"mask dtype: expected bool, got {sample['mask'].dtype}"
    assert sample['difficulty'].dtype == torch.long, f"difficulty dtype: expected long, got {sample['difficulty'].dtype}"
    assert isinstance(sample['length'], int), f"length type: expected int, got {type(sample['length'])}"

    # Assert: mask.sum() == length
    assert sample['mask'].sum().item() == sample['length'], \
        f"Mask sum {sample['mask'].sum().item()} != length {sample['length']}"

    # Assert: padded region is actually padded
    mask = sample['mask']
    chart = sample['chart']
    audio = sample['audio']

    if len(mask) > sample['length']:  # Only test if there's padding
        padding_start = sample['length']
        # Check chart padding is zeros
        assert torch.all(chart[padding_start:] == 0), "Chart padding should be all zeros"
        # Check audio padding is zeros
        assert torch.all(audio[padding_start:] == 0), "Audio padding should be all zeros"


def test_dataloader_batch():
    """Test 4: Batch smoke test"""
    chart_path = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_chart.sm')
    audio_dir = os.path.join(os.path.dirname(__file__), 'fixtures')

    dataset = StepManiaDataset(
        chart_files=[chart_path],
        audio_dir=audio_dir,
        parser=_fixture_parser(),
        max_sequence_length=1200
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Pull one batch - should not raise exceptions
    batch = next(iter(dataloader))

    # Assert: batch shapes are correct
    assert batch['chart'].shape == (1, 1200, 4), f"Chart batch shape: expected (1, 1200, 4), got {batch['chart'].shape}"
    assert batch['audio'].shape == (1, 1200, 23), f"Audio batch shape: expected (1, 1200, 23), got {batch['audio'].shape}"
    assert batch['mask'].shape == (1, 1200), f"Mask batch shape: expected (1, 1200), got {batch['mask'].shape}"
    assert batch['difficulty'].shape == (1,), f"Difficulty batch shape: expected (1,), got {batch['difficulty'].shape}"


def test_cache_identity_rejects_stale_index():
    """Regression for the INDEX-KEYED cache footgun (notes/cache_index_bug.md): a cache entry written for one
    song must NOT be served when the same index later maps to a DIFFERENT song (subset/--match probes)."""
    import tempfile, shutil
    chart_path = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_chart.sm')
    audio_dir = os.path.join(os.path.dirname(__file__), 'fixtures')
    cache_dir = tempfile.mkdtemp()
    try:
        ds = StepManiaDataset(chart_files=[chart_path], audio_dir=audio_dir, parser=_fixture_parser(),
                              max_sequence_length=1200, cache_dir=cache_dir)
        _ = ds[0]                                        # writes sample_000000.pt stamped with song-0 identity
        assert ds._load_from_cache(0) is not None, "freshly-cached sample must round-trip"
        # Simulate index 0 now mapping to a DIFFERENT song (a different file set at the same index):
        ds.valid_samples[0] = {**ds.valid_samples[0], 'chart_file': '/some/other/song.sm'}
        assert ds._load_from_cache(0) is None, "a stale index (different song identity) must be REJECTED"
        assert ds._is_cached(0) is False, "warm_cache must re-warm a stale index, not skip it"
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
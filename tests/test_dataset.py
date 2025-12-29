"""Test 3: Dataset sample contract test + Test 4: Batch smoke test"""

import os
import sys
import torch

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from torch.utils.data import DataLoader
from src.data.dataset import StepManiaDataset

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
        max_sequence_length=1200
    )

    assert len(dataset) > 0, "Dataset should have at least one sample"

    # Get sample
    sample = dataset[0]

    # Assert: keys exist
    required_keys = {'chart', 'audio', 'mask', 'difficulty', 'length'}
    assert set(sample.keys()) == required_keys, f"Expected keys {required_keys}, got {set(sample.keys())}"

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
        max_sequence_length=1200
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Pull one batch - should not raise exceptions
    batch = next(iter(dataloader))

    # Assert: batch shapes are correct
    assert batch['chart'].shape == (1, 1200, 4), f"Chart batch shape: expected (1, 1200, 4), got {batch['chart'].shape}"
    assert batch['audio'].shape == (1, 1200, 13), f"Audio batch shape: expected (1, 1200, 13), got {batch['audio'].shape}"
    assert batch['mask'].shape == (1, 1200), f"Mask batch shape: expected (1, 1200), got {batch['mask'].shape}"
    assert batch['difficulty'].shape == (1,), f"Difficulty batch shape: expected (1,), got {batch['difficulty'].shape}"
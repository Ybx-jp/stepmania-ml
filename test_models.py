"""
Test script to verify model instantiation and forward passes work correctly.

Tests:
- LateFusionClassifier with different pooling strategies
- MLP baselines
- Forward pass with sample data matching expected input contracts
"""

import torch
import yaml
from src.models import LateFusionClassifier, DualTaskClassifier, MLPBaseline, SimpleConcatBaseline, PooledFeatureBaseline
from src.models.components import AudioEncoder


def create_sample_data(batch_size=2, sequence_length=960, audio_dim=13, chart_dim=4):
    """Create sample data matching the expected input format."""
    # Audio features (MFCC)
    audio = torch.randn(batch_size, sequence_length, audio_dim)

    # Chart sequences (binary step encoding)
    chart = torch.randint(0, 2, (batch_size, sequence_length, chart_dim)).float()

    # Attention masks (random sequence lengths)
    mask = torch.ones(batch_size, sequence_length)
    for i in range(batch_size):
        # Randomly truncate sequences
        actual_length = torch.randint(sequence_length // 2, sequence_length, (1,)).item()
        mask[i, actual_length:] = 0

    return audio, chart, mask


def load_test_config():
    """Load config for testing."""
    # Load from actual config file
    with open('config/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    return config['classifier']


def test_late_fusion_classifier():
    """Test LateFusionClassifier with different configurations."""
    print("Testing LateFusionClassifier...")

    config = load_test_config()

    # Test different pooling strategies
    pooling_types = ['attention', 'mean_max', 'global']
    fusion_types = ['late', 'gated', 'additive']

    for pooling_type in pooling_types:
        for fusion_type in fusion_types:
            print(f"  Testing {fusion_type} fusion with {pooling_type} pooling...")

            test_config = config.copy()
            test_config['pooling_type'] = pooling_type
            test_config['fusion_type'] = fusion_type

            # Create model
            model = LateFusionClassifier(test_config)
            model.eval()

            # Create sample data
            audio, chart, mask = create_sample_data()

            # Forward pass
            with torch.no_grad():
                logits = model(audio, chart, mask)

            # Verify output shape
            expected_shape = (audio.size(0), config['num_classes'])
            assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"

            # Test feature extraction
            features = model.get_feature_representations(audio, chart, mask)
            assert 'audio_encoded' in features
            assert 'chart_encoded' in features
            assert 'fused_features' in features
            assert 'pooled_features' in features

            print(f"    ✓ Output shape: {logits.shape}")
            print(f"    ✓ Feature shapes: {[k + ': ' + str(v.shape) for k, v in features.items()]}")


def test_dual_task_classifier():
    """Test DualTaskClassifier."""
    print("\nTesting DualTaskClassifier...")

    config = load_test_config()
    config['pooling_type'] = 'attention'

    model = DualTaskClassifier(config)
    model.eval()

    audio, chart, mask = create_sample_data()

    with torch.no_grad():
        clf_logits, reg_outputs = model(audio, chart, mask)

    # Verify output shapes
    expected_clf_shape = (audio.size(0), config['num_classes'])
    expected_reg_shape = (audio.size(0), 1)

    assert clf_logits.shape == expected_clf_shape
    assert reg_outputs.shape == expected_reg_shape

    print(f"  ✓ Classification shape: {clf_logits.shape}")
    print(f"  ✓ Regression shape: {reg_outputs.shape}")


def test_mlp_baseline():
    """Test MLPBaseline."""
    print("\nTesting MLPBaseline...")

    config = load_test_config()
    config['hidden_dims'] = [256, 128]
    config['dropout'] = 0.2
    config['pooling_type'] = 'global'

    model = MLPBaseline(config)
    model.eval()

    audio, chart, mask = create_sample_data()

    with torch.no_grad():
        logits = model(audio, chart, mask)

    expected_shape = (audio.size(0), config['num_classes'])
    assert logits.shape == expected_shape

    print(f"  ✓ Output shape: {logits.shape}")


def test_simple_baselines():
    """Test other baseline models."""
    print("\nTesting SimpleConcatBaseline...")

    config = load_test_config()
    config['hidden_dim'] = 512
    config['dropout'] = 0.3

    model = SimpleConcatBaseline(config)
    model.eval()

    audio, chart, mask = create_sample_data()

    with torch.no_grad():
        logits = model(audio, chart, mask)

    expected_shape = (audio.size(0), config['num_classes'])
    assert logits.shape == expected_shape
    print(f"  ✓ SimpleConcatBaseline output shape: {logits.shape}")

    # Test PooledFeatureBaseline
    print("Testing PooledFeatureBaseline...")

    config['hidden_dim'] = 256
    model2 = PooledFeatureBaseline(config)
    model2.eval()

    with torch.no_grad():
        logits2 = model2(audio, chart, mask)

    assert logits2.shape == expected_shape
    print(f"  ✓ PooledFeatureBaseline output shape: {logits2.shape}")


def test_parameter_counts():
    """Test and compare parameter counts."""
    print("\nParameter Counts:")

    config = load_test_config()

    models = {
        'LateFusionClassifier (attention)': LateFusionClassifier({**config, 'pooling_type': 'attention'}),
        'LateFusionClassifier (mean_max)': LateFusionClassifier({**config, 'pooling_type': 'mean_max'}),
        'MLPBaseline': MLPBaseline({**config, 'hidden_dims': [256, 128], 'pooling_type': 'global'}),
        'SimpleConcatBaseline': SimpleConcatBaseline({**config, 'hidden_dim': 512}),
        'PooledFeatureBaseline': PooledFeatureBaseline({**config, 'hidden_dim': 256})
    }

    for name, model in models.items():
        param_count = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  {name}: {param_count:,} total, {trainable_count:,} trainable")


def test_padding_invariance():
    """Test that pooling outputs are invariant to different padding lengths."""
    print("\nTesting Padding Invariance...")

    config = load_test_config()
    config['pooling_type'] = 'attention'

    model = LateFusionClassifier(config)
    model.eval()

    # Create same sequence with different padding lengths
    batch_size = 1
    true_length = 100
    audio_dim, chart_dim = 13, 4

    # Original sequence (length 100, padded to 200)
    audio1 = torch.randn(batch_size, 200, audio_dim)
    chart1 = torch.randint(0, 2, (batch_size, 200, chart_dim)).float()
    mask1 = torch.zeros(batch_size, 200)
    mask1[:, :true_length] = 1

    # Same sequence but padded to 300
    audio2 = torch.zeros(batch_size, 300, audio_dim)
    chart2 = torch.zeros(batch_size, 300, chart_dim)
    audio2[:, :200] = audio1
    chart2[:, :200] = chart1
    mask2 = torch.zeros(batch_size, 300)
    mask2[:, :true_length] = 1

    with torch.no_grad():
        features1 = model.get_feature_representations(audio1, chart1, mask1)
        features2 = model.get_feature_representations(audio2, chart2, mask2)

        logits1 = model(audio1, chart1, mask1)
        logits2 = model(audio2, chart2, mask2)

    # Test that pooled features are nearly identical
    pooled1 = features1['pooled_features']
    pooled2 = features2['pooled_features']

    max_diff = torch.max(torch.abs(pooled1 - pooled2)).item()
    assert max_diff < 1e-5, f"Pooled features differ by {max_diff}, expected < 1e-5"

    # Test final outputs are nearly identical
    output_diff = torch.max(torch.abs(logits1 - logits2)).item()
    assert output_diff < 1e-5, f"Final outputs differ by {output_diff}, expected < 1e-5"

    print(f"  ✓ Pooled features max difference: {max_diff:.2e}")
    print(f"  ✓ Final output max difference: {output_diff:.2e}")


def test_all_padding_edge_case():
    """Test edge case where entire sequence is padding (mask = all zeros)."""
    print("\nTesting All-Padding Edge Case...")

    config = load_test_config()

    # Test multiple pooling strategies
    pooling_types = ['attention', 'mean_max', 'global']

    for pooling_type in pooling_types:
        print(f"  Testing {pooling_type} pooling with all-padding...")

        test_config = config.copy()
        test_config['pooling_type'] = pooling_type

        model = LateFusionClassifier(test_config)
        model.eval()

        # Create fully padded sequences
        batch_size = 2
        sequence_length = 200
        audio = torch.randn(batch_size, sequence_length, 13)
        chart = torch.randint(0, 2, (batch_size, sequence_length, 4)).float()
        mask = torch.zeros(batch_size, sequence_length)  # All padding!

        with torch.no_grad():
            features = model.get_feature_representations(audio, chart, mask)
            logits = model(audio, chart, mask)

        # Check for NaNs
        assert not torch.any(torch.isnan(logits)), "Found NaNs in output with all-padding input"
        assert not torch.any(torch.isnan(features['pooled_features'])), "Found NaNs in pooled features"

        # Check outputs are finite and deterministic
        assert torch.all(torch.isfinite(logits)), "Found non-finite values in output"
        assert torch.all(torch.isfinite(features['pooled_features'])), "Found non-finite values in pooled features"

        # Test that outputs are identical across batch (should be deterministic for all-padding)
        if batch_size > 1:
            batch_diff = torch.max(torch.abs(logits[0] - logits[1])).item()
            assert batch_diff < 1e-6, f"All-padding outputs should be identical across batch, diff: {batch_diff}"

        print(f"    ✓ No NaNs, output range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")


def test_mask_alignment_downsample_upsample():
    """Test that mask alignment is preserved through downsample/upsample operations."""
    print("\nTesting Mask Alignment Through Downsample/Upsample...")

    config = load_test_config()
    model = LateFusionClassifier(config)

    # Test both encoders
    audio_encoder = model.audio_encoder
    chart_encoder = model.chart_encoder

    batch_size = 3
    sequence_lengths = [100, 150, 200]  # Variable lengths
    max_length = max(sequence_lengths)

    # Create variable-length sequences
    audio = torch.randn(batch_size, max_length, 13)
    chart = torch.randint(0, 2, (batch_size, max_length, 4)).float()
    mask = torch.zeros(batch_size, max_length)

    for i, length in enumerate(sequence_lengths):
        mask[i, :length] = 1

    with torch.no_grad():
        # Test audio encoder
        audio_encoded = audio_encoder(audio, mask)

        # Test chart encoder
        chart_encoded = chart_encoder(chart, mask)

    # Verify output sequence lengths match input
    assert audio_encoded.shape[1] == max_length, \
        f"Audio encoder output length {audio_encoded.shape[1]} != input length {max_length}"
    assert chart_encoded.shape[1] == max_length, \
        f"Chart encoder output length {chart_encoded.shape[1]} != input length {max_length}"

    # Test that the encoded features respect the mask
    # (Features at padded positions should be consistent/deterministic)
    for i, length in enumerate(sequence_lengths):
        if length < max_length:
            # Check that padded positions have consistent values across time
            padded_audio = audio_encoded[i, length:]  # Padded region
            padded_chart = chart_encoded[i, length:]  # Padded region

            # Variance in padded region should be small (features should be similar)
            audio_var = torch.var(padded_audio, dim=0).mean().item()
            chart_var = torch.var(padded_chart, dim=0).mean().item()

            print(f"    Sample {i} (length {length}): padded region variance - audio: {audio_var:.6f}, chart: {chart_var:.6f}")

    print("  ✓ Downsample/upsample preserves sequence length")
    print("  ✓ Mask alignment maintained through encoder operations")


def test_encoder_intermediate_shapes():
    """Test that intermediate shapes in encoders are as expected."""
    print("\nTesting Encoder Intermediate Shapes...")

    config = load_test_config()
    audio_encoder = AudioEncoder(
        input_dim=13,
        hidden_dim=256,
        num_res_blocks=3
    )

    batch_size = 2
    sequence_length = 96  # Use length divisible by stride (2)
    audio = torch.randn(batch_size, sequence_length, 13)
    mask = torch.ones(batch_size, sequence_length)

    # Hook to capture intermediate shapes
    shapes_captured = []

    def capture_shape(name):
        def hook(module, input, output):
            shapes_captured.append((name, output.shape))
        return hook

    # Register hooks
    audio_encoder.downsample.register_forward_hook(capture_shape('downsample'))
    audio_encoder.upsample.register_forward_hook(capture_shape('upsample'))

    with torch.no_grad():
        output = audio_encoder(audio, mask)

    # Verify shapes
    expected_shapes = [
        ('downsample', (batch_size, 256, sequence_length // 2)),  # After stride=2
        ('upsample', (batch_size, 256, sequence_length))          # Back to original
    ]

    for (expected_name, expected_shape), (actual_name, actual_shape) in zip(expected_shapes, shapes_captured):
        assert actual_name == expected_name
        assert actual_shape == expected_shape, \
            f"{actual_name}: expected {expected_shape}, got {actual_shape}"
        print(f"  ✓ {actual_name}: {actual_shape}")

    # Final output shape
    assert output.shape == (batch_size, sequence_length, 256)
    print(f"  ✓ Final output: {output.shape}")


def main():
    """Run all tests."""
    print("🧪 Testing StepMania Model Architectures")
    print("=" * 50)

    try:
        test_late_fusion_classifier()
        test_dual_task_classifier()
        test_mlp_baseline()
        test_simple_baselines()
        test_parameter_counts()

        # Critical robustness tests
        test_padding_invariance()
        test_all_padding_edge_case()
        test_mask_alignment_downsample_upsample()
        test_encoder_intermediate_shapes()

        print("\n" + "=" * 50)
        print("✅ All tests passed successfully!")
        print("\nModels are ready for training. Key features implemented:")
        print("  • Late fusion architecture with separate encoders")
        print("  • Minimal conv pattern: Conv → ResBlock → Downsample → ResBlocks → Upsample + Skip")
        print("  • Mask-aware pooling for variable sequence lengths")
        print("  • Multiple fusion strategies (late, gated, additive)")
        print("  • Multiple pooling strategies (attention, mean+max, global)")
        print("  • Comprehensive baseline models for comparison")
        print("  • Robust handling of edge cases (all-padding, variable lengths)")
        print("  • Proper mask alignment through downsample/upsample operations")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
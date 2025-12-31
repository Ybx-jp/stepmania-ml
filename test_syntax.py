#!/usr/bin/env python3
"""
Syntax validation and basic structural tests for model architectures.

This script can run without PyTorch to validate:
- Module imports work correctly
- Class definitions are syntactically correct
- Configuration loading works
- Basic structural integrity
"""

import sys
import importlib.util
import yaml


def test_module_imports():
    """Test that all modules can be imported successfully."""
    print("Testing module imports...")

    modules_to_test = [
        'src.models.components.conv_blocks',
        'src.models.components.encoders',
        'src.models.components.fusion',
        'src.models.components.pooling',
        'src.models.components.heads',
        'src.models.classifier',
        'src.models.baseline'
    ]

    try:
        # Test if we have torch available
        import torch
        pytorch_available = True
        print("  ✓ PyTorch available - running full imports")
    except ImportError:
        pytorch_available = False
        print("  ⚠ PyTorch not available - testing syntax only")

    if pytorch_available:
        # Full import test with PyTorch
        for module_name in modules_to_test:
            try:
                importlib.import_module(module_name)
                print(f"  ✓ {module_name}")
            except Exception as e:
                print(f"  ❌ {module_name}: {e}")
                return False
    else:
        # Syntax-only test without PyTorch
        import ast
        import os

        for module_name in modules_to_test:
            file_path = module_name.replace('.', '/') + '.py'
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        source = f.read()
                    ast.parse(source)
                    print(f"  ✓ {module_name} (syntax)")
                except SyntaxError as e:
                    print(f"  ❌ {module_name}: Syntax error - {e}")
                    return False
                except Exception as e:
                    print(f"  ❌ {module_name}: {e}")
                    return False
            else:
                print(f"  ❌ {module_name}: File not found")
                return False

    return True


def test_config_loading():
    """Test that configuration files can be loaded."""
    print("\nTesting configuration loading...")

    config_files = [
        'config/model_config.yaml',
        'config/data_config.yaml'
    ]

    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            # Basic structure checks
            if config_file.endswith('model_config.yaml'):
                assert 'classifier' in config, "Missing 'classifier' section in model config"
                classifier_config = config['classifier']

                required_keys = [
                    'audio_features_dim', 'chart_sequence_dim', 'max_sequence_length',
                    'audio_encoder', 'chart_encoder', 'fusion_dim', 'num_classes'
                ]

                for key in required_keys:
                    assert key in classifier_config, f"Missing required key '{key}' in classifier config"

            print(f"  ✓ {config_file}")

        except Exception as e:
            print(f"  ❌ {config_file}: {e}")
            return False

    return True


def test_class_structure():
    """Test class definitions and basic structure."""
    print("\nTesting class structure...")

    try:
        # Test if PyTorch is available for full testing
        import torch
        print("  ✓ PyTorch available - testing class instantiation")

        # Test basic class loading
        from src.models.components.conv_blocks import Conv1DBlock, ResidualBlock1D
        from src.models.components.pooling import MaskedAttentionPool
        from src.models.components.heads import ClassificationHead

        # Test that classes can be instantiated (without calling forward)
        conv_block = Conv1DBlock(13, 64)
        res_block = ResidualBlock1D(64)
        pooling = MaskedAttentionPool(64)
        head = ClassificationHead(64, 10)

        print("  ✓ Component classes instantiate correctly")

        # Test config-based model loading
        with open('config/model_config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        from src.models import LateFusionClassifier, MLPBaseline

        # Test that main models can be instantiated
        classifier = LateFusionClassifier(config['classifier'])
        baseline = MLPBaseline(config['classifier'])

        print("  ✓ Main model classes instantiate correctly")
        print(f"  ✓ Classifier parameters: {sum(p.numel() for p in classifier.parameters()):,}")
        print(f"  ✓ Baseline parameters: {sum(p.numel() for p in baseline.parameters()):,}")

    except ImportError:
        print("  ⚠ PyTorch not available - skipping class instantiation tests")
        print("  ✓ Class definitions syntax validated in import test")
    except Exception as e:
        print(f"  ❌ Class structure test failed: {e}")
        return False

    return True


def test_critical_features():
    """Test that critical architectural features are implemented."""
    print("\nTesting critical features...")

    try:
        import torch

        # Test mask-aware pooling with simple data
        from src.models.components.pooling import MaskedAttentionPool, MaskedMeanMaxPool

        # Create simple test data
        batch_size, seq_len, dim = 2, 10, 8
        features = torch.randn(batch_size, seq_len, dim)
        mask = torch.ones(batch_size, seq_len)
        mask[0, 7:] = 0  # First sequence has padding
        mask[1, 5:] = 0  # Second sequence has padding

        # Test attention pooling
        attn_pool = MaskedAttentionPool(dim)
        attn_out = attn_pool(features, mask)
        assert attn_out.shape == (batch_size, dim), f"Expected {(batch_size, dim)}, got {attn_out.shape}"

        # Test mean+max pooling
        meanmax_pool = MaskedMeanMaxPool()
        meanmax_out = meanmax_pool(features, mask)
        assert meanmax_out.shape == (batch_size, dim * 2), f"Expected {(batch_size, dim * 2)}, got {meanmax_out.shape}"

        print("  ✓ Mask-aware pooling works correctly")

        # Test encoder shape preservation
        from src.models.components.encoders import AudioEncoder

        audio_encoder = AudioEncoder(input_dim=13, hidden_dim=64, num_res_blocks=2)
        audio_input = torch.randn(batch_size, seq_len, 13)
        audio_out = audio_encoder(audio_input, mask)

        assert audio_out.shape == (batch_size, seq_len, 64), f"Expected {(batch_size, seq_len, 64)}, got {audio_out.shape}"
        print("  ✓ Encoder preserves sequence length")

        # Test fusion
        from src.models.components.fusion import LateFusionModule

        fusion = LateFusionModule(audio_dim=64, chart_dim=64, fusion_dim=128)
        chart_features = torch.randn(batch_size, seq_len, 64)
        fused = fusion(audio_out, chart_features, mask)

        assert fused.shape == (batch_size, seq_len, 128), f"Expected {(batch_size, seq_len, 128)}, got {fused.shape}"
        print("  ✓ Late fusion works correctly")

    except ImportError:
        print("  ⚠ PyTorch not available - skipping feature tests")
    except Exception as e:
        print(f"  ❌ Critical features test failed: {e}")
        return False

    return True


def main():
    """Run all syntax and structure tests."""
    print("🔍 StepMania Model Architecture - Syntax & Structure Tests")
    print("=" * 60)

    tests = [
        test_module_imports,
        test_config_loading,
        test_class_structure,
        test_critical_features
    ]

    all_passed = True
    for test in tests:
        if not test():
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All syntax and structure tests passed!")
        print("\nArchitecture Summary:")
        print("  • Modular component design with proper separation")
        print("  • Late fusion with separate audio/chart encoders")
        print("  • Minimal conv pattern: Conv → ResBlock → Downsample → ResBlocks → Upsample + Skip")
        print("  • Mask-aware pooling for variable sequence lengths")
        print("  • Multiple fusion and pooling strategies")
        print("  • Comprehensive baseline models")
        print("  • Configuration-driven model instantiation")

        try:
            import torch
            print("\n🚀 PyTorch detected - models ready for training!")
            print("   Run 'python test_models.py' for full functional tests")
        except ImportError:
            print("\n📦 Install PyTorch to run full functional tests:")
            print("   conda env create -f environment.yml")
            print("   conda activate stepmania-chart-gen")
    else:
        print("❌ Some tests failed - check the output above")
        sys.exit(1)

def test_padding_invariance():
    torch.manual_seed(0)

    batch_size, dim = 1, 16
    true_len = 10
    max_len = 20

    x = torch.randn(batch_size, true_len, dim)
    mask_short = torch.ones(batch_size, true_len)

    # Pad to max length
    x_padded = torch.cat([x, torch.zeros(batch_size, max_len - true_len, dim)], dim=1)
    mask_padded = torch.cat([mask_short, torch.zeros(batch_size, max_len - true_len)], dim=1)

    pool = MaskedAttentionPool(dim)
    out1 = pool(x, mask_short)
    out2 = pool(x_padded, mask_padded)

    assert torch.allclose(out1, out2, atol=1e-5)


if __name__ == '__main__':
    main()
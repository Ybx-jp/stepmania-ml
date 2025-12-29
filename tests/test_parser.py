"""Test 1: Parser → tensor shape + values"""

import os
import sys
import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.stepmania_parser import StepManiaParser

def test_parser_tensor_shape_and_values():
    """Test parser produces correct tensor shape and values"""
    parser = StepManiaParser()

    # Use fixture
    fixture_path = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_chart.sm')

    # Parse the test chart
    result = parser.process_chart(fixture_path)
    assert result is not None, "Parser should successfully process test fixture"

    chart, chart_tensors = result
    assert len(chart_tensors) > 0, "Should have at least one chart tensor"

    # Test first tensor
    tensor = chart_tensors[0]

    # Assert: output shape is (T, 4)
    assert tensor.ndim == 2, f"Expected 2D tensor, got {tensor.ndim}D"
    assert tensor.shape[1] == 4, f"Expected 4 panels, got {tensor.shape[1]}"

    # Assert: T > 0
    T = tensor.shape[0]
    assert T > 0, f"Expected positive timesteps, got {T}"

    # Assert: values are in {0,1}
    unique_values = np.unique(tensor)
    assert np.all(np.isin(unique_values, [0, 1])), f"Expected only 0,1 values, got {unique_values}"

    # Assert: no timestep has >2 active panels (current enforcement)
    max_simultaneous = np.max(np.sum(tensor, axis=1))
    assert max_simultaneous <= 2, f"Expected ≤2 simultaneous notes, got {max_simultaneous}"
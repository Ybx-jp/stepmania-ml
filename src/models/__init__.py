"""
StepMania neural network models for difficulty classification.

Models:
- LateFusionClassifier: Main model with late fusion architecture
- DualTaskClassifier: Classifier with both classification and regression heads
- MLPBaseline: Simple MLP baseline
- SimpleConcatBaseline: Even simpler flattening baseline
- PooledFeatureBaseline: Statistical features baseline
"""

from .classifier import LateFusionClassifier
from .baseline import MLPBaseline, SimpleConcatBaseline, PooledFeatureBaseline

__all__ = [
    'LateFusionClassifier',
    'MLPBaseline',
    'SimpleConcatBaseline',
    'PooledFeatureBaseline'
]
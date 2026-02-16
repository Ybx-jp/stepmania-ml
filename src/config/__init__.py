"""Configuration management for the StepMania chart generator."""

from .paths import (
    PROJECT_ROOT,
    DATA_DIR,
    CHECKPOINT_DIR,
    CONFIG_DIR,
    CACHE_DIR,
    get_checkpoint_path,
    list_experiments
)

__all__ = [
    'PROJECT_ROOT',
    'DATA_DIR',
    'CHECKPOINT_DIR',
    'CONFIG_DIR',
    'CACHE_DIR',
    'get_checkpoint_path',
    'list_experiments',
]

"""
Centralized path management for the StepMania chart generator.

Provides auto-detected project root and standard directories.
Works from notebooks, scripts, or anywhere in the project.
"""

from pathlib import Path
import os
from typing import List, Dict, Optional


def _find_project_root() -> Path:
    """
    Auto-detect project root by looking for setup.py.

    Searches upward from this file until finding setup.py or reaching filesystem root.

    Returns:
        Path to project root directory
    """
    # Check environment variable first
    if 'PROJECT_ROOT' in os.environ:
        return Path(os.environ['PROJECT_ROOT'])

    # Start from this file's directory
    current = Path(__file__).resolve().parent

    # Search upward for setup.py
    while current != current.parent:  # Not at filesystem root
        if (current / 'setup.py').exists():
            return current
        current = current.parent

    # Fallback: assume we're in src/config/ and go up 2 levels
    return Path(__file__).resolve().parent.parent.parent


# Project root
PROJECT_ROOT = _find_project_root()

# Standard directories
DATA_DIR = PROJECT_ROOT / 'data'
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
CONFIG_DIR = PROJECT_ROOT / 'config'
CACHE_DIR = PROJECT_ROOT / 'cache'
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'

# Ensure cache directory exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_checkpoint_path(experiment_name: str, checkpoint: str = 'best_val_loss.pt') -> Optional[Path]:
    """
    Find checkpoint file for a given experiment.

    Args:
        experiment_name: Name of experiment (e.g., 'classifier_baseline', 'contrastive_experiment_b')
        checkpoint: Checkpoint filename (default: 'best_val_loss.pt')

    Returns:
        Path to checkpoint file, or None if not found

    Examples:
        >>> path = get_checkpoint_path('classifier_baseline')
        >>> path = get_checkpoint_path('contrastive', 'last.pt')
    """
    # Try exact match first
    checkpoint_path = CHECKPOINT_DIR / experiment_name / checkpoint
    if checkpoint_path.exists():
        return checkpoint_path

    # Try fuzzy match (find directories containing experiment_name)
    if CHECKPOINT_DIR.exists():
        for exp_dir in CHECKPOINT_DIR.iterdir():
            if exp_dir.is_dir() and experiment_name.lower() in exp_dir.name.lower():
                checkpoint_path = exp_dir / checkpoint
                if checkpoint_path.exists():
                    return checkpoint_path

    return None


def list_experiments() -> List[Dict[str, str]]:
    """
    List all available experiments with metadata.

    Returns:
        List of experiment dictionaries with keys:
        - name: Experiment directory name
        - path: Full path to experiment directory
        - checkpoints: List of checkpoint files
        - best_exists: Whether best_val_loss.pt exists
        - last_exists: Whether last.pt exists

    Examples:
        >>> experiments = list_experiments()
        >>> for exp in experiments:
        ...     print(f"{exp['name']}: {len(exp['checkpoints'])} checkpoints")
    """
    if not CHECKPOINT_DIR.exists():
        return []

    experiments = []

    for exp_dir in sorted(CHECKPOINT_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue

        # Get list of checkpoint files
        checkpoints = [f.name for f in exp_dir.glob('*.pt')]

        experiments.append({
            'name': exp_dir.name,
            'path': str(exp_dir),
            'checkpoints': checkpoints,
            'best_exists': 'best_val_loss.pt' in checkpoints,
            'last_exists': 'last.pt' in checkpoints,
        })

    return experiments


# Print info on import (helpful for debugging)
if __name__ != '__main__':
    # Only print if not running as script
    pass
else:
    # When run as script, print diagnostics
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data dir: {DATA_DIR} (exists: {DATA_DIR.exists()})")
    print(f"Checkpoints dir: {CHECKPOINT_DIR} (exists: {CHECKPOINT_DIR.exists()})")
    print(f"Config dir: {CONFIG_DIR} (exists: {CONFIG_DIR.exists()})")
    print(f"Cache dir: {CACHE_DIR} (exists: {CACHE_DIR.exists()})")
    print(f"\nAvailable experiments:")
    for exp in list_experiments():
        print(f"  - {exp['name']}: {len(exp['checkpoints'])} checkpoints")

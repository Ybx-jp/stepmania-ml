# StepMania Chart Generator

An ML-based system for generating StepMania dance charts using difficulty classification and diffusion models.

## Project Overview

**Primary Goal (Phase 1)**: Build a robust difficulty classification model that learns the relationship between StepMania chart patterns, audio features, and difficulty ratings.

**Research Milestone**: Accurate difficulty prediction demonstrates that neural networks can learn meaningful musical-motor coordination patterns from dance game data.

### Two-Phase Research Approach
1. **Phase 1 (Current Focus)**: Difficulty Classification Model
   - Input: Chart step sequences + audio features (MFCC, tempo)
   - Output: Difficulty score prediction (1-10 scale)
   - Success criteria: >80% accuracy on held-out test set
   - Deliverable: Trained classifier that understands chart complexity

2. **Phase 2 (Future Work)**: Generative Chart Creation
   - Leverage Phase 1 backbone for conditional generation
   - Use diffusion models for step sequence generation
   - Condition on audio features and target difficulty

## Project Structure

```
stepmania-chart-generator/
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   ├── models/            # Model architectures
│   ├── training/          # Training scripts
│   ├── evaluation/        # Evaluation and metrics
│   └── utils/             # Utilities
├── config/                # Configuration files
├── data/                  # Data storage
├── notebooks/             # Jupyter notebooks for exploration
├── scripts/               # Executable scripts
└── tests/                 # Test modules
```

## Setup

### Phase 1 Environment Setup

```bash
# Create conda environment (minimal dependencies)
conda env create -f environment.yml
conda activate stepmania-chart-gen

# Install package in development mode
pip install -e .

# For development with Jupyter
pip install -e .[dev]

# Register Jupyter kernel
python -m ipykernel install --user --name stepmania-chart-gen --display-name "StepMania Chart Gen"
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import librosa; print(f'Librosa: {librosa.__version__}')"
```

### Future Phase Dependencies

When you're ready for Phase 2 or advanced features:
```bash
# Install full dependency set
pip install -e .[full]
```

## Usage

### Phase 1 Development Workflow

#### 1. Data Exploration (Start Here)
```bash
jupyter lab notebooks/01_data_exploration.ipynb
```

#### 2. Data Preparation
```bash
python scripts/preprocess_data.py
```

#### 3. Classification Model Training
```bash
python scripts/run_classifier_training.py
```

#### 4. Model Evaluation
```bash
jupyter lab notebooks/04_classification_eval.ipynb
```

### Future Phase 2 Usage
```bash
# Advanced hyperparameter optimization (install .[full] first)
python scripts/optimize_hyperparams.py --model classifier

# Diffusion model training
python scripts/run_diffusion_training.py

# Chart generation
python scripts/generate_charts.py --audio_file song.wav --difficulty 7
```

## Configuration

- `config/data_config.yaml`: Data processing settings
- `config/model_config.yaml`: Model architecture parameters
- `docs/chart_representation.md`: Data format specifications

## Development

**Phase 1 Getting Started**:
1. `notebooks/01_data_exploration.ipynb`: Explore StepMania chart formats
2. `notebooks/02_feature_analysis.ipynb`: Analyze audio features
3. `notebooks/03_model_experiments.ipynb`: Prototype classification models

**Key Documentation**:
- `docs/chart_representation.md`: How charts become tensors

## Scope

- **Target**: 4-panel DDR-style charts only
- **Difficulty**: Beginner to Hard (levels 1-10)
- **Song Length**: 90-120 seconds
- **Patterns**: Basic steps and jumps (no holds/rolls initially)
- **Timing**: Fixed BPM songs, 4/4 time signature
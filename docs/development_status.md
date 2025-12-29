# Development Status - Project Onboarding

*Last updated: December 28, 2024*

## Project Overview

This project implements an ML-based system for generating StepMania dance charts using a two-phase approach:
1. **Phase 1 (Current)**: Difficulty classification model
2. **Phase 2 (Future)**: Generative chart creation using diffusion models

**Current Focus**: Building a robust difficulty classification model that learns the relationship between StepMania chart patterns, audio features, and difficulty ratings.

## Phase 1 Goals

- **Input**: Chart step sequences + audio features (MFCC, tempo)
- **Output**: Difficulty score prediction (1-10 scale)
- **Success Criteria**: >80% accuracy on held-out test set
- **Deliverable**: Trained classifier that understands chart complexity

## Development Status

### ✅ COMPLETED: Data Layer (FROZEN)

The data processing pipeline is **complete and frozen** for Phase 1 development:

#### Core Components
- **StepMania Parser** (`src/data/stepmania_parser.py`)
  - Parses .sm/.ssc chart files
  - Converts charts to `(timesteps, 4)` binary tensors
  - 16th note resolution alignment with audio
  - Fixed BPM validation for Phase 1 scope

- **Audio Feature Extractor** (`src/data/audio_features.py`)
  - Extracts MFCC features synchronized with chart timesteps
  - Produces `(timesteps, 13)` feature tensors
  - Automatic hop_length calculation for perfect alignment

- **Dataset Class** (`src/data/dataset.py`)
  - PyTorch Dataset following clean patterns
  - Joint padding/truncation with attention masks
  - Lazy loading with robust error handling
  - Cache support (stubbed for future optimization)

- **Data Splits Utility** (`src/utils/data_splits.py`)
  - Train/validation/test split creation
  - Dataset instantiation helpers

#### Data Contracts & Format
- **Chart Tensors**: `(sequence_length, 4)` - binary encoding for [Left, Down, Up, Right]
- **Audio Features**: `(sequence_length, 13)` - MFCC coefficients
- **Attention Masks**: `(sequence_length,)` - boolean masks for padding
- **Alignment**: Hard assertion ensures chart and audio timestep synchronization

#### Phase 1 Scope (Enforced)
- **Target**: 4-panel DDR-style charts only
- **Patterns**: Basic steps and jumps (no holds/rolls)
- **BPM**: Fixed BPM songs only (no tempo changes)
- **Difficulties**: 1-10 range (Beginner to Hard)
- **Resolution**: 16th note timestep resolution

### 🔧 IN PROGRESS: Model Architecture

**Next Development Priority**: Phase 1 classification model components

#### Planned Components
- Audio encoder (CNN/LSTM for MFCC processing)
- Chart encoder (embedding + sequence processing)
- Fusion module (combine audio + chart representations)
- Classification head (difficulty prediction 1-10)

### 📋 TODO: Training & Evaluation

#### Training Pipeline
- Training loop implementation
- Hyperparameter optimization (using Optuna)
- Model checkpointing and resumption
- Metrics tracking and logging

#### Evaluation Framework
- Accuracy metrics for difficulty prediction
- Confusion matrix analysis
- Model performance visualization

## Project Structure

```
stepmania-chart-generator/
├── src/
│   ├── data/           # ✅ COMPLETE - Data processing pipeline
│   ├── models/         # 🔧 NEXT - Model architectures
│   ├── training/       # 📋 TODO - Training scripts
│   ├── evaluation/     # 📋 TODO - Evaluation tools
│   └── utils/          # ✅ PARTIAL - Utilities
├── config/             # ✅ COMPLETE - Configuration files
├── docs/               # ✅ COMPLETE - Documentation
├── notebooks/          # 📋 TODO - Development notebooks
└── scripts/           # 📋 TODO - Executable scripts
```

## Environment Setup

**Phase 1 Dependencies** (minimal for classification):
```bash
conda env create -f environment.yml
conda activate stepmania-chart-gen
pip install -e .
```

Core dependencies: PyTorch, librosa, numpy, pandas, scikit-learn

**Future Dependencies**: diffusers, transformers, optuna (install with `pip install -e .[full]`)

## Key Design Principles

### Data Layer Philosophy
- **Minimal validation**: Performs only essential checks, advanced playability validation deferred
- **Alignment-first**: Prioritizes chart/audio synchronization over complex pattern analysis
- **Clean separation**: Parser handles format conversion, dataset handles ML preparation
- **Error resilience**: Graceful fallbacks for corrupted data

### Phase 1 Constraints
- **Fixed scope**: No variable BPM, holds, or advanced patterns
- **Simple patterns**: Steps and jumps only - sufficient for difficulty learning
- **Padding strategy**: Fixed sequence length with attention masks for variable song lengths

## Development Workflow

### For New Team Members

1. **Start Here**: Review `notebooks/01_data_exploration.ipynb` (when created)
2. **Understand formats**: Read `docs/chart_representation.md`
3. **Test data pipeline**: Run dataset loading to verify your environment
4. **Current work**: Model architecture in `src/models/`

### Phase 1 Development Path
1. ✅ Data layer implementation (complete)
2. 🔧 Model architecture components (in progress)
3. 📋 Training pipeline setup
4. 📋 Evaluation and metrics
5. 📋 Hyperparameter optimization
6. 📋 Model analysis and validation

## Important Notes

### Data Layer is Frozen
The data processing pipeline is **frozen for Phase 1**. Do not modify:
- Core parsing logic
- Alignment calculations
- Dataset return format
- Validation rules

Any data changes should be discussed as Phase 2 enhancements.

### Dependencies Strategy
- **Phase 1**: Minimal dependencies for faster development
- **Phase 2**: Full dependency set when needed for diffusion models
- **Development**: Use `pip install -e .[dev]` for Jupyter support

## Getting Started Checklist

- [ ] Clone repository and set up environment
- [ ] Read `docs/chart_representation.md` for data format understanding
- [ ] Review Phase 1 scope in main README
- [ ] Test data pipeline with sample data
- [ ] Understand current model architecture needs (see `config/model_config.yaml`)
- [ ] Check current development priorities in this document

## Contact & Resources

- **Primary Documentation**: `docs/chart_representation.md`
- **Configuration**: `config/` directory for all settings
- **Issue Tracking**: Focus on Phase 1 classification model completion
- **Development Status**: This document (updated regularly)

Welcome to the team! The foundation is solid - time to build the classification model.
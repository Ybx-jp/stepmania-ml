# ML Project Conventions

This project follows the ML Workbench methodology. All ML work must adhere to these conventions.

## Project Overview

- **Task**: Multi-class classification (4 difficulty classes: Beginner, Easy, Medium, Hard)
- **Architecture**: LateFusionClassifier with Conv1D backbone, dual audio/chart encoders
- **Baselines**: MLPBaseline, PooledFeatureBaseline (in `src/models/baseline.py`)
- **Data**: StepMania charts (.sm/.ssc files) + audio features (MFCC, onset, spectral)
- **Primary metric**: Macro F1 (imbalanced classes with class weighting)
- **MLflow experiment name**: `stepmania-difficulty-classifier`

## Experiment Methodology

1. **Baseline first**: Always establish a simple baseline before building complex models
2. **Measure before optimizing**: Define success metrics before writing training code
3. **One change at a time**: Each experiment should change exactly one variable from the previous run
4. **Log everything**: All experiments tracked in MLflow with hyperparameters and metrics

## Training Loop Pattern

Every PyTorch training loop must follow this structure:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    # Training phase
    model.train()
    for batch in train_loader:
        inputs, targets = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Validation phase
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            # compute metrics ...

    # Checkpoint best model based on validation metric
    if val_metric > best_val_metric:
        best_val_metric = val_metric
        torch.save(model.state_dict(), "best_model.pth")
```

**Non-negotiable rules:**
- `model.train()` before training batches
- `model.eval()` before validation/inference
- `torch.no_grad()` wrapping all validation/inference
- `optimizer.zero_grad()` before each forward pass
- Save best model on **validation** metric, not training metric
- **Early stopping**: stop after `--patience` (default 3) epochs with no validation-metric improvement

## Reproducibility

Every script must call `set_seed()` before any stochastic operations:

```python
from src.utils.reproducibility import set_seed
set_seed()  # Default seed: 42
```

This seeds: `torch`, `torch.cuda`, `numpy`, `random`, and sets `cudnn.deterministic = True`.

## Data Handling

- **Stratified splits** for imbalanced classification (pass `stratify_labels` to `create_data_splits()`)
- **Split before preprocessing**: `train_test_split` first, then `fit_transform` on training data only
- **Never `fit_transform` on validation or test data** -- this is data leakage
- **DataLoader config**: `num_workers=min(8, cpu_count)`, `pin_memory=True` for GPU

## Device Management

- Always use: `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- Never hardcode `torch.device('cuda')`
- Move both model and data to the same device

## Loss Function Rules

- `nn.CrossEntropyLoss()` -- already includes softmax. **Do not** add `nn.Softmax` to model output
- `nn.BCEWithLogitsLoss()` -- already includes sigmoid. **Do not** add `nn.Sigmoid` to model output
- Use `nn.BCELoss()` only if model already outputs sigmoid probabilities

## Evaluation Standards

- **Classification**: Always compute per-class metrics (precision, recall, F1), not just accuracy
- **Imbalanced data**: Use macro F1 as primary metric, never accuracy alone
- **Always compare to baseline** with improvement percentage
- **Test set evaluated exactly once** -- no iterating on test set metrics
- Use `scripts/evaluate.py` for standardized evaluation with all artifacts

## Key Commands

```bash
# Train classifier
python scripts/train.py --config config/model_config.yaml --data_dir data/ --audio_dir data/

# Train baseline
python scripts/train.py --config config/model_config.yaml --data_dir data/ --audio_dir data/ --model_type mlp_baseline

# Evaluate
python scripts/evaluate.py --checkpoint checkpoints/<exp>/best_val_loss.pt --config config/model_config.yaml --data_dir data/ --audio_dir data/

# Run tests
pytest tests/
```

## File Organization

```
stepmania-chart-generator/
├── config/                    # YAML configs (model, data, experiments)
├── scripts/
│   ├── train.py               # Training with MLflow, seed, model selection
│   └── evaluate.py            # Evaluation with plots and metrics
├── src/
│   ├── config/                # Path management, ExperimentConfig dataclass
│   ├── data/                  # Dataset, parser, audio features, groove radar
│   ├── evaluation/            # Evaluation utilities (compute_metrics, load_and_evaluate)
│   ├── losses/                # Contrastive and ordinal losses
│   ├── models/                # LateFusionClassifier, baselines, components
│   ├── training/              # BaseTrainer, Trainer, ContrastiveTrainer, callbacks
│   ├── utils/                 # Reproducibility, data splits, audio I/O
│   └── visualization/         # Plotting functions (confusion matrix, etc.)
├── checkpoints/               # Saved model weights
├── outputs/                   # Evaluation plots and artifacts
├── mlruns/                    # MLflow experiment tracking
└── tests/                     # Unit tests
```

## Reference Materials

For detailed methodology, architecture selection, and evaluation patterns:
- `~/Notebooks/ML_Fundamentals_Reference.md`
- `~/Notebooks/PyTorch_Fundamentals_Reference.md`
- `~/Notebooks/PyTorch_Techniques_Tools_Reference.md`
- `~/Notebooks/Complete_ML_DL_Reference_Overview.md`

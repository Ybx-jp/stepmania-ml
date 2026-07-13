# ML Project Conventions

This project follows the ML Workbench methodology. All ML work must adhere to these conventions.

## Project Overview

- **Task**: Audio-conditioned StepMania chart **generation** — given a song + a target difficulty,
  write a playable chart (onset → panel pattern → tap/hold type), with groove/style conditioning.
- **Deployed model**: `LayeredTypedChartGenerator` (`src/generation/typed_model.py`), checkpoint
  `checkpoints/gen_motif_v2_48th_cont/best_val.pt` on 42-dim `highres_v2` features (48th grid).
- **Canonical decode**: `src/generation/decode_defaults.py` + `decode_harness.py` are the single
  source for the decode palette / tau pipeline. Consult the `generation-defaults` and
  `conditioning-mechanics` skills BEFORE setting or measuring any generation knob.
- **Primary metrics**: onset F1 and phase shares offline; **the user's ear is the deciding vote**
  (playtests in `notes/playtest_log.md`). A learned taste critic supplements both.
- **Data**: StepMania charts (.sm/.ssc) + audio features. Four feature-cache generations exist —
  `python tools/cache.py status` is the map; never mix a cache with the wrong checkpoint.
- **Legacy sub-project**: the difficulty **classifier** (LateFusionClassifier in `src/models/`,
  `scripts/train.py`, MLflow experiment `stepmania-difficulty-classifier`). It seeded the project,
  warm-started the generator's audio encoder, and backs the critic — maintained, not active.

## Experiment Methodology

1. **Baseline first**: Always establish a simple baseline before building complex models
2. **Measure before optimizing**: Define success metrics before writing training code
3. **One change at a time**: Each experiment should change exactly one variable from the previous run
4. **Log everything**: All experiments tracked in MLflow with hyperparameters and metrics

## Documentation Discipline

**Never assert volatile external state as fact in durable documents** (HANDOFF, memories, lineage files, notes,
commit/PR bodies). This includes PR merge/open/closed status, what's "pushed", branch existence, deployment/release
status, CI/check results, or "currently running" jobs. Such claims are stale the moment they're written and have
repeatedly misled later sessions (e.g. a HANDOFF asserting "PR #42 OPEN" long after it merged).

- **Record the durable identifier, not the transient state**: write "PR #42 (`release/v0.1.0-prep` → `main`)",
  not "PR #42 is open". Reference the branch/PR/commit so a reader can look it up.
- **Instruct verification at read time**: pair any such reference with "verify current state via `gh pr view <n>`
  / `git`" rather than stating the state. When you yourself need the state, RUN the command — never rely on a
  doc's or memory's claim about it.
- Facts that ARE durable (a commit SHA, a design decision, a measured result, a file path) may be asserted; the
  rule targets state that changes outside the document.

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
# Generate a chart for your own song (BYO audio; --bpm strongly recommended)
python scripts/generate.py --audio song.ogg --difficulty Hard --bpm 174 --out MyGenerated

# Canonical dataset-bound export (the playtest/eval path; bare defaults = the deployed config)
python scripts/export_typed_samples.py --data_dir data/ --audio_dir data/

# Validate the canonical defaults & repo layout
python tools/check_export_defaults.py
python tools/check_repo_layout.py

# Feature-cache map / drift check
python tools/cache.py status

# Legacy classifier train/evaluate
python scripts/train.py --config config/model_config.yaml --data_dir data/ --audio_dir data/
python scripts/evaluate.py --checkpoint checkpoints/<exp>/best_val_loss.pt --config config/model_config.yaml --data_dir data/ --audio_dir data/

# Run tests (includes layout + canonical-default enforcement)
pytest tests/
```

## File Organization

```
stepmania-chart-generator/
├── config/                    # YAML configs (model, data, experiments)
├── scripts/                   # Public CLIs: generate.py (BYO song), batch_generate.py,
│                              #   pull_audio.py, train.py / evaluate.py (legacy classifier)
├── src/
│   ├── data/                  # Parser, dataset, audio features, groove radar, offset/meter detect
│   ├── generation/            # typed_model, decode_defaults/harness (CANONICAL), sm_writer, manifold
│   ├── models/                # Legacy classifier (LateFusionClassifier, baselines) — critic backbone
│   ├── training/, losses/     # Trainers, callbacks, contrastive/ordinal losses
│   └── utils/, visualization/ # Reproducibility, splits, audio I/O, plots
├── experiments/               # Research code — conventions in experiments/README.md
│   ├── generation_typed/      # Active generator lab: live trainers + canonical exporter
│   ├── probes/                # Cross-cutting standalone probes (run from repo root)
│   ├── realism_critic/        # Taste/realism critic
│   └── archive/               # Retired trainers (byte-faithful; do not modernize)
├── tools/                     # check_export_defaults.py, check_repo_layout.py, cache.py, chart_ui.py
├── notes/                     # Findings per experiment + playtest log (notes/INDEX.md is the map)
├── cache/                     # Feature caches + fitted artifacts (tools/cache.py status)
├── checkpoints/               # Trained weights (deployed: gen_motif_v2_48th_cont)
├── outputs/                   # Generated sets, probe results, evaluation artifacts
└── tests/                     # Unit + regression tests (KV-cache bit-identity, layout, defaults)
```

**Layout rules are enforced** by `tools/check_repo_layout.py` (run via pytest): no `.py` at repo
root; new trainers must be added to its allowlist deliberately; no result CSVs/logs in `cache/`.

## Reference Materials

For detailed methodology, architecture selection, and evaluation patterns:
- `~/Notebooks/ML_Fundamentals_Reference.md`
- `~/Notebooks/PyTorch_Fundamentals_Reference.md`
- `~/Notebooks/PyTorch_Techniques_Tools_Reference.md`
- `~/Notebooks/Complete_ML_DL_Reference_Overview.md`

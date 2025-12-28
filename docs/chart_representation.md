# Chart Data Representation

This document defines how StepMania charts are converted to tensor representations for machine learning models.

## Overview

Charts are represented as sequences of timesteps, where each timestep contains information about which arrows should be pressed. This representation must align with audio features for training and generation.

## Chart Tensor Format

### Basic Structure
```
Chart Tensor Shape: (sequence_length, num_panels)
- sequence_length: Number of timesteps in the chart
- num_panels: 4 (Left, Down, Up, Right)
```

### Panel Encoding
```
Panel Index Mapping:
- 0: Left arrow
- 1: Down arrow
- 2: Up arrow
- 3: Right arrow
```

### Timestep Values
```
Binary Encoding (Phase 1):
- 0: No step on this panel at this timestep
- 1: Step on this panel at this timestep

Example timestep: [0, 1, 0, 1] = Down + Right (jump)
```

## Time Resolution

### Target Resolution
- **16th note resolution**: Most common for beginner-intermediate charts
- **32nd note resolution**: For advanced patterns (future consideration)

### BPM Alignment
```
Timesteps per beat = resolution_factor
- 16th notes: 4 timesteps per beat
- 32nd notes: 8 timesteps per beat

Total timesteps = (song_length_seconds * BPM / 60) * timesteps_per_beat
```

### Example Calculation
```
Song: 120 seconds, 140 BPM, 16th note resolution
Total timesteps = (120 * 140 / 60) * 4 = 280 * 4 = 1120 timesteps
Chart tensor shape: (1120, 4)
```

## Audio Feature Alignment

### Feature Extraction Windows
```
Audio features extracted with hop_length matching chart timesteps:
hop_length = sr * 60 / (BPM * timesteps_per_beat)

For 140 BPM, 16th notes, sr=22050:
hop_length = 22050 * 60 / (140 * 4) = 236 samples
```

### Synchronized Representations
```
Chart tensor:     (1120, 4)     - timestep × panel
Audio features:   (1120, 13)    - timestep × MFCC coefficients
```

## Data Preprocessing Pipeline

### 1. Parse StepMania File
```python
# Extract from .sm/.ssc files:
- BPM information
- Note data (beat positions and panels)
- Audio file reference
```

### 2. Convert to Timestep Grid
```python
# For each note in chart:
timestep = round(beat_position * timesteps_per_beat)
chart_tensor[timestep, panel_index] = 1
```

### 3. Extract Audio Features
```python
# Using librosa with aligned hop_length:
mfcc_features = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    hop_length=hop_length,
    n_mfcc=13
)
```

### 4. Validate Alignment
```python
assert chart_tensor.shape[0] == mfcc_features.shape[1]
```

## Pattern Types (Phase 1 Scope)

### Supported Patterns
1. **Single Steps**: One panel active per timestep
2. **Jumps**: Two panels active simultaneously
3. **Empty Steps**: No panels active (rest)

### Pattern Examples
```
Single step (Down):  [0, 1, 0, 0]
Jump (Left+Right):   [1, 0, 0, 1]
Empty timestep:      [0, 0, 0, 0]
```

### Excluded Patterns (Future Phases)
- Holds (extended presses)
- Rolls (rapid alternating presses)
- Mines (panels to avoid)

## Data Validation

### Sanity Checks
1. **Density**: Max 2 simultaneous steps (jump constraint)
2. **Playability**: No impossible body movements
3. **Timing**: No conflicting timestep assignments
4. **Alignment**: Chart and audio feature lengths match

### Quality Filters
- Minimum gap between complex patterns
- Maximum consecutive jumps
- Reasonable overall step density for difficulty level

## Model Input Format

### Classification Model Input
```python
# Concatenated features per timestep
input_features = torch.cat([
    chart_tensor,      # (seq_len, 4)
    audio_features.T,  # (seq_len, 13)
], dim=1)              # (seq_len, 17)

# Or separate encoding:
chart_encoding = model.chart_encoder(chart_tensor)
audio_encoding = model.audio_encoder(audio_features.T)
```

### Expected Tensor Dimensions
```
Batch processing:
- Charts: (batch_size, sequence_length, 4)
- Audio: (batch_size, sequence_length, 13)
- Labels: (batch_size,) for difficulty scores
```

## Future Considerations

### Phase 2 Extensions
- Variable BPM handling
- Hold/roll representation
- Multi-difficulty conditioning
- Attention mechanism compatibility

### Advanced Encoding
- Continuous values for note intensity
- Timing offset encoding for humanization
- Multi-panel pattern embeddings

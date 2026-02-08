# Data Flow Quick Reference

## Complete Pipeline: Data Loading to Training

```
START: select subject .mat file
  │
  ├─→ STAGE 1: FORMAT DETECTION & CONVERSION
  │     ├─ Load .mat file with scipy
  │     ├─ Detect format (raw or standardized)
  │     └─ If raw: auto-convert → save standardized format
  │
  ├─→ STAGE 2: DATA EXTRACTION & VALIDATION
  │     ├─ Extract all_data (900, 62, 800) and all_label (900, 1)
  │     ├─ Validate shapes and timepoints
  │     ├─ Check label values {0, 1, 2}
  │     └─ Calculate class distribution
  │
  ├─→ STAGE 3: OPTIONAL CLASS BALANCING
  │     ├─ Default: reduce_rest_ratio = 1.0 (no change)
  │     └─ If < 1.0: downsample Rest class
  │
  ├─→ STAGE 4: CREATE DATA BUNDLE
  │     └─ Return EEGDataBundle (all numpy arrays)
  │
  ├─→ STAGE 5: TRAIN/TEST SPLIT
  │     ├─ 80% training: (720, 62, 800)
  │     └─ 20% testing: (180, 62, 800)
  │
  ├─→ STAGE 6: CONVERT TO PYTORCH TENSORS
  │     ├─ Numpy → float32 tensor
  │     ├─ Add channel dimension: (B, 1, 62, 800)
  │     └─ Move to CUDA/CPU device
  │
  ├─→ STAGE 7: CREATE DATALOADERS
  │     ├─ Wrap in TensorDataset
  │     ├─ Wrap in DataLoader (batch_size=64, shuffle=True/False)
  │     └─ Ready for training loop
  │
  └─→ TRAINING READY
        train_loader: 12 batches of ~64 samples
        test_loader: 3 batches of ~64 samples
        Each batch: (X, Y) where X shape is (B, 1, 62, 800)
```

---

## Key Files

| File | Purpose |
|------|---------|
| `src/mi3_eeg/main.py` | Entry point, orchestrates pipeline |
| `src/mi3_eeg/dataset.py` | Data loading, tensors, dataloaders |
| `src/mi3_eeg/data_formatting/dataformatter.py` | Raw→standardized conversion |
| `src/mi3_eeg/config.py` | Configuration classes |

---

## Data Format Specifications

### Input: .mat File (File on Disk)

**Standardized Format (Preferred):**
```
all_data: shape (900, 62, 800)
  - 900 EEG trials
  - 62 channels
  - 800 timepoints (200 Hz × 4 seconds)
all_label: shape (900, 1)
  - 0 = Rest (300 samples)
  - 1 = Elbow (300 samples)
  - 2 = Hand (300 samples)
sampling_rate: 200
source_file: original .mat path
```

**Raw Format (Auto-Converted):**
```
task_data: shape (15, 40, 62, 800)
  - 15 sessions
  - 40 trials per session
  - 62 channels
  - 800 timepoints
task_label: shape (15, 40)
  - Values: {1=Elbow, 2=Hand}
rest_data: shape (300, 62, 800)
  - 300 rest trials
```

---

## Memory & Device Usage

### Before Training

| Stage | Shape | Dtype | Memory | Location |
|-------|-------|-------|--------|----------|
| .mat file | - | - | 7-10 GB | Disk |
| Numpy arrays | (900, 62, 800) | float64 | ~210 MB | RAM |
| After split | (720, 62, 800) + (180, 62, 800) | float64 | ~210 MB | RAM |
| PyTorch tensors | (720, 1, 62, 800) | float32 | ~105 MB | CPU then CUDA |
| In DataLoader | Batches of (64, 1, 62, 800) | float32 | ~30 MB per batch | CUDA/CPU |

### GPU Requirements

- **Minimum:** 2 GB VRAM
- **Recommended:** 4-8 GB VRAM for single-subject training
- **Batch size:** 64 samples (adjustable)

---

## Key Transformations

### 1. Shape Changes: EEG Data

```
Raw .mat load
  ↓
all_data: (900, 62, 800)  ← 900 trials, 62 channels, 800 timepoints
  ↓
Train split: (720, 62, 800)  ← 80% of data
  ↓
Numpy → Tensor: (720, 62, 800) dtype float32
  ↓
Add channel dim: (720, 1, 62, 800)  ← For Conv2D
  ↓
In DataLoader: Batches of (64, 1, 62, 800)  ← During training
```

### 2. Shape Changes: Labels

```
Raw .mat load
  ↓
all_label: (900, 1)  ← Each sample has 1 label
  ↓
Train split: (720,)  ← Flattened for training
  ↓
Numpy → Tensor: (720,) dtype int64
  ↓
In DataLoader: Batches of (64,)  ← During training
```

---

## Important Parameters

### Always Used

```python
sampling_rate = 200 Hz          # 4-second trials = 800 timepoints
num_channels = 62               # EEG electrode count
num_classes = 3                 # Rest, Elbow, Hand
test_size = 0.2                 # 20% test split
random_seed = 42                # Reproducibility
```

### Configurable

```python
reduce_rest_ratio = 1.0         # 1.0 = no change (default)
batch_size = 64                 # Samples per batch
dropout = 0.4                   # Regularization
learning_rate = 0.01            # Optimizer
epochs = 600                    # Max training iterations (can override)
device = "cuda"                 # Or "cpu" if no GPU
```

---

## Error Handling

### Data Not Found
```
→ Check Datasets/MI3/derivatives/
→ If raw format files present: auto-convert
→ If no files: graceful exit with download instructions
```

### Format Detection
```
→ Check for all_data + all_label (standardized)
→ Check for task_data + task_label + rest_data (raw)
→ If neither: raise ValueError with available keys
```

### Tensor Conversion
```
→ Numpy float64 → PyTorch float32 (reduced precision)
→ Labels flattened if needed
→ Moved to device (CPU or CUDA)
```

---

## Logging Examples

When running `python -m mi3_eeg.main`:

```
MI3 EEG Motor Imagery Classification Pipeline
Using device: cuda
STAGE 1: Loading and Preprocessing Data
Auto-selecting first available subject: sub-001_eeg200hz.mat
Loading dataset from: .../Datasets/MI3/derivatives/sub-001_eeg200hz.mat
Detected format: standardized
Original data shape: (900, 62, 800)
Original class distribution: {'Rest': 300, 'Elbow': 300, 'Hand': 300}
Split: 720 train samples, 180 test samples
Created DataLoader: 720 samples, batch_size=64, shuffle=True
Created DataLoader: 180 samples, batch_size=64, shuffle=False
STAGE 2: Training Models
Training lenet model...
```

---

## Pipeline Execution

### Basic Training

```bash
# Single subject (auto-selected)
python -m mi3_eeg.main

# Specific subject
python -m mi3_eeg.main --subject-file sub-001_eeg200hz.mat

# Custom epochs
python -m mi3_eeg.main --epochs 50

# Use CPU
python -m mi3_eeg.main --device cpu
```

### Batch Processing

```bash
# All subjects
python -m mi3_eeg.run_all_subjects

# Custom epochs for all
python -m mi3_eeg.run_all_subjects --epochs 50
```

---

## Quick Checklist

- [ ] Dataset downloaded to `Datasets/MI3/derivatives/`
- [ ] Files with names like `sub-001_eeg200hz.mat`
- [ ] Python environment activated (`.venv/Scripts/Activate.ps1`)
- [ ] CUDA available: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Run: `python -m mi3_eeg.main`
- [ ] Training should start automatically

---

## More Details

See `DATA_FLOW_ANALYSIS.md` for complete technical documentation including:
- 14 detailed stages of processing
- Error handling flow
- Configuration parameter reference
- Memory usage analysis
- Logging output examples

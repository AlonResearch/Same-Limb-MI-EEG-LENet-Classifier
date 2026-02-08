# MI3 EEG Data Flow: Complete Processing Pipeline

## Overview
This document traces the data flow from raw files through to model training input, detailing all processing logic, transformations, and key decision points.

---

## 1. ENTRY POINT: File Selection

### Location: `src/mi3_eeg/main.py` (lines 60-75)

```python
if subject_file is None or subject_file == "":
    paths = Paths.from_here()
    available_files = sorted(paths.dataset_derivatives.glob("*.mat"))
    
    if not available_files:
        logger.error(f"No .mat files found in {paths.dataset_derivatives}")
        return
    
    subject_file = available_files[0].name  # Auto-select first
```

**Input Path:** `Datasets/MI3/derivatives/*.mat`
**Files Expected:** 
- `sub-XXX_eeg200hz.mat` (standardized format - preferred)
- `sub-XXX_eeg90hz.mat` (older 90Hz sampling)
- Raw format files (auto-detected and converted)

**Output:** Selected filename (e.g., `sub-001_eeg200hz.mat`)

---

## 2. FORMAT DETECTION

### Location: `src/mi3_eeg/dataset.py:load_mat_from_derivatives()` (lines 105-145)

### Step 2.1: Load .mat File with Error Handling
```python
mat_data = scio.loadmat(str(mat_path))
```

**Error Handling:**
- `FileNotFoundError`: Dataset file missing
- `ValueError`: Corrupted or unsupported MATLAB format
- `Exception`: Unexpected errors with context

### Step 2.2: Format Detection
```python
data_format = detect_format(mat_data)
```

**Detection Logic** (`dataformatter.py:detect_format()`, lines 225-246):

```
IF 'all_data' AND 'all_label' in mat_data:
    → Format: STANDARDIZED ✓
ELIF 'task_data' AND 'task_label' AND 'rest_data' in mat_data:
    → Format: RAW (needs conversion)
ELSE:
    → ValueError: Cannot determine format
```

**Keys Present:**
- **Standardized Format:**
  - `all_data`: Shape (900, 62, 800)
  - `all_label`: Shape (900, 1) or (900,)
  - `sampling_rate`: Integer Hz value
  - `source_file`: String (optional)

- **Raw Format:**
  - `task_data`: Shape (15, 40, 62, 800)
  - `task_label`: Shape (15, 40)
  - `rest_data`: Shape (300, 62, 800)

---

## 3. RAW FORMAT CONVERSION (if needed)

### Location: `src/mi3_eeg/data_formatting/dataformatter.py`

### Step 3.1: Extract Components
If raw format detected, extract:
- **task_data**: 15 sessions × 40 trials/session
  - Shape: (15, 40, 62, 800)
  - Classes: Elbow (1) and Hand (2)
  
- **task_label**: Motor imagery class labels
  - Shape: (15, 40)
  - Values: {1, 2}
  
- **rest_data**: Rest condition EEG
  - Shape: (300, 62, 800)
  - Will become class 0

### Step 3.2: Reshape & Flatten
```python
# Flatten task data structure
task_data_flat = task_data.reshape(-1, channels, timepoints)
# Result: (600, 62, 800) - 600 total task trials = 15×40

# Flatten task labels
task_label_flat = task_label.flatten()
# Result: (600,) - 600 labels

# Create rest labels (all zeros)
rest_label = np.zeros(rest_data.shape[0], dtype=...)
# Result: (300,) - 300 rest samples
```

**Class Distribution After Reshape:**
- Task data: 600 trials
  - Elbow (1): 300 trials
  - Hand (2): 300 trials
- Rest data: 300 trials
  - Rest (0): 300 trials

### Step 3.3: Concatenate All Data
```python
all_data = np.concatenate([rest_data, task_data_flat], axis=0)
# Shape: (900, 62, 800)

all_label = np.concatenate([rest_label, task_label_flat], axis=0)
# Shape: (900,) → Reshaped to (900, 1)
```

**Result:**
- **all_data**: (900, 62, 800)
  - 900 samples total
  - 62 EEG channels
  - 800 timepoints (200Hz × 4 seconds)
  
- **all_label**: (900, 1)
  - Rest (0): 300 samples
  - Elbow (1): 300 samples
  - Hand (2): 300 samples

### Step 3.4: Infer Sampling Rate
```python
if timepoints == 800:
    sampling_rate = 200  # 800 points / 4 seconds
elif timepoints == 360:
    sampling_rate = 90   # 360 points / 4 seconds
else:
    sampling_rate = int(timepoints / 4)  # Estimate from timepoints
```

### Step 3.5: Save Standardized Format
```python
output_filename = f"{subject_id}_eeg{sampling_rate}hz.mat"
# Example: sub-001_eeg200hz.mat

save_dict = {
    'all_data': all_data,
    'all_label': all_label,
    'sampling_rate': sampling_rate,
    'source_file': str(input_path),
}
scio.savemat(str(output_path), save_dict)
```

**Saved to:** `Datasets/MI3/derivatives/sub-XXX_eeg200hz.mat`

---

## 4. LOAD STANDARDIZED DATA

### Location: `src/mi3_eeg/dataset.py:load_mat_from_derivatives()` (lines 236-260)

### Step 4.1: Extract Data & Labels
```python
try:
    all_data = mat_data["all_data"]      # (900, 62, 800)
    all_label = mat_data["all_label"]    # (900, 1) or (900,)
except KeyError as e:
    raise KeyError(f"Required key {e} not found")

# Ensure labels are flattened then reshaped
all_label = np.atleast_1d(all_label).flatten()
if all_label.ndim == 1:
    all_label = all_label.reshape(-1, 1)  # (900, 1)
```

### Step 4.2: Validate Timepoints (Optional)
```python
if validate_timepoints and expected_sampling_rate is not None:
    timepoints = all_data.shape[2]  # 800
    expected_timepoints = expected_sampling_rate * 4  # 200 × 4 = 800
    tolerance = expected_timepoints * 0.1  # 80
    
    if abs(timepoints - expected_timepoints) > tolerance:
        logger.warning(f"Timepoint mismatch...")
```

### Step 4.3: Class Distribution Analysis
```python
original_dist = _calculate_class_distribution(all_label)
# Returns: {'Rest': 300, 'Elbow': 300, 'Hand': 300}

logger.info(f"Original class distribution: {original_dist}")
```

### Step 4.4: Verify Label Values
```python
unique_labels = np.unique(all_label)  # Should be [0, 1, 2]
expected_labels = np.array([0, 1, 2])

if not np.array_equal(unique_labels, expected_labels):
    logger.warning(f"Label mismatch! Got {unique_labels}")
```

---

## 5. OPTIONAL: CLASS BALANCING

### Location: `src/mi3_eeg/dataset.py:_balance_rest_class()` (lines 310-335)

**Trigger:** Only if `reduce_rest_ratio < 1.0`
**Default:** `reduce_rest_ratio = 1.0` (no reduction)

### If Balancing Enabled:
```python
if reduce_rest_ratio < 1.0:
    all_data, all_label = _balance_rest_class(
        all_data, all_label, reduce_rest_ratio, random_seed
    )
```

### Balancing Logic:
```python
# Find indices for each class
rest_indices = np.where(label_flat == 0)[0]      # 300 indices
other_indices = np.where(label_flat != 0)[0]     # 600 indices

# Randomly select subset of Rest samples
num_rest_to_keep = int(len(rest_indices) * keep_ratio)
selected_rest_indices = np.random.choice(
    rest_indices, size=num_rest_to_keep, replace=False
)

# Combine: selected Rest + all other classes
balanced_indices = np.concatenate((selected_rest_indices, other_indices))
np.random.shuffle(balanced_indices)

# Apply indexing
balanced_data = data[balanced_indices]
balanced_labels = labels[balanced_indices]
```

**Example with `reduce_rest_ratio = 0.5`:**
- Before: Rest=300, Elbow=300, Hand=300 (900 total)
- After: Rest=150, Elbow=300, Hand=300 (750 total)

---

## 6. CREATE DATA BUNDLE

### Location: `src/mi3_eeg/dataset.py:load_mat_from_derivatives()` (lines 284-304)

```python
channel_count = all_data.shape[1]           # 62
num_classes = len(np.unique(all_label))     # 3
actual_sampling_rate = inferred_sampling_rate  # 200

return EEGDataBundle(
    data=all_data,                          # (900, 62, 800)
    labels=all_label,                       # (900, 1)
    channel_count=channel_count,            # 62
    num_classes=num_classes,                # 3
    sample_rate=actual_sampling_rate,       # 200
    class_distribution=class_dist,          # {'Rest': 300, ...}
)
```

**EEGDataBundle Dataclass:**
```python
@dataclass(frozen=True)
class EEGDataBundle:
    data: np.ndarray                       # Raw EEG (numpy)
    labels: np.ndarray                     # Class labels (numpy)
    channel_count: int = 62                # EEG channels
    num_classes: int = 3                   # Rest, Elbow, Hand
    sample_rate: int = 200                 # Hz
    class_distribution: dict               # Class counts
```

---

## 7. TRAIN/TEST SPLIT

### Location: `src/mi3_eeg/dataset.py:prepare_data_loaders()` (lines 396-440)

```python
from sklearn.model_selection import train_test_split

train_data, test_data, train_labels, test_labels = train_test_split(
    data_bundle.data,          # (900, 62, 800)
    data_bundle.labels,        # (900, 1)
    test_size=0.2,             # 20% test split
    shuffle=True,              # Randomize order
    random_state=42,           # Reproducible split
)
```

**Split Result:**
- **Training Set:**
  - Data: (720, 62, 800)
  - Labels: (720, 1)
  - Classes: ~240 Rest, ~240 Elbow, ~240 Hand
  
- **Test Set:**
  - Data: (180, 62, 800)
  - Labels: (180, 1)
  - Classes: ~60 Rest, ~60 Elbow, ~60 Hand

---

## 8. CREATE DATALOADERS

### Location: `src/mi3_eeg/dataset.py:create_data_loader()` (lines 349-395)

### Step 8.1: Convert Labels to Tensor
```python
label_tensor = torch.LongTensor(labels.flatten()).to(device)
# Shape: (720,) for training, (180,) for testing
# Dtype: int64 (standard for classification)
# Device: CUDA or CPU
```

### Step 8.2: Ensure Data Shape
```python
if data.shape[1] >= data.shape[2]:
    logger.debug("Swapping axes to get (samples, channels, timepoints)")
    data = data.swapaxes(1, 2)
# Expected output: (samples, 62, 800)
```

### Step 8.3: Convert Data to Tensor
```python
data_tensor = torch.tensor(data, dtype=torch.float32)
# Shape: (720, 62, 800) for training
# Dtype: float32
# NOT on device yet
```

### Step 8.4: Add Channel Dimension for Conv2D
```python
data_tensor = torch.unsqueeze(data_tensor, dim=1).to(device)
# Shape before: (720, 62, 800)
# Shape after:  (720, 1, 62, 800)
#                 └─ batch
#                    └─ new channel dimension for Conv2D
#                       └─ original data
```

**Why Add Channel Dimension?**
- PyTorch Conv2D expects: (batch, channels, height, width)
- EEG data naturally: (batch, channels, timepoints)
- Adding 1 channel: (batch, 1, channels, timepoints)
- Allows Conv2D to process spatial electrode patterns

### Step 8.5: Create TensorDataset
```python
dataset = data_utils.TensorDataset(data_tensor, label_tensor)
# Pairs: (sample_i, label_i)
# Length: 720 (training) or 180 (testing)
```

### Step 8.6: Wrap in DataLoader
```python
loader = data_utils.DataLoader(
    dataset=dataset,
    batch_size=64,           # 64 samples per batch
    shuffle=shuffle,         # True for training, False for testing
    drop_last=False,         # Keep incomplete last batch
)
```

**Training DataLoader Behavior:**
```
Number of batches: 720 / 64 = 11.25 → 12 batches
Batch sizes:
  - Batches 0-10: 64 samples each (704 samples)
  - Batch 11: 16 samples (720 - 704)

Iteration yields:
  (X_batch, Y_batch)
  X_batch shape: (64, 1, 62, 800) or (16, 1, 62, 800) for last batch
  Y_batch shape: (64,) or (16,)
```

**Test DataLoader Behavior:**
```
Number of batches: 180 / 64 = 2.8 → 3 batches
Batch sizes:
  - Batches 0-1: 64 samples each (128 samples)
  - Batch 2: 52 samples (180 - 128)

Iteration yields:
  (X_batch, Y_batch)
  X_batch shape: (64, 1, 62, 800) or (52, 1, 62, 800) for last batch
  Y_batch shape: (64,) or (52,)
```

---

## 9. DATA FLOW SUMMARY TABLE

| Stage | Input Format | Output Format | Shape | Dtype | Device |
|-------|--------------|---------------|-------|-------|--------|
| 1. Raw Mat File | MATLAB struct | Dict with keys | - | - | Disk |
| 2. After Load | np.ndarray | np.ndarray | (900, 62, 800) | float64 | RAM |
| 3. Labels Extract | np.ndarray | np.ndarray | (900, 1) | int64 | RAM |
| 4. Train/Test Split | np.ndarray | np.ndarray | (720, 62, 800) / (180, 62, 800) | float64 | RAM |
| 5. Convert to Tensor | np.ndarray | torch.Tensor | (720, 62, 800) | float32 | CPU |
| 6. Add Channel Dim | torch.Tensor | torch.Tensor | (720, 1, 62, 800) | float32 | CUDA/CPU |
| 7. In DataLoader | torch.Tensor | Batch tuple | (64, 1, 62, 800) | float32 | CUDA/CPU |

---

## 10. CONFIGURATION PARAMETERS

### DataConfig (src/mi3_eeg/config.py, lines 79-107)
```python
@dataclass(frozen=True)
class DataConfig:
    mat_filename: str = ""              # e.g., "sub-001_eeg200hz.mat"
    subject_id: str = ""                # e.g., "sub-001"
    sampling_rate: int = 200            # Hz (200 or 90)
    bandpass_filter: tuple = (7, 35)    # Filter range (Hz)
    num_channels: int = 62              # EEG channels
    num_classes: int = 3                # Rest, Elbow, Hand
    class_names: tuple = ("Rest", "Elbow", "Hand")
    reduce_rest_ratio: float = 1        # 1.0 = no reduction
    test_size: float = 0.2              # 20% test split
    random_seed: int = 42               # Reproducibility
    expected_sampling_rate: int | None = None
    validate_timepoints: bool = True
```

### TrainingConfig (src/mi3_eeg/config.py, lines 110-132)
```python
@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 600               # Max training epochs
    batch_size: int = 64            # Batch size
    learning_rate: float = 0.01     # Optimizer LR
    dropout: float = 0.4            # Dropout probability
    early_stopping_patience: int = 200
    early_stopping_min_delta: float = 5e-4
    device: str = "cuda"            # Device selection
```

---

## 11. LOGGING OUTPUT EXAMPLE

When running `python -m mi3_eeg.main`:

```
MI3 EEG Motor Imagery Classification Pipeline
Using device: cuda
STAGE 1: Loading and Preprocessing Data
Auto-selecting first available subject: sub-001_eeg200hz.mat
Dataset: sub-001_eeg200hz.mat, Subject: sub-001, Sampling: 200Hz, Test split: 20%
Loading dataset from: .../Datasets/MI3/derivatives/sub-001_eeg200hz.mat
Detected format: standardized
Original data shape: (900, 62, 800)
Original label shape: (900, 1)
Original class distribution: {'Rest': 300, 'Elbow': 300, 'Hand': 300}
Data shape: (900, 62, 800), Class distribution: {'Rest': 300, 'Elbow': 300, 'Hand': 300}
Split: 720 train samples, 180 test samples
Created DataLoader: 720 samples, batch_size=64, shuffle=True
Created DataLoader: 180 samples, batch_size=64, shuffle=False
STAGE 2: Training Models
Training lenet model...
```

---

## 12. KEY TRANSFORMATIONS SUMMARY

### Numpy Arrays (In Memory)
```
Load .mat file
    ↓
Validate & extract all_data, all_label
    ↓
Optional: Balance Rest class (downsample)
    ↓
Create EEGDataBundle (all numpy arrays)
    ↓
Train/Test split with sklearn
    ↓
Two numpy array pairs: (train_data, train_labels), (test_data, test_labels)
```

### PyTorch Tensors (Device-Ready)
```
Numpy array (720, 62, 800)
    ↓
torch.tensor(..., dtype=float32)  [still CPU]
    ↓
torch.unsqueeze(dim=1)  →  (720, 1, 62, 800)
    ↓
.to(device)  →  Move to CUDA/CPU
    ↓
TensorDataset pairs data with labels
    ↓
DataLoader batches and shuffles
    ↓
Yields batches: X(B, 1, 62, 800), Y(B,) during iteration
```

---

## 13. ERROR HANDLING FLOW

```
File not found
    ↓ Check derivatives folder
    ↓ Auto-detect raw format files
    ↓ Auto-convert if found
    ↓ Retry loading
    ↓ If still fails: Graceful exit with helpful message

Corrupted .mat file
    ↓ ValueError caught
    ↓ Log detailed error message
    ↓ Suggest re-download
    ↓ Graceful exit

Unexpected error during processing
    ↓ Exception caught with context
    ↓ Full traceback logged
    ↓ Graceful exit with error code 1
```

---

## 14. READY FOR TRAINING

At the end of Stage 1, the pipeline produces:

```python
# Objects passed to train_model():
train_loader: torch.utils.data.DataLoader
    - Type: DataLoader
    - Length: 12 batches (720 samples / 64 batch size)
    - Each iteration yields: (X, Y)
      - X: torch.Tensor shape (B, 1, 62, 800), dtype float32
      - Y: torch.Tensor shape (B,), dtype int64
    - Device: CUDA (if available)
    - Shuffled: True

test_loader: torch.utils.data.DataLoader
    - Type: DataLoader  
    - Length: 3 batches (180 samples / 64 batch size)
    - Each iteration yields: (X, Y)
      - X: torch.Tensor shape (B, 1, 62, 800), dtype float32
      - Y: torch.Tensor shape (B,), dtype int64
    - Device: CUDA (if available)
    - Shuffled: False

training_config: TrainingConfig
    - epochs: 50 (or specified)
    - batch_size: 64
    - learning_rate: 0.01
    - device: "cuda" or "cpu"
    - early_stopping_patience: 200
    
model_config: ModelConfig
    - channel_count: 62
    - classes_num: 3
    - drop_out: 0.4

model: LENet instance
    - Input: (B, 1, 62, 800)
    - Output: (B, 3) - logits for 3 classes
```

### Next Step: Model Training
```python
history = train_model(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    config=training_config,
    save_path=paths.models / "sub-001_lenet_best.pth",
)
```

---

## References

- **Dataset Module:** `src/mi3_eeg/dataset.py`
- **Format Conversion:** `src/mi3_eeg/data_formatting/dataformatter.py`
- **Configuration:** `src/mi3_eeg/config.py`
- **Main Pipeline:** `src/mi3_eeg/main.py`
- **MIT3 Dataset Paper:** Multi-channel EEG recording during motor imagery (Nature Sci Data 2020)

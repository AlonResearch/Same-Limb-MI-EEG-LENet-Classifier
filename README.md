# Same Limb MI-EEG LENet Classifier

A modular, production-ready PyTorch package for classifying motor imagery EEG signals from the MI3 dataset using deep learning.

## 🎯 Overview

Motor-imagery EEG trials from the MI3 dataset are classified with the LENet architecture:

- **LENet** – A CNN that is lightweight and efficient by using Convolutional Classification Block with multi-scale temporal, spatial, and feature fusion layers

**Key Features:**
- ✅ Modular, testable architecture following best practices
- ✅ BIDS-compliant dataset structure
- ✅ Comprehensive logging and monitoring
- ✅ Automated training with early stopping
- ✅ Rich visualization suite
- ✅ GPU/CUDA acceleration support
- ✅ 63 unit tests ensuring reliability (including CUDA tests)

## 📁 Project Structure

```
Same-Limb-MI-EEG-LENet-Classifier/
├── .pytest_cache/            # Pytest cache directory
├── .venv/                    # Virtual environment
├── Datasets/                 # BIDS-formatted MI3 dataset
│   └── MI3/
│       └── derivatives/      # Preprocessed .mat files (not in repo)
├── models/                   # Saved model weights (subject-specific)
│   ├── sub-001_lenet_best.pth
│   ├── sub-001_lenet_final.pth
│   └── ...                   # Models for all 24 subjects
├── Notebooks/                # Exploratory notebooks
├── reports/                  # Training outputs
│   ├── figures/              # Plots and visualizations
│   │   ├── sub-XXX_lenet_confusion_matrix.png
│   │   ├── sub-XXX_lenet_training_curves.png
│   │   └── ...
│   ├── logs/                 # Training logs
│   └── metrics/              # Evaluation results (JSON)
│       ├── sub-XXX_lenet_results.json
│       └── ...               # Results for all 24 subjects
├── src/                      # Source code
│   └── mi3_eeg/              # Main package
│       ├── config.py         # Configuration and paths
│       ├── dataset.py        # Data loading and preprocessing
│       ├── evaluation.py     # Metrics and evaluation
│       ├── logger.py         # Centralized logging
│       ├── main.py           # Pipeline orchestrator
│       ├── model.py          # Neural network architectures
│       ├── run_all_subjects.py  # Batch processing script for all subjects
│       ├── train.py          # Training orchestration
│       └── visualization.py  # Plotting and visualization
├── tests/                    # Unit tests (63 tests)
├── .gitignore                # Git ignore file
├── OpenVScodeHere.bat        # Batch file to open VS Code
├── pyproject.toml            # Project metadata & dependencies
├── README.md                 # This file
├── setup_env.ps1             # Windows setup script
├── setup_env.sh              # Linux/Mac setup script
└── uv.lock                   # UV lock file
```

**Note:** Raw sourcedata files and large dataset files are not tracked in the repository due to size constraints. Download preprocessed `.mat` files separately (see Dataset section below).

## 💻 Environment & Requirements

### System Requirements
- **Python:** 3.11 or 3.12
- **GPU:** NVIDIA GPU with CUDA 12.4+ (recommended)
  - GTX 1060 or better for training
  - 4GB+ VRAM recommended
- **RAM:** 8GB+ system memory
- **Storage:** 2GB for dependencies + dataset

### Core Dependencies
- **PyTorch:** 2.5.1+ with CUDA 12.4 support (configured via `pyproject.toml`)
- **NumPy:** 1.24+ for numerical operations
- **scikit-learn:** 1.3+ for metrics and data splitting
- **matplotlib:** 3.7+ for visualizations
- **scipy:** 1.11+ for .mat file loading
- **pandas:** 2.0+ for data handling

PyTorch with CUDA 12.4 is automatically installed through the custom PyTorch index configured in `pyproject.toml`.

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- CUDA 12.4+ (for GPU acceleration)
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip
  - **uv** is a fast Python package installer
  - It's optional but recommended for faster dependency installation

### Installation (Recommended: uv)

#### Option 1: Automated Setup (Easiest)

<details>
<summary><b>🪟 Windows (PowerShell)</b></summary>

```powershell
# Install uv if you don't have it
pip install uv

# Clone and setup
git clone <repository-url>
cd Same-Limb-MI-EEG-LENet-Classifier
.\setup_env.ps1
```

The setup script will:
- ✅ Verify uv is installed (or install if missing)
- ✅ Create virtual environment with `uv sync`
- ✅ Install all dependencies (including PyTorch CUDA 12.4)
- ✅ Install optional dependencies (test, lint, notebook)
- ✅ Verify installation and CUDA

**After setup completes, activate the virtual environment:**
```powershell
.\.venv\Scripts\Activate.ps1
```

**VS Code users:** Select the Python interpreter from `.venv` (Ctrl+Shift+P → "Python: Select Interpreter")

</details>

<details>
<summary><b>🐧 Linux/Mac</b></summary>

```bash
# Install uv if you don't have it
pip install uv

# Clone and setup
git clone <repository-url>
cd Same-Limb-MI-EEG-LENet-Classifier
chmod +x setup_env.sh
./setup_env.sh
```

The setup script will:
- ✅ Verify uv is installed (or install if missing)
- ✅ Create virtual environment with `uv sync`
- ✅ Install all dependencies (including PyTorch CUDA 12.4)
- ✅ Install optional dependencies (test, lint, notebook)
- ✅ Verify installation and CUDA

**After setup completes, activate the virtual environment:**
```bash
source .venv/bin/activate
```

**VS Code users:** Select the Python interpreter from `.venv` (Cmd+Shift+P → "Python: Select Interpreter")

</details>

<details>
<summary><b>❌ Troubleshooting: pip install uv fails</b></summary>

If you encounter errors when running `pip install uv`, try these solutions:

**Solution 1: Upgrade pip first**
```bash
python -m pip install --upgrade pip
pip install uv
```

**Solution 2: Use python -m pip explicitly**
```bash
python -m pip install --user uv
```

**Solution 3: If you have permission issues (especially on Linux/Mac)**
```bash
# Install for current user only
pip install --user uv

# Then add to PATH (Linux/Mac)
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

**Solution 4: Use uv bootstrap (direct installation)**
If all else fails, download uv directly from: https://github.com/astral-sh/uv/releases
- Extract to a folder in your PATH or run directly with the full path
- Verify: `uv --version`

**If still having issues:**
You can proceed with **Option 2 (Manual Setup)** or **Alternative Installation (Standard pip)** below without uv.

</details>

#### Option 2: Manual Setup

<details>
<summary><b>🪟 Windows (PowerShell)</b></summary>

1. **Install uv package manager (optional):**
```powershell
pip install uv
```
If this fails, see the troubleshooting section above.

2. **Clone the repository:**
```powershell
git clone <repository-url>
cd Same-Limb-MI-EEG-LENet-Classifier
```

3. **Create and activate virtual environment:**
```powershell
uv venv
.\.venv\Scripts\Activate.ps1
```

4. **Install all dependencies (includes PyTorch CUDA 12.4):**
```powershell
uv sync --all-extras
```

5. **Verify installation:**
```powershell
# Check package
python -c "import mi3_eeg; print(f'mi3_eeg v{mi3_eeg.__version__}')"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run tests
pytest tests/
```

</details>

<details>
<summary><b>🐧 Linux/Mac</b></summary>

1. **Install uv package manager (optional):**
```bash
pip install uv
```
If this fails, see the troubleshooting section above.

2. **Clone the repository:**
```bash
git clone <repository-url>
cd Same-Limb-MI-EEG-LENet-Classifier
```

3. **Create and activate virtual environment:**
```bash
uv venv
source .venv/bin/activate
```

4. **Install all dependencies (includes PyTorch CUDA 12.4):**
```bash
uv sync --all-extras
```

5. **Verify installation:**
```bash
# Check package
python -c "import mi3_eeg; print(f'mi3_eeg v{mi3_eeg.__version__}')"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run tests
pytest tests/
```

</details>

### Alternative Installation (Standard pip)

<details>
<summary><b>🪟 Windows (PowerShell)</b></summary>

If you prefer using pip without uv or if uv installation fails:

```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install package
pip install -e ".[test]"
```

**Note:** Using `uv` is recommended as it's faster and handles dependencies better, but standard `pip` works too.

</details>

<details>
<summary><b>🐧 Linux/Mac</b></summary>

If you prefer using pip without uv or if uv installation fails:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install package
pip install -e ".[test]"
```

**Note:** Using `uv` is recommended as it's faster and handles dependencies better, but standard `pip` works too.

</details>

### Running the Full Pipeline

**Important:** Always ensure your virtual environment is activated before running commands!

#### Processing All Subjects

To train models on all subjects in the dataset:
```bash
python -m mi3_eeg.run_all_subjects
```

This script will:
- ✅ Automatically detect all `*_eeg200hz.mat` files in the derivatives folder
- ✅ Skip subjects that have already been processed
- ✅ Train each subject with 50 epochs
- ✅ Save results to `reports/metrics/sub-XXX_lenet_results.json`
- ✅ Generate figures in `reports/figures/`
- ✅ Save trained models in `models/sub-XXX_lenet_*.pth`

The script will display progress and automatically continue if a subject fails.

#### Activating the Virtual Environment

<details>
<summary><b>🪟 Windows (PowerShell)</b></summary>

```powershell
.\.venv\Scripts\Activate.ps1
```

**VS Code:** The virtual environment should be automatically detected. If not:
1. Press `Ctrl+Shift+P`
2. Type "Python: Select Interpreter"
3. Choose the interpreter from `.venv` folder
4. Open a new terminal (it will auto-activate)

</details>

<details>
<summary><b>🐧 Linux/Mac</b></summary>

```bash
source .venv/bin/activate
```

**VS Code:** The virtual environment should be automatically detected. If not:
1. Press `Cmd+Shift+P`
2. Type "Python: Select Interpreter"
3. Choose the interpreter from `.venv` folder
4. Open a new terminal (it will auto-activate)

</details>

#### Running the Pipeline

Train a single subject with default settings (GPU):
```bash
python -m mi3_eeg.main
```

Train specific subject file with custom settings:
```bash
python -m mi3_eeg.main --subject-file sub-001_eeg200hz.mat --epochs 50 --device cuda
```

Use CPU if GPU is not available:
```bash
python -m mi3_eeg.main --device cpu
```

Process all subjects at once:
```bash
python -m mi3_eeg.run_all_subjects
```

### Troubleshooting Installation

<details>
<summary><b>❌ Issue: ModuleNotFoundError: No module named 'mi3_eeg'</b></summary>

This means the virtual environment is not activated. Solution:
```bash
# Windows
.\.venv\Scripts\Activate.ps1

# Linux/Mac
source .venv/bin/activate

# Verify it's activated (you should see (.venv) in your prompt)
python -c "import mi3_eeg; print('✅ Package found!')"
```

**VS Code Users:** If using the integrated terminal:
1. Open Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`)
2. Select "Python: Select Interpreter"
3. Choose the `.venv` interpreter
4. Open a new terminal (it will auto-activate the venv)

</details>

<details>
<summary><b>❌ Issue: Corrupted virtual environment</b></summary>

Remove and recreate the virtual environment:
```bash
# Windows PowerShell
Remove-Item -Recurse -Force .venv

# Linux/Mac
rm -rf .venv

# Then follow installation steps again
uv venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

</details>

<details>
<summary><b>❌ Issue: PyTorch CPU version installed instead of CUDA</b></summary>

The project is configured to automatically install PyTorch with CUDA 12.4 support through the PyTorch index specified in `pyproject.toml`. If you're seeing the CPU version:

```bash
# Check current PyTorch version
python -c "import torch; print(torch.__version__)"

# Should show something like: 2.6.0+cu124

# If it shows +cpu instead, reinstall:
# 1. Remove venv
rm -rf .venv  # Linux/Mac
Remove-Item -Recurse -Force .venv  # Windows

# 2. Run setup script again
./setup_env.sh  # Linux/Mac
.\setup_env.ps1  # Windows
```

The `pyproject.toml` contains:
```toml
[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true
```

This ensures PyTorch with CUDA 12.4 is always installed automatically.

</details>

<details>
<summary><b>❌ Issue: CUDA not detected despite installation</b></summary>

Verify CUDA installation and configuration:
```bash
# 1. Verify CUDA is installed on system
nvidia-smi

# 2. Check PyTorch version and CUDA support
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else "N/A"}')"

# Expected output:
# PyTorch: 2.6.0+cu124
# CUDA Available: True
# CUDA Version: 12.4

# If CUDA not detected:
# 1. Check NVIDIA drivers are up-to-date (requires 525+ for CUDA 12.4)
nvidia-smi  # Check driver version at top

# 2. Verify PyTorch CUDA build is installed (should show +cu124)
python -c "import torch; print(torch.__version__)"

# 3. If showing CPU build (+cpu), reinstall environment
rm -rf .venv && ./setup_env.sh  # Linux/Mac
Remove-Item -Recurse -Force .venv; .\setup_env.ps1  # Windows
```

**Note:** The project automatically configures PyTorch with CUDA 12.4 through `pyproject.toml`. The setup scripts handle this automatically.

</details>

### Using in Python

```python
from mi3_eeg import (
    load_dataset_from_config,
    prepare_data_loaders,
    create_model,
    train_model,
    evaluate_model,
)

# Load data
data_bundle = load_dataset_from_config()
train_loader, test_loader = prepare_data_loaders(data_bundle)

# Create and train model
model = create_model("lenet", device="cuda")
history = train_model(model, train_loader, test_loader)

# Evaluate
results = evaluate_model(model, test_loader)
print(f"Accuracy: {results.overall_accuracy * 100:.2f}%")
```

## 📊 Dataset

### MI3 Dataset Setup

**IMPORTANT:** The preprocessed dataset files are not included in this repository due to their large size (several GB). You must download them separately.

#### Downloading the Dataset

1. **Access the MI3 Dataset:**
   - Dataset paper: [Motor Imagery Dataset of Same Limb during Motor Execution and Motor Imagery](https://doi.org/10.1038/s41597-023-02020-0)
   - OpenNeuro: [https://openneuro.org/datasets/ds004148](https://openneuro.org/datasets/ds004148)
   - Direct download of preprocessed files is available from the dataset source

2. **Download preprocessed derivatives:**
   - Navigate to the dataset's `derivatives/` folder
   - Download the preprocessed `.mat` files (200Hz sampling rate):
     - `sub-001_eeg200hz.mat` through `sub-024_eeg200hz.mat` (24 subjects total)
   - Optional: Download `sub-011_eeg.mat` and `sub-011_eeg90hz.mat` for testing
   
   **Note:** As of January 2026, all 24 subjects (sub-001 through sub-024) have been processed with results available in `reports/metrics/`.

3. **Place files in your local repository:**
   ```bash
   # Create the derivatives directory if it doesn't exist
   mkdir -p Datasets/MI3/derivatives
   
   # Copy downloaded .mat files to:
   Datasets/MI3/derivatives/
   ```

4. **Verify the structure:**
   ```
   Datasets/MI3/derivatives/
   ├── sub-001_eeg200hz.mat
   ├── sub-002_eeg200hz.mat
   ├── ...
   ├── sub-024_eeg200hz.mat
   ├── sub-011_eeg.mat (optional)
   └── sub-011_eeg90hz.mat (optional)
   ```

**Note:** A small example file (`sub-011_eeg90hz.mat`) may be included in the repository for testing purposes. This file contains 90Hz downsampled, bandpass filtered (7-35Hz) data with shape (965 samples, 62 channels, 360 timepoints) and classes: Rest (0), Elbow (1), Hand (2).

### MI3 Dataset Structure (BIDS Format)

The project expects BIDS-formatted MI3 data in `Datasets/MI3/`:

- **derivatives/** – Preprocessed MATLAB .mat files (200Hz sampling, not tracked in git)
  - `sub-XXX_eeg200hz.mat` – Preprocessed EEG data files
  - Shape: (~900 samples, 62 channels, 800 timepoints)
  - Classes: Rest (0), Elbow (1), Hand (2)
  - **Important:** These files are not included in the repository due to size. Download separately (see above).

**Note:** Raw sourcedata files (`.cnt` format) are not required for training and are not included in the repository.

### Data Processing Pipeline

1. Python loads from derivatives → Class balancing → PyTorch tensors
2. Train/test split (80/20) → DataLoaders → Model training

## 🧪 Development

### Running Tests

```bash
# All tests (including CUDA tests if GPU available)
pytest

# Specific module
pytest tests/test_model.py -v

# CUDA tests only
pytest tests/test_cuda.py -v

# With coverage
pytest --cov=mi3_eeg --cov-report=html

# Show CUDA info
pytest tests/test_cuda.py::test_cuda_info_display -v -s
```

**Note:** CUDA tests automatically skip if GPU is not available, so all tests should pass on any machine.

### Code Quality

```bash
# Format code (if ruff installed)
ruff format .

# Check linting
ruff check .

# Type checking (if mypy installed)
mypy src/
```

## 📈 Results

### Completed Subjects

**✅ All 24 subjects processed** (sub-001 through sub-024)

Results are available in:
- `reports/metrics/sub-XXX_lenet_results.json` - Detailed metrics for each subject
- `reports/figures/sub-XXX_lenet_*.png` - Visualizations (confusion matrices, training curves)
- `models/sub-XXX_lenet_*.pth` - Trained model weights

### Typical Performance

Performance on MI3 dataset varies by subject:

| Model      | Overall Acc | Rest Acc | Elbow Acc | Hand Acc |
|------------|-------------|----------|-----------|----------|
| LENet (CCB)| ~75-80%     | ~98%     | ~50-60%   | ~65-75%  |

**Training Configuration:**
- Epochs: 50 per subject
- Batch size: 64
- Learning rate: 0.01
- Early stopping: Enabled (patience: 50)
- Device: CUDA (GPU acceleration)

*Results vary based on random initialization, data splits, and subject-specific characteristics.*

## 🔧 Configuration

Key configurations in `src/mi3_eeg/config.py`:

### DataConfig
- `mat_filename`: Dataset file name
- `sampling_rate`: 90 Hz
- `bandpass_filter`: (7, 35) Hz
- `num_channels`: 62
- `test_size`: 0.2 (20% test split)

### TrainingConfig
- `epochs`: 1000 (with early stopping)
- `batch_size`: 64
- `learning_rate`: 0.01
- `dropout`: 0.35
- `early_stopping_patience`: 50

### ModelConfig
- `classes_num`: 3 (Rest, Elbow, Hand)
- `channel_count`: 62
- `drop_out`: 0.35

## 📚 Module Documentation

### `config.py`
Configuration dataclasses for paths, data, training, and model parameters.

### `dataset.py`
- Load EEG data from BIDS derivatives
- Class balancing and preprocessing
- PyTorch DataLoader creation

### `model.py`
- LENet architecture with CCB (Convolutional Classification Block)
- Model factory functions
- Weight initialization and serialization

### `train.py`
- Training orchestration with early stopping
- Epoch-level train/validation loops
- Training history tracking

### `evaluation.py`
- Model evaluation and metrics
- Confusion matrix computation
- Per-class accuracy analysis

### `visualization.py`
- Training curve plots
- Confusion matrix visualizations
- Model comparison charts

### `main.py`
- End-to-end pipeline orchestrator
- CLI interface
- Logging and artifact management

## 🚧 Future Improvements

- [ ] Additional model architectures (RNN, Transformer, Attention)
- [ ] Transfer learning capabilities
- [ ] Hyperparameter optimization with Optuna
- [ ] Cross-subject validation
- [ ] Real-time inference pipeline
- [ ] Web-based demo interface

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass (`pytest`)
5. Submit a pull request

## 📞 Contact

For questions or issues, please open a GitHub issue or contact artalon.contact@gmail.com

---

**Built with ❤️ for EEG research**

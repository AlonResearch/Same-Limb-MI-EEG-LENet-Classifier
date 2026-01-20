# MI3 EEG Motor Imagery Classification

A modular, production-ready PyTorch package for classifying motor imagery EEG signals from the MI3 dataset using deep learning.

## 🎯 Overview

Motor-imagery EEG trials from the MI3 dataset are classified with lightweight LENet variants:

- **LENet (CCB)** – Convolutional Classification Block matching the MI2/BCI IV-2a architecture
- **LENet_FCL** – Fully Connected Layer variant with dynamic sizing for higher-channel recordings

**Key Features:**
- ✅ Modular, testable architecture following best practices
- ✅ BIDS-compliant dataset structure
- ✅ Comprehensive logging and monitoring
- ✅ Automated training with early stopping
- ✅ Rich visualization suite
- ✅ GPU/CUDA acceleration support
- ✅ 71 unit tests ensuring reliability (including CUDA tests)

## 📁 Project Structure

```
MI-EEG-Final-ML-Proj/
├── src/mi3_eeg/              # Main package
│   ├── config.py             # Configuration and paths
│   ├── logger.py             # Centralized logging
│   ├── dataset.py            # Data loading and preprocessing
│   ├── model.py              # Neural network architectures
│   ├── train.py              # Training orchestration
│   ├── evaluation.py         # Metrics and evaluation
│   ├── visualization.py      # Plotting and visualization
│   └── main.py               # Pipeline orchestrator
├── tests/                    # Unit tests (60+ tests)
├── Datasets/                 # BIDS-formatted MI3 dataset
│   └── MI3/
│       ├── sourcedata/       # Raw .cnt files (immutable)
│       ├── derivatives/      # Preprocessed .mat files
│       └── code/             # MATLAB preprocessing scripts
├── data/                     # PyTorch processing cache
│   ├── tensors/              # Cached tensors
│   └── splits/               # Train/test splits
├── models/                   # Saved model weights
├── reports/                  # Training outputs
│   ├── figures/              # Plots and visualizations
│   ├── metrics/              # Evaluation results (JSON)
│   └── logs/                 # Training logs
├── Notebooks/                # Exploratory notebooks
└── pyproject.toml            # Project metadata & dependencies
```

## 💻 Environment & Requirements

### System Requirements
- **Python:** 3.11 or 3.12
- **GPU:** NVIDIA GPU with CUDA 12.4+ (recommended)
  - GTX 1060 or better for training
  - 4GB+ VRAM recommended
- **RAM:** 8GB+ system memory
- **Storage:** 2GB for dependencies + dataset

### Core Dependencies
- **PyTorch:** 2.5.1+ with CUDA 12.4 support
- **NumPy:** 2.4+ for numerical operations
- **scikit-learn:** 1.8+ for metrics and data splitting
- **matplotlib:** 3.10+ for visualizations
- **scipy:** 1.16+ for .mat file loading
- **pandas:** 2.3+ for data handling

All dependencies are managed through `pyproject.toml` and automatically installed.

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- CUDA 12.4+ (for GPU acceleration)
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip
  - **uv** is a fast Python package installer (install with: `pip install uv`)
  - It's optional but recommended for faster dependency installation

### Installation (Recommended: uv)

#### Option 1: Automated Setup (Easiest)

**Windows (PowerShell):**
```powershell
# Install uv if you don't have it
pip install uv

# Clone and setup
git clone <repository-url>
cd MI-EEG-Final-ML-Proj
.\setup_env.ps1
```

**Linux/Mac:**
```bash
# Install uv if you don't have it
pip install uv

# Clone and setup
git clone <repository-url>
cd MI-EEG-Final-ML-Proj
chmod +x setup_env.sh
./setup_env.sh
```

The setup script will:
- ✅ Verify uv is installed (or install if missing)
- ✅ Create virtual environment
- ✅ Install PyTorch with CUDA 12.4
- ✅ Install all dependencies
- ✅ Verify installation and CUDA

**After setup completes, activate the virtual environment:**
```powershell
# Windows
.\.venv\Scripts\Activate.ps1

# Linux/Mac
source .venv/bin/activate
```

**VS Code users:** Select the Python interpreter from `.venv` (Ctrl+Shift+P → "Python: Select Interpreter")

#### Option 2: Manual Setup

1. **Install uv package manager (if not already installed):**
```bash
pip install uv
```

2. **Clone the repository:**
```bash
git clone <repository-url>
cd MI-EEG-Final-ML-Proj
```

3. **Create and activate virtual environment:**
```bash
uv venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

4. **Install PyTorch with CUDA support:**
```bash
# For CUDA 12.4
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

5. **Install the package with dependencies:**
```bash
uv pip install -e ".[test]"
```

6. **Verify installation:**
```bash
# Check package
python -c "import mi3_eeg; print(f'mi3_eeg v{mi3_eeg.__version__}')"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run tests
pytest tests/
```

### Alternative Installation (Standard pip)

If you prefer using pip without uv:

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows (or source .venv/bin/activate on Linux/Mac)

# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install package
pip install -e ".[test]"
```

**Note:** Using `uv` is recommended as it's faster and handles dependencies better, but standard `pip` works too.

### Running the Full Pipeline

**Important:** Always ensure your virtual environment is activated before running commands!

#### Activating the Virtual Environment

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

**VS Code:** The virtual environment should be automatically detected. If not:
1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type "Python: Select Interpreter"
3. Choose the interpreter from `.venv` folder

#### Running the Pipeline

Train both models with default settings (GPU):
```bash
python -m mi3_eeg.main
```

Train specific model with custom settings:
```bash
python -m mi3_eeg.main --models lenet --epochs 500 --device cuda
```

Use CPU if GPU is not available:
```bash
python -m mi3_eeg.main --device cpu
```

### Troubleshooting Installation

**Issue: ModuleNotFoundError: No module named 'mi3_eeg'**

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

**Issue: Corrupted virtual environment**
```bash
# Remove and recreate
Remove-Item -Recurse -Force .venv  # Windows PowerShell
# or
rm -rf .venv  # Linux/Mac

# Then follow installation steps again
uv venv
.venv\Scripts\activate
```

**Issue: Corrupted virtual environment**
```bash
# Remove and recreate
Remove-Item -Recurse -Force .venv  # Windows PowerShell
# or
rm -rf .venv  # Linux/Mac

# Then follow installation steps again
uv venv
.venv\Scripts\activate
```

**Issue: PyTorch CPU version installed instead of CUDA**
```bash
# Uninstall CPU version
uv pip uninstall torch

# Install CUDA version
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**Issue: CUDA not detected**
```bash
# Verify CUDA installation
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

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

### MI3 Dataset Structure (BIDS Format)

The project expects BIDS-formatted MI3 data in `Datasets/MI3/`:

- **sourcedata/** – Raw CNT acquisitions (immutable)
- **derivatives/** – Preprocessed MATLAB .mat files
  - `sub-011_eeg90hz.mat` – 90Hz downsampled, bandpass filtered (7-35Hz)
  - Shape: (965 samples, 62 channels, 360 timepoints)
  - Classes: Rest (0), Elbow (1), Hand (2)
- **code/** – MATLAB preprocessing scripts (`MI3_process.m`)

### Data Processing Pipeline

1. Raw CNT files → MATLAB preprocessing → .mat files (derivatives)
2. Python loads from derivatives → Class balancing → PyTorch tensors
3. Train/test split (80/20) → DataLoaders → Model training

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

Typical performance on MI3 dataset (sub-011):

| Model      | Overall Acc | Rest Acc | Elbow Acc | Hand Acc |
|------------|-------------|----------|-----------|----------|
| LENet (CCB)| ~75-80%     | ~98%     | ~50-60%   | ~65-75%  |
| LENet_FCL  | ~70-75%     | ~98%     | ~60-70%   | ~35-45%  |

*Results vary based on random initialization and data splits.*

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
- LENet and LENet_FCL architectures
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

## 🎓 Citation

If you use this code, please cite:

```bibtex
@software{mi3_eeg_2026,
  title = {MI3 EEG Motor Imagery Classification},
  author = {MI3-EEG Team},
  year = {2026},
  version = {1.0.0}
}
```

## 📄 License

[Specify your license here]

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

For questions or issues, please open a GitHub issue or contact [your contact info].

---

**Built with ❤️ for EEG research**

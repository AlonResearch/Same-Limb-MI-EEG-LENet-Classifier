# Same Limb MI-EEG LENet Classifier

A modular, production-ready PyTorch package for classifying motor imagery EEG signals from the MI3 dataset using deep learning, with comprehensive group-level analysis capabilities.

## 🎯 Overview

Motor-imagery EEG trials from the MI3 dataset are classified with the LENet architecture and analyzed at both individual and group levels:

- **LENet (CCB)** – A CNN architecture with Convolutional Classification Block featuring multi-scale temporal, spatial, and feature fusion layers
- **Group-Level Analysis** – Comprehensive statistical analysis, time-frequency decomposition (TFR), and topographical brain mapping across all subjects

**Key Features:**
- ✅ Modular, testable architecture following best practices
- ✅ BIDS-compliant dataset structure
- ✅ Subject-specific and full-cohort training modes
- ✅ Comprehensive group-level EEG analysis pipeline
- ✅ Time-frequency analysis with ERD/ERS mapping
- ✅ Topographical brain mapping with MNE
- ✅ Statistical testing (ANOVA, pairwise t-tests, FDR correction)
- ✅ Modular visualization system with caching
- ✅ Automated training with early stopping
- ✅ GPU/CUDA acceleration support
- ✅ 63+ unit tests ensuring reliability

## � Quick Navigation

<details open>
<summary><b>Click to expand table of contents</b></summary>

### Project Setup & Infrastructure
- [📁 Project Structure](#-project-structure) – Directory layout and file organization
- [💻 Environment & Requirements](#-environment--requirements) – System requirements and dependencies
- [🚀 Quick Start](#-quick-start) – Installation and setup instructions

### Pipelines & Usage
- [🔄 Complete Pipeline Workflow](#-complete-pipeline-workflow) – All 4 pipelines explained
  - [Pipeline A: Subject-Level Training](#pipeline-a-subject-level-training)
  - [Pipeline B: Classification Analysis](#pipeline-b-classification-analysis-depends-on-pipeline-a)
  - [Pipeline C: Time-Frequency & Topographical Analysis](#pipeline-c-time-frequency--topographical-analysis--independent)
  - [Pipeline D: Visualization Regeneration](#pipeline-d-visualization-regeneration-optional)
- [🔀 Pipeline Combinations](#-pipeline-combinations) – Different execution options

### Examples & API
- [📚 Examples](#-examples) – Command-line usage and Python API
- [🔍 Troubleshooting](#-troubleshooting) – Common issues and solutions

### Development
- [🧪 Testing](#-testing) – Running unit tests
- [📚 Code Documentation](#-code-documentation) – Key modules and architecture
- [🤝 Contributing](#-contributing) – Contribution guidelines

</details>

---

## �📁 Project Structure

### Root Level (Configuration & Data)
```
Same-Limb-MI-EEG-LENet-Classifier/
├── Datasets/                          # 📊 BIDS-formatted input data
│   └── MI3/derivatives/               # Preprocessed .mat files (download separately)
│       ├── sub-001_eeg200hz.mat
│       └── ...                        # sub-001 through sub-025
│
├── models/                            # 🧠 Trained neural network weights
│   ├── sub-001_lenet_best.pth         # Per-subject models (25 total)
│   ├── sub-001_lenet_final.pth
│   └── ...
│
└── reports/                           # 📈 Analysis outputs
    ├── figures/                       # Training visualizations
    ├── logs/                          # Training logs (.txt)
    ├── metrics/                       # Performance metrics (.json, .csv)
    └── group_analysis/                # Post-training group analysis
        ├── figures/                   # Performance summary plots
        ├── tfr_analysis/              # Time-frequency & topography
        └── statistics/                # Statistical test results
```

### Source Code (`src/mi3_eeg/`)

**Core Training Pipeline:**
```
├── main.py                            # 🔧 Single-subject training orchestrator
├── run_all_subjects.py                # 🔁 Batch training for all subjects
├── train.py                           # Training loops + early stopping
├── model.py                           # LENet architecture
├── dataset.py                         # Data loading & preprocessing
├── evaluation.py                      # Metrics computation
├── visualization.py                   # Training plots (curves, confusion matrices)
├── config.py                          # Configuration dataclasses
├── logger.py                          # Centralized logging
└── __init__.py                        # Package exports
```

**Analysis Submodule (`analysis/`):**
```
└── analysis/                          # 📊 Group-level analysis (independent pipeline)
    ├── group_analysis.py              # Main orchestrator (4-step workflow)
    ├── time_frequency.py              # TFR computation + ERD/ERS
    ├── topography.py                  # Topographical brain mapping
    ├── statistical_tests.py           # ANOVA, t-tests, FDR correction
    ├── tfr_visualization.py           # Modular TFR plotting
    ├── regenerate_visualizations.py   # Cache-based viz regeneration
    ├── metrics_aggregator.py          # Cross-subject metrics aggregation
    ├── visualization.py               # Analysis-specific plots
    └── __init__.py                    # Analysis exports
```

### Testing & Configuration (Root)
```
├── tests/                             # 🧪 Unit tests (63+ tests)
│   ├── test_*.py                      # Test modules
│   └── conftest.py                    # Pytest configuration
│
├── pyproject.toml                     # ⚙️ Project metadata & dependencies
├── setup_env.ps1                      # Windows setup automation
├── setup_env.sh                       # Linux/Mac setup automation
├── uv.lock                            # Locked dependency versions
├── README.md                          # This file
└── .gitignore / .gitattributes       # Git configuration
```

### Key Points
- **Data**: Download preprocessed `.mat` files to `Datasets/MI3/derivatives/`
- **Training outputs**: Saved to `models/` and `reports/{figures,metrics,logs}/`
- **Analysis outputs**: Grouped in `reports/group_analysis/` (separate from training)
- **All source code**: In `src/mi3_eeg/` (including `analysis/` submodule)

## 🔄 Complete Pipeline Workflow

The project has **4 independent pipelines** with flexible execution options:

### Pipeline A: Subject-Level Training

Train individual neural network models (no external dependencies needed):

```bash
python -m mi3_eeg.run_all_subjects
```

**Timing:** 30-60 minutes

**Process:**
1. Loads each subject's EEG data from `Datasets/MI3/derivatives/`
2. Trains a LENet model on that subject's data
3. Evaluates on held-out test set
4. Saves model weights and per-subject metrics

**Outputs:**
- `models/sub-XXX_lenet_best.pth` - Best model per subject (25 models)
- `models/sub-XXX_lenet_final.pth` - Final model per subject
- `reports/metrics/sub-XXX_lenet_results.json` - Per-subject performance
- `reports/figures/sub-XXX_lenet_*.png` - Confusion matrices, training curves

**Results by Subject:**
- Accuracy ranges from 37-75% across subjects
- Mean accuracy: 51.47% ± 8.19%

---

### Pipeline B: Classification Analysis (Depends on Pipeline A)

Aggregate training results and perform statistical analysis on classification performance:

```bash
python -m mi3_eeg.analysis.group_analysis --analysis-type classification
```

**Timing:** 2-5 minutes

**Dependencies:** ⚠️ **Requires Pipeline A** - must run after training completes

**3-Step Analysis:**

1. **Metrics Aggregation**
   - Loads all per-subject results from `reports/metrics/`
   - Aggregates accuracy, precision, recall, F1-score
   - Creates summary: `lenet_all_subjects_metrics.csv`

2. **Performance Visualization**
   - Accuracy distribution histogram (all 25 subjects)
   - Subject ranking bar chart
   - Per-class accuracy boxplots
   - Saves to `reports/group_analysis/figures/`

3. **Statistical Testing**
   - One-way ANOVA across classes (Rest/Elbow/Hand)
   - Pairwise t-tests with FDR correction (Benjamini-Hochberg)
   - Effect size calculations (Cohen's d, η²)
   - Saves to `reports/group_analysis/statistics/`

**Outputs:**
```
reports/group_analysis/figures/
├── lenet_accuracy_distribution.png
├── lenet_subject_ranking.png
└── lenet_class_accuracy_boxplot.png

reports/group_analysis/statistics/
├── lenet_classification_statistics.json
└── lenet_classification_statistics.txt
```

**Key Findings:**
- **One-way ANOVA:** F(2,72) = 5.83, **p = 0.0045** ✓✓✓
- **Effect Size:** η² = 0.139 (moderate)
- Rest significantly easier to classify than motor imagery (p=0.0005)

---

### Pipeline C: Time-Frequency & Topographical Analysis ⭐ **INDEPENDENT**

Analyze raw EEG data with no dependency on trained models:

```bash
python -m mi3_eeg.analysis.group_analysis --analysis-type tfr
```

**Timing:** 15-30 minutes

**Dependencies:** ✅ NONE - only needs raw `.mat` files

**🚀 Can run in parallel with Pipeline A!** Since they use different data sources, you can save 20+ minutes by running them simultaneously.

**Analysis:**

1. **Time-Frequency Decomposition**
   - Loads raw EEG from `Datasets/MI3/derivatives/`
   - Morlet wavelet decomposition (4-40 Hz)
   - Event-related desynchronization/synchronization (ERD/ERS)
   - Batch processing: Memory-efficient (~2-3 GB per batch)

2. **Topographical Mapping**
   - Brain maps for Alpha (8-13 Hz) and Beta (13-30 Hz) bands
   - Standard 10-20 electrode montage
   - MNE-based visualization

3. **Caching**
   - Joblib-based caching at `~/.cache/mi3_eeg/tfr/`
   - Saves `aggregated_tfr_data.pkl` for fast regeneration

**Outputs:**
```
reports/group_analysis/tfr_analysis/
├── group_time_frequency_maps.png (4.27 MB)
├── group_topomap_alpha.png (1.79 MB)
└── group_topomap_beta.png (1.72 MB)
```

**Key Findings:**
- Clear mu rhythm desynchronization (8-13 Hz) during motor imagery
- Beta band (13-30 Hz) modulation over motor cortex
- Topographical localization consistent with contralateral motor areas

---

### Pipeline D: Visualization Regeneration (Optional)

Regenerate TFR visualizations from cached data without recomputation:

```bash
python -m mi3_eeg.analysis.regenerate_visualizations
```

**Timing:** 10-30 seconds

**Dependencies:** Requires Pipeline C to have run at least once (for cache)

**Use Cases:**
- Tweak plot styling (DPI, colors, layout)
- Export to different formats
- Generate custom visualizations
- Fast iteration (~seconds instead of minutes)

---

## 🔀 Pipeline Combinations

**Option 1: Training Only**
```
Pipeline A → Done (30-60 min)
Get 25 trained models + per-subject results
```

**Option 2: Training + Classification Analysis**
```
Pipeline A → Pipeline B → Done (32-65 min)
Get trained models + cross-subject classification statistics
```

**Option 3: EEG Analysis Only (No Training)**
```
Pipeline C → Done (15-30 min)
Get TFR/topography without training models (quick analysis!)
```

**Option 4: All Analysis (Recommended for papers) - Parallel Execution**
```
Terminal 1: python -m mi3_eeg.run_all_subjects          # Pipeline A (30-60 min)
Terminal 2: python -m mi3_eeg.analysis.group_analysis --analysis-type tfr  # Pipeline C (15-30 min, runs in parallel!)
Terminal 3: (after A) python -m mi3_eeg.analysis.group_analysis --analysis-type classification  # Pipeline B (2-5 min)

Total Time: ~50 minutes (instead of 73 minutes sequential!)
Time Saved: ~23 minutes by running A & C in parallel
```

**Option 5: Tweak Visualizations**
```
Pipeline C → Pipeline D → Done (seconds)
Quickly regenerate TFR plots after adjusting parameters
```

## 💻 Environment & Requirements

### System Requirements
- **Python:** 3.11 or 3.12
- **GPU:** NVIDIA GPU with CUDA 12.4+ (recommended)
  - GTX 1060 or better for training
  - 4GB+ VRAM for single-subject training
  - 8GB+ VRAM for full-cohort training
- **RAM:** 
  - 8GB+ for subject-specific training
  - 16GB+ for full-cohort training
  - 32GB+ recommended for group TFR analysis (25 subjects)
- **Storage:** 
  - 2GB for dependencies
  - 7-10GB for dataset (25+ subjects @ 200Hz)
  - 1-2GB for models and results

### Core Dependencies
- **PyTorch:** 2.5.1+ with CUDA 12.4 (auto-configured via `pyproject.toml`)
- **NumPy:** 1.24+ for numerical operations
- **SciPy:** 1.11+ for signal processing & .mat file loading
- **scikit-learn:** 1.3+ for metrics and data splitting
- **MNE:** 1.11+ for EEG topography and TFR
- **matplotlib:** 3.7+ / **seaborn:** 0.13+ for visualizations
- **pandas:** 2.0+ for data aggregation
- **joblib:** 1.3+ for TFR caching
- **tqdm:** 4.67+ for progress bars
- **h5py:** 3.9+ for HDF5 .mat files
- **pywavelets:** 1.5+ for wavelet transforms

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

## 🔍 Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'mi3_eeg'**
```bash
# Solution: Activate virtual environment
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate  # Linux/Mac

# Verify it's activated (you should see (.venv) in your prompt)
python -c "import mi3_eeg; print('✅ Package found!')"
```

**2. CUDA not detected**
```bash
# Check NVIDIA driver
nvidia-smi

# Verify PyTorch CUDA build
python -c "import torch; print(torch.__version__)"  # Should show +cu124

# Verify CUDA availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Reinstall if needed (remove venv and run setup again)
rm -rf .venv && ./setup_env.sh  # Linux/Mac
Remove-Item -Recurse -Force .venv; .\setup_env.ps1  # Windows
```

**3. Out of memory during TFR analysis**
```bash
# TFR analysis (Pipeline C) is memory-intensive but independent
# Can run in parallel with training (Pipeline A)

# Solution 1: Run TFR while training (different terminals)
# Terminal 1: python -m mi3_eeg.run_all_subjects
# Terminal 2: python -m mi3_eeg.analysis.group_analysis --analysis-type tfr

# Solution 2: Analyze fewer subjects at a time
python -m mi3_eeg.analysis.group_analysis --analysis-type tfr --subjects sub-001 sub-002 sub-003

# Solution 3: Close other applications to free up RAM

# Solution 4: Increase system virtual memory (Windows: pagefile.sys)

# Note: First run computes TFR and caches results. Subsequent runs are fast.
```

**4. Slow TFR computation**
```bash
# This is normal for first run (15-30 minutes for 25 subjects)
# TFR analysis can run in parallel with training to save overall time

# Subsequent runs use joblib cache at ~/.cache/mi3_eeg/tfr/
# Cache location: 
#   Windows: C:\Users\<username>\.cache\mi3_eeg\tfr\
#   Linux/Mac: ~/.cache/mi3_eeg/tfr/

# To clear cache and force recomputation:
rm -rf ~/.cache/mi3_eeg/tfr/  # Linux/Mac
Remove-Item -Recurse -Force $env:USERPROFILE\.cache\mi3_eeg\tfr\  # Windows
```

**5. Missing analysis outputs**
```bash
# Check which pipelines have completed

# Training outputs (Pipeline A):
ls models/                          # Should have sub-001_lenet_*.pth files
ls reports/metrics/                 # Should have sub-001_lenet_results.json files

# Classification analysis (Pipeline B):
ls reports/group_analysis/figures/              # Performance plots
ls reports/group_analysis/statistics/           # Statistical results

# TFR analysis (Pipeline C):
ls reports/group_analysis/tfr_analysis/        # TFR & topomaps

# If missing, run the corresponding pipeline:
python -m mi3_eeg.analysis.group_analysis --analysis-type tfr          # Pipeline C
python -m mi3_eeg.analysis.group_analysis --analysis-type classification # Pipeline B
```

**6. Import errors for analysis modules**
```bash
# Verify analysis submodule is installed
python -c "from mi3_eeg.analysis import run_group_analysis; print('✅ Analysis module found!')"

# If error, reinstall package in development mode
pip install -e .
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
from mi3_eeg.analysis import (
    run_group_analysis,
    plot_group_erd_ers_maps,
    plot_group_topomaps,
    regenerate_from_cache,
)

# Train single subject
data_bundle = load_dataset_from_config()
train_loader, test_loader = prepare_data_loaders(data_bundle)
model = create_model("lenet", device="cuda")
history = train_model(model, train_loader, test_loader)

# Evaluate
results = evaluate_model(model, test_loader)
print(f"Accuracy: {results.overall_accuracy * 100:.2f}%")

# Run group analysis (after training all subjects)
run_group_analysis(model_name="lenet")

# Or regenerate visualizations from cache
regenerate_from_cache()
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
     - `sub-001_eeg200hz.mat` through `sub-025_eeg200hz.mat` (25 subjects total)
   - Optional: Download additional files for testing
   
   **Note:** As of February 2026, all 25 subjects (sub-001 through sub-025) have been processed with results available in `reports/metrics/`.

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
   └── sub-025_eeg200hz.mat
   ```

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

**✅ All 25 subjects processed** (sub-001 through sub-025)

Results are available in:
- `reports/metrics/sub-XXX_lenet_results.json` - Detailed metrics for each subject
- `reports/figures/sub-XXX_lenet_*.png` - Visualizations (confusion matrices, training curves)
- `models/sub-XXX_lenet_*.pth` - Trained model weights

### Training Results (25 Subjects)

#### Subject-Specific Training Performance
- **Mean Overall Accuracy:** 51.47% ± 8.19%
- **Range:** 37.22% - 75.00%
- **Median:** 51.11%
- **Per-Class Performance:**
  - Rest: 58.33% ± 16.95% (best)
  - Hand: 50.42% ± 12.60%
  - Elbow: 45.92% ± 15.41%

#### Statistical Significance
- **One-Way ANOVA:** F(2,72) = 5.83, **p = 0.0045** ✓✓✓
  - Significant difference between classes
  - Effect size: η² = 0.139
- **Pairwise Comparisons (FDR-corrected):**
  - Rest vs Elbow: **p = 0.0005**, d = 0.81 (strong effect) ✓✓✓
  - Rest vs Hand: **p = 0.029**, d = 0.47 (moderate effect) ✓
  - Elbow vs Hand: p = 0.27 (not significant)

**Interpretation:** Rest condition significantly easier to classify than active motor imagery (Elbow/Hand). Hand imagery shows better performance than Elbow, but difference is not statistically significant.

### Time-Frequency Analysis

- **Frequency Range:** 4-40 Hz (37 frequencies)
- **Time Window:** 0-4 seconds (800 timepoints @ 200Hz)
- **ERD/ERS Baseline:** Rest condition
- **Key Findings:**
  - Clear mu rhythm desynchronization (8-13 Hz) during motor imagery
  - Beta band (13-30 Hz) modulation over motor cortex
  - Topographical localization consistent with contralateral motor areas

### Model Performance

| Model              | Accuracy | Rest   | Elbow  | Hand   | Notes              |
|--------------------|----------|--------|--------|--------|---------------------|
| LENet (per-subject)| 51.47%   | 58.33% | 45.92% | 50.42% | 25 subjects        |
| Chance Level       | 33.33%   | 33.33% | 33.33% | 33.33% | Random baseline    |

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
- `mat_filename`: Dataset file pattern
- `sampling_rate`: 200 Hz
- `num_channels`: 62
- `test_size`: 0.2 (20% test split)
- `class_balance_method`: "downsample"

### TrainingConfig
- `epochs`: 50 (default for per-subject)
- `batch_size`: 64
- `learning_rate`: 0.01
- `dropout`: 0.35
- `early_stopping_patience`: 50

### GroupAnalysisConfig
- `tfr_freqs`: (4.0, 40.0, 1.0) Hz
- `sampling_rate`: 160.0 Hz (resampled for TFR)
- `use_cache`: True (joblib caching)
- `baseline_method`: "percent" (ERD/ERS)
- `electrodes_of_interest`: ["C3", "Cz", "C4"] (motor cortex)

## 📚 Module Documentation

### Core Modules

- **`config.py`**: Configuration dataclasses, paths, hyperparameters
- **`dataset.py`**: BIDS data loading, class balancing, DataLoaders
- **`model.py`**: LENet architecture, model factory, serialization
- **`train.py`**: Training loops, early stopping, history tracking
- **`evaluation.py`**: Metrics computation, confusion matrices
- **`visualization.py`**: Training curves, confusion matrix plots
- **`main.py`**: Single-subject training pipeline orchestrator
- **`run_all_subjects.py`**: Batch processing for all subjects

### Analysis Submodule (`mi3_eeg.analysis`)

- **`group_analysis.py`**: Main orchestrator for 4-step analysis pipeline
- **`time_frequency.py`**: Morlet wavelet TFR, ERD/ERS computation, batch processing
- **`topography.py`**: MNE-based brain mapping, 10-20 montage, electrode positioning
- **`statistical_tests.py`**: ANOVA, t-tests, FDR correction, effect sizes
- **`tfr_visualization.py`**: Modular TFR and topomap plotting functions
- **`regenerate_visualizations.py`**: Standalone visualization regeneration from cache
- **`metrics_aggregator.py`**: Cross-subject metrics aggregation
- **`visualization.py`**: Analysis-specific plots

## � Quick Start Examples

### Example 1: Train Single Subject
```bash
# Activate environment
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate  # Linux/Mac

# Train subject 001 (Pipeline A)
python -m mi3_eeg.main --subject-file sub-001_eeg200hz.mat --epochs 50 --device cuda
```

### Example 2: Train All Subjects (Pipeline A)
```bash
# Process all 25 subjects with individual models
python -m mi3_eeg.run_all_subjects
```

### Example 3: Analyze Raw EEG (Pipeline C - Independent!)
```bash
# Analyze raw EEG with TFR/topography - no training needed
# Can run in parallel with Example 2
python -m mi3_eeg.analysis.group_analysis --analysis-type tfr
```

### Example 4: Classification Analysis (Pipeline B - After Training)
```bash
# After training completes, analyze classification results
python -m mi3_eeg.analysis.group_analysis --analysis-type classification

# Or run all analyses together
python -m mi3_eeg.analysis.group_analysis
```

### Example 5: Regenerate TFR Visualizations (Pipeline D)
```bash
# Fast regeneration from cached data (no recomputation)
python -m mi3_eeg.analysis.regenerate_visualizations
```

### Example 6: Parallel Execution (Recommended Workflow)
```bash
# Terminal 1: Start training (Pipeline A - takes 30-60 min)
python -m mi3_eeg.run_all_subjects

# Terminal 2: Meanwhile, analyze raw EEG (Pipeline C - 15-30 min)
# No need to wait for training to finish!
python -m mi3_eeg.analysis.group_analysis --analysis-type tfr

# After both complete: Classification stats (Pipeline B - 2-5 min)
python -m mi3_eeg.analysis.group_analysis --analysis-type classification
```

### Example 7: Python API (Training - Pipeline A)
```python
from mi3_eeg import (
    load_dataset_from_config,
    prepare_data_loaders,
    create_model,
    train_model,
    evaluate_model,
)

# Load subject data
data_bundle = load_dataset_from_config()
train_loader, test_loader = prepare_data_loaders(data_bundle)

# Train and evaluate
model = create_model("lenet", device="cuda")
history = train_model(model, train_loader, test_loader)
results = evaluate_model(model, test_loader)
print(f"Subject Accuracy: {results.overall_accuracy * 100:.2f}%")
```

### Example 8: Python API (EEG Analysis - Pipeline C)
```python
from mi3_eeg.analysis import group_analysis

# Analyze raw EEG - no training required!
group_analysis.run_group_analysis(
    analysis_type="tfr"
)
```

### Example 9: Python API (Classification - Pipeline B)
```python
from mi3_eeg.analysis import group_analysis

# Analyze classification results (after training)
group_analysis.run_group_analysis(
    analysis_type="classification"
)
```

## 🚧 Future Improvements

- [ ] Additional model architectures (RNN, Transformer, Attention)
- [ ] Transfer learning capabilities
- [ ] Hyperparameter optimization with Optuna
- [ ] Cross-subject validation
- [ ] Real-time inference pipeline
- [ ] Web-based demo interface
- [ ] Export to ONNX for deployment
- [ ] Additional time-frequency analysis methods (Hilbert transform, multitaper)
- [ ] Interactive visualization dashboards

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

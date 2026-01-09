#!/bin/bash
# Setup script for MI3 EEG Project (Linux/Mac)
# This script sets up the Python environment with GPU-enabled PyTorch

echo "============================================================"
echo "MI3 EEG Environment Setup"
echo "============================================================"
echo ""

# Check if uv is installed
echo "Checking for uv..."
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found. Installing..."
    pip install uv
else
    echo "✅ uv found"
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
uv venv

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source .venv/bin/activate

# Install PyTorch with CUDA
echo ""
echo "Installing PyTorch with CUDA 12.4..."
uv pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install package with dependencies
echo ""
echo "Installing MI3-EEG package..."
uv pip install -e ".[test]"

# Verify installation
echo ""
echo "============================================================"
echo "Verifying Installation"
echo "============================================================"

# Check package
echo ""
echo "Checking package installation..."
python -c "import mi3_eeg; print(f'✅ mi3_eeg v{mi3_eeg.__version__} installed')"

# Check PyTorch and CUDA
echo ""
echo "Checking PyTorch and CUDA..."
python -c "import torch; print(f'✅ PyTorch {torch.__version__}'); print(f'✅ CUDA available: {torch.cuda.is_available()}'); print(f'✅ GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '⚠️  No GPU detected (will use CPU)')"

echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "Run tests with: pytest"
echo "Train models with: python -m mi3_eeg.main"
echo ""

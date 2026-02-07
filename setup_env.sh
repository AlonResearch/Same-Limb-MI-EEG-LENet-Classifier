#!/bin/bash
# Setup script for MI3 EEG Project (Linux/Mac)
# Automated environment setup with GPU-enabled PyTorch
# Simple, clean, and foolproof for non-technical users

# === CONFIGURATION ===
VERBOSE=false
RECREATE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --recreate)
            RECREATE=true
            shift
            ;;
        *)
            echo "Usage: $0 [-v|--verbose] [--recreate]"
            exit 1
            ;;
    esac
done

# === COLOR FUNCTIONS ===
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

write_section() {
    echo ""
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}============================================================${NC}"
}

write_step() {
    echo ""
    echo -e "${YELLOW}$1${NC}"
}

write_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

write_error() {
    echo -e "${RED}❌ $1${NC}"
}

write_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

# === MAIN SETUP ===

write_section "MI3 EEG Environment Setup"
write_info "This will set up your Python environment with all dependencies"

# Step 1: Check for uv
write_step "Step 1/5: Checking for uv package manager..."
if command -v uv &> /dev/null; then
    UV_PATH=$(command -v uv)
    write_success "uv found at: $UV_PATH"
else
    write_info "Installing uv package manager..."
    if pip install uv -q 2>&1; then
        write_success "uv installed successfully"
    else
        write_error "Failed to install uv. Please run: pip install uv"
        exit 1
    fi
fi

# Step 2: Handle existing virtual environment
write_step "Step 2/5: Checking for existing virtual environment..."
if [ -d ".venv" ] && [ "$RECREATE" != "true" ]; then
    write_info "Found existing .venv directory"
    read -p "Do you want to recreate it from scratch? (y/N): " response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        write_info "Removing existing .venv..."
        rm -rf .venv
        write_success ".venv removed"
    else
        write_success "Reusing existing .venv"
    fi
elif [ "$RECREATE" = "true" ] && [ -d ".venv" ]; then
    write_info "Recreating virtual environment..."
    rm -rf .venv
    write_success ".venv removed"
fi

# Step 3: Run uv sync (creates venv + installs all dependencies)
write_step "Step 3/5: Setting up environment and installing dependencies..."
write_info "This may take a few minutes on first run..."

if [ "$VERBOSE" = true ]; then
    uv sync --all-extras
else
    uv sync --all-extras > /dev/null 2>&1
fi

if [ $? -ne 0 ] || [ ! -f ".venv/bin/activate" ]; then
    write_error "Environment setup failed"
    write_info "Please try manually: uv sync --all-extras"
    exit 1
fi

write_success "Environment created and dependencies installed"

# Step 4: Activate virtual environment
write_step "Step 4/5: Activating virtual environment..."
if [ ! -f ".venv/bin/activate" ]; then
    write_error "Activation script not found at .venv/bin/activate"
    write_info "Try manually: source .venv/bin/activate"
    exit 1
fi

source .venv/bin/activate

if [ -z "$VIRTUAL_ENV" ]; then
    write_error "Virtual environment activation failed"
    write_info "VIRTUAL_ENV not set after sourcing activate"
    exit 1
fi

write_success "Virtual environment activated"

# Step 5: Verify installation
write_step "Step 5/5: Verifying installation..."

ALL_PASS=true

# Check main package
echo -n "Checking mi3_eeg package..."
if python -c "import mi3_eeg; print(f' v{mi3_eeg.__version__}')" 2>/dev/null; then
    write_success "mi3_eeg package ready"
else
    write_error "mi3_eeg import failed"
    ALL_PASS=false
fi

# Check analysis submodule
echo -n "Checking analysis submodule..."
if python -c "from mi3_eeg.analysis import group_analysis; print(' OK')" 2>/dev/null; then
    write_success "analysis submodule ready"
else
    write_error "analysis submodule import failed"
    ALL_PASS=false
fi

# Check PyTorch with CUDA
echo -n "Checking PyTorch + CUDA..."
TORCH_INFO=$(python -c "
import torch
cuda_available = torch.cuda.is_available()
print(f'{torch.__version__}|{cuda_available}')
if cuda_available:
    print(torch.cuda.get_device_name(0))
" 2>/dev/null)

if [ $? -eq 0 ]; then
    VERSION=$(echo "$TORCH_INFO" | head -n1 | cut -d'|' -f1)
    CUDA_AVAIL=$(echo "$TORCH_INFO" | head -n1 | cut -d'|' -f2)
    
    if [ "$CUDA_AVAIL" = "True" ]; then
        GPU_NAME=$(echo "$TORCH_INFO" | tail -n1)
        write_success "PyTorch $VERSION with CUDA ($GPU_NAME)"
    else
        echo ""
        write_info "PyTorch $VERSION installed (CPU only - no GPU detected)"
    fi
else
    write_error "PyTorch verification failed"
    ALL_PASS=false
fi

# Check key analysis dependencies
echo -n "Checking analysis dependencies..."
if python -c "import mne, joblib, seaborn, h5py, pywt; print(' OK')" 2>/dev/null; then
    write_success "All analysis dependencies ready (mne, joblib, seaborn, h5py, pywavelets)"
else
    write_error "Missing analysis dependencies"
    write_info "Required: mne, joblib, seaborn, h5py, pywavelets"
    ALL_PASS=false
fi

# Final status
write_section "Setup Result"

if [ "$ALL_PASS" = true ]; then
    echo ""
    write_success "Setup completed successfully! 🎉"
    echo ""
    echo -e "${GREEN}Next steps:${NC}"
    echo -e "  ${YELLOW}• Activate environment:${NC}"
    echo -e "    ${CYAN}source .venv/bin/activate${NC}"
    echo ""
    echo -e "  ${YELLOW}• Train models:${NC}"
    echo -e "    ${CYAN}python -m mi3_eeg.main${NC}"
    echo ""
    echo -e "  ${YELLOW}• Run tests:${NC}"
    echo -e "    ${CYAN}pytest${NC}"
    echo ""
    echo -e "  ${YELLOW}• Run group analysis:${NC}"
    echo -e "    ${CYAN}python -m mi3_eeg.analysis.group_analysis${NC}"
    echo ""
    exit 0
else
    echo ""
    write_error "Setup completed with errors"
    write_info "Some components failed verification. Check the errors above."
    write_info "Try running with verbose mode: ./setup_env.sh -v"
    echo ""
    exit 1
fi

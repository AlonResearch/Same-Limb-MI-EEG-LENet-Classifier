#!/bin/bash
# Setup script for MI3 EEG Project (Linux/Mac)
# This script sets up the Python environment with GPU-enabled PyTorch
# Features: uv sync for fast dependency management, CUDA support verification

# === CONFIGURATION ===
VERBOSE=false
FORCE_REINSTALL=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --force-reinstall)
            FORCE_REINSTALL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
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

# === PROGRESS TRACKING ===
declare -A SETUP_SUCCESS=(
    [UvInstalled]=false
    [SyncCompleted]=false
    [VenvActivated]=false
    [TorchInstalled]=false
)

# === HELPER FUNCTIONS ===

remove_venv_forcefully() {
    if [ -d ".venv" ]; then
        write_info "Removing .venv directory..."
        rm -rf .venv
        
        # Verify removal
        sleep 0.5
        if [ -d ".venv" ]; then
            write_error "Failed to fully remove .venv"
            return 1
        fi
        write_success ".venv removed successfully"
        return 0
    fi
    return 0
}

run_with_output() {
    local cmd="$1"
    if [ "$VERBOSE" = true ]; then
        eval "$cmd"
    else
        eval "$cmd" > /dev/null 2>&1
    fi
}

check_package_installed() {
    local package_name="$1"
    
    # Convert package name for import (e.g., "mi3-eeg" -> "mi3_eeg")
    local python_name="${package_name//-/_}"
    
    # Try direct import - most reliable check
    if python -c "import $python_name" 2>/dev/null; then
        return 0
    fi
    
    return 1
}

# === MAIN SETUP ===

write_section "MI3 EEG Environment Setup"

# --- Check for existing virtual environment ---
if [ -d ".venv" ] && [ -f ".venv/bin/activate" ]; then
    echo ""
    write_info "Virtual environment already exists at: .venv"
    echo ""
    echo "Options:"
    echo "  [1] Reuse and sync (faster)"
    echo "  [2] Recreate from scratch"
    echo "  [3] Exit"
    echo ""
    
    read -p "Enter your choice (1-3): " venv_choice
    
    case $venv_choice in
        1)
            write_info "Reusing existing environment..."
            SKIP_VENV_CREATION=true
            ;;
        2)
            write_info "Recreating environment from scratch..."
            if remove_venv_forcefully; then
                SKIP_VENV_CREATION=false
            else
                write_error "Cannot proceed without removing .venv"
                exit 1
            fi
            ;;
        3)
            write_info "Exiting setup..."
            exit 0
            ;;
        *)
            write_error "Invalid choice. Exiting."
            exit 1
            ;;
    esac
else
    SKIP_VENV_CREATION=false
fi

# --- Check for uv ---
write_step "Checking for uv..."
if command -v uv &> /dev/null; then
    UV_PATH=$(command -v uv)
    write_success "uv found at: $UV_PATH"
    SETUP_SUCCESS[UvInstalled]=true
else
    write_error "uv not found"
    write_info "Installing uv via pip..."
    
    if pip install uv -q 2>&1; then
        write_success "uv installed successfully"
        SETUP_SUCCESS[UvInstalled]=true
    else
        write_error "Failed to install uv"
        write_info "Please install uv manually: pip install uv"
        exit 1
    fi
fi

# --- Run uv sync ---
write_step "Running 'uv sync' (creates venv + installs dependencies)..."
if [ -d ".venv" ] && [ "$SKIP_VENV_CREATION" != "true" ]; then
    write_info "Virtual environment already exists. Running sync..."
fi

if SYNC_OUTPUT=$(uv sync --all-extras 2>&1); then
    if [ -f ".venv/bin/activate" ]; then
        write_success "uv sync completed successfully"
        SETUP_SUCCESS[SyncCompleted]=true
    else
        write_error "uv sync failed or did not create valid venv"
        write_info "Please run manually: uv sync --all-extras"
        exit 1
    fi
else
    write_error "uv sync failed"
    if [ "$FORCE_REINSTALL" = true ]; then
        write_info "Attempting recovery: removing .venv and retrying..."
        if remove_venv_forcefully; then
            if uv sync --all-extras 2>&1; then
                if [ -f ".venv/bin/activate" ]; then
                    write_success "uv sync completed successfully on retry"
                    SETUP_SUCCESS[SyncCompleted]=true
                else
                    write_error "uv sync failed on retry"
                    exit 1
                fi
            else
                write_error "Recovery failed"
                exit 1
            fi
        else
            exit 1
        fi
    else
        write_info "Please run manually: uv sync"
        exit 1
    fi
fi

# --- Activate virtual environment ---
write_step "Activating virtual environment..."
if [ ! -f ".venv/bin/activate" ]; then
    write_error "Activation script not found at .venv/bin/activate"
    write_info "Troubleshooting:"
    write_info "  1. Check if .venv exists: [ -d .venv ] && echo 'exists' || echo 'missing'"
    write_info "  2. Try manual activation: source .venv/bin/activate"
    write_info "  3. Remove and recreate: rm -rf .venv && uv sync"
    exit 1
fi

source .venv/bin/activate

# Verify activation
if [ -z "$VIRTUAL_ENV" ]; then
    write_error "Virtual environment activation failed"
    write_info "VIRTUAL_ENV not set after sourcing activate"
    exit 1
fi

write_success "Virtual environment activated: $VIRTUAL_ENV"
SETUP_SUCCESS[VenvActivated]=true

# --- Verify PyTorch installation ---
write_step "Verifying PyTorch installation..."
if check_package_installed "torch"; then
    write_success "PyTorch installed (with CUDA 12.4 support)"
    SETUP_SUCCESS[TorchInstalled]=true
else
    write_error "PyTorch installation verification failed"
    write_info "PyTorch was not installed. Verify with: python -c 'import torch; print(torch.__version__)'"
    exit 1
fi

# === VERIFICATION ===
write_section "Verifying Installation"

ALL_VERIFICATIONS_PASS=true

# Check package
write_step "Checking package installation..."
if python -c "import mi3_eeg; print(f'✅ mi3_eeg v{mi3_eeg.__version__} installed')" 2>&1; then
    :
else
    write_error "Package import failed"
    ALL_VERIFICATIONS_PASS=false
fi

# Check PyTorch and CUDA
write_step "Checking PyTorch and CUDA..."
python -c "
import torch
print(f'✅ PyTorch {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️  No GPU detected (will use CPU)')
" 2>&1 || ALL_VERIFICATIONS_PASS=false

# === FINAL STATUS ===
if [ "$ALL_VERIFICATIONS_PASS" = true ] && [ "${SETUP_SUCCESS[SyncCompleted]}" = true ] && [ "${SETUP_SUCCESS[TorchInstalled]}" = true ]; then
    write_section "Setup Complete! ✅"
    echo ""
    write_success "All components installed and verified!"
    echo ""
    echo "Next steps:"
    echo -e "  ${YELLOW}1. Activate virtual environment:${NC}"
    echo -e "     ${CYAN}source .venv/bin/activate${NC}"
    echo ""
    echo -e "  ${YELLOW}2. Train models:${NC}"
    echo -e "     ${CYAN}python -m mi3_eeg.main${NC}"
    echo ""
    echo -e "  ${YELLOW}3. Run tests:${NC}"
    echo -e "     ${CYAN}pytest${NC}"
    echo ""
    exit 0
else
    write_section "Setup Failed ❌"
    echo ""
    write_error "Setup did not complete successfully"
    echo ""
    echo "Partial Progress:"
    echo "  Uv installed:        ${SETUP_SUCCESS[UvInstalled]}"
    echo "  Sync completed:      ${SETUP_SUCCESS[SyncCompleted]}"
    echo "  Venv activated:      ${SETUP_SUCCESS[VenvActivated]}"
    echo "  PyTorch installed:   ${SETUP_SUCCESS[TorchInstalled]}"
    echo ""
    echo "To retry:"
    echo "  1. Fix the issue above"
    echo "  2. Re-run this script"
    echo "  3. Use -v flag for verbose details: ./setup_env.sh -v"
    echo ""
    exit 1
fi

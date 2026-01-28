#!/bin/bash
# Setup script for MI3 EEG Project (Linux/Mac)
# This script sets up the Python environment with GPU-enabled PyTorch
# Features: Error handling, validation, recovery options, verbose logging

# === CONFIGURATION ===
VERBOSE=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
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
    [VenvCreated]=false
    [VenvActivated]=false
    [TorchInstalled]=false
    [PackageInstalled]=false
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

# === MAIN SETUP ===

write_section "MI3 EEG Environment Setup"

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

# --- Create virtual environment with automatic fallback ---
write_step "Creating virtual environment..."
VENV_CREATION_SUCCESS=false

# Try uv venv first
write_info "Attempting 'uv venv'..."
if VENV_OUTPUT=$(uv venv 2>&1); then
    if [ -f ".venv/bin/activate" ]; then
        write_success "Virtual environment created successfully"
        SETUP_SUCCESS[VenvCreated]=true
        VENV_CREATION_SUCCESS=true
    else
        write_error "uv venv: Activate script not found"
        VENV_OUTPUT="venv created but bin/activate not found"
    fi
else
    write_error "uv venv failed"
fi

# Fallback to uv sync if uv venv failed
if [ "$VENV_CREATION_SUCCESS" = false ]; then
    write_error "Falling back to 'uv sync'..."
    
    if remove_venv_forcefully; then
        write_info "Running 'uv sync'..."
        if SYNC_OUTPUT=$(uv sync 2>&1); then
            if [ -f ".venv/bin/activate" ]; then
                write_success "uv sync completed successfully"
                SETUP_SUCCESS[VenvCreated]=true
                VENV_CREATION_SUCCESS=true
            else
                write_error "uv sync failed or created invalid venv"
                if [ -n "$SYNC_OUTPUT" ]; then
                    echo -e "${RED}Error details: $SYNC_OUTPUT${NC}"
                fi
                write_info "Please run manually: uv sync"
                exit 1
            fi
        else
            write_error "uv sync failed"
            if [ -n "$SYNC_OUTPUT" ]; then
                echo -e "${RED}Error details: $SYNC_OUTPUT${NC}"
            fi
            exit 1
        fi
    else
        write_error "Could not remove .venv for fallback"
        write_info "Please manually run: rm -rf .venv && uv sync"
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
    write_info "  3. Remove and recreate: rm -rf .venv && uv venv"
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

# --- Install PyTorch with CUDA ---
write_step "Installing PyTorch with CUDA 12.4..."
if run_with_output "uv pip install torch --index-url https://download.pytorch.org/whl/cu124 2>&1"; then
    write_success "PyTorch installed successfully"
    SETUP_SUCCESS[TorchInstalled]=true
else
    write_error "PyTorch installation failed"
    write_info "Partial progress saved: PyTorch installation failed, but venv is intact."
    write_info "You can retry with: uv pip install torch --index-url https://download.pytorch.org/whl/cu124"
    exit 1
fi

# --- Install MI3-EEG package ---
write_step "Installing MI3-EEG package with dependencies..."
if run_with_output "uv pip install -e '.[test]' 2>&1"; then
    write_success "MI3-EEG package installed successfully"
    SETUP_SUCCESS[PackageInstalled]=true
else
    write_error "Package installation failed"
    write_info "Partial progress saved: PyTorch is installed, but package installation failed."
    write_info "You can retry with: uv pip install -e '.[test]'"
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
if [ "$ALL_VERIFICATIONS_PASS" = true ] && [ "${SETUP_SUCCESS[PackageInstalled]}" = true ]; then
    write_section "Setup Complete! ✅"
    echo ""
    write_success "All components installed and verified!"
    echo ""
    echo "Next steps:"
    echo -e "  ${CYAN}Train models:${NC}    python -m mi3_eeg.main"
    echo -e "  ${CYAN}Run tests:${NC}       pytest"
    echo ""
    exit 0
else
    write_section "Setup Failed ❌"
    echo ""
    write_error "Setup did not complete successfully"
    echo ""
    echo "Partial Progress:"
    echo "  Uv installed:        ${SETUP_SUCCESS[UvInstalled]}"
    echo "  Venv created:        ${SETUP_SUCCESS[VenvCreated]}"
    echo "  Venv activated:      ${SETUP_SUCCESS[VenvActivated]}"
    echo "  PyTorch installed:   ${SETUP_SUCCESS[TorchInstalled]}"
    echo "  Package installed:   ${SETUP_SUCCESS[PackageInstalled]}"
    echo ""
    echo "To retry:"
    echo "  1. Fix the issue above"
    echo "  2. Re-run this script"
    echo "  3. Use -v flag for verbose details: ./setup_env.sh -v"
    echo ""
    exit 1
fi

# Setup script for MI3 EEG Project (Windows PowerShell)
# Automated environment setup with GPU-enabled PyTorch
# Simple, clean, and foolproof for non-technical users

# === CONFIGURATION ===
param(
    [switch]$Verbose = $false,
    [switch]$Recreate = $false
)

$ErrorActionPreference = "Continue"

# === HELPER FUNCTIONS ===

function Write-Section {
    param([string]$Title)
    Write-Host "`n============================================================" -ForegroundColor Cyan
    Write-Host $Title -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
}

function Write-Step {
    param([string]$Message)
    Write-Host "`n$Message" -ForegroundColor Yellow
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ️  $Message" -ForegroundColor Cyan
}

# === MAIN SETUP ===

Write-Section "MI3 EEG Environment Setup"
Write-Info "This will set up your Python environment with all dependencies"

# Step 1: Check for uv
Write-Step "Step 1/5: Checking for uv package manager..."
try {
    $uvPath = Get-Command uv -ErrorAction Stop
    Write-Success "uv found at: $($uvPath.Source)"
} catch {
    Write-Info "Installing uv package manager..."
    try {
        pip install uv -q
        Write-Success "uv installed successfully"
    } catch {
        Write-Error-Custom "Failed to install uv. Please run: pip install uv"
        exit 1
    }
}

# Step 2: Handle existing virtual environment
Write-Step "Step 2/5: Checking for existing virtual environment..."
if ((Test-Path .\.venv) -and -not $Recreate) {
    Write-Info "Found existing .venv directory"
    $response = Read-Host "Do you want to recreate it from scratch? (y/N)"
    if ($response -match '^[Yy]') {
        Write-Info "Removing existing .venv..."
        Remove-Item -Path .\.venv -Recurse -Force
        Write-Success ".venv removed"
    } else {
        Write-Success "Reusing existing .venv"
    }
} elseif ($Recreate -and (Test-Path .\.venv)) {
    Write-Info "Recreating virtual environment..."
    Remove-Item -Path .\.venv -Recurse -Force
    Write-Success ".venv removed"
}

# Step 3: Run uv sync (creates venv + installs all dependencies)
Write-Step "Step 3/5: Setting up environment and installing dependencies..."
Write-Info "This may take a few minutes on first run..."

try {
    if ($Verbose) {
        & uv sync --all-extras
    } else {
        & uv sync --all-extras 2>&1 | Out-Null
    }
    
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path .\.venv\Scripts\Activate.ps1)) {
        throw "uv sync failed or created invalid environment"
    }
    
    Write-Success "Environment created and dependencies installed"
} catch {
    Write-Error-Custom "Environment setup failed: $_"
    Write-Info "Please try manually: uv sync --all-extras"
    exit 1
}

# Step 4: Activate virtual environment
Write-Step "Step 4/5: Activating virtual environment..."
try {
    & .\.venv\Scripts\Activate.ps1
    
    if (-not $env:VIRTUAL_ENV) {
        throw "VIRTUAL_ENV not set after activation"
    }
    
    Write-Success "Virtual environment activated"
} catch {
    Write-Error-Custom "Activation failed: $_"
    Write-Info "Try manually: .\.venv\Scripts\Activate.ps1"
    exit 1
}

# Step 5: Verify installation
Write-Step "Step 5/5: Verifying installation..."

$allPass = $true

# Check main package
Write-Host "`nChecking mi3_eeg package..." -NoNewline
try {
    python -c "import mi3_eeg; print(f' v{mi3_eeg.__version__}')" -ErrorAction Stop
    Write-Success "mi3_eeg package ready"
} catch {
    Write-Error-Custom "mi3_eeg import failed"
    $allPass = $false
}

# Check analysis submodule
Write-Host "Checking analysis submodule..." -NoNewline
try {
    python -c "from mi3_eeg.analysis import group_analysis; print(' OK')" -ErrorAction Stop
    Write-Success "analysis submodule ready"
} catch {
    Write-Error-Custom "analysis submodule import failed"
    $allPass = $false
}

# Check PyTorch with CUDA
Write-Host "Checking PyTorch + CUDA..." -NoNewline
try {
    $torchInfo = python -c @"
import torch
cuda_available = torch.cuda.is_available()
print(f'{torch.__version__}|{cuda_available}')
if cuda_available:
    print(torch.cuda.get_device_name(0))
"@ -ErrorAction Stop
    
    $parts = $torchInfo -split '\|'
    $version = $parts[0]
    $cudaAvail = $parts[1]
    
    if ($cudaAvail -eq "True") {
        $gpuName = $parts[2]
        Write-Success "PyTorch $version with CUDA ($gpuName)"
    } else {
        Write-Host ""
        Write-Info "PyTorch $version installed (CPU only - no GPU detected)"
    }
} catch {
    Write-Error-Custom "PyTorch verification failed"
    $allPass = $false
}

# Check key analysis dependencies
Write-Host "Checking analysis dependencies..." -NoNewline
try {
    python -c @"
import mne, joblib, seaborn, h5py, pywt
print(' OK')
"@ -ErrorAction Stop
    Write-Success "All analysis dependencies ready (mne, joblib, seaborn, h5py, pywavelets)"
} catch {
    Write-Error-Custom "Missing analysis dependencies"
    Write-Info "Required: mne, joblib, seaborn, h5py, pywavelets"
    $allPass = $false
}

# Final status
Write-Section "Setup Result"

if ($allPass) {
    Write-Host ""
    Write-Success "Setup completed successfully! 🎉"
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Green
    Write-Host "  • Activate environment:" -ForegroundColor Yellow
    Write-Host "    .\.venv\Scripts\Activate.ps1" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  • Train models:" -ForegroundColor Yellow
    Write-Host "    python -m mi3_eeg.main" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  • Run tests:" -ForegroundColor Yellow
    Write-Host "    pytest" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  • Run group analysis:" -ForegroundColor Yellow
    Write-Host "    python -m mi3_eeg.analysis.group_analysis" -ForegroundColor Cyan
    Write-Host ""
    exit 0
} else {
    Write-Host ""
    Write-Error-Custom "Setup completed with errors"
    Write-Info "Some components failed verification. Check the errors above."
    Write-Info "Try running with verbose mode: .\setup_env.ps1 -Verbose"
    Write-Host ""
    exit 1
}

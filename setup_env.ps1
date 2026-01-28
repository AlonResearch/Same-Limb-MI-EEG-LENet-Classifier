# Setup script for MI3 EEG Project (Windows PowerShell)
# This script sets up the Python environment with GPU-enabled PyTorch
# Features: Error handling, validation, recovery options, exit codes

# === CONFIGURATION ===
param(
    [switch]$Verbose = $false
)

# Set error handling policy - Continue on error for better recovery
$ErrorActionPreference = "Continue"
$VerbosePreference = if ($Verbose) { "Continue" } else { "SilentlyContinue" }

# Track setup success state
$script:SetupSuccess = $false
$script:PartialSuccess = @{
    UvInstalled = $false
    VenvCreated = $false
    VenvActivated = $false
    TorchInstalled = $false
    PackageInstalled = $false
}

# === HELPER FUNCTIONS ===

function Write-Section {
    param([string]$Title)
    Write-Host "`n" -NoNewline
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host ("=" * 59) -ForegroundColor Cyan
    Write-Host $Title -ForegroundColor Cyan
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host ("=" * 59) -ForegroundColor Cyan
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

function Show-VenvRecoveryMenu {
    Write-Host "`n" -ForegroundColor Yellow
    Write-Host "Virtual environment creation failed." -ForegroundColor Red
    Write-Host "Possible causes:" -ForegroundColor Yellow
    Write-Host "  • File permissions (antivirus/security software blocking access)" -ForegroundColor Yellow
    Write-Host "  • Corrupted .venv directory from previous failed attempt" -ForegroundColor Yellow
    Write-Host "  • Insufficient disk space" -ForegroundColor Yellow
    Write-Host "`nRecovery options:" -ForegroundColor Cyan
    Write-Host "  [1] Remove .venv and retry" -ForegroundColor Cyan
    Write-Host "  [2] Skip venv creation and retry (use existing)" -ForegroundColor Cyan
    Write-Host "  [3] Exit and fix manually" -ForegroundColor Cyan
    Write-Host "  [4] Fallback to uv sync" -ForegroundColor Cyan
    Write-Host ""
    
    $choice = Read-Host "Enter your choice (1-4)"
    return $choice
}function Remove-VenvForcefully {
    if (Test-Path .\.venv) {
        Write-Host "Removing .venv directory..." -ForegroundColor Yellow
        try {
            Remove-Item -Path .\.venv -Recurse -Force -ErrorAction Continue
            # Verify removal
            Start-Sleep -Milliseconds 500
            if (Test-Path .\.venv) {
                Write-Error-Custom "Failed to fully remove .venv"
                return $false
            }
            Write-Success ".venv removed successfully"
            return $true
        } catch {
            Write-Error-Custom "Error during removal: $_"
            return $false
        }
    }
    return $true
}

# === MAIN SETUP ===

Write-Section "MI3 EEG Environment Setup"

# --- Check for uv ---
Write-Step "Checking for uv..."
try {
    $uvPath = Get-Command uv -ErrorAction Stop
    Write-Success "uv found at: $($uvPath.Source)"
    $script:PartialSuccess.UvInstalled = $true
} catch {
    Write-Error-Custom "uv not found"
    Write-Host "Installing uv via pip..." -ForegroundColor Yellow
    try {
        pip install uv -q
        Write-Success "uv installed successfully"
        $script:PartialSuccess.UvInstalled = $true
    } catch {
        Write-Error-Custom "Failed to install uv: $_"
        Write-Host "Please install uv manually: pip install uv" -ForegroundColor Yellow
        exit 1
    }
}

# --- Create virtual environment with automatic fallback ---
Write-Step "Creating virtual environment..."
$venvCreationSuccess = $false

# Try uv venv first
try {
    Write-Host "Attempting 'uv venv'..." -ForegroundColor Yellow
    $output = & uv venv 2>&1
    
    if (Test-Path .\.venv\Scripts\Activate.ps1) {
        Write-Success "Virtual environment created successfully"
        $script:PartialSuccess.VenvCreated = $true
        $venvCreationSuccess = $true
    } else {
        throw "venv created but Activate.ps1 not found"
    }
} catch {
    Write-Error-Custom "uv venv failed: $_"
    if ($output) {
        Write-Host "Error details: $output" -ForegroundColor Red
    }
    
    # Automatic fallback to uv sync
    Write-Host "`nFalling back to 'uv sync'..." -ForegroundColor Yellow
    try {
        if (Remove-VenvForcefully) {
            Write-Host "Running 'uv sync'..." -ForegroundColor Yellow
            $syncOutput = & uv sync 2>&1
            
            if ($LASTEXITCODE -eq 0 -and (Test-Path .\.venv\Scripts\Activate.ps1)) {
                Write-Success "uv sync completed successfully"
                $script:PartialSuccess.VenvCreated = $true
                $venvCreationSuccess = $true
            } else {
                Write-Error-Custom "uv sync failed or created invalid venv"
                if ($syncOutput) {
                    Write-Host "Error details: $syncOutput" -ForegroundColor Red
                }
                Write-Host "Please run manually: uv sync" -ForegroundColor Yellow
                exit 1
            }
        } else {
            Write-Error-Custom "Could not remove .venv for fallback"
            Write-Host "Please manually run: Remove-Item .\.venv -Recurse -Force; uv sync" -ForegroundColor Yellow
            exit 1
        }
    } catch {
        Write-Error-Custom "Fallback failed: $_"
        exit 1
    }
}

# --- Activate virtual environment ---
Write-Step "Activating virtual environment..."
try {
    $activateScript = ".\.venv\Scripts\Activate.ps1"
    if (-not (Test-Path $activateScript)) {
        throw "Activation script not found at $activateScript"
    }
    
    & $activateScript
    
    # Verify activation
    if ($env:VIRTUAL_ENV) {
        Write-Success "Virtual environment activated: $env:VIRTUAL_ENV"
        $script:PartialSuccess.VenvActivated = $true
    } else {
        throw "Virtual environment activation did not set VIRTUAL_ENV"
    }
} catch {
    Write-Error-Custom "Virtual environment activation failed: $_"
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Check if .venv exists: Test-Path .\.venv" -ForegroundColor Yellow
    Write-Host "  2. Try manual activation: .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host "  3. Remove and recreate: Remove-Item .\.venv -Recurse" -ForegroundColor Yellow
    exit 1
}

# --- Install PyTorch with CUDA ---
Write-Step "Installing PyTorch with CUDA 12.4..."
try {
    uv pip install torch --index-url https://download.pytorch.org/whl/cu124 2>&1 | Write-Verbose
    
    # Verify PyTorch installation
    python -c "import torch; print(f'Installed: torch {torch.__version__}')" 2>&1 | Write-Verbose
    Write-Success "PyTorch installed successfully"
    $script:PartialSuccess.TorchInstalled = $true
} catch {
    Write-Error-Custom "PyTorch installation failed: $_"
    Write-Host "Partial progress saved: PyTorch installation failed, but venv is intact." -ForegroundColor Yellow
    Write-Host "You can retry with: uv pip install torch --index-url https://download.pytorch.org/whl/cu124" -ForegroundColor Yellow
    exit 1
}

# --- Install MI3-EEG package ---
Write-Step "Installing MI3-EEG package with dependencies..."
try {
    uv pip install -e ".[test]" 2>&1 | Write-Verbose
    
    # Verify package installation
    python -c "import mi3_eeg; print(f'Installed: mi3_eeg {mi3_eeg.__version__}')" 2>&1 | Write-Verbose
    Write-Success "MI3-EEG package installed successfully"
    $script:PartialSuccess.PackageInstalled = $true
} catch {
    Write-Error-Custom "Package installation failed: $_"
    Write-Host "Partial progress saved: PyTorch is installed, but package installation failed." -ForegroundColor Yellow
    Write-Host "You can retry with: uv pip install -e '.[test]'" -ForegroundColor Yellow
    exit 1
}

# === VERIFICATION ===
Write-Section "Verifying Installation"

$allVerificationsPass = $true

# Check package
Write-Step "Checking package installation..."
try {
    python -c "import mi3_eeg; print(f'✅ mi3_eeg v{mi3_eeg.__version__} installed')"
} catch {
    Write-Error-Custom "Package import failed: $_"
    $allVerificationsPass = $false
}

# Check PyTorch and CUDA
Write-Step "Checking PyTorch and CUDA..."
try {
    python -c @"
import torch
print(f'✅ PyTorch {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️  No GPU detected (will use CPU)')
"@
} catch {
    Write-Error-Custom "PyTorch verification failed: $_"
    $allVerificationsPass = $false
}

# === FINAL STATUS ===
if ($allVerificationsPass -and $script:PartialSuccess.PackageInstalled) {
    $script:SetupSuccess = $true
    Write-Section "Setup Complete! ✅"
    Write-Host "`nNext steps:" -ForegroundColor Green
    Write-Host "  Train models:    python -m mi3_eeg.main" -ForegroundColor Cyan
    Write-Host "  Run tests:       pytest" -ForegroundColor Cyan
    Write-Host ""
    exit 0
} else {
    Write-Section "Setup Failed ❌"
    Write-Host "`nPartial Progress:" -ForegroundColor Yellow
    Write-Host "  Uv installed:        $($script:PartialSuccess.UvInstalled)" -ForegroundColor Yellow
    Write-Host "  Venv created:        $($script:PartialSuccess.VenvCreated)" -ForegroundColor Yellow
    Write-Host "  Venv activated:      $($script:PartialSuccess.VenvActivated)" -ForegroundColor Yellow
    Write-Host "  PyTorch installed:   $($script:PartialSuccess.TorchInstalled)" -ForegroundColor Yellow
    Write-Host "  Package installed:   $($script:PartialSuccess.PackageInstalled)" -ForegroundColor Yellow
    Write-Host "`nTo retry:" -ForegroundColor Cyan
    Write-Host "  1. Fix the issue above" -ForegroundColor Cyan
    Write-Host "  2. Re-run this script" -ForegroundColor Cyan
    Write-Host "  3. Use --Verbose flag for details: . .\setup_env.ps1 -Verbose" -ForegroundColor Cyan
    Write-Host ""
    exit 1
}

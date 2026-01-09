# Setup script for MI3 EEG Project (Windows PowerShell)
# This script sets up the Python environment with GPU-enabled PyTorch

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "MI3 EEG Environment Setup" -ForegroundColor Cyan
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host ""

# Check if uv is installed
Write-Host "Checking for uv..." -ForegroundColor Yellow
if (!(Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "❌ uv not found. Installing..." -ForegroundColor Red
    pip install uv
} else {
    Write-Host "✅ uv found" -ForegroundColor Green
}

# Create virtual environment
Write-Host "`nCreating virtual environment..." -ForegroundColor Yellow
uv venv

# Activate virtual environment
Write-Host "`nActivating virtual environment..." -ForegroundColor Yellow
. .\.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA
Write-Host "`nInstalling PyTorch with CUDA 12.4..." -ForegroundColor Yellow
uv pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install package with dependencies
Write-Host "`nInstalling MI3-EEG package..." -ForegroundColor Yellow
uv pip install -e ".[test]"

# Verify installation
Write-Host "`n" -NoNewline
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "Verifying Installation" -ForegroundColor Cyan
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan

# Check package
Write-Host "`nChecking package installation..." -ForegroundColor Yellow
python -c "import mi3_eeg; print(f'✅ mi3_eeg v{mi3_eeg.__version__} installed')"

# Check PyTorch and CUDA
Write-Host "`nChecking PyTorch and CUDA..." -ForegroundColor Yellow
python -c @"
import torch
print(f'✅ PyTorch {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️  No GPU detected (will use CPU)')
"@

Write-Host "`n" -NoNewline
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "`nRun tests with: pytest" -ForegroundColor Cyan
Write-Host "Train models with: python -m mi3_eeg.main" -ForegroundColor Cyan
Write-Host ""

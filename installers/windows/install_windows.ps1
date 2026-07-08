$ErrorActionPreference = "Stop"

if (Get-Variable PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

Write-Host "Installing Neuron Segmentation Assistant dependencies..."
Write-Host ""

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$ProjectRoot = $ProjectRoot.Path

$InferenceDir = Join-Path $ProjectRoot "yolo-inference"
$VenvDir = Join-Path $ProjectRoot "venv"
$ConfigDir = Join-Path $env:USERPROFILE ".neuron-segmentation-assistant"
$ConfigFile = Join-Path $ConfigDir "config.properties"

$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$InferScript = Join-Path $InferenceDir "infer_one.py"
$RetrainScript = Join-Path $InferenceDir "retrain_model.py"
$CompareScript = Join-Path $InferenceDir "compare_models.py"

$RequirementsFile = Join-Path $InferenceDir "requirements.txt"
$CudaRequirementsFile = Join-Path $InferenceDir "requirements-windows-cuda.txt"
$LegacyCudaRequirementsFile = Join-Path $InferenceDir "requirements-windows-cuda-legacy.txt"

Write-Host "Project root: $ProjectRoot"
Write-Host "Inference directory: $InferenceDir"
Write-Host ""

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Python is not installed or is not available in PATH."
    Write-Host "Please install Python 3.10+ and check 'Add Python to PATH' during installation."
    pause
    exit 1
}

if (-not (Test-Path $InferenceDir)) {
    Write-Host "ERROR: Inference directory not found:"
    Write-Host $InferenceDir
    pause
    exit 1
}

if (-not (Test-Path $RequirementsFile)) {
    Write-Host "ERROR: requirements.txt not found:"
    Write-Host $RequirementsFile
    pause
    exit 1
}

Write-Host "Detected GPU information:"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    try {
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    }
    catch {
        Write-Host "NVIDIA GPU detected, but nvidia-smi could not be queried."
    }
}
else {
    Write-Host "nvidia-smi not found. NVIDIA CUDA may not be available."
}
Write-Host ""

Write-Host "Select PyTorch installation mode:"
Write-Host "  0 - CPU only. Safest option."
Write-Host "  1 - NVIDIA CUDA modern. Recommended for newer NVIDIA GPUs."
Write-Host "  2 - NVIDIA CUDA legacy. Recommended for older NVIDIA GPUs such as MX250."
Write-Host ""

$TorchChoice = Read-Host "Choose installation mode [0/1/2]"

if ($TorchChoice -ne "0" -and $TorchChoice -ne "1" -and $TorchChoice -ne "2") {
    Write-Host "Invalid option. Using CPU-only mode."
    $TorchChoice = "0"
}

Write-Host ""
Write-Host "Virtual environment setup..."

$ResetVenv = Read-Host "Remove existing virtual environment and recreate it? [y/N]"

if ($ResetVenv -match "^[Yy]$") {
    if (Test-Path $VenvDir) {
        Write-Host "Removing existing virtual environment:"
        Write-Host $VenvDir

        try {
            Remove-Item -Recurse -Force $VenvDir -ErrorAction Stop
            Write-Host "Existing virtual environment removed."
        }
        catch {
            Write-Host ""
            Write-Host "ERROR: The virtual environment could not be removed."
            Write-Host "Close Fiji, VS Code, Python terminals or any process using this venv and try again."
            Write-Host $_.Exception.Message
            pause
            exit 1
        }
    }
    else {
        Write-Host "No existing virtual environment found. A new one will be created."
    }
}

if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating Python virtual environment..."
    python -m venv $VenvDir
}
else {
    Write-Host "Virtual environment already exists. It will be reused:"
    Write-Host $VenvDir
}

if (-not (Test-Path $PythonExe)) {
    Write-Host ""
    Write-Host "ERROR: The virtual environment exists but python.exe was not found:"
    Write-Host $PythonExe
    Write-Host "The venv may be corrupted. Run the installer again and choose to remove the existing venv."
    pause
    exit 1
}

Write-Host ""
Write-Host "Upgrading pip..."
& $PythonExe -m pip install --upgrade pip

Write-Host ""
Write-Host "Removing previous PyTorch installation if present..."

$OldErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"

try {
    foreach ($Package in @("torch", "torchvision", "torchaudio")) {
        & $PythonExe -m pip show $Package *> $null

        if ($LASTEXITCODE -eq 0) {
            Write-Host "Uninstalling $Package..."
            & $PythonExe -m pip uninstall -y $Package
        }
        else {
            Write-Host "$Package is not installed. Skipping."
        }
    }
}
finally {
    $ErrorActionPreference = $OldErrorActionPreference
}

Write-Host ""
Write-Host "Installing PyTorch..."

if ($TorchChoice -eq "1") {
    if (-not (Test-Path $CudaRequirementsFile)) {
        Write-Host "ERROR: CUDA requirements file not found:"
        Write-Host $CudaRequirementsFile
        pause
        exit 1
    }

    Write-Host "Installing PyTorch with CUDA modern support..."
    & $PythonExe -m pip install -r $CudaRequirementsFile
}
elseif ($TorchChoice -eq "2") {
    if (-not (Test-Path $LegacyCudaRequirementsFile)) {
        Write-Host "ERROR: Legacy CUDA requirements file not found:"
        Write-Host $LegacyCudaRequirementsFile
        pause
        exit 1
    }

    Write-Host "Installing PyTorch with CUDA legacy support..."
    & $PythonExe -m pip install -r $LegacyCudaRequirementsFile
}
else {
    Write-Host "Installing PyTorch CPU-only..."
    & $PythonExe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
}

Write-Host ""
Write-Host "Installing remaining Python dependencies..."
& $PythonExe -m pip install -r $RequirementsFile

Write-Host ""
Write-Host "Checking PyTorch installation..."

$TorchCheckScript = Join-Path $env:TEMP "check_neuron_torch_cuda.py"

$TorchCheckCode = @'
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    try:
        x = torch.zeros(1, device="cuda")
        print("CUDA tensor test: OK")
    except Exception as e:
        print("CUDA tensor test: FAILED")
        print(str(e))
        raise
else:
    print("GPU: CPU only")
'@

$Utf8NoBom = New-Object System.Text.UTF8Encoding -ArgumentList $false
[System.IO.File]::WriteAllText($TorchCheckScript, $TorchCheckCode, $Utf8NoBom)

& $PythonExe $TorchCheckScript

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "WARNING: PyTorch was installed, but the CUDA check failed."
    Write-Host "The plugin may still work using CPU mode."
}
Write-Host ""
Write-Host "Creating configuration file..."

New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null

$PythonConfig = $PythonExe.Replace("\", "/")
$InferConfig = $InferScript.Replace("\", "/")
$RetrainConfig = $RetrainScript.Replace("\", "/")
$CompareConfig = $CompareScript.Replace("\", "/")

$ConfigContent = @"
python=$PythonConfig
script=$InferConfig
retrain_script=$RetrainConfig
compare_script=$CompareConfig
"@

$Utf8NoBom = New-Object System.Text.UTF8Encoding -ArgumentList $false
[System.IO.File]::WriteAllText($ConfigFile, $ConfigContent, $Utf8NoBom)

Write-Host ""
Write-Host "Installation completed successfully."
Write-Host "Configuration saved to:"
Write-Host $ConfigFile
Write-Host ""
Write-Host "You can now open Fiji and run:"
Write-Host "Plugins > Neuron Segmentation > Single Image Segmentation"
Write-Host "Plugins > Neuron Segmentation > Transfer Learning Assistant"
Write-Host "Plugins > Neuron Segmentation > Model Comparison / External Validation"
Write-Host ""

pause
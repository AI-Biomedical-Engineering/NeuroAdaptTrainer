$ErrorActionPreference = "Stop"

Write-Host "Installing Neuron Segmentation Assistant dependencies..."

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
$RequirementsFile = Join-Path $InferenceDir "requirements.txt"

Write-Host "Project root: $ProjectRoot"
Write-Host "Inference directory: $InferenceDir"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Python is not installed or is not available in PATH."
    Write-Host "Please install Python 3.10+ and check 'Add Python to PATH' during installation."
    pause
    exit 1
}

if (-not (Test-Path $RequirementsFile)) {
    Write-Host "ERROR: requirements.txt not found:"
    Write-Host $RequirementsFile
    pause
    exit 1
}

Write-Host "Creating Python virtual environment..."

python -m venv $VenvDir

Write-Host "Upgrading pip..."

& $PythonExe -m pip install --upgrade pip

Write-Host "Installing Python dependencies..."

& $PythonExe -m pip install -r $RequirementsFile

Write-Host "Creating configuration file..."

New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null

$PythonConfig = $PythonExe.Replace("\", "/")
$InferConfig = $InferScript.Replace("\", "/")
$RetrainConfig = $RetrainScript.Replace("\", "/")

$ConfigContent = @"
python=$PythonConfig
script=$InferConfig
retrain_script=$RetrainConfig
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
Write-Host "or:"
Write-Host "Plugins > Neuron Segmentation > Transfer Learning Assistant"
Write-Host ""

pause
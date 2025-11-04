# PowerShell one-click runner for STT_KR web (Windows)
# - Creates venv in .venv
# - Installs requirements
# - Sets external model caches to %LOCALAPPDATA%\stt_kr_cache
# - Runs Django dev server on 127.0.0.1:8001

$ErrorActionPreference = "Stop"

function Ensure-Dir($path) {
    if (-not (Test-Path -LiteralPath $path)) {
        New-Item -ItemType Directory -Path $path | Out-Null
    }
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
$WebRoot = $PSScriptRoot
$VenvPath = Join-Path $WebRoot ".venv"
$PythonExe = Join-Path $VenvPath "Scripts/python.exe"
$CacheRoot = Join-Path $env:LOCALAPPDATA "stt_kr_cache"

# External caches (HuggingFace/Torch)
$env:HF_HOME = $CacheRoot
$env:HUGGINGFACE_HUB_CACHE = $CacheRoot
$env:TORCH_HOME = $CacheRoot
Ensure-Dir $CacheRoot

# Create venv if missing
if (-not (Test-Path -LiteralPath $PythonExe)) {
    Write-Host "[setup] Creating virtual environment at $VenvPath" -ForegroundColor Cyan
    try {
        & py -3 -m venv $VenvPath
    } catch {
        Write-Warning "Python launcher 'py' not found. Trying 'python'..."
        & python -m venv $VenvPath
    }
}

# Install dependencies
Write-Host "[setup] Upgrading pip and installing requirements" -ForegroundColor Cyan
& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install -r (Join-Path $WebRoot "requirements.txt")

# Run migrations (safe no-op if none)
$ManageDir = Join-Path $WebRoot "STT_KR"
Push-Location $ManageDir
try {
    Write-Host "[migrate] Applying migrations" -ForegroundColor Cyan
    & $PythonExe manage.py migrate

    # Start server
    Write-Host "[run] Starting dev server at http://127.0.0.1:8001/" -ForegroundColor Green
    & $PythonExe manage.py runserver 127.0.0.1:8001
} finally {
    Pop-Location
}

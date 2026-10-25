# Run: powershell -File scripts/setup.ps1 (or pwsh -File scripts/setup.ps1).
# Installs dependencies only; does not start services or alter execution policy.
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable $($Arguments -join ' ')"
    }
}

foreach ($tool in @('ffmpeg', 'ffprobe', 'node', 'npm')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "$tool is required on PATH. See docs/SETUP.md."
    }
}
if (-not (Test-Path '.venv/Scripts/python.exe')) {
    $pythonLauncher = if (Get-Command py -ErrorAction SilentlyContinue) { 'py' } elseif (Get-Command python -ErrorAction SilentlyContinue) { 'python' } else { throw 'Install Python 3.11 or 3.12.' }
    Invoke-Checked -Executable $pythonLauncher -Arguments @('-c', 'import sys; assert (3, 11) <= sys.version_info < (3, 13), "Python 3.11 or 3.12 is required"')
    Invoke-Checked -Executable $pythonLauncher -Arguments @('-m', 'venv', '.venv')
}
$voxguardPython = (Resolve-Path '.venv/Scripts/python.exe').Path
Invoke-Checked -Executable $voxguardPython -Arguments @('-c', 'import sys; assert (3, 11) <= sys.version_info < (3, 13), "Python 3.11 or 3.12 is required"')
Invoke-Checked -Executable $voxguardPython -Arguments @('-m', 'pip', 'install', 'torch==2.6.0', '--index-url', 'https://download.pytorch.org/whl/cpu')
Invoke-Checked -Executable $voxguardPython -Arguments @('-m', 'pip', 'install', '-r', 'backend/requirements.txt', '-r', 'ml/requirements.txt', 'pytest', 'httpx')
Invoke-Checked -Executable $voxguardPython -Arguments @('-m', 'pip', 'check')
Invoke-Checked -Executable $voxguardPython -Arguments @('-m', 'ml.download_model')
Push-Location frontend
try { Invoke-Checked -Executable 'npm' -Arguments @('ci') } finally { Pop-Location }
Write-Host 'Setup complete. See docs/SETUP.md to start the backend and frontend.'

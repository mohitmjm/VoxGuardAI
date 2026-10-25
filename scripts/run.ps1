# Run the backend from any working directory. Frontend runs in a second terminal.
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$voxguardPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (-not (Test-Path $voxguardPython)) { throw 'Run scripts/setup.ps1 first.' }
& $voxguardPython scripts/preflight.py
if ($LASTEXITCODE -ne 0) { throw 'Preflight failed. Resolve its message before starting.' }
& $voxguardPython -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
if ($LASTEXITCODE -ne 0) { throw 'Backend stopped with an error.' }

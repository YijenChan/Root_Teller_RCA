param(
    [int]$Port = 4315,
    [string]$Workspace = ""
)

$ErrorActionPreference = "Stop"
$SystemRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $SystemRoot
if (-not $Workspace) { $Workspace = $RepoRoot }
$PythonCandidates = @(
    (Join-Path $RepoRoot ".venv\Scripts\python.exe"),
    (Join-Path $Workspace ".venv\Scripts\python.exe")
)
$Python = $PythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $Python) { $Python = (Get-Command python.exe -ErrorAction SilentlyContinue).Source }
if (-not $Python) { throw "Python 3.12 runtime not found. See system/README.md for setup instructions." }

$env:ROOTTELLER_WORKSPACE = $Workspace
$env:PYTHONPATH = "$(Join-Path $RepoRoot 'src')"

Write-Host "Root-Teller: http://127.0.0.1:$Port" -ForegroundColor Cyan
& $Python -m uvicorn root_teller_system.app:app --app-dir $SystemRoot --host 127.0.0.1 --port $Port

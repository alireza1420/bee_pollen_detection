param(
    [string]$Python = "C:\Python312\python.exe"
)

$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param(
        [scriptblock]$Command
    )

    & $Command
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

if (-not (Test-Path $Python)) {
    throw "Python 3.12 was not found at '$Python'. Install Python 3.12 or pass -Python <path>."
}

$TempDir = Join-Path (Get-Location) ".tmp"
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null
$env:TEMP = $TempDir
$env:TMP = $TempDir

if (-not (Test-Path ".venv")) {
    Invoke-Checked { & $Python -m venv --system-site-packages .venv }
} else {
    $VenvConfig = ".venv\pyvenv.cfg"
    if (Test-Path $VenvConfig) {
        $Config = Get-Content $VenvConfig
        $Config = $Config -replace "include-system-site-packages = false", "include-system-site-packages = true"
        Set-Content -Path $VenvConfig -Value $Config
    }
}

$VenvPython = Join-Path (Get-Location) ".venv\Scripts\python.exe"

Invoke-Checked { & $VenvPython -m pip install --upgrade pip setuptools wheel }
Invoke-Checked { & $VenvPython -m pip install --no-cache-dir --force-reinstall --no-deps -r requirements-local.txt }
Invoke-Checked { & $VenvPython verify_env.py }

Write-Host ""
Write-Host "Environment ready."
Write-Host "Activate it with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"

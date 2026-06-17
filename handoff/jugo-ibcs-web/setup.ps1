# One-time setup for Windows PowerShell
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$pyExe = $null
foreach ($ver in @("3.12", "3.11", "3.10")) {
    try {
        & py "-$ver" -c "import sys; assert (3,10) <= sys.version_info[:2] <= (3,12)"
        $pyExe = @("py", "-$ver")
        break
    } catch { }
}

if (-not $pyExe) {
    foreach ($name in @("python3.12", "python3.11", "python3.10", "python")) {
        if (Get-Command $name -ErrorAction SilentlyContinue) {
            try {
                & $name -c "import sys; assert (3,10) <= sys.version_info[:2] <= (3,12)"
                $pyExe = @($name)
                break
            } catch { }
        }
    }
}

if (-not $pyExe) {
    Write-Error "Python 3.10-3.12 is required. Install from https://www.python.org/downloads/"
}

Write-Host "Using $($pyExe -join ' ')"
& @pyExe --version

if (-not (Test-Path ".venv")) {
    & @pyExe -m venv .venv
}

& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

if (-not (Test-Path ".env")) {
    Copy-Item .env.example .env
}

Write-Host ""
Write-Host "Setup complete. Next steps:"
Write-Host "  1. Edit .env and set ROBOFLOW_API_KEY"
Write-Host "  2. Run: .\start.ps1"
Write-Host "  3. Open: http://127.0.0.1:8000"

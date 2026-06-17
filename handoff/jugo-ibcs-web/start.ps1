# Start the web app (Windows PowerShell)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    Write-Error "No .venv found. Run setup.bat or setup.ps1 first."
}

& .\.venv\Scripts\Activate.ps1

if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        $line = $_.Trim()
        if ($line -eq "" -or $line.StartsWith("#")) { return }
        $parts = $line -split "=", 2
        if ($parts.Count -eq 2) {
            $name = $parts[0].Trim()
            $value = $parts[1].Trim()
            Set-Item -Path "env:$name" -Value $value
        }
    }
}

if (-not $env:ROBOFLOW_API_KEY) {
    Write-Warning "ROBOFLOW_API_KEY is not set. Edit .env and add your key."
}

Write-Host "Starting Jugo IBCS Analysis at http://127.0.0.1:8000"
Write-Host "Press Ctrl+C to stop the server."
Write-Host ""
uvicorn web.server:app --host 127.0.0.1 --port 8000

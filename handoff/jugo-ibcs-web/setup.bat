@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "PYEXE="
for %%V in (3.12 3.11 3.10) do (
  if not defined PYEXE (
    py -%%V -c "import sys; assert (3,10) <= sys.version_info[:2] <= (3,12)" 2>nul && set "PYEXE=py -%%V"
  )
)
if not defined PYEXE (
  for %%P in (python3.12 python3.11 python3.10 python) do (
    if not defined PYEXE (
      %%P -c "import sys; assert (3,10) <= sys.version_info[:2] <= (3,12)" 2>nul && set "PYEXE=%%P"
    )
  )
)

if not defined PYEXE (
  echo Python 3.10-3.12 is required.
  echo Install from https://www.python.org/downloads/ and check "Add python.exe to PATH".
  pause
  exit /b 1
)

echo Using !PYEXE!
!PYEXE! --version

if not exist .venv (
  !PYEXE! -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

if not exist .env copy .env.example .env

echo.
echo Setup complete. Next steps:
echo   1. Edit .env and set ROBOFLOW_API_KEY
echo   2. Double-click start.bat  (or run start.ps1 in PowerShell)
echo   3. Open http://127.0.0.1:8000
echo.
pause

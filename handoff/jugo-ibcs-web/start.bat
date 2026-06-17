@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

if not exist .venv (
  echo No .venv found. Run setup.bat first.
  pause
  exit /b 1
)

call .venv\Scripts\activate.bat

if exist .env (
  for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
    if not "%%~A"=="" set "%%~A=%%~B"
  )
)

if "%ROBOFLOW_API_KEY%"=="" (
  echo Warning: ROBOFLOW_API_KEY is not set. Edit .env and add your key.
)

echo Starting Jugo IBCS Analysis at http://127.0.0.1:8000
echo Press Ctrl+C to stop the server.
echo.
uvicorn web.server:app --host 127.0.0.1 --port 8000

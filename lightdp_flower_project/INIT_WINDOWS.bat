@echo off
setlocal
cd /d "%~dp0"

if "%PYTHON%"=="" set "PYTHON=py -3.11"
%PYTHON% -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] < (3,13) else 1)"
if errorlevel 1 (
  echo Python 3.10, 3.11, or 3.12 is required. Set PYTHON to your Python launcher.
  exit /b 1
)

if not exist .venv\Scripts\python.exe (
  %PYTHON% -m venv .venv
  if errorlevel 1 exit /b 1
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1
python -m pip install -e .
if errorlevel 1 exit /b 1

echo.
echo Setup complete. Activate with: .venv\Scripts\activate
echo Then start the smoke run with: flwr run . --stream
endlocal

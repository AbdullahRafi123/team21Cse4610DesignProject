@echo off
setlocal
cd /d "%~dp0" || exit /b 1
set "PROJECT_ROOT=%CD%"

if not defined PYTHON (
  for %%V in (3.12 3.11 3.13 3.10) do (
    py -%%V -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] < (3,14) else 1)" >nul 2>&1
    if not errorlevel 1 if not defined PYTHON set "PYTHON=py -%%V"
  )
)
%PYTHON% -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] < (3,14) else 1)" >nul 2>&1
if errorlevel 1 (
  echo Python 3.10, 3.11, 3.12, or 3.13 is required.
  echo Set PYTHON to an installed interpreter, for example: set "PYTHON=py -3.13"
  exit /b 1
)

rem Keep the virtual environment outside the checkout to avoid Ray path issues.
if not defined VENV_DIR set "VENV_DIR=%USERPROFILE%\.venvs\lightdp-flower-project"
for %%I in ("%VENV_DIR%") do if not exist "%%~dpI" mkdir "%%~dpI"
if errorlevel 1 (
  echo Could not create the virtual environment directory: "%VENV_DIR%"
  exit /b 1
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
  %PYTHON% -m venv "%VENV_DIR%"
  if errorlevel 1 exit /b 1
)

"%VENV_DIR%\Scripts\python.exe" -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] < (3,14) else 1)" >nul 2>&1
if errorlevel 1 (
  echo Existing virtual environment uses an unsupported Python version.
  echo Remove "%VENV_DIR%" or set VENV_DIR to a new path, then rerun initialize.bat.
  exit /b 1
)

"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
"%VENV_DIR%\Scripts\python.exe" -m pip install -e "%PROJECT_ROOT%"
if errorlevel 1 exit /b 1

echo.
echo Setup complete. Activate in Command Prompt with:
echo call "%VENV_DIR%\Scripts\activate.bat"
echo Activate in PowerShell with:
echo ^& "%VENV_DIR%\Scripts\Activate.ps1"
echo Then, from the repository directory, run:
echo flwr run . --stream
endlocal

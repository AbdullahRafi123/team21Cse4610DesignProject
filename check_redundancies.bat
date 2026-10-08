@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0" || exit /b 1

if defined VENV_DIR if exist "%VENV_DIR%\Scripts\python.exe" (
  "%VENV_DIR%\Scripts\python.exe" scripts\check_redundancies.py %*
  exit /b !errorlevel!
)
if exist "%USERPROFILE%\.venvs\lightdp-flower-project\Scripts\python.exe" (
  "%USERPROFILE%\.venvs\lightdp-flower-project\Scripts\python.exe" scripts\check_redundancies.py %*
  exit /b !errorlevel!
)
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" scripts\check_redundancies.py %*
  exit /b !errorlevel!
)
where py >nul 2>&1
if not errorlevel 1 (
  py -3 scripts\check_redundancies.py %*
  exit /b !errorlevel!
)
where python >nul 2>&1
if not errorlevel 1 (
  python scripts\check_redundancies.py %*
  exit /b !errorlevel!
)

echo Python was not found. Run initialize.bat first or install Python 3.
exit /b 1

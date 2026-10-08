@echo off
setlocal
cd /d "%~dp0.." || exit /b 1
if not defined VENV_DIR set "VENV_DIR=%USERPROFILE%\.venvs\lightdp-flower-project"

if not exist "%VENV_DIR%\Scripts\flwr.exe" (
  call setup\initialize.bat
  if errorlevel 1 goto :failed
)

"%VENV_DIR%\Scripts\flwr.exe" run . --stream
set "RUN_EXIT=%errorlevel%"
echo.
if "%RUN_EXIT%"=="0" (
  echo Smoke run finished successfully.
) else (
  echo Smoke run failed with exit code %RUN_EXIT%.
)
pause
exit /b %RUN_EXIT%

:failed
set "RUN_EXIT=%errorlevel%"
echo Setup failed with exit code %RUN_EXIT%.
pause
exit /b %RUN_EXIT%

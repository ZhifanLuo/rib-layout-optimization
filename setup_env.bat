@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_LAUNCHER=py"
where py >nul 2>nul
if errorlevel 1 set "PYTHON_LAUNCHER=python"

echo Creating the local Python environment in:
echo     %~dp0.venv
%PYTHON_LAUNCHER% -m venv "%~dp0.venv"
if errorlevel 1 goto :failed

echo Installing required packages...
"%~dp0.venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed
"%~dp0.venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto :failed

echo.
echo Environment setup completed successfully.
pause
exit /b 0

:failed
echo.
echo [ERROR] Environment setup failed.
pause
exit /b 1

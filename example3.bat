@echo off
setlocal
cd /d "%~dp0"
set "PYTHONDONTWRITEBYTECODE=1"

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Project Python was not found:
    echo         %PYTHON_EXE%
    echo Please double-click setup_env.bat first.
    pause
    exit /b 1
)

echo Running rib-layout optimization Example 3...
echo Results will be written to: %~dp0results\example_3
echo.
"%PYTHON_EXE%" example3.py
set "RUN_STATUS=%ERRORLEVEL%"

echo.
if not "%RUN_STATUS%"=="0" (
    echo [ERROR] Example 3 failed with exit code %RUN_STATUS%.
) else (
    echo Example 3 completed successfully.
)
pause
exit /b %RUN_STATUS%

@echo off
setlocal
cd /d "%~dp0"
set "PYTHONDONTWRITEBYTECODE=1"

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Project Python was not found.
    echo Please double-click setup_env.bat first.
    pause
    exit /b 1
)

echo Running all rib-layout optimization examples in order: 1, 2, 3, 4.
echo Results will be written to: %~dp0results
echo.

for %%N in (1 2 3 4) do (
    echo ============================================================
    echo Starting Example %%N...
    echo ============================================================
    "%PYTHON_EXE%" "example%%N.py"
    if errorlevel 1 goto :failed
    echo.
    echo Example %%N completed successfully.
    echo.
)

echo ============================================================
echo All four examples completed successfully.
echo ============================================================
pause
exit /b 0

:failed
set "RUN_STATUS=%ERRORLEVEL%"
echo.
echo [ERROR] The current example failed with exit code %RUN_STATUS%.
echo Remaining examples were not started.
pause
exit /b %RUN_STATUS%

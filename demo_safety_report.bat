@echo off
echo Safety Report Generator Demo
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Run the demo
echo Running Safety Report Generator demo...
echo.
python demo_safety_report.py

echo.
echo Demo completed. Check the 'reports' directory for generated files.
pause

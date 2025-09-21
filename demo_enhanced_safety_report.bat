@echo off
echo Enhanced Safety Report Generator Demo
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Run the enhanced demo
echo Running Enhanced Safety Report Generator demo...
echo.
python demo_enhanced_safety_report.py

echo.
echo Enhanced demo completed. Check the 'reports' directory for improved files.
pause

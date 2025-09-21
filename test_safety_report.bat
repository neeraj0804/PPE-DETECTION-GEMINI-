@echo off
echo Testing Safety Report Generator...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Install required packages if not already installed
echo Installing required packages...
pip install google-generativeai reportlab pandas seaborn openpyxl

REM Run the test
echo.
echo Running Safety Report Generator test...
python test_safety_report.py

echo.
echo Test completed. Check the 'reports' directory for generated files.
pause

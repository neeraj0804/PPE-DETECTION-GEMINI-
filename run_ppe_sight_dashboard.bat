@echo off
echo Starting PPE Sight Dashboard...
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
pip install plotly

REM Run the PPE Sight Dashboard
echo.
echo Launching PPE Sight Dashboard...
echo The dashboard will open in your default web browser.
echo.
echo Press Ctrl+C to stop the application.
echo.

streamlit run ppe_sight_app.py --server.port 8501 --server.address localhost

pause

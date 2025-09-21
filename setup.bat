@echo off
echo ===== PPE Detection System Setup =====
echo.
echo This script will set up the environment and install all required dependencies.
echo.

REM Check if Python is installed
python --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.8 or higher from https://www.python.org/downloads/
    echo.
    pause
    exit /b
)

echo Python is installed. Installing required packages...
echo.

REM Install PyTorch with CUDA support
echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

REM Install other dependencies
echo.
echo Installing other dependencies...
pip install opencv-python numpy ultralytics pillow matplotlib streamlit optree PyQt5

REM Create required directories
echo.
echo Creating required directories...
mkdir models 2>nul
mkdir output 2>nul
mkdir recordings 2>nul
mkdir processed_videos 2>nul

echo.
echo ===== Setup Complete =====
echo.
echo Next steps:
echo 1. Make sure your dataset is downloaded to: C:\Users\np080\Downloads\PPE-DETECTION.v26i.yolov11
echo 2. Run run_training.bat to train the model
echo 3. Run run_testing.bat to test the model
echo 4. Run run_app.bat, run_web_app.bat, or run_desktop_app.bat to use the application
echo.
echo If you encounter any issues, please refer to CUDA_TROUBLESHOOTING.md
echo.
pause 
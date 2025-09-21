@echo off
echo ===================================
echo Safety Model Testing Script
echo ===================================

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python 3.8 or higher.
    goto :eof
)

:: Check if required files exist
if not exist "safety_data.yaml" (
    echo safety_data.yaml file not found. This file is required for testing.
    goto :eof
)

:: Check if model exists
if not exist "models\safety\best_safety_model.pt" (
    echo Model not found at models\safety\best_safety_model.pt
    echo Did you train the model first?
    
    :: Ask if they want to use yolov8n.pt instead
    set /p USE_DEFAULT=Do you want to use yolov8n.pt for testing instead? (Y/N): 
    
    if /i "%USE_DEFAULT%" neq "Y" (
        echo Testing cancelled.
        goto :eof
    )
    
    set MODEL=yolov8n.pt
) else (
    set MODEL=models\safety\best_safety_model.pt
)

:: Check if external dataset exists
if not exist "C:\Users\np080\Downloads\safety.v1i.yolov8" (
    echo Safety dataset not found at C:\Users\np080\Downloads\safety.v1i.yolov8
    echo Please ensure the dataset is downloaded and available at this location.
    goto :eof
)

:: Set parameters
set CONF=0.25
set IOU=0.45
set IMG_SIZE=640

:: Choose test source
echo ===================================
echo Please select a test source:
echo ===================================
echo 1. Test dataset folder
echo 2. Custom image/video file
echo 3. Custom folder
echo ===================================
set /p SOURCE_CHOICE=Enter your choice (1-3): 

if "%SOURCE_CHOICE%"=="1" (
    set SOURCE=C:\Users\np080\Downloads\safety.v1i.yolov8\test\images
    echo Using test dataset: %SOURCE%
) else if "%SOURCE_CHOICE%"=="2" (
    set /p SOURCE=Enter path to image or video file: 
    if not exist "%SOURCE%" (
        echo File not found: %SOURCE%
        goto :eof
    )
) else if "%SOURCE_CHOICE%"=="3" (
    set /p SOURCE=Enter path to folder containing images or videos: 
    if not exist "%SOURCE%" (
        echo Folder not found: %SOURCE%
        goto :eof
    )
) else (
    echo Invalid choice. Exiting.
    goto :eof
)

:: Set save options
set /p SAVE_TXT=Do you want to save detection results as text files? (Y/N): 
if /i "%SAVE_TXT%"=="Y" (
    set SAVE_TXT_FLAG=--save-txt
) else (
    set SAVE_TXT_FLAG=
)

set /p SAVE_CONF=Do you want to save confidence scores in text files? (Y/N): 
if /i "%SAVE_CONF%"=="Y" (
    set SAVE_CONF_FLAG=--save-conf
) else (
    set SAVE_CONF_FLAG=
)

:: Create output directory
set OUTPUT_DIR=output\safety_results_%date:~-4,4%%date:~-7,2%%date:~-10,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set OUTPUT_DIR=%OUTPUT_DIR: =0%
mkdir "%OUTPUT_DIR%" 2>nul

echo ===================================
echo Testing Configuration:
echo ===================================
echo - Model: %MODEL%
echo - Source: %SOURCE%
echo - Confidence Threshold: %CONF%
echo - IoU Threshold: %IOU%
echo - Image Size: %IMG_SIZE%
echo - Output Directory: %OUTPUT_DIR%
echo - Save Text Results: %SAVE_TXT%
echo - Save Confidence: %SAVE_CONF%
echo ===================================

:: Ask for confirmation
set /p CONFIRM=Do you want to start testing with these settings? (Y/N): 

if /i "%CONFIRM%" neq "Y" (
    echo Testing cancelled.
    goto :eof
)

:: Run testing
echo Starting testing...
python test_safety_model.py --model %MODEL% --data safety_data.yaml --img-size %IMG_SIZE% --conf %CONF% --iou %IOU% --source "%SOURCE%" --save-dir "%OUTPUT_DIR%" %SAVE_TXT_FLAG% %SAVE_CONF_FLAG%

if %errorlevel% neq 0 (
    echo Testing failed with error code %errorlevel%.
    goto :eof
)

echo ===================================
echo Testing completed!
echo ===================================
echo Results saved to: %OUTPUT_DIR%
echo ===================================

:: Open the output directory
explorer "%OUTPUT_DIR%"

pause 
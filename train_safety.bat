@echo off
echo ===================================
echo Safety Model Training Script
echo ===================================

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python 3.8 or higher.
    goto :eof
)

:: Check if required files exist
if not exist "safety_data.yaml" (
    echo safety_data.yaml file not found. This file is required for training.
    goto :eof
)

if not exist "yolov8n.pt" (
    echo Downloading YOLOv8n base model...
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
)

:: Check if external dataset exists
if not exist "C:\Users\np080\Downloads\safety.v1i.yolov8" (
    echo Safety dataset not found at C:\Users\np080\Downloads\safety.v1i.yolov8
    echo Please ensure the dataset is downloaded and available at this location.
    goto :eof
)

echo Checking dataset structure...
if not exist "C:\Users\np080\Downloads\safety.v1i.yolov8\train\images" (
    echo Training image directory not found at expected location.
    goto :eof
)

if not exist "C:\Users\np080\Downloads\safety.v1i.yolov8\valid\images" (
    echo Validation image directory not found at expected location.
    goto :eof
)

echo Dataset structure looks good!

:: Create models/safety directory if it doesn't exist
if not exist "models\safety" (
    mkdir "models\safety"
)

:: Set parameters
set EPOCHS=100
set BATCH_SIZE=16
set IMG_SIZE=640
set MODEL=yolov8n.pt
set PATIENCE=30
set OPTIMIZER=Adam
set LR=0.001

echo ===================================
echo Training Configuration:
echo ===================================
echo - Epochs: %EPOCHS%
echo - Batch Size: %BATCH_SIZE%
echo - Image Size: %IMG_SIZE%
echo - Base Model: %MODEL%
echo - Early Stopping Patience: %PATIENCE%
echo - Optimizer: %OPTIMIZER%
echo - Learning Rate: %LR%
echo ===================================

:: Ask for confirmation
set /p CONFIRM=Do you want to start training with these settings? (Y/N): 

if /i "%CONFIRM%" neq "Y" (
    echo Training cancelled.
    goto :eof
)

:: Run training
echo Starting training...
python train_safety_model.py --data safety_data.yaml --epochs %EPOCHS% --batch %BATCH_SIZE% --img %IMG_SIZE% --weights %MODEL% --patience %PATIENCE% --optimizer %OPTIMIZER% --lr0 %LR% --augment

if %errorlevel% neq 0 (
    echo Training failed with error code %errorlevel%.
    goto :eof
)

echo ===================================
echo Training completed!
echo ===================================
echo You can test the model with: python test_safety_model.py --data safety_data.yaml
echo ===================================

pause 
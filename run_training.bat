@echo off
echo Starting PPE Detection Model Training with NVIDIA GTX 1650...
echo *** ENHANCED HIGH-ACCURACY CONFIGURATION (Target: 90%+ mAP) ***
echo.

REM Check for data parameter
set DATA_FILE=data.yaml
set DATA_ARG=

:parse_args
if "%~1"=="" goto continue
if /i "%~1"=="--data" (
    set DATA_FILE=%~2
    set DATA_ARG=--data %~2
    shift
    shift
    goto parse_args
) else (
    shift
    goto parse_args
)

:continue
echo Using dataset configuration: %DATA_FILE%
echo.

set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

REM Check if CUDA is available
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo.
echo Starting training with high-accuracy parameters...
echo This training uses:
echo  - YOLOv8s model (more powerful than YOLOv8n)
echo  - 150 epochs (allows model to fully converge)
echo  - 640x640 resolution (higher detail for PPE detection)
echo  - Strong data augmentation (prevents overfitting)
echo  - Cosine learning rate schedule (better optimization)
echo.

REM Primary high-accuracy training approach
python train_ppe_model.py --data %DATA_FILE% --epochs 150 --batch 8 --img 640 --weights yolov8s.pt --optimizer AdamW --lr0 0.001 --lrf 0.01 --weight-decay 0.0005 --augment --device 0 --save-period 10 --cos-lr --patience 50 --exist-ok

REM If the above command fails, try with smaller batch size but keep the image resolution
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo First attempt failed. Trying with smaller batch size but maintaining high resolution...
    python train_ppe_model.py --data %DATA_FILE% --epochs 150 --batch 4 --img 640 --weights yolov8s.pt --optimizer AdamW --lr0 0.001 --lrf 0.01 --weight-decay 0.0005 --augment --device 0 --save-period 10 --cos-lr --patience 50 --exist-ok
)

REM If still fails, try YOLOv8n but with higher epochs and other optimizations
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Second attempt failed. Trying with YOLOv8n but with enhanced training parameters...
    python train_ppe_model.py --data %DATA_FILE% --epochs 200 --batch 8 --img 640 --weights yolov8n.pt --optimizer AdamW --lr0 0.001 --lrf 0.01 --weight-decay 0.0005 --augment --device 0 --save-period 10 --cos-lr --patience 50 --exist-ok
)

REM Copy best model to models directory
echo.
echo Training completed. Copying best model to models/best_ppe_model.pt
copy /Y "runs\train\exp*\weights\best.pt" "models\best_ppe_model.pt"

echo.
echo Training completed. Check the runs/train directory for results.
echo.
echo If you encountered memory errors, please run fix_memory.bat for instructions
echo on how to increase your system's virtual memory.
pause 
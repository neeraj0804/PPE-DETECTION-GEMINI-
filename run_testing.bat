@echo off
echo Starting PPE Detection Model Testing with NVIDIA GTX 1650...
echo.

set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

REM Check if model exists
if not exist "models\best_ppe_model.pt" (
    echo Error: Model file not found at models\best_ppe_model.pt
    echo Please train the model first by running run_training.bat
    echo.
    pause
    exit /b
)

REM Check if CUDA is available
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo.
echo Starting testing with reduced memory usage...
python test_ppe_model.py --weights models/best_ppe_model.pt --source 0 --conf 0.25 --device 0

REM If the above command fails, try with CPU
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo GPU testing failed. Your system may not have enough GPU memory.
    echo Trying with CPU instead (this will be slower)...
    python test_ppe_model.py --weights models/best_ppe_model.pt --source 0 --conf 0.25 --device cpu
)

echo.
echo Testing completed.
echo.
echo If you encountered memory errors, please run fix_memory.bat for instructions
echo on how to increase your system's virtual memory.
pause 
@echo off
echo =============================================
echo PPE Detection Dataset Splitter
echo =============================================
echo.
echo This script will split the dataset into:
echo - 70%% training data
echo - 30%% validation data
echo.

REM Activate the Python environment if needed
REM call conda activate your_env_name
REM OR
REM call venv\Scripts\activate

echo Checking Python installation...
python --version
if %ERRORLEVEL% neq 0 (
    echo Error: Python is not installed or not in the PATH.
    echo Please install Python and try again.
    exit /b 1
)

echo.
echo Splitting dataset...
python split_dataset.py --data_yaml data.yaml --train_ratio 0.7

if %ERRORLEVEL% neq 0 (
    echo.
    echo Error: Failed to split the dataset.
    exit /b 1
) else (
    echo.
    echo Dataset successfully split!
    echo.
    echo To train the model with the split dataset, run:
    echo run_training.bat --data data_split.yaml
)

echo.
pause 
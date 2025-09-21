# PPE Detection System

This application uses computer vision and machine learning to detect whether workers are wearing proper Personal Protective Equipment (PPE) in real-time using camera feeds.

## Features

- Real-time PPE detection using YOLO object detection
- GPU acceleration for faster training and inference
- Support for multiple camera feeds
- Detection of various PPE items:
  - Helmets
  - Safety vests
- Detection of PPE violations:
  - Missing helmets
  - Missing vests
- Alert system for PPE violations
- User-friendly web and desktop interfaces
- **NEW: AI-Powered Safety Report Generator**
  - Comprehensive safety reports with AI analysis
  - PDF and Excel export capabilities
  - Visual compliance charts and timeline analysis
  - Real-time data collection and session management
  - Professional reporting for safety audits
- **NEW: PPE Sight Dashboard**
  - Modern web-based dashboard interface
  - Real-time safety monitoring and analytics
  - Interactive charts and visualizations
  - Responsive design for all devices
  - Integrated with AI-powered reporting


## Step-by-Step Guide to Run the Program

### Step 1: Download the Dataset

1. Make sure the dataset is downloaded to: `C:\Users\np080\Downloads\PPE-DETECTION.v26i.yolov11`
2. The dataset should have the following structure:
   ```
   PPE-DETECTION.v26i.yolov11/
   ├── train/
   │   ├── images/
   │   │   ├── image1.jpg
   │   │   ├── image2.jpg
   │   │   └── ...
   │   └── labels/
   │       ├── image1.txt
   │       ├── image2.txt
   │       └── ...
   └── data.yaml
   ```

### Step 2: Set Up the Environment

1. Run the setup script to install all required dependencies:
   ```
   setup.bat
   ```
   This will:
   - Install PyTorch with CUDA support
   - Install other required packages
   - Create necessary directories

2. If you encounter CUDA issues, run:
   ```
   fix_memory.bat
   ```
   and follow the instructions to increase your virtual memory.

### Step 3: Train the Model

1. Run the training script:
   ```
   run_training.bat
   ```
   This will:
   - Train the model using the downloaded dataset
   - Save the best model to `models/best_ppe_model.pt`
   - Use GPU acceleration if available

2. The training process will take some time depending on your hardware.
   - With GPU: ~30 minutes to a few hours
   - With CPU: Several hours to days

### Step 4: Test the Model

1. Run the testing script:
   ```
   run_testing.bat
   ```
   This will:
   - Load the trained model
   - Open your webcam
   - Detect PPE items in real-time
   - Display compliance status

2. Press 'q' to quit the testing application.

### Step 5: Run the Application

1. Run the main application:
   ```
   run_app.bat
   ```
   This will:
   - Load the trained model
   - Open your webcam
   - Detect PPE items in real-time
   - Display compliance status
   - Allow recording videos

2. Or run the web interface for a more user-friendly experience:
   ```
   run_web_app.bat
   ```
   This will:
   - Open a web interface in your browser
   - Provide options to upload images/videos
   - Allow live detection from webcam
   - Provide settings to customize detection

3. Or run the desktop application for a native experience:
   ```
   run_desktop_app.bat
   ```
   This will:
   - Open a desktop application with a modern UI
   - Provide all the features of the web interface
   - Run natively without requiring a browser
   - Offer better performance for video processing

4. Press 'q' to quit the main application, close the browser tab for the web interface, or close the window for the desktop application.

## Troubleshooting

If you encounter any issues:

1. **CUDA errors**: Run `check_cuda.bat` to diagnose CUDA issues, then follow the instructions in `CUDA_TROUBLESHOOTING.md`.

2. **Memory errors**: Run `fix_memory.bat` and follow the instructions to increase your virtual memory.

3. **Missing model**: If you get an error about missing model file, make sure you've run the training script first.

4. **Dataset path**: If the dataset path is incorrect, update it in `data.yaml`.

## Project Structure

- `app.py`: Main application entry point
- `desktop_app.py`: Desktop application using PyQt5
- `ppe_detector.py`: Core PPE detection logic
- `camera_utils.py`: Camera handling utilities
- `web_app.py`: Streamlit web interface
- `train_ppe_model.py`: Script for training the model
- `test_ppe_model.py`: Script for testing the model
- `data.yaml`: Dataset configuration
- `models/`: Directory for model weights
- `training/`: Training utilities and documentation
- `run_training.bat`: Script to run the training process with GPU
- `run_testing.bat`: Script to run the testing process with GPU
- `run_app.bat`: Script to run the main application with GPU
- `run_web_app.bat`: Script to run the web interface with GPU
- `run_desktop_app.bat`: Script to run the desktop application with GPU
- `setup.bat`: Script to set up the environment and install dependencies
- `check_cuda.bat`: Script to check CUDA availability
- `fix_memory.bat`: Script to fix memory issues
- `CUDA_TROUBLESHOOTING.md`: Guide for resolving CUDA issues

### Safety Report Generator Files

- `safety_report_generator.py`: Core Safety Report Generator module
- `test_safety_report.py`: Test script for the Safety Report Generator
- `demo_safety_report.py`: Demo script showing how to use the feature
- `test_safety_report.bat`: Batch file to run tests
- `demo_safety_report.bat`: Batch file to run demo
- `SAFETY_REPORT_GENERATOR.md`: Detailed documentation for the feature
- `reports/`: Directory for generated safety reports
  - `pdf/`: PDF safety reports
  - `excel/`: Excel reports
  - `charts/`: Compliance and timeline charts

### PPE Sight Dashboard Files

- `ppe_sight_app.py`: Integrated web dashboard application
- `run_ppe_sight_dashboard.bat`: Batch file to run the dashboard
- `PPE_SIGHT_DASHBOARD.md`: Comprehensive dashboard documentation
- **Features**: Modern UI, real-time monitoring, analytics, reporting

## License

MIT 
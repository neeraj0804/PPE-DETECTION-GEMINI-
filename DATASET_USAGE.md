# Using the Downloaded PPE Detection Dataset

This guide explains how to use the downloaded PPE detection dataset from Roboflow to train and test the PPE detection model.

## Dataset Information

The downloaded dataset (`PPE-DETECTION.v26i.yolov11`) contains:

- Images of workers with and without PPE items
- Annotations for 5 classes:
  - `No helmet` (class 0)
  - `No vest` (class 1)
  - `Person` (class 2)
  - `helmet` (class 3)
  - `vest` (class 4)

## Setup

1. Make sure the dataset is downloaded to: `C:\Users\np080\Downloads\PPE-DETECTION.v26i.yolov11`
2. The application is configured to use this dataset location in the `data.yaml` file

## GPU Acceleration

The training and inference processes are optimized to use your NVIDIA GPU:

1. The application automatically detects and uses your NVIDIA GPU
2. Training is significantly faster with GPU acceleration (10-20x speedup)
3. Real-time detection achieves higher FPS with GPU support
4. All batch files are configured to use the GPU by default

### Requirements for GPU Acceleration

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- PyTorch with CUDA support

## Training the Model

To train the model using the downloaded dataset with GPU acceleration:

1. Run the training script:
   ```
   run_training.bat
   ```

   This will:
   - Use the `data.yaml` file that points to the downloaded dataset
   - Train for 50 epochs with batch size 8
   - Use Adam optimizer with learning rate 0.001
   - Apply data augmentation to improve model robustness
   - **Use your NVIDIA GPU for faster training**

2. The trained model will be saved to:
   - `runs/train/exp/weights/best.pt` (original location)
   - `models/best_ppe_model.pt` (copied for easy access)

## Testing the Model

To test the trained model with GPU acceleration:

1. Run the testing script:
   ```
   run_testing.bat
   ```

   This will:
   - Use the trained model from `models/best_ppe_model.pt`
   - Open your webcam for real-time detection
   - Display detection results with bounding boxes and compliance status
   - **Use your NVIDIA GPU for faster inference**

## Using the Model in the Application

The application is configured to use the trained model with GPU acceleration by default:

1. Run the main application with GPU acceleration:
   ```
   run_app.bat
   ```

2. Or run the web interface with GPU acceleration:
   ```
   run_web_app.bat
   ```

## Understanding the Detection Results

The model detects:
- People
- Helmets and safety vests
- Missing helmets and vests

The application determines compliance based on:
- If a person is detected
- If required PPE items (helmet, vest) are detected
- If "No helmet" or "No vest" violations are detected

## Customizing the Model

You can customize the training process by editing:
- `data.yaml` - to point to different dataset locations
- `train_ppe_model.py` - to change training parameters
- `ppe_detector.py` - to modify detection logic and compliance rules

## Switching Between GPU and CPU

If you need to switch between GPU and CPU:

1. In the web interface, use the "Computation Device" dropdown in the sidebar
2. For command-line applications, use the `--device` parameter:
   - For GPU: `--device 0`
   - For CPU: `--device cpu`

Example:
```
python app.py --device cpu  # Use CPU
python app.py --device 0    # Use GPU
``` 
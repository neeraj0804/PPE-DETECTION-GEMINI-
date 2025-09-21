import cv2
import time
import threading
import numpy as np
from queue import Queue

class CameraManager:
    """
    Utility class to manage camera feeds for PPE detection.
    Supports multiple cameras, video files, and RTSP streams.
    """
    
    def __init__(self, max_cameras=4):
        """
        Initialize the camera manager.
        
        Args:
            max_cameras (int): Maximum number of cameras to support
        """
        self.cameras = {}
        self.max_cameras = max_cameras
        self.frame_queues = {}
        self.stop_flags = {}
        self.threads = {}
    
    def add_camera(self, camera_id, source):
        """
        Add a camera to the manager.
        
        Args:
            camera_id (str): Unique identifier for the camera
            source: Camera index, video file path, or RTSP URL
            
        Returns:
            bool: True if camera was added successfully, False otherwise
        """
        if len(self.cameras) >= self.max_cameras:
            print(f"Error: Maximum number of cameras ({self.max_cameras}) reached")
            return False
        
        if camera_id in self.cameras:
            print(f"Error: Camera ID '{camera_id}' already exists")
            return False
        
        # Try to open the camera
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open camera source {source}")
            return False
        
        # Store the camera
        self.cameras[camera_id] = {
            'source': source,
            'cap': cap,
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS)
        }
        
        # Create a queue for frames
        self.frame_queues[camera_id] = Queue(maxsize=30)  # Buffer 30 frames max
        self.stop_flags[camera_id] = False
        
        # Start a thread to read frames
        self.threads[camera_id] = threading.Thread(
            target=self._read_frames,
            args=(camera_id,),
            daemon=True
        )
        self.threads[camera_id].start()
        
        print(f"Added camera '{camera_id}' with source {source}")
        return True
    
    def _read_frames(self, camera_id):
        """
        Thread function to continuously read frames from a camera.
        
        Args:
            camera_id (str): Camera identifier
        """
        cap = self.cameras[camera_id]['cap']
        
        while not self.stop_flags[camera_id]:
            ret, frame = cap.read()
            
            if not ret:
                # Try to reconnect for RTSP streams
                if isinstance(self.cameras[camera_id]['source'], str) and self.cameras[camera_id]['source'].startswith('rtsp'):
                    print(f"Lost connection to camera {camera_id}, attempting to reconnect...")
                    cap.release()
                    time.sleep(2)
                    cap = cv2.VideoCapture(self.cameras[camera_id]['source'])
                    self.cameras[camera_id]['cap'] = cap
                    continue
                else:
                    # End of video file or camera disconnected
                    print(f"Camera {camera_id} disconnected or end of video file")
                    break
            
            # If queue is full, remove oldest frame
            if self.frame_queues[camera_id].full():
                try:
                    self.frame_queues[camera_id].get_nowait()
                except:
                    pass
            
            # Add new frame to queue
            try:
                self.frame_queues[camera_id].put(frame, block=False)
            except:
                pass
            
            # Small delay to prevent CPU overuse
            time.sleep(0.01)
    
    def get_frame(self, camera_id, timeout=1):
        """
        Get the latest frame from a camera.
        
        Args:
            camera_id (str): Camera identifier
            timeout (float): Timeout in seconds
            
        Returns:
            numpy.ndarray or None: The latest frame or None if no frame is available
        """
        if camera_id not in self.cameras:
            print(f"Error: Camera ID '{camera_id}' not found")
            return None
        
        try:
            return self.frame_queues[camera_id].get(timeout=timeout)
        except:
            return None
    
    def remove_camera(self, camera_id):
        """
        Remove a camera from the manager.
        
        Args:
            camera_id (str): Camera identifier
            
        Returns:
            bool: True if camera was removed successfully, False otherwise
        """
        if camera_id not in self.cameras:
            print(f"Error: Camera ID '{camera_id}' not found")
            return False
        
        # Stop the thread
        self.stop_flags[camera_id] = True
        if self.threads[camera_id].is_alive():
            self.threads[camera_id].join(timeout=2)
        
        # Release the camera
        self.cameras[camera_id]['cap'].release()
        
        # Remove from dictionaries
        del self.cameras[camera_id]
        del self.frame_queues[camera_id]
        del self.stop_flags[camera_id]
        del self.threads[camera_id]
        
        print(f"Removed camera '{camera_id}'")
        return True
    
    def get_camera_info(self, camera_id=None):
        """
        Get information about cameras.
        
        Args:
            camera_id (str, optional): Camera identifier. If None, returns info for all cameras.
            
        Returns:
            dict: Camera information
        """
        if camera_id is not None:
            if camera_id not in self.cameras:
                print(f"Error: Camera ID '{camera_id}' not found")
                return None
            
            # Return a copy of the info without the cap object
            info = self.cameras[camera_id].copy()
            del info['cap']
            return info
        
        # Return info for all cameras
        all_info = {}
        for cam_id, cam_info in self.cameras.items():
            info = cam_info.copy()
            del info['cap']
            all_info[cam_id] = info
        
        return all_info
    
    def release_all(self):
        """
        Release all cameras and stop all threads.
        """
        for camera_id in list(self.cameras.keys()):
            self.remove_camera(camera_id)
        
        print("Released all cameras")

class VideoRecorder:
    """
    Utility class to record video from camera feeds.
    """
    
    def __init__(self, output_path, fps=30, resolution=(640, 480)):
        """
        Initialize the video recorder.
        
        Args:
            output_path (str): Path to save the video file
            fps (int): Frames per second
            resolution (tuple): Video resolution (width, height)
        """
        self.output_path = output_path
        self.fps = fps
        self.resolution = resolution
        self.writer = None
        self.is_recording = False
        self.start_time = None
    
    def start(self, fourcc='mp4v'):
        """
        Start recording.
        
        Args:
            fourcc (str): FourCC code for the video codec
            
        Returns:
            bool: True if recording started successfully, False otherwise
        """
        if self.is_recording:
            print("Already recording")
            return False
        
        # Create the video writer
        fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc_code,
            self.fps,
            self.resolution
        )
        
        if not self.writer.isOpened():
            print(f"Error: Could not create video writer for {self.output_path}")
            return False
        
        self.is_recording = True
        self.start_time = time.time()
        print(f"Started recording to {self.output_path}")
        return True
    
    def write_frame(self, frame):
        """
        Write a frame to the video file.
        
        Args:
            frame (numpy.ndarray): Frame to write
            
        Returns:
            bool: True if frame was written successfully, False otherwise
        """
        if not self.is_recording or self.writer is None:
            return False
        
        # Resize frame if needed
        if frame.shape[1] != self.resolution[0] or frame.shape[0] != self.resolution[1]:
            frame = cv2.resize(frame, self.resolution)
        
        self.writer.write(frame)
        return True
    
    def stop(self):
        """
        Stop recording.
        
        Returns:
            float: Duration of the recording in seconds
        """
        if not self.is_recording or self.writer is None:
            return 0.0
        
        duration = time.time() - self.start_time
        self.writer.release()
        self.writer = None
        self.is_recording = False
        print(f"Stopped recording to {self.output_path} (duration: {duration:.2f}s)")
        return duration 
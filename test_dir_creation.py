#!/usr/bin/env python3
"""
Test script to verify directory creation for training experiments.
"""

import os
from pathlib import Path
import time

def test_directory_creation():
    """Test that directories are properly created."""
    # Project directory
    project_dir = "runs/train"
    
    # Create a timestamped experiment name
    experiment_name = f"test_exp_{int(time.time())}"
    
    # Create project directory
    os.makedirs(project_dir, exist_ok=True)
    print(f"Created project directory: {project_dir}")
    
    # Create experiment directory
    exp_dir = os.path.join(project_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Created experiment directory: {exp_dir}")
    
    # Create a test file in the experiment directory
    test_file = os.path.join(exp_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("This is a test file to verify directory creation.")
    print(f"Created test file: {test_file}")
    
    # List the contents of the project directory
    print("\nContents of project directory:")
    for item in os.listdir(project_dir):
        item_path = os.path.join(project_dir, item)
        if os.path.isdir(item_path):
            print(f"  - Directory: {item}")
        else:
            print(f"  - File: {item}")
    
    # Verify the test file exists
    if os.path.exists(test_file):
        print(f"\nTest file exists at: {test_file}")
        print("Directory creation test passed!")
    else:
        print("\nTest file not found. Directory creation test failed!")

if __name__ == "__main__":
    print("Testing directory creation for training experiments...")
    test_directory_creation() 
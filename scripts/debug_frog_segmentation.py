#!/usr/bin/env python
"""
Debug script for frog segmentation tool.

This script checks various potential issues that could prevent the
interactive segmentation window from displaying properly.

Usage:
    python scripts/debug_frog_segmentation.py [image_path]
"""

import os
import sys
import platform
import importlib
import subprocess
import traceback

def check_matplotlib_backend():
    """Check the matplotlib backend configuration."""
    print("\n=== Checking Matplotlib Backend ===")
    try:
        import matplotlib
        print(f"Matplotlib version: {matplotlib.__version__}")
        print(f"Current backend: {matplotlib.get_backend()}")
        
        # Try to set a different backend
        try:
            matplotlib.use('TkAgg')
            print("Successfully switched to TkAgg backend")
        except Exception as e:
            print(f"Failed to switch to TkAgg backend: {e}")
            
        # Check available backends
        print("\nAvailable backends:")
        for backend in matplotlib.rcsetup.all_backends:
            try:
                matplotlib.use(backend)
                print(f"  {backend}: Available")
            except Exception:
                print(f"  {backend}: Not available")
                
        # Reset to default
        matplotlib.use('Agg')  # Non-interactive backend for testing
    except Exception as e:
        print(f"Error checking matplotlib: {e}")
        traceback.print_exc()

def check_display_environment():
    """Check the display environment variables and settings."""
    print("\n=== Checking Display Environment ===")
    
    # Check OS
    print(f"Operating System: {platform.system()} {platform.release()}")
    
    # Check environment variables related to display
    display_vars = ['DISPLAY', 'WAYLAND_DISPLAY', 'QT_QPA_PLATFORM']
    for var in display_vars:
        print(f"{var}: {os.environ.get(var, 'Not set')}")
    
    # Check if running in WSL
    if platform.system() == "Linux" and "microsoft" in platform.release().lower():
        print("Running in Windows Subsystem for Linux (WSL)")
        print("Note: GUI applications in WSL may require additional configuration")
    
    # Check if running in a remote session
    if os.environ.get('SSH_CONNECTION'):
        print("Running in SSH session - X11 forwarding may be required")

def check_gui_toolkit():
    """Check if required GUI toolkits are available."""
    print("\n=== Checking GUI Toolkits ===")
    
    toolkits = ['tkinter', 'PyQt5', 'PySide2', 'wx']
    
    for toolkit in toolkits:
        try:
            importlib.import_module(toolkit)
            print(f"{toolkit}: Available")
        except ImportError:
            print(f"{toolkit}: Not available")

def check_image_loading(image_path):
    """Check if the image can be loaded properly."""
    print(f"\n=== Checking Image Loading: {image_path} ===")
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image file does not exist: {image_path}")
        return
    
    print(f"Image file exists: {image_path}")
    print(f"File size: {os.path.getsize(image_path)} bytes")
    
    try:
        import cv2
        import numpy as np
        
        # Try to load with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            print("ERROR: OpenCV failed to load the image")
        else:
            print(f"OpenCV loaded image successfully: {img.shape}")
            
            # Check if image has content
            if np.mean(img) < 1.0:
                print("WARNING: Image appears to be mostly black or empty")
    except Exception as e:
        print(f"Error loading image with OpenCV: {e}")
        traceback.print_exc()
    
    try:
        from PIL import Image
        
        # Try to load with PIL
        img = Image.open(image_path)
        print(f"PIL loaded image successfully: {img.size}, mode: {img.mode}")
    except Exception as e:
        print(f"Error loading image with PIL: {e}")
        traceback.print_exc()

def check_sam_model():
    """Check if the SAM model can be loaded."""
    print("\n=== Checking SAM Model ===")
    
    try:
        from segment_anything import sam_model_registry
        print("Successfully imported segment_anything")
        
        # Check if config module can be imported
        try:
            from src.config import PATHS
            print("Successfully imported config")
            
            # Check if model file exists
            model_path = os.path.join(PATHS['downloads'], "sam_vit_h_4b8939.pth")
            if os.path.exists(model_path):
                print(f"SAM model file exists: {model_path}")
                print(f"File size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
            else:
                print(f"ERROR: SAM model file not found: {model_path}")
        except Exception as e:
            print(f"Error importing config: {e}")
            traceback.print_exc()
    except Exception as e:
        print(f"Error importing segment_anything: {e}")
        traceback.print_exc()

def test_simple_plot():
    """Test if a simple matplotlib plot can be displayed."""
    print("\n=== Testing Simple Matplotlib Plot ===")
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        print("Creating a simple plot...")
        plt.figure(figsize=(5, 5))
        plt.plot(np.random.rand(10), np.random.rand(10), 'o')
        plt.title("Test Plot")
        
        print("Attempting to display the plot...")
        plt.show(block=False)
        plt.pause(2)
        plt.close()
        print("Plot displayed and closed successfully")
    except Exception as e:
        print(f"Error displaying simple plot: {e}")
        traceback.print_exc()

def check_script_paths():
    """Check if the script paths are correct."""
    print("\n=== Checking Script Paths ===")
    
    # Check current working directory
    print(f"Current working directory: {os.getcwd()}")
    
    # Check if the scripts directory exists
    if os.path.exists("scripts"):
        print("scripts directory exists")
        script_files = os.listdir("scripts")
        print(f"Files in scripts directory: {script_files}")
    else:
        print("ERROR: scripts directory not found")
    
    # Check if the src directory exists
    if os.path.exists("src"):
        print("src directory exists")
        if os.path.exists("src/tools"):
            print("src/tools directory exists")
            tool_files = os.listdir("src/tools")
            print(f"Files in src/tools directory: {tool_files}")
        else:
            print("ERROR: src/tools directory not found")
    else:
        print("ERROR: src directory not found")
    
    # Check Python path
    print("\nPython path:")
    for path in sys.path:
        print(f"  {path}")

def run_command_with_timeout(cmd, timeout=10):
    """Run a command with timeout and return the output."""
    try:
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout
        )
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return "", "Command timed out"
    except Exception as e:
        return "", str(e)

def test_script_execution():
    """Test if the script can be executed properly."""
    print("\n=== Testing Script Execution ===")
    
    # Test importing the module
    try:
        from scripts.frog_segmentation import interactive_segmentation
        print("Successfully imported interactive_segmentation function")
    except Exception as e:
        print(f"Error importing interactive_segmentation: {e}")
        traceback.print_exc()
    
    # Test running the script with --help
    print("\nTesting script execution with --help:")
    stdout, stderr = run_command_with_timeout(["python", "-m", "scripts.frog_segmentation", "--help"])
    if stderr and not "usage" in stdout.lower():
        print(f"Error running script with --help: {stderr}")
    else:
        print("Script executed successfully with --help")

def main():
    """Main function to run all checks."""
    print("=== Frog Segmentation Debugging Tool ===")
    print(f"Python version: {sys.version}")
    
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Try to find an image in the default location
        from src.config import PATHS
        default_dir = os.path.join(PATHS['downloads'], "whole-frog")
        if os.path.exists(default_dir):
            jpg_files = [f for f in os.listdir(default_dir) if f.lower().endswith('.jpg')]
            if jpg_files:
                image_path = os.path.join(default_dir, jpg_files[0])
                print(f"Using first found image: {image_path}")
            else:
                print("No JPG files found in default directory")
                image_path = None
        else:
            print(f"Default directory not found: {default_dir}")
            image_path = None
    
    # Run all checks
    check_script_paths()
    check_matplotlib_backend()
    check_display_environment()
    check_gui_toolkit()
    
    if image_path:
        check_image_loading(image_path)
    
    check_sam_model()
    test_simple_plot()
    test_script_execution()
    
    print("\n=== Debugging Complete ===")
    print("\nPossible solutions if the window doesn't appear:")
    print("1. Try running with a different matplotlib backend:")
    print("   import matplotlib; matplotlib.use('TkAgg'); import matplotlib.pyplot as plt")
    print("2. Make sure you have a GUI toolkit installed (tkinter, PyQt5, etc.)")
    print("3. If running remotely, ensure X11 forwarding is enabled")
    print("4. Try running the script with the -i flag to keep Python interactive:")
    print("   python -i scripts/frog_segmentation.py path/to/image.jpg")
    print("5. Check if you're running in a virtual environment and it has all required packages")

if __name__ == "__main__":
    main() 
# Real-Time-Semantic-Segmentation-Project
This project implements a real-time semantic segmentation system using a deep learning model. The system captures live video feed from a camera and performs pixel-wise classification to identify various objects and their boundaries in the scene.
## Setup and Installation
1. **Dependencies**: 
   - Python 3.x
   - OpenCV
   - PyTorch
   - PIL
   - torchvision
   - scipy
   - numpy

2. **Installing Required Libraries**: 
  ```
  pip install opencv-python-headless torch torchvision Pillow numpy scipy
  ```
3. **Model Weights**: 
Download the model weights (`encoder_epoch.pth` and `decoder_epoch.pth`) from the provided link and place them in the project directory.

4. **Running the Application**:
- Execute the main script to start the real-time semantic segmentation:
  ```
  python main_script.py
  ```

## Usage
- The application opens a window displaying the live camera feed.
- Press `Space` to toggle real-time mode.
- Press `TAB` to toggle between different visualizations.
- Press `1-9` or `a-f` to choose specific classes for segmentation visualization.
- Press `Esc` to exit the application.
- Press `s` to save the current frame and segmentation result.

## Project Purpose
The purpose of this project is to demonstrate the capabilities of semantic segmentation in real-time applications. It can be used for educational purposes or as a base for more complex computer vision projects.

## Credits and Acknowledgments
- Model training and architecture are based on [CSAILVision's Semantic Segmentation PyTorch](https://github.com/CSAILVision/semantic-segmentation-pytorch/).
- Data used: [ADE20K MIT Scene Parsing Benchchmark](https://groups.csail.mit.edu/vision/datasets/ADE20K/).

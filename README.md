# Traffic Sign Classification Using Classical Digital Image Processing

This project implements a full traffic sign classification pipeline using only classical digital image processing techniques. All steps are developed from scratch in NumPy without the use of machine learning models or high-level vision libraries.

## Project Scope

### Dataset
- Cropped RGB traffic sign images from 6 to 8 selected classes
- Approximately 100 images per class used for training and evaluation

## Components

### 1. Preprocessing and Filtering
- Mean Filter (3×3)
- Gaussian Filter
- Median Filter
- Adaptive Median Filter
- Unsharp Masking / High-Boost Filtering

### 2. Color Segmentation
- Manual RGB to HSV conversion
- Red and blue hue thresholding
- Binary mask generation and refinement using:
  - Morphological operations (erosion, dilation, opening)
  - Connected component filtering for noise removal
  - Hole filling

### 3. Edge Detection
- Manual implementation of the Canny Edge Detector:
  - Gradient computation (Sobel)
  - Non-maximum suppression
  - Double thresholding
  - Edge tracking by hysteresis

### 4. Geometric Normalization
- Image rotation and scaling using affine transformations
- All transformations implemented using NumPy

### 5. Feature Extraction
- Harris Corner Detection
- Circularity: 4π × Area / (Perimeter²)
- Aspect Ratio (width/height)
- Extent (region area / bounding box area)
- Average Hue

### 6. Rule-Based Classification
- A set of if-else rules that use extracted features to assign ClassId
- Distinguishes visually similar signs using combined color and shape heuristics

## Output Files

- `traffic_sign_classification.ipynb`: Full implementation
- `pipeline.py`: Pipeline

## Dependencies

- Python 3.7+
- NumPy
- OpenCV (for image loading only)
- Matplotlib
- Pillow

Install dependencies:

```bash
pip install numpy opencv-python matplotlib

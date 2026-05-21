# 🌄 Panorama Stitching
An advanced panorama stitching pipeline built with Python and OpenCV. It stitches multiple overlapping images into a seamless wide-angle panorama using cylindrical projection, feature matching, homography estimation, seam blending, and automatic horizon correction. 

## ✨ Features

* Cylindrical projection to reduce perspective distortion
* SIFT feature detection and descriptor extraction
* Feature matching using BFMatcher
* RANSAC-based homography estimation with affine fallback
* Global exposure compensation for brightness balancing
* Seam finding for smooth transitions
* Multi-band blending using Laplacian pyramids
* Automatic horizon/tilt correction
* Debug output images for feature match visualization

## 📁 Project Structure

Panaroma-stitching/
│
├── main.py              # Entry point and CLI handling
├── stitcher.py          # Main panorama stitching pipeline
├── utils.py             # Helper functions and image utilities
├── projection.py        # Cylindrical projection logic
├── features.py          # Feature detection and matching
├── homography.py        # Homography estimation
├── warper.py            # Image warping and canvas generation
├── exposure.py          # Exposure compensation
├── seam_blend.py        # Seam finding and blending
└── README.md


## ⚙️ Installation

Clone the repository and install dependencies:

git clone https://github.com/jasman5/Panaroma-stitching.git
cd Panaroma-stitching

pip install opencv-python numpy scipy

## 🚀 Usage

Basic usage:

python main.py --input img1.jpg img2.jpg img3.jpg --output panorama.jpg

### Available Arguments

| Argument        | Required | Description                             |
| --------------- | -------- | --------------------------------------- |
| `--input`       | ✅        | Input images in left-to-right order     |
| `--output`      | ✅        | Output panorama image path              |
| `--focal`       | ❌        | Focal length for cylindrical projection |
| `--no-cylinder` | ❌        | Disable cylindrical projection          |


## 📌 Examples

### Basic Stitching
python main.py --input left.jpg center.jpg right.jpg --output pano.jpg

### Using Custom Focal Length
python main.py --input img1.jpg img2.jpg --output pano.jpg --focal 800

### Disable Cylindrical Projection
python main.py --input img1.jpg img2.jpg --output pano.jpg --no-cylinder

## 🔍 Stitching Pipeline

Input Images
     ↓
Cylindrical Projection
     ↓
Feature Detection (SIFT)
     ↓
Feature Matching
     ↓
Homography Estimation (RANSAC)
     ↓
Warping onto Common Canvas
     ↓
Exposure Compensation
     ↓
Seam Finding & Blending
     ↓
Auto Crop & Horizon Correction
     ↓
Final Panorama


## 📷 Tips for Better Results

* Capture images left to right
* Keep 30–50% overlap between images
* Lock camera exposure and focus
* Avoid moving objects during capture
* Rotate around a fixed point for minimal distortion

## 🛠️ Tech Stack

* Python 3.x
* OpenCV (`cv2`)
* NumPy
* SciPy
**Jasman Kaur** — [GitHub Profile](https://github.com/jasman5?utm_source=chatgpt.com)

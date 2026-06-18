# License Plate Recognition System

A Computer Vision project that detects vehicle license plates using YOLO and extracts plate numbers using EasyOCR.

## Features

* License Plate Detection with YOLO
* OCR-based Text Recognition
* Automatic OCR Error Correction
* Stable Plate Prediction Across Frames
* Video Annotation and Export
* Real-Time Visualization

## Project Structure

```text
License-Plate-Recognition/
│
├── numberPlate_recognition.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── weights/
├── input/
└── output/
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Place YOLO weights in:

```text
weights/license_best.pt
```

2. Place input video in:

```text
input/vehicle_video.mp4
```

3. Run:

```bash
python numberPlate_recognition.py
```

4. Output video will be saved in:

```text
output/output_with_licensev3.mp4
```

## Technologies Used

* Python
* OpenCV
* Ultralytics YOLO
* EasyOCR
* NumPy

## Author

Pawan Kumar Vishwakarma

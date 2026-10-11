# Cars

A Python-based application for object detection in video streams using the YOLOv8 model from Ultralytics.

## Architecture Overview

The `Cars` repository is structured to perform real-time object detection on video files. The core functionality is encapsulated within the `detect_objects.py` script, which leverages the YOLOv8 model to identify and count objects in a given video file. The architecture can be summarized as follows:

1. **Model Loading**: The YOLOv8n model is loaded using the Ultralytics library.
2. **Video Capture**: A video file (`video.mp4`) is opened for processing.
3. **Frame Processing**: Every 10th frame of the video is processed to detect objects, optimizing performance by reducing unnecessary computations.
4. **Object Detection**: The YOLOv8 model predicts and identifies objects within each frame.
5. **Result Compilation**: Detected object counts are compiled and printed in descending order.

## Tech Stack

- **Programming Language**: Python
- **Machine Learning Framework**: Ultralytics YOLOv8
- **Video Processing Library**: OpenCV (cv2)
- **Data Structures**: Collections (defaultdict)

## Getting Started / Setup Guide

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/Cars.git
   cd Cars
   ```

2. **Install Dependencies**:
   Ensure you have Python installed, then install the required packages using pip:
   ```bash
   pip install ultralytics opencv-python
   ```

3. **Prepare Video File**:
   Place your video file in the repository directory or update the `detect_objects.py` script to point to the correct path.

4. **Run the Application**:
   Execute the object detection script:
   ```bash
   python detect_objects.py
   ```

5. **Output**:
   The script will print the count of detected objects in the format: "count label", e.g., "5 person", "2 car".

## Additional Notes

- Ensure that the video file path specified in `detect_objects.py` is correct.
- For better performance, consider using a more powerful model (e.g., YOLOv8m or YOLOv8l) at the expense of increased computational resources.

```

This README provides a clear and concise overview of the project, its architecture, required dependencies, and setup instructions.
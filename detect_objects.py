import torch
import cv2
from collections import defaultdict
import sys
import os

# Add yolov5 directory to path
sys.path.insert(0, './yolov5')

# Load YOLOv5 model
model = torch.hub.load('yolov5', 'yolov5s', source='local')

# Open video
cap = cv2.VideoCapture('video.mp4')
frame_count = 0
detections = defaultdict(int)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 10 != 0:
        continue

    results = model(frame)
    for *xyxy, conf, cls in results.xyxy[0]:
        label = model.names[int(cls)]
        detections[label] += 1

cap.release()

# Output results
for label, count in sorted(detections.items(), key=lambda x: -x[1]):
    print(f"{count} {label}")

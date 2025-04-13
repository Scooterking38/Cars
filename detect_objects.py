import torch
import cv2
from collections import defaultdict

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
model.eval()

# Open video
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

frame_count = 0
detections_count = defaultdict(int)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 10 != 0:  # Process every 10th frame
        continue

    results = model(frame)
    for det in results.pred[0]:
        class_id = int(det[5])
        label = model.names[class_id]
        detections_count[label] += 1

cap.release()

# Print results
for label, count in sorted(detections_count.items(), key=lambda x: -x[1]):
    print(f"{count} {label}")

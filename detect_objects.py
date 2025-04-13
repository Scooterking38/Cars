from ultralytics import YOLO
import cv2
from collections import defaultdict

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Small and fast

# Open video
cap = cv2.VideoCapture('video.mp4')
frame_count = 0
counts = defaultdict(int)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Only analyze every 10th frame for speed
    frame_count += 1
    if frame_count % 10 != 0:
        continue

    # Run detection
    results = model.predict(frame, verbose=False)
    boxes = results[0].boxes

    if boxes:
        for cls_id in boxes.cls.tolist():
            label = model.names[int(cls_id)]
            counts[label] += 1

cap.release()

# Output format: "5 person", "2 car", etc.
for label, count in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"{count} {label}")

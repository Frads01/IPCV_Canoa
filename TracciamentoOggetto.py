from collections import defaultdict

import cv2
from ultralytics import YOLO
import numpy as np
# Load the YOLOv9 model
model = YOLO('yolov9e.pt')  # Load an official Segment model

# open the video file
cap = cv2.VideoCapture("cut.mp4")
if not cap.isOpened():
    print("Error: Could not open video stream")
ret, frame = cap.read()
fourcc = cv2.VideoWriter.fourcc(*'mp4v')
out = cv2.VideoWriter('track.mp4', fourcc, 24.0, (frame.shape[1], frame.shape[0]), True)
track_history = defaultdict(lambda: [])

while cap.isOpened():
    roi = frame[4*frame.shape[0]//10:frame.shape[0], 0:frame.shape[1]]
    #cv2.imshow('frame', roi)
    # Read a frame from the video
    if ret:
        # Run YOLOv9 tracking on the frame, persisting tracks between frames
        conf = 0.2
        iou = 0.3
        tracker = "bytetrack.yaml"
        results = model.track(roi, persist=True, conf=conf, iou=iou, show=False, classes=[0])
        annotated_frame = results[0].plot()
        frame[4 * frame.shape[0] // 10:frame.shape[0], 0:frame.shape[1]] = annotated_frame
        cv2.imshow('Tracking', frame)
        #out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()
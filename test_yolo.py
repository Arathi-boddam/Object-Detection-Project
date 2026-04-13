from ultralytics import YOLO
import cv2
import time

# Load model
model = YOLO("yolov8n.pt")

# Load image
img = cv2.imread("test.jpg")

# Check if image loaded
if img is None:
    print("❌ Image not found. Make sure test.jpg is in the folder.")
    exit()

# Measure latency
start = time.time()
results = model(img)
end = time.time()

print("YOLO Latency:", (end - start) * 1000, "ms")

# Save output
results[0].save(filename="output_yolo.jpg")

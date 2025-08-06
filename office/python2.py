import cv2
from ultralytics import YOLO
import torch

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # You can replace with yolov8s.pt, yolov8m.pt etc.

# Set device (GPU if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Load video
video_path = "D:\\coding\\office\\front_kriti.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection on the full frame
    results = model.predict(source=frame, conf=0.4, verbose=False, device=device)

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            label = model.names[class_id]
            conf = float(box.conf[0])

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("YOLOv8 Detection", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

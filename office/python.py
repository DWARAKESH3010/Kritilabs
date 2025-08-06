from ultralytics import YOLO
import cv2
import torch

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load model
model = YOLO(r"D:\coding\office\best.pt")
model.to(device)

# Load video
cap = cv2.VideoCapture(r"D:\coding\office\video.mp4")

# Get frame dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define output video writer
out = cv2.VideoWriter("D:/coding/office/output_detected.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame, device=device)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = model.names[cls]

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    out.write(frame)                      # ✅ Save to video file
    cv2.imshow("YOLOv8 Detection", frame)

    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
out.release()
cv2.destroyAllWindows()

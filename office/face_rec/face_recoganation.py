import cv2
import os
from deepface import DeepFace
from ultralytics import YOLO
import torch
import numpy as np

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO("yolov8n-face-lindevs.pt").to(device)

# Load reference image and compute its embedding using SFace
reference_img_path = r"D:\coding\office\face_rec\ref.jpg"
print("[INFO] Building reference embedding using SFace...")
reference_embedding = DeepFace.represent(img_path=reference_img_path, model_name='SFace')[0]["embedding"]

# Create folder for matched faces
output_folder = r"D:\coding\office\face_rec\captured_faces"
os.makedirs(output_folder, exist_ok=True)

# Start video capture (0 for webcam or replace with video file path)
cap = cv2.VideoCapture(0)
face_id = 0

print("[INFO] Starting real-time face recognition...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.4, verbose=False, device=device)

    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        face_crop = frame[y1:y2, x1:x2]

        # Resize face crop for consistent embedding input
        resized_face = cv2.resize(face_crop, (224, 224))

        try:
            # Get embedding of detected face
            face_embedding = DeepFace.represent(
                img_path=resized_face,
                model_name='SFace',
                enforce_detection=False
            )[0]["embedding"]

            # Calculate cosine similarity
            cosine_similarity = np.dot(reference_embedding, face_embedding) / (
                np.linalg.norm(reference_embedding) * np.linalg.norm(face_embedding)
            )

            # If match is above threshold
            if cosine_similarity > 0.4:
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"MATCHED ({cosine_similarity:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Save matched face
                filename = os.path.join(output_folder, f"match_{face_id}.jpg")
                cv2.imwrite(filename, face_crop)
                print(f"[MATCH FOUND] Saved {filename}")
                face_id += 1

        except Exception as e:
            print(f"[WARNING] Face processing failed: {e}")

    cv2.imshow("YOLO + DeepFace Match", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
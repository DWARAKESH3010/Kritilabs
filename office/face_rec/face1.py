import cv2
import torch
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# === [1] Load YOLOv8n-face detector ===
yolo_model = YOLO("yolov8n-face.pt")  # Use 'yolov8n-face-lindevs.pt' if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model.to(device)

# === [2] Load Real-ESRGAN ===
sr_model = RealESRGANer(
    scale=2,
    model_path='D:\coding\office\face_rec\RealESRGAN_x2plus.pth',
    model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=True if device == 'cuda' else False
)

# === [3] Initialize InsightFace recognizer ===
face_app = FaceAnalysis(name='buffalo_l', root='insightface_models')
face_app.prepare(ctx_id=0 if device == 'cuda' else -1)

# === [4] Load reference image and get embedding ===
ref_img = cv2.imread("D:\coding\office\face_rec\ref.jpg")
ref_faces = face_app.get(ref_img)
if not ref_faces:
    raise ValueError("No face found in reference image.")
ref_embedding = ref_faces[0].embedding

# === [5] Start Webcam and Process ===
cap = cv2.VideoCapture(0)  # or replace with a video path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model.predict(frame, conf=0.5)[0]
    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box.tolist())
        face_crop = frame[y1:y2, x1:x2]

        # Upscale if face is small
        if min(face_crop.shape[:2]) < 100:
            try:
                face_crop, _ = sr_model.enhance(face_crop, outscale=2)
            except:
                continue

        # Get embedding and compare
        faces = face_app.get(face_crop)
        if not faces:
            continue

        face_emb = faces[0].embedding
        dist = np.linalg.norm(ref_embedding - face_emb)

        label = f"Match ({dist:.2f})" if dist < 1.0 else "Unknown"
        color = (0, 255, 0) if dist < 1.0 else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

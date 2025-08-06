import cv2
import numpy as np
from tensorflow.keras.models import load_model

# === CONFIG ===
video_path = r"D:\coding\office\litter_road.mp4"
output_path = r"D:\coding\office\detected_litter.avi"
min_contour_area = 400
resize_shape = (64, 64)  # Match CNN input
model = load_model("litter_classifier.h5")  # Pre-trained binary classifier

# === Load video ===
cap = cv2.VideoCapture(video_path)
ret, bg_frame = cap.read()
if not ret:
    print("❌ Failed to load video")
    exit()

# Prepare background
bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
bg_gray = cv2.GaussianBlur(bg_gray, (21, 21), 0)

# Output setup
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    diff = cv2.absdiff(bg_gray, gray)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < min_contour_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        roi = frame[y:y+h, x:x+w]

        try:
            resized = cv2.resize(roi, resize_shape)
            normalized = resized / 255.0
            input_tensor = np.expand_dims(normalized, axis=0)

            pred = model.predict(input_tensor)[0][0]
            if pred > 0.5:
                label = f"Litter ({pred:.2f})"
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        except:
            pass

    cv2.imshow("Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

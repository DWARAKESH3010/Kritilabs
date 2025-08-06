import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# Load known encodings
with open("encodings/insight_encodings.pkl", "rb") as f:
    known_encodings, known_names = pickle.load(f)

# Init model
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0)  # 0 for GPU, -1 for CPU

# Recognition threshold
SIMILARITY_THRESHOLD = 0.5  # Lower = stricter

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = face_app.get(frame)

    for face in faces:
        emb = face.embedding.reshape(1, -1)
        similarities = cosine_similarity(emb, np.array(known_encodings))[0]
        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]

        if best_score >= SIMILARITY_THRESHOLD:
            name = known_names[best_match_idx]
            color = (0, 255, 0)  # Green for known
        else:
            name = "Unknown"
            color = (0, 0, 255)  # 🔴 Red for unknown

        # Draw bounding box and name
        box = face.bbox.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(frame, f"{name} ({best_score:.2f})", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

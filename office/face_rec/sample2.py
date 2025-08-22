import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque, Counter

# Load known encodings
with open("encodings/insight_encodings.pkl", "rb") as f:
    known_encodings, known_names = pickle.load(f)

# Init model
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0)  # Use 0 for GPU, -1 for CPU

# Recognition threshold (adjusted for stability)
SIMILARITY_THRESHOLD = 0.20  # Higher = less flickering

# Keep last 10 predictions (smoothing buffer)
prediction_history = deque(maxlen=10)

# Load video
video_path = r"D:\coding\office\face_rec\office.mp4"  # or RTSP URL
cap = cv2.VideoCapture(video_path)

# Get video properties
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

# Define VideoWriter to save the output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'mp4v' for .mp4
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

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
        else:
            name = "Unknown"

        # Add prediction to history
        prediction_history.append(name)

        # Get the smoothed name (most frequent in history)
        final_name = Counter(prediction_history).most_common(1)[0][0]

        # Draw on frame
        box = face.bbox.astype(int)
        color = (0, 255, 0) if final_name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(frame, f"{final_name} ({best_score:.2f})",
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    resized = cv2.resize(frame, (1200, 800))
    # Show live preview
    cv2.imshow("Face Recognition", resized)

    # Save frame to output video
    out.write(frame)

    if cv2.waitKey(1) == 27:  # ESC to exit
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

import cv2
import os
import pickle
import numpy as np
from insightface.app import FaceAnalysis

# Init model

face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0)
known_encodings = []
known_names = []

known_path = r"D:\coding\office\face_rec\faces"
for file in os.listdir(known_path):
    name = os.path.splitext(file)[0]
    img = cv2.imread(os.path.join(known_path, file))
    faces = face_app.get(img)
    if faces:
        known_encodings.append(faces[0].embedding)
        known_names.append(name)

# Save encodings
with open("encodings/insight_encodings.pkl", "wb") as f:
    pickle.dump((known_encodings, known_names), f)

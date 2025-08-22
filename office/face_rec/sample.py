import cv2

video_path = "rtsp://admin:cctv%40123@192.168.8.208:554"
cap = cv2.VideoCapture(video_path)

output_path = r"D:\coding\office\face_rec\output.mp4"

# Get frame properties
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30   # fallback to 30 if fps not found
frame_width = 1200
frame_height = 600

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')   # use 'avc1' if 'mp4v' doesn’t work
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    resized = cv2.resize(frame, (frame_width, frame_height))

    # Show window
    cv2.imshow("Face Recognition", resized)

    # Write frame to video file
    out.write(resized)

    # Exit with ESC
    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video saved at: {output_path}")
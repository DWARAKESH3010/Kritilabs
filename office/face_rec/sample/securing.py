import cv2
import os

# Paths
video_dir = r"D:\coding\office\face_rec\sample\video"
output_dir = r"D:\coding\office\face_rec\sample\images"

# Create output folder if not exists
os.makedirs(output_dir, exist_ok=True)

# Supported video extensions
video_extensions = ('.mp4', '.avi', '.mov', '.mkv')

# Global frame counter across all videos
frame_counter = 0

# Loop through all videos in the folder
for file in os.listdir(video_dir):
    if file.lower().endswith(video_extensions):
        video_path = os.path.join(video_dir, file)
        video_name = os.path.splitext(file)[0]  # filename without extension

        print(f"Processing: {video_path}")

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # frames per second
        frame_interval = fps * 2              # capture every 2 seconds

        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if count % frame_interval == 0:
                frame_filename = os.path.join(
                    output_dir,
                    f"{video_name}_frame_{frame_counter:06d}.jpg"
                )
                cv2.imwrite(frame_filename, frame)
                print(f"Saved {frame_filename}")
                frame_counter += 1

            count += 1

        cap.release()

print("All videos processed ✅")

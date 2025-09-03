import cv2
import os

# Paths
video_dir = r"D:\coding\office\face_rec\sample\video"
image_dir = r"D:\coding\office\face_rec\sample\image"

# Create base image directory if not exists
os.makedirs(image_dir, exist_ok=True)

# Loop over all videos in the video folder
for video_file in os.listdir(video_dir):
    if video_file.endswith((".mp4", ".avi", ".mov", ".mkv")):  # video formats
        video_path = os.path.join(video_dir, video_file)
        
        # Create a subfolder for each video inside image_dir
        video_name = os.path.splitext(video_file)[0]
        video_image_dir = os.path.join(image_dir, video_name)
        os.makedirs(video_image_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)  # frames per second
        if fps == 0:
            print(f"⚠️ Skipping {video_file} (cannot read FPS)")
            continue
        
        frame_interval = int(fps * 2)    # every 2 seconds
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                img_name = f"frame{saved_count}.jpg"
                img_path = os.path.join(video_image_dir, img_name)
                cv2.imwrite(img_path, frame)
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        print(f"✅ Saved {saved_count} frames from {video_file}")

print("🎉 All videos processed successfully!")

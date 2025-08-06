import cv2

# === INPUT VIDEO ===
video_path = r'D:\coding\office\Kriti\2f_kriti.mp4'  # Uploaded video path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Could not open video.")
    exit()

# === BACKGROUND SUBTRACTOR ===
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# === OUTPUT CONFIG ===
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_path = '/mnt/data/bgsub_output.mp4'

fourcc = cv2.VideoWriter_fourcc(*'mp4 v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === APPLY BACKGROUND SUBTRACTION ===
    fg_mask = bg_subtractor.apply(frame)

    # Optional: convert mask to 3 channels to overlay
    fg_colored = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
    output = cv2.addWeighted(frame, 0.7, fg_colored, 0.3, 0)

    # === DISPLAY ===
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Foreground Mask', fg_mask)
    cv2.imshow('Overlay (Blended)', output)

    out.write(output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === CLEANUP ===
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Background subtraction video saved to: {output_path}")

import cv2
import numpy as np

# --- Configuration ---
video_path = r'D:\coding\office\chinese.mp4'

# --- Capture video ---
cap = cv2.VideoCapture(video_path)

# Read the first frame to define background and mask
ret, first_frame = cap.read()
if not ret:
    print("Failed to read video.")
    cap.release()
    exit()

# Define polygon ROI points manually (clockwise or counter-clockwise)
# Example: a triangle or quadrilateral
polygon_pts = np.array([[200, 200], [500, 300], [450, 400], [250, 400]])

# Create a black mask same size as frame
mask = np.zeros(first_frame.shape[:2], dtype=np.uint8)
cv2.fillPoly(mask, [polygon_pts], 255)  # Fill the polygon with white (255)

# Convert first frame to gray and blur
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_gray = cv2.GaussianBlur(first_gray, (21, 21), 0)

# Apply the polygon mask to the background
first_gray_masked = cv2.bitwise_and(first_gray, first_gray, mask=mask)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Apply the same mask to the current frame
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)

    # Subtract masked regions
    frame_delta = cv2.absdiff(first_gray_masked, gray_masked)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Draw polygon on the original frame for reference
    display_frame = frame.copy()
    cv2.polylines(display_frame, [polygon_pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # Show results
    cv2.imshow("Frame", display_frame)
    cv2.imshow("Delta in Polygon", frame_delta)
    cv2.imshow("Threshold in Polygon", thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

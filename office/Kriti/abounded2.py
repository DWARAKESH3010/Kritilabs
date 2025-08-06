import cv2
import numpy as np

# === CONFIG ===
background_image_path = r"D:\coding\office\Kriti\first_image.png"
video_path = r"D:\coding\office\Kriti\front_kriti.mp4"
output_video_path = r"D:\coding\office\Kriti\output_result.mp4"
resize_dim = (1280, 720)
min_persistence_frames = 230  # ~5 seconds if 30 FPS

# === LOAD BACKGROUND ===
background = cv2.imread(background_image_path)
if background is None:
    print("❌ Error: Could not load background image.")
    exit()
background = cv2.resize(background, resize_dim)
gray_bg = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
gray_bg = cv2.GaussianBlur(gray_bg, (21, 21), 0)

# === VIDEO SETUP ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Error: Could not open video.")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_video_path, fourcc, fps, resize_dim)

# === TRACKING STORAGE ===
object_tracker = {}  # key = object ID, value = [x, y, w, h, frame_count]
next_object_id = 0

def get_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

# === PROCESS EACH FRAME ===
frame_index = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_index += 1
    frame = cv2.resize(frame, resize_dim)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    diff = cv2.absdiff(gray_bg, gray)
    _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    matched_ids = set()
    for cnt in contours:
        if cv2.contourArea(cnt) < 400:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        matched = False

        for object_id, data in object_tracker.items():
            old_x, old_y, old_w, old_h, count = data
            iou = get_iou((x, y, w, h), (old_x, old_y, old_w, old_h))
            if iou > 0.5:
                object_tracker[object_id] = [x, y, w, h, count + 1]
                matched_ids.add(object_id)
                matched = True
                break

        if not matched:
            object_tracker[next_object_id] = [x, y, w, h, 1]
            matched_ids.add(next_object_id)
            next_object_id += 1

    object_tracker = {obj_id: data for obj_id, data in object_tracker.items() if obj_id in matched_ids}

    # === DRAW DETECTIONS ===
    persistent_detected = False
    persistent_count = 0
    for obj_id, (x, y, w, h, count) in object_tracker.items():
        if count >= min_persistence_frames:
            persistent_detected = True
            persistent_count += 1

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Put "Unattended Object" label above the bounding box
            label = "Unattended Object"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            label_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
            label_w, label_h = label_size
            cv2.rectangle(frame, (x, y - label_h - 6), (x + label_w + 6, y), (0, 0, 255), -1)
            cv2.putText(frame, label, (x + 3, y - 5), font, font_scale, (255, 255, 255), thickness)

    # === DISPLAY COUNT ===
    if persistent_detected:
        count_text = f"Detected: {persistent_count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        text_size, _ = cv2.getTextSize(count_text, font, font_scale, font_thickness)
        text_w, text_h = text_size

        cv2.rectangle(frame, (10, 10), (10 + text_w + 10, 10 + text_h + 10), (0, 0, 255), -1)
        cv2.putText(frame, count_text, (15, 10 + text_h), font, font_scale, (255, 255, 255), font_thickness)

    # === DISPLAY ===
    cv2.imshow("Detection", frame)
    cv2.imshow("Background (Reference)", background)
    foreground = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow("Foreground (Detected Changes)", foreground)

    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === CLEANUP ===
cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Saved abandoned-object detection video:", output_video_path)

import cv2
import time
from datetime import datetime
import os
import numpy as np
from ultralytics import YOLO
import sqlite3


# --- CONFIG ---
#cam_url = "rtsp://admin:cctv%40123@192.168.8.208:554"
cam_url = r"D:\coding\office\face_rec\sample.mp4"

image_save_dir = "./images"
os.makedirs(image_save_dir, exist_ok=True)

# --- YOLO MODEL ---
yolo = YOLO("yolov8s.pt")  # model

last_save_time = time.time()
last_hash = None

frame_skip = 3
frame_count = 0
last_results = None


# server post 

conn = None
cursor = None

device_data = {
    "packetType" : "CD",
    "version" : "1.0.001",    
    "timeStamp": "000000000000",
    "eventName": "X",
    "eventData": "X",
    "identifier": "X",
}

# Function to update timestamp and post to server
def update_timestamp_and_post_to_server(current_timestamp, packet, event, e_data, camera_id):
    print(f"Updating timestamp and posting to server with status: {event} and data: {e_data}")
    device_data["timeStamp"] = str(current_timestamp)
    device_data["packetType"] = packet

    if event is None and e_data is None:
        device_data.pop("eventName", None)
        device_data.pop("eventData", None)
    else:
        device_data["eventName"] = event
        device_data["eventData"] = e_data

    device_data["identifier"] = camera_id
    print(f"Device data prepared for posting: {device_data}")
    cursor.execute("INSERT INTO data (content) VALUES (?)",(str(device_data),))
    conn.commit()
    

def Connect_to_db():
    global conn
    global cursor
    conn = sqlite3.connect("shared.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name 
        FROM sqlite_master 
        WHERE type='table' AND name=?;
    """, ('data',))
    result = cursor.fetchone()

    if result == None:    
	    cursor.execute("""
		CREATE TABLE IF NOT EXISTS data (
		    id INTEGER PRIMARY KEY AUTOINCREMENT,
		    content TEXT NOT NULL
		)
		""")    


def hamming_distance(hash1, hash2):
    return np.count_nonzero(hash1 != hash2)

def connect_camera():
    """Try to connect to camera, return None if failed"""
    try:
        cap = cv2.VideoCapture(cam_url)
        if cap.isOpened():
            print("✅ Connected to camera.")
            return cap
        else:
            cap.release()
            return None
    except Exception as e:
        print(f"⚠️ Camera connection error: {e}")
        return None

def create_no_signal_frame():
    """Create a frame to display when camera is not available"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add "NO SIGNAL" text
    text = "NO CAMERA SIGNAL"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    color = (0, 0, 255)  # Red
    thickness = 3
    
    # Get text size to center it
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (640 - text_width) // 2
    y = (480 + text_height) // 2
    
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    # Add timestamp
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add reconnection message
    reconnect_msg = "Attempting to reconnect..."
    cv2.putText(frame, reconnect_msg, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return frame

# Create fullscreen window
window_name = "Person Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# --- MAIN LOOP (NEVER EXITS) ---
cap = None


last_connection_attempt = 0
connection_retry_interval = 5  # seconds
last_person_count = None

Connect_to_db()
print("🔄 Starting persistent camera monitoring. Press 'q' to quit.")

while True:
    current_time = time.time()
    # Try to connect/reconnect to camera if needed
    if cap is None or not cap.isOpened():
        if current_time - last_connection_attempt >= connection_retry_interval:
            if cap is not None:
                cap.release()
            cap = connect_camera()
            last_connection_attempt = current_time
            
        if cap is None or not cap.isOpened():
            # Show "NO SIGNAL" frame
            no_signal_frame = create_no_signal_frame()
            cv2.imshow(window_name, no_signal_frame)
            
            if cv2.waitKey(1000) & 0xFF == ord('q'):  # Check for 'q' every second
                break
            continue

    # Try to read frame
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame, will attempt reconnection...")
        if cap is not None:
            cap.release()
            cap = None
        continue

    frame_count += 1

    # --- Run YOLO detection ---
    try:
        if frame_count % (frame_skip + 1) == 0 or last_results is None:
            results = yolo.predict(
                frame,
                imgsz=320,
                conf=0.5,
                iou=0.45,
                classes=[0],  # only persons
                verbose=False
            )
            last_results = results
        else:
            results = last_results

        if results is None or len(results) == 0:
            annotated_frame = frame.copy()
            person_count = 0
            boxes = None
        else:
            annotated_frame = frame.copy()
            boxes = results[0].boxes
            person_count = len(boxes) if boxes is not None else 0

        # === TOP-RIGHT PERSON COUNT LABEL ===
        label_text = f"No of persons in view: {person_count}"
        (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        top_right_x = annotated_frame.shape[1] - label_w - 20
        top_right_y = 40

        # draw filled blue background rectangle
        cv2.rectangle(
            annotated_frame,
            (top_right_x - 5, top_right_y - label_h - 5),
            (top_right_x + label_w + 5, top_right_y + 5),
            (255, 0, 0),  # blue
            -1
        )

        # draw white text on top
        cv2.putText(
            annotated_frame,
            label_text,
            (top_right_x, top_right_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),  # white
            2,
            cv2.LINE_AA
        )

        # === DRAW DETECTION BOXES WITH CONF SCORE ===
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])  # confidence score
                cls_id = int(box.cls[0])   # class id
                label = results[0].names[cls_id]  # get class name
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = label.title()
                conf_text = f"{label} {conf:.2f}"
                (conf_w, conf_h), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

                # background rectangle for conf text
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - conf_h - 6),
                    (x1 + conf_w + 6, y1),
                    (255, 0, 0),
                    -1
                )

                # put confidence text
                cv2.putText(
                    annotated_frame,
                    conf_text,
                    (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

        # --- Save image every 5s if unique ---
        if person_count != last_person_count and person_count != 0:  
            if (time.time() - last_save_time >= 5):
                current_timestamp = datetime.now().strftime("%H%M%S%d%m%y")
                device_data["timeStamp"] = str(current_timestamp)
                device_data["eventName"] = "PC"
                device_data["eventData"] = person_count

                image_name = f"{device_data['eventName']}_{device_data['eventData']}_{current_timestamp}.jpg"
                image_path = os.path.join(image_save_dir, image_name)

                print(f"Saving image with name: {image_name}")
                cv2.imwrite(image_path, annotated_frame)
                print(f"💾 Saved: {image_name}")
                update_timestamp_and_post_to_server(current_timestamp, "CD", "PC", person_count, "iPhone")
                last_save_time = time.time()
            else:
                print("⚡ Skipped: count changed but interval not reached or count=0")

        last_person_count = person_count

        # Display fullscreen
        cv2.imshow(window_name, annotated_frame)
        # Only exit if user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"⚠️ Error during processing: {e}")
        # Continue running even if there's an error

# Cleanup
if cap is not None:
    cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np

# === INPUT CONFIG ===
background_image_path = r"D:\coding\office\abounded\cleaned-room.png"
current_image_path = r"D:\coding\office\abounded\uncleaned-room.png"
resize_dim = (1536, 1024)

# === LOAD IMAGES ===
background = cv2.imread(background_image_path)
current = cv2.imread(current_image_path)

if background is None or current is None:
    print("❌ Error: Could not load images.")
    exit()

# === Resize and Convert to Grayscale ===
background = cv2.resize(background, resize_dim)
current = cv2.resize(current, resize_dim)

gray_bg = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
gray_curr = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

# === Blur to reduce noise ===
gray_bg = cv2.GaussianBlur(gray_bg, (21, 21), 0)
gray_curr = cv2.GaussianBlur(gray_curr, (21, 21), 0)

# === Compute Absolute Difference ===
diff = cv2.absdiff(gray_bg, gray_curr)
_, mask = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

# === Morphological operations to clean noise ===
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# === Find Contours ===
contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# === Draw Detected Objects ===
output_img = current.copy()
object_found = False
for cnt in contours:
    if cv2.contourArea(cnt) < 300:
        continue
    x, y, w, h = cv2.boundingRect(cnt)
    object_found = True
    cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

# === Label if object found ===
if object_found:
    label_text = "Unclean"
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Larger font scale and thickness
    font_scale = 1.2
    font_thickness = 3

    # Get size of the text
    text_size, _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
    text_w, text_h = text_size

    # Increase padding
    box_x, box_y = 10, 10
    padding = 20

    # Draw larger red rectangle
    cv2.rectangle(output_img, 
                  (box_x, box_y), 
                  (box_x + text_w + padding, box_y + text_h + padding), 
                  (0, 0, 255), 
                  -1)

    # Put larger white text
    cv2.putText(output_img, 
                label_text, 
                (box_x + padding // 2, box_y + text_h + padding // 4), 
                font, 
                font_scale, 
                (255, 255, 255), 
                font_thickness)


# === Show and Save ===
cv2.imshow("Abandoned Object Detection", output_img)
cv2.imwrite(r"D:\coding\office\abounded\output_result_image.png", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("✅ Saved output image with detection box.")

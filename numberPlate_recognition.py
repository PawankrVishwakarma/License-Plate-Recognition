import os
from ultralytics import YOLO
import cv2
import numpy as np 
import easyocr
import re
from collections import defaultdict, deque

WEIGHTS_PATH = "OCR/License_Plate_Recognition/weights/license_best.pt"
if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(
        f"YOLO weights file not found: {WEIGHTS_PATH}. "
        "Place the file in the OCR folder or update WEIGHTS_PATH."
    )

model = YOLO(WEIGHTS_PATH)
reader = easyocr.Reader(['en'], gpu=True)

# Regex: 2 letter + 2 numbers + 3 letters
plate_pattern = re.compile(r"[A-Z]{2}[0-9]{2}[A-Z]{3}$")

def correct_plate_format(ocr_text):
    mapping_num_to_alpha = {"0":"O", "1":"I", "5":"S", "8":"B"}
    mapping_alpha_to_num = {"O":"0", "I":"1", "Z":"2", "S":"5", "B":"8"}

    ocr_text = ocr_text.upper().replace(" ", "")
    if len(ocr_text) != 7:
        return ""  # discard if wrong length

    corrected = []
    for i, ch in enumerate(ocr_text):
        if i < 2 or i >= 4:  # alphabet positions
            if ch.isdigit() and ch in mapping_num_to_alpha:
                corrected.append(mapping_num_to_alpha[ch])
            elif ch.isalpha():
                corrected.append(ch)
            else:
                return ""  # invalid char
        else:  # numeric positions
            if ch.isalpha() and ch in mapping_alpha_to_num:
                corrected.append(mapping_alpha_to_num[ch])
            elif ch.isdigit():
                corrected.append(ch)
            else:
                return ""  # invalid char
    return "".join(corrected)


def recognize_plate(plate_crop):
    if plate_crop is None or plate_crop.size == 0:
        return ""

    ocr_results = reader.readtext(plate_crop, detail=0, paragraph=False)
    if not ocr_results:
        return ""

    joined = "".join(ocr_results).upper()
    corrected = correct_plate_format(joined)
    if corrected:
        return corrected

    for item in ocr_results:
        candidate = correct_plate_format(item.upper())
        if candidate:
            return candidate

    return ""


# Buffer to store the last 10 predictions for each box ID
plate_history = defaultdict(lambda: deque(maxlen=10)) 
plate_final = {}

def get_box_id(x1, y1, x2, y2):
    # Use rounded coordinates as a pseudo ID
    return f"{int(x1/10)}_{int(y1/10)}_{int(x2/10)}_{int(y2/10)}"

def get_stable_plate(box_id, new_text):
    if new_text:
        plate_history[box_id].append(new_text)
        # Majority vote
        most_common = max(set(plate_history[box_id]), key=plate_history[box_id].count)
        plate_final[box_id] = most_common
    return plate_final.get(box_id, "")

# Video for inference
input_video = "OCR/License_Plate_Recognition/input-output/vehicle_video.mp4"
output_video = "OCR/License_Plate_Recognition/input-output/output_with_license.mp4"

cap = cv2.VideoCapture(input_video)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc,
                      cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(3)), int(cap.get(4))))

CONF_THRESH = 0.3

# Operating frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run object detection
    results = model(frame, verbose=False)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf.cpu().numpy())
            if conf < CONF_THRESH:
                continue
            
            # Extract coordinates and crop license plate
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
            plate_crop = frame[y1:y2, x1:x2]
            
            # OCR with correction
            text = recognize_plate(plate_crop)
            
            # Stabilize text using history
            box_id = get_box_id(x1, y1, x2, y2)
            stable_text = get_stable_plate(box_id, text)
            
            # Draw rectangle and overlay text
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Overlay zoomed-in plate above detected plate
            if plate_crop.size > 0:
                overlay_h, overlay_w = 150, 400
                plate_resized = cv2.resize(plate_crop, (overlay_w, overlay_h))

                oy1 = max(0, y1 - overlay_h - 40)
                ox1 = x1
                oy2, ox2 = oy1 + overlay_h, ox1 + overlay_w

                if oy2 <= frame.shape[0] and ox2 <= frame.shape[1]:
                    frame[oy1:oy2, ox1:ox2] = plate_resized
                    
                    # Show stabilized OCR text above overlay
                    if stable_text:
                        cv2.putText(frame, stable_text, (ox1, oy1 - 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 6)  # black outline
                        cv2.putText(frame, stable_text, (ox1, oy1 - 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)  # white text
                        
    out.write(frame)
    cv2.imshow("Annotated Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Annotated video saved as", output_video)
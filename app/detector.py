import cv2
import requests
from ultralytics import YOLO

# load model pcb
model = YOLO("best.pt")

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    detections = []

    for r in results:
        boxes = r.boxes

        if boxes is None:
            continue

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            name = model.names[cls]

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class": name,
                "confidence": conf,
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                        }
                            })

    
    if detections:
        try:
            requests.post(
                "http://127.0.0.1:8000/detections",
                json=detections
            )
            print("sent:", detections)

        except Exception as e:
            print("api error:", e)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
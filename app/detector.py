# import cv2
# import requests
# import base64
# import time
# from ultralytics import YOLO

# model = YOLO("best.pt")
# cap = cv2.VideoCapture(0)

# prev_time = 0

# # ===== counting =====
# line_y = 300  # vị trí line
# count = 0

# # lưu ID đã đếm rồi (tránh đếm trùng)
# counted_ids = set()

# while True:
#     start_time = time.time()

#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model(frame)

#     detections = []

#     for r in results:
#         boxes = r.boxes
#         if boxes is None:
#             continue

#         for box in boxes:
#             cls = int(box.cls[0])
#             conf = float(box.conf[0])
#             name = model.names[cls]

#             x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

#             detections.append({
#                 "class": name,
#                 "confidence": conf,
#                 "bbox": {
#                     "x1": x1,
#                     "y1": y1,
#                     "x2": x2,
#                     "y2": y2
#                 }
#             })

#             # 🔥 vẽ luôn bbox lên frame (để stream)
#             label = f"{name} {conf:.2f}"
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
#             cv2.putText(
#                 frame,
#                 label,
#                 (x1, y1 - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.5,
#                 (0,255,0),
#                 2
#             )

#     # ===== FPS =====
#     fps = 1 / (time.time() - start_time)

#     cv2.putText(
#         frame,
#         f"FPS: {fps:.2f}",
#         (10, 30),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1,
#         (0, 0, 255),
#         2
#     )

#     # ===== encode ảnh (giảm chất lượng để nhẹ hơn) =====
#     encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
#     _, buffer = cv2.imencode('.jpg', frame, encode_param)
#     img_base64 = base64.b64encode(buffer).decode('utf-8')

#     # ===== gửi API (giảm tần suất gửi để tránh lag) =====
#     try:
#         # chỉ gửi mỗi ~100ms (~10 FPS network)
#         if time.time() - prev_time > 0.1:
#             requests.post(
#                 "http://127.0.0.1:8000/detections",
#                 json={
#                     "image": img_base64,
#                     "detections": detections,
#                     "fps": fps
#                 },
#                 timeout=0.2
#             )
#             prev_time = time.time()

#             print(f"sent | FPS: {fps:.2f}")

#     except Exception as e:
#         print("api error:", e)

#     # ===== debug local =====
#     cv2.imshow("Detector", frame)

#     if cv2.waitKey(1) == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import requests
import base64
import time
from ultralytics import YOLO

# ===== model =====
model = YOLO("best.pt")

cap = cv2.VideoCapture(0)

# ===== counting =====
line_y = 1000  # vị trí line
count = 0

# lưu ID đã đếm rồi (tránh đếm trùng)
counted_ids = set()

prev_time = 0

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # ===== TRACKING (quan trọng) =====
    results = model.track(frame, persist=True)

    detections = []

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            # ===== tracking ID =====
            track_id = int(box.id[0]) if box.id is not None else -1

            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # ===== center point =====
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # ===== COUNTING LOGIC =====
            if track_id not in counted_ids:
                if cy > line_y:   # đi qua line
                    count += 1
                    counted_ids.add(track_id)

            detections.append({
                "id": track_id,
                "class": name,
                "confidence": conf,
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }
            })

            # ===== vẽ bbox + ID =====
            label = f"ID:{track_id} {name} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                2
            )

            # vẽ center
            cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)

    # ===== vẽ line =====
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (255,0,0), 2)

    # ===== hiển thị count =====
    cv2.putText(
        frame,
        f"Count: {count}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        3
    )

    # ===== FPS =====
    fps = 1 / (time.time() - start_time)
    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    # ===== encode ảnh =====
    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # ===== gửi API =====
    try:
        if time.time() - prev_time > 0.1:
            requests.post(
                "http://127.0.0.1:8000/detections",
                json={
                    "image": img_base64,
                    "detections": detections,
                    "count": count,
                    "fps": fps
                },
                timeout=0.2
            )
            prev_time = time.time()

    except Exception as e:
        print("api error:", e)

    cv2.imshow("Tracking + Counting", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
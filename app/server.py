from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
import threading
import base64
import numpy as np

app = FastAPI()

latest_frame = None
latest_detections = []
lock = threading.Lock()

@app.get("/")
def root():
    return {"msg": "AI vision system running"}

@app.post("/detections")
def add_detection(data: dict):
    global latest_frame, latest_detections

    with lock:
        latest_frame = data.get("image")  # base64
        latest_detections = data.get("detections", [])

    return {"status": "ok"}

@app.get("/detections")
def get_detections():
    return latest_detections


def generate_frames():
    global latest_frame

    while True:
        if latest_frame is None:
            continue

        with lock:
            img_bytes = base64.b64decode(latest_frame)

        frame = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
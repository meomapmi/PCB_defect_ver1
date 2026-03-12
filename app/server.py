# from fastapi import FastAPI
# from typing import List

# app = FastAPI()

# # lưu kết quả detection mới nhất
# latest_result = []

# @app.get("/")
# def root():
#     return {"msg": "AI vision system running"}

# @app.post("/detections")
# def add_detection(data: List[dict]):
#     global latest_result
#     latest_result = data
#     return {"status": "ok"}

# @app.get("/detections")
# def get_detections():
#     return latest_result

from fastapi import FastAPI
from typing import List

app = FastAPI()

latest_result = []

@app.get("/")
def root():
    return {"msg": "AI vision system running"}

@app.post("/detections")
def add_detection(data: List[dict]):

    global latest_result

    if len(data) > 0:
        latest_result.extend(data)

        if len(latest_result) > 200:
            latest_result = latest_result[-200:]

    return {"status": "ok"}

@app.get("/detections")
def get_detections():
    return latest_result
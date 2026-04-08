import streamlit as st
import requests
import time
import pandas as pd

st.title("🔥 AI Vision Realtime Dashboard")

# ===== CONTROL =====
threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# ===== VIDEO STREAM =====
st.markdown("### 📹 Live Stream")
st.image("http://127.0.0.1:8000/video_feed")

# ===== DATA =====
placeholder = st.empty()
chart_placeholder = st.empty()

history = []

while True:
    try:
        r = requests.get("http://127.0.0.1:8000/detections")
        detections = r.json()

        # filter theo threshold
        filtered = [
            d for d in detections if d["confidence"] >= threshold
        ]

        # hiển thị json
        placeholder.json(filtered)

        # lưu history
        history.append(len(filtered))
        if len(history) > 50:
            history = history[-50:]

        # chart
        df = pd.DataFrame(history, columns=["detections"])
        chart_placeholder.line_chart(df)

        time.sleep(1)

    except:
        st.write("API error")
        time.sleep(1)
import streamlit as st
import requests
from streamlit_autorefresh import st_autorefresh

st.title("AI Vision Dashboard")

st_autorefresh(interval=1000, key="datarefresh")

try:
    r = requests.get("http://127.0.0.1:8000/detections")
    data = r.json()
    st.json(data)

except:
    st.write("cannot connect to api")
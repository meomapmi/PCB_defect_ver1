# PCB_defect_ver1
Real-time pcb defect detection system using yolo, fastapi, and streamlit

This project implements a real-time PCB defect detection system using YOLOv8.
The system captures images from a camera, detects PCB defects, sends the detection results to a FastAPI server, and visualizes them on a Streamlit dashboard.

Main components:
- YOLOv8 for defect detection
- FastAPI for backend API
- Streamlit for monitoring dashboard
## System architecture
Camera → YOLOv8 Detector → FastAPI Server → Streamlit Dashboard
## Project structure

PCB_defect_ver1/
│
├── app/
│   ├── detector.py
│   └── server.py
│
├── dashboard/
│   └── dashboard.py
│
├── model/
│   └── train.py
│
├── requirements.txt
└── test_camera.py
## Installation
Clone the repository: git clone https://github.com/meomapmi/PCB_defect_ver1.git
cd PCB_defect_ver1
Install dependencies: pip install -r requirements.txt
## Run the system
Start the API server: uvicorn app.server:app --reload

Start the detector: python app/detector.py

Start the dashboard: streamlit run dashboard/dashboard.py
## Future Improvements
- save detection results to database
- display detected images on dashboard
- support multiple cameras
- deploy system with Docker